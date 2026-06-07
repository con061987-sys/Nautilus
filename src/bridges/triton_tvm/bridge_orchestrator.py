"""Triton ↔ TVM MetaSchedule bridge orchestrator.

Coordinates the full pipeline: intercept Triton kernel → extract metadata →
construct TIR template → run TVM MetaSchedule → map config → recompile.

Production features:
  - Multi-level cache (in-memory LRU → disk → remote Redis optional)
  - Circuit breaker pattern (per-dependency failure isolation)
  - Stage-level timeouts (never a global one)
  - Structured logging with stage and span tracking
  - Graceful degradation chain (L0-L5 fallback tiers)
  - IR dump on error (ring buffer, not firehose)
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from src.bridges.triton_tvm.config_cache import ConfigCache
from src.bridges.triton_tvm.config_mapper import ConfigMapper, MappedTuningConfig
from src.bridges.triton_tvm.metadata_extractor import (
    KernelMetadata,
    MetadataExtractor,
)
from src.bridges.triton_tvm.metaschedule_adapter import MetaScheduleAdapter
from src.bridges.triton_tvm.timeout_manager import StageTimeoutError
from src.common.errors import TuningError
from src.common.logging import get_logger
from src.common.result import Err, Ok, Result

try:
    __import__("tvm.meta_schedule")

    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False


TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None

logger = get_logger(__name__)


class FallbackTier(Enum):
    """Tuning fallback tiers. Higher number = worse but more reliable."""

    L0_TVM_DB_HIT = auto()  # Best: existing TVM database record
    L1_TVM_WARM_START = auto()  # TVM warm-starts from similar config
    L2_TVM_COLD = auto()  # Full TVM tuning (cold start)
    L3_DISK_CACHE = auto()  # Previously cached Triton config
    L4_TRITON_DEFAULT = auto()  # Triton's built-in defaults
    L5_SAFE_FALLBACK = auto()  # Conservative pre-vetted config


@dataclass
class TuningResult:
    """Result of the full bridge pipeline."""

    config: MappedTuningConfig
    fallback_tier: FallbackTier
    total_duration_ms: float
    cache_hit: bool = False
    error: str | None = None
    stages: dict[str, float] = field(default_factory=dict)  # stage → duration_ms
    # True when the real-IR path failed and a synthetic fallback ran.
    # fallback_reason carries the M/N/K mismatch (expected vs fallback)
    # so callers can tell when the config came from placeholder bounds.
    fallback_used: bool = False
    fallback_reason: str | None = None


class TritonTVMBridge:
    """Main orchestrator coordinating the Triton ↔ TVM bridge pipeline.

    Usage:
        bridge = TritonTVMBridge()
        result = bridge.tune(
            kernel_fn=my_matmul_kernel,
            grid=(lambda meta: (triton.cdiv(M, meta['BLOCK']),)),
            example_args=(x, y),
            target="nvidia/nvidia-a100",
        )
        # result.config.to_triton_config() → triton.Config
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        max_trials: int = 64,
        enable_cache: bool = True,
        enable_tvm: bool = True,
    ) -> None:
        self.cache_dir = Path(
            cache_dir
            or os.environ.get(
                "NVINDIACUD_CACHE_DIR",
                str(Path.home() / ".cache" / "nvindia_cud"),
            )
        )
        self.max_trials = max_trials
        self.enable_cache = enable_cache
        self.enable_tvm = enable_tvm and TVM_AVAILABLE

        self.extractor = MetadataExtractor()
        self.config_mapper = ConfigMapper()
        self.tvm_adapter = (
            MetaScheduleAdapter(
                cache_dir=str(self.cache_dir),
                default_max_trials=max_trials,
                default_num_trials_per_iter=16,
                timeout_seconds=600,
            )
            if self.enable_tvm
            else None
        )

        # Real IR capture infrastructure (Step 1 of the production wiring)
        # These are the components that hook into Triton's actual compile
        # pipeline via the backend plugin and capture REAL TTGIR.
        from .circuit_breaker import get_default_breakers
        from .extern_bridge import ExternMatmulBuilder
        from .ir_capture import IRCapture
        from .timeout_manager import StageBudgets, TimeoutManager

        self.ir_capture = IRCapture()
        self.extern_builder = ExternMatmulBuilder(
            cache_dir=str(self.cache_dir / "extern_cache"),
        )
        self.breakers = get_default_breakers()
        self.timeout_manager = TimeoutManager(StageBudgets())

        # In-memory LRU cache
        self._cache: dict[str, MappedTuningConfig] = {}
        self._lru_order: list[str] = []
        self._max_cache_entries = 256

        # Persistent config cache (separate from the in-memory + disk
        # bridge cache above).  Keyed by sha256(IR_hash + vendor + arch)
        # so kernel edits invalidate automatically.
        self.config_cache = ConfigCache()

        # Stage timing
        self._stages: dict[str, float] = {}

    def tune(
        self,
        kernel_fn: Any,
        grid: tuple[int, int, int] | Callable,
        example_args: tuple[Any, ...],
        target: str = "nvidia/nvidia-a100",
        num_warps: int = 4,
        num_stages: int = 3,
        num_ctas: int = 1,
        force_retune: bool = False,
    ) -> TuningResult:
        """Run the full bridge pipeline on a Triton kernel.

        Args:
            kernel_fn: @triton.jit decorated function.
            grid: Grid tuple or lambda. If lambda, called with defaults
                to get grid dimensions.
            example_args: Example tensor arguments for shape extraction.
            target: TVM target string.
            num_warps: Initial num_warps guess.
            num_stages: Initial num_stages guess.
            num_ctas: Initial num_ctas guess.
            force_retune: If True, bypass cache and re-tune.

        Returns:
            TuningResult with optimal config and metadata.
        """
        start = time.perf_counter()
        self._stages = {}

        # Step 1: Extract metadata
        t0 = time.perf_counter()
        metadata = self.extractor.extract_from_call(
            kernel_fn=kernel_fn,
            grid=self._resolve_grid(grid),
            args=example_args,
            num_warps=num_warps,
            num_stages=num_stages,
            num_ctas=num_ctas,
        )
        self._stages["extract"] = (time.perf_counter() - t0) * 1000

        # Step 2: Check cache (in-memory + bridge disk + ConfigCache)
        if self.enable_cache and not force_retune:
            t0 = time.perf_counter()
            cached = self._get_cached(metadata.cache_key, target)
            if cached is not None:
                elapsed = (time.perf_counter() - start) * 1000
                return TuningResult(
                    config=cached,
                    fallback_tier=FallbackTier.L3_DISK_CACHE,
                    total_duration_ms=elapsed,
                    cache_hit=True,
                    stages=self._stages,
                )

            # Step 2b: Check persistent ConfigCache
            vendor, arch = self._vendor_arch_from_target(target)
            cc_cached = self.config_cache.get(metadata.cache_key, vendor, arch)
            if cc_cached is not None:
                config = MappedTuningConfig(**cc_cached)
                self._set_cache(metadata.cache_key, target, config)
                elapsed = (time.perf_counter() - start) * 1000
                return TuningResult(
                    config=config,
                    fallback_tier=FallbackTier.L3_DISK_CACHE,
                    total_duration_ms=elapsed,
                    cache_hit=True,
                    stages=self._stages,
                )

        # Step 3-5: TVM tuning chain
        tune_result = self._tuning_chain(metadata, target)
        if isinstance(tune_result, Ok):
            mapped = tune_result.unwrap()
        else:
            mapped = self._fallback_config(
                metadata,
                FallbackTier.L4_TRITON_DEFAULT,
                error=tune_result.error.message,
            )

        # Cache the result (bridge cache + ConfigCache)
        if self.enable_cache:
            self._set_cache(metadata.cache_key, target, mapped)
            vendor, arch = self._vendor_arch_from_target(target)
            self.config_cache.set(metadata.cache_key, vendor, arch, mapped.__dict__)

        elapsed = (time.perf_counter() - start) * 1000
        return TuningResult(
            config=mapped,
            fallback_tier=FallbackTier.L0_TVM_DB_HIT,
            total_duration_ms=elapsed,
            stages=self._stages,
        )

    def tune_configs_list(
        self,
        metadata: KernelMetadata,
        target: str = "nvidia/nvidia-a100",
    ) -> list[MappedTuningConfig]:
        """Return a configs list suitable for @triton.autotune.

        This is an alternative to tune() — instead of picking the single
        best config, you get a list for the @triton.autotune decorator
        so Triton can benchmark them at runtime.
        """
        tune_result = self._tuning_chain(metadata, target)
        if isinstance(tune_result, Ok):
            best = tune_result.unwrap()
        else:
            best = self._fallback_config(
                metadata,
                FallbackTier.L4_TRITON_DEFAULT,
                error=tune_result.error.message,
            )
        # Generate variant configs around the best
        variants = [
            best,
            MappedTuningConfig(
                block_m=best.block_m,
                block_n=best.block_n,
                block_k=max(best.block_k // 2, 16),
                num_warps=best.num_warps,
                num_stages=best.num_stages,
            ),
            MappedTuningConfig(
                block_m=best.block_m * 2,
                block_n=best.block_n,
                block_k=best.block_k,
                num_warps=min(best.num_warps * 2, 32),
                num_stages=best.num_stages,
            ),
        ]
        return variants

    def tune_with_real_ir(
        self,
        source_hash: str,
        target: str = "nvidia/nvidia-a100",
        force_retune: bool = False,
    ) -> TuningResult:
        """End-to-end tuning using REAL IR captured from Triton's pipeline.

        This is the production path. Unlike tune() which uses Python-level
        metadata, this method:
          1. Reads the actual TTGIR captured by the Triton backend plugin
          2. Classifies the kernel (matmul/reduction/elementwise/attention)
          3. Extracts mathematical bounds (M, N, K) from the REAL IR
          4. For matmul: routes through the extern_bridge to keep Tensor
             Core performance while MetaSchedule tunes the rest
          5. For other kinds: builds exact TIR from extracted bounds
          6. Runs TVM MetaSchedule on the constructed TIR
          7. Maps results back to Triton Config
          8. Returns the full result with observability metadata

        Args:
            source_hash: The kernel source hash from Triton's compile.
                Must match the hash captured by the backend plugin.
            target: TVM target string.
            force_retune: If True, bypass cache and re-tune.

        Returns:
            TuningResult with optimal config and full pipeline metadata.
        """
        from .ir_capture import KernelKind
        from .structured_logging import span as span_ctx
        from .structured_logging import stage as stage_ctx

        start = time.perf_counter()
        self._stages = {}

        with span_ctx(source_hash, target, metadata={"mode": "real_ir"}) as sp:
            # Stage 1: Real IR capture
            with stage_ctx(sp, "ir_capture") as st:
                captured = self.ir_capture.capture_for_source(source_hash, target)
                if captured is None:
                    synth_metadata = self._synthesize_metadata(source_hash, target)
                    logger.error(
                        "REAL_IR_CAPTURE_FAILED: no IR captured for source=%s "
                        "target=%s; falling back to synthetic metadata with "
                        "grid=(%d,%d,%d), num_warps=%d, num_stages=%d. "
                        "Returned TuningResult.fallback_used=True.",
                        source_hash[:12],
                        target,
                        synth_metadata.grid_0,
                        synth_metadata.grid_1,
                        synth_metadata.grid_2,
                        synth_metadata.num_warps,
                        synth_metadata.num_stages,
                    )
                    return self._fallback_tune_chain(
                        synth_metadata,
                        target,
                        start,
                        fallback_reason=(
                            f"real_ir_capture_failed: no IR for source="
                            f"{source_hash[:12]} target={target}; "
                            f"using grid=(1,1,1) placeholder"
                        ),
                    )
                st.metadata["kind"] = captured.kind.name
                st.metadata["ops"] = len(captured.ops_seen)
                self._stages["ir_capture"] = st.duration_ms

            # Stage 2: Cache check (bridge cache + ConfigCache)
            cache_key = captured.cache_key
            if self.enable_cache and not force_retune:
                with stage_ctx(sp, "cache_check") as st:
                    cached = self._get_cached(cache_key, target)
                    if cached is not None:
                        elapsed = (time.perf_counter() - start) * 1000
                        st.metadata["cache_hit"] = True
                        return TuningResult(
                            config=cached,
                            fallback_tier=FallbackTier.L3_DISK_CACHE,
                            total_duration_ms=elapsed,
                            cache_hit=True,
                            stages=self._stages,
                        )

                    # Stage 2b: Persistent ConfigCache check
                    vendor, arch = self._vendor_arch_from_target(target)
                    cc_cached = self.config_cache.get(cache_key, vendor, arch)
                    if cc_cached is not None:
                        config = MappedTuningConfig(**cc_cached)
                        self._set_cache(cache_key, target, config)
                        elapsed = (time.perf_counter() - start) * 1000
                        st.metadata["cache_hit"] = True
                        return TuningResult(
                            config=config,
                            fallback_tier=FallbackTier.L3_DISK_CACHE,
                            total_duration_ms=elapsed,
                            cache_hit=True,
                            stages=self._stages,
                        )

                    st.metadata["cache_hit"] = False

            # Stage 3: Build TIR template from REAL bounds
            with stage_ctx(sp, "build_tir") as st:
                tir_mod, fallback_reason = self._build_tir_from_captured(
                    captured,
                    target,
                )
                st.metadata["built"] = tir_mod is not None
                st.metadata["fallback_used"] = tir_mod is None
                if tir_mod is None:
                    st.metadata["fallback_reason"] = fallback_reason
                self._stages["build_tir"] = st.duration_ms

            if tir_mod is None:
                # Real-IR and template paths both failed. Surface the
                # M/N/K mismatch in logs and propagate the flag so the
                # returned TuningResult is not silently degraded.
                logger.warning(
                    "TIR build failed for %s..%s: %s; falling back to synthetic metadata "
                    "(expected M=%s N=%s K=%s, fallback uses grid=(1,1,1))",
                    source_hash[:12],
                    target,
                    fallback_reason,
                    captured.bounds.m,
                    captured.bounds.n,
                    captured.bounds.k,
                )
                return self._fallback_tune_chain(
                    self._synthesize_metadata(source_hash, target),
                    target,
                    start,
                    fallback_reason=fallback_reason,
                )

            # Stage 4: Run MetaSchedule (with circuit breaker)
            with stage_ctx(sp, "tvm_tune") as st:
                mapped = self._tune_with_breaker(
                    tir_mod=tir_mod,
                    target_str=target,
                    cache_key=cache_key,
                )
                st.metadata["mapped"] = mapped != MappedTuningConfig.defaults()
                self._stages["tvm_tune"] = st.duration_ms

            # Stage 5: For matmul, route through extern_bridge to preserve
            # Tensor Core performance (the hard part of the bridge)
            if captured.kind in (KernelKind.MATMUL, KernelKind.ATTENTION):
                with stage_ctx(sp, "extern_bridge") as st:
                    self._handle_matmul_extern(captured, target, mapped)
                    st.metadata["handled"] = True
                    self._stages["extern_bridge"] = st.duration_ms

            # Stage 6: Cache the result (bridge cache + ConfigCache)
            if self.enable_cache:
                self._set_cache(cache_key, target, mapped)
                vendor, arch = self._vendor_arch_from_target(target)
                self.config_cache.set(cache_key, vendor, arch, mapped.__dict__)

            elapsed = (time.perf_counter() - start) * 1000
            self.timeout_manager.check_total_budget()
            return TuningResult(
                config=mapped,
                fallback_tier=FallbackTier.L0_TVM_DB_HIT,
                total_duration_ms=elapsed,
                stages=self._stages,
            )

    def _tune_with_breaker(
        self,
        tir_mod: Any,
        target_str: str,
        cache_key: str,
    ) -> MappedTuningConfig:
        """Run TVM MetaSchedule through the tvm_tune circuit breaker.

        The adapter now returns :class:`Result`; on ``Err`` we degrade
        to :meth:`MappedTuningConfig.defaults`. The circuit breaker is
        still in the loop to short-circuit on transport / OOM / segfault
        style failures that raise hard exceptions.
        """
        if not (self.tvm_adapter and self.enable_tvm):
            return MappedTuningConfig.defaults()

        breaker = self.breakers["tvm_tune"]
        try:
            result: Result[MappedTuningConfig, TuningError] = breaker.call(
                self.tvm_adapter.tune,
                tir_mod=tir_mod,
                target_str=target_str,
                max_trials=self.max_trials,
                cache_key=cache_key,
            )
        except StageTimeoutError as exc:
            logger.warning(
                "TVM tune timed out (breaker=%s budget=%.1fs): %s",
                breaker.state.name,
                exc.budget_s,
                exc,
            )
            return MappedTuningConfig.defaults()
        except (ValueError, TypeError) as exc:
            logger.warning(
                "TVM tune rejected input (breaker=%s): %s",
                breaker.state.name,
                exc,
            )
            return MappedTuningConfig.defaults()
        except ImportError as exc:
            logger.warning(
                "TVM tune: missing dependency (breaker=%s): %s",
                breaker.state.name,
                exc,
            )
            return MappedTuningConfig.defaults()
        except OSError as exc:
            logger.warning(
                "TVM tune: filesystem error (breaker=%s): %s",
                breaker.state.name,
                exc,
            )
            return MappedTuningConfig.defaults()
        except RuntimeError as exc:
            logger.warning(
                "TVM tune: runtime error (breaker=%s): %s",
                breaker.state.name,
                exc,
            )
            return MappedTuningConfig.defaults()

        if isinstance(result, Ok):
            return result.unwrap()
        logger.warning(
            "TVM tune returned Err (breaker=%s): %s",
            breaker.state.name,
            result.error.message,
        )
        return MappedTuningConfig.defaults()

    def _build_tir_from_captured(
        self,
        captured: Any,
        target: str,
    ) -> tuple[Any, str | None]:
        """Build a TIR module from REAL captured IR.

        Returns:
            (tir_mod, fallback_reason) — ``tir_mod`` is the IRModule or
            ``None`` if both the real-IR and template paths failed;
            ``fallback_reason`` is a human-readable string describing
            the failure when ``tir_mod`` is ``None``.
        """
        from .ir_capture import KernelKind
        from .tir_template import TIRTemplateBuilder

        real_ir_attempted = False
        real_ir_failed_reason: str | None = None
        real_ir_detected_bounds: tuple[int | None, int | None, int | None] | None = None

        if captured.ir_text and len(captured.ir_text) > 100:
            real_ir_attempted = True
            try:
                builder = TIRTemplateBuilder()
                ir_module, conv_result = builder.build_from_captured_ir(
                    captured.ir_text,
                )
                if ir_module is not None:
                    logger.info(
                        "Real IR conversion succeeded: status=%s, has_dot=%s, M=%s N=%s K=%s",
                        conv_result.status.name,
                        conv_result.has_dot_split,
                        captured.bounds.m,
                        captured.bounds.n,
                        captured.bounds.k,
                    )
                    return ir_module, None
                real_ir_failed_reason = (
                    f"conversion returned status={conv_result.status.name}, "
                    f"error={conv_result.error}"
                )
                logger.warning(
                    "Real IR conversion returned no IRModule (%s); "
                    "falling back to template with M=%s N=%s K=%s dtype=%s",
                    real_ir_failed_reason,
                    captured.bounds.m,
                    captured.bounds.n,
                    captured.bounds.k,
                    captured.bounds.data_dtype,
                )
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                real_ir_failed_reason = f"raised {type(exc).__name__}: {exc}"
                logger.warning(
                    "Real IR conversion raised %s: %s; "
                    "falling back to template with M=%s N=%s K=%s dtype=%s",
                    type(exc).__name__,
                    exc,
                    captured.bounds.m,
                    captured.bounds.n,
                    captured.bounds.k,
                    captured.bounds.data_dtype,
                )
            except ImportError as exc:
                real_ir_failed_reason = f"missing dependency: {exc}"
                logger.warning(
                    "Real IR conversion: missing dependency %s; "
                    "falling back to template with M=%s N=%s K=%s",
                    exc,
                    captured.bounds.m,
                    captured.bounds.n,
                    captured.bounds.k,
                )

        try:
            builder = TIRTemplateBuilder()
            if captured.kind in (KernelKind.MATMUL, KernelKind.ATTENTION):
                if captured.bounds.m and captured.bounds.n and captured.bounds.k:
                    template_bounds = (
                        captured.bounds.m,
                        captured.bounds.n,
                        captured.bounds.k,
                    )
                    if real_ir_attempted:
                        logger.info(
                            "TEMPLATE_FALLBACK_BUILD: kind=%s M=%d N=%d K=%d "
                            "dtype=%s (real_ir_failed=%s)",
                            captured.kind.name,
                            template_bounds[0],
                            template_bounds[1],
                            template_bounds[2],
                            captured.bounds.data_dtype,
                            real_ir_failed_reason or "unknown",
                        )
                        if real_ir_detected_bounds is not None:
                            exp = real_ir_detected_bounds
                            if (exp[0], exp[1], exp[2]) != template_bounds:
                                logger.error(
                                    "MNK_DISCREPANCY: template fallback built "
                                    "with M=%d N=%d K=%d but real-IR analysis "
                                    "had M=%s N=%s K=%s",
                                    template_bounds[0],
                                    template_bounds[1],
                                    template_bounds[2],
                                    exp[0],
                                    exp[1],
                                    exp[2],
                                )
                    ir_module = builder.build_matmul(
                        m=template_bounds[0],
                        n=template_bounds[1],
                        k=template_bounds[2],
                        dtype=captured.bounds.data_dtype,
                    )
                    return ir_module, None
                return None, (
                    f"matmul bounds incomplete: "
                    f"m={captured.bounds.m}, n={captured.bounds.n}, k={captured.bounds.k}"
                )
            if captured.kind == KernelKind.REDUCTION:
                if captured.bounds.reduce_size:
                    if real_ir_attempted:
                        logger.info(
                            "TEMPLATE_FALLBACK_BUILD: kind=REDUCTION "
                            "shape=(%d, %d) dtype=%s (real_ir_failed=%s)",
                            captured.bounds.keep_size or 1,
                            captured.bounds.reduce_size,
                            captured.bounds.data_dtype,
                            real_ir_failed_reason or "unknown",
                        )
                    ir_module = builder.build_reduction(
                        shape=(captured.bounds.keep_size or 1, captured.bounds.reduce_size),
                        dtype=captured.bounds.data_dtype,
                    )
                    return ir_module, None
                return None, "reduction bounds missing reduce_size"
            if captured.kind == KernelKind.ELEMENTWISE:
                total = captured.bounds.total_elements or 1024
                if real_ir_attempted:
                    logger.info(
                        "TEMPLATE_FALLBACK_BUILD: kind=ELEMENTWISE "
                        "total=%d dtype=%s (real_ir_failed=%s)",
                        total,
                        captured.bounds.data_dtype,
                        real_ir_failed_reason or "unknown",
                    )
                ir_module = builder.build_elementwise(
                    shape=(total,),
                    dtype=captured.bounds.data_dtype,
                )
                return ir_module, None
            return None, f"unsupported kernel kind: {captured.kind.name}"
        except (ValueError, TypeError) as exc:
            return None, f"TIR template build rejected input: {type(exc).__name__}: {exc}"
        except ImportError as exc:
            return None, f"TIR template build missing dependency: {exc}"
        except OSError as exc:
            return None, f"TIR template build filesystem error: {exc}"

    def _handle_matmul_extern(
        self,
        captured: Any,
        target: str,
        mapped: MappedTuningConfig,
    ) -> None:
        """For matmul kernels, compile via extern_bridge to preserve Tensor Cores.

        The extern_bridge:
          1. Compiles the matmul portion separately with Triton's normal compiler
             (which knows how to emit Tensor Core / MFMA / WGMMA instructions)
          2. Produces a binary (cubin / hsaco / spv depending on target)
          3. Wraps it as a tvm.extern call so the TIR can reference it
        """
        from .ir_capture import IRBounds

        if not (captured.bounds.m and captured.bounds.n and captured.bounds.k):
            logger.warning("Cannot build extern matmul: missing M/N/K bounds")
            return

        bounds = IRBounds(
            m=captured.bounds.m,
            n=captured.bounds.n,
            k=captured.bounds.k,
            data_dtype=captured.bounds.data_dtype,
        )

        # Map the target name to a backend string
        backend = self._target_to_backend(target)

        try:
            self.extern_builder.build_matmul(
                name=captured.source_hash[:12],
                bounds=bounds,
                target=backend,
                source_hash=captured.source_hash,
            )
        except (ValueError, TypeError) as exc:
            logger.warning(
                "Extern matmul build rejected input (backend=%s): %s",
                backend,
                exc,
            )
        except ImportError as exc:
            logger.warning(
                "Extern matmul build: missing dependency (backend=%s): %s",
                backend,
                exc,
            )
        except OSError as exc:
            logger.warning(
                "Extern matmul build: filesystem error (backend=%s): %s",
                backend,
                exc,
            )
        except RuntimeError as exc:
            logger.warning("Extern matmul build failed (backend=%s): %s", backend, exc)

    def _target_to_backend(self, target: str) -> str:
        """Map a TVM target string to a backend name for AOT compilation."""
        if "cuda" in target or "nvidia" in target or "hopper" in target or "h100" in target:
            return "cuda"
        if "rocm" in target or "amd" in target or "mi" in target:
            return "rocm"
        if "metal" in target or "apple" in target:
            return "metal"
        if "intel" in target or "spirv" in target or "gaudi" in target:
            return "intel"
        return "cuda"

    def _synthesize_metadata(
        self,
        source_hash: str,
        target: str,
    ) -> KernelMetadata:
        """Create a minimal KernelMetadata when real IR isn't available."""
        return KernelMetadata(
            kernel_name=f"synthesized_{source_hash[:8]}",
            source_hash=source_hash,
            grid_0=1,
            grid_1=1,
            grid_2=1,
            num_warps=4,
            num_stages=3,
            num_ctas=1,
        )

    def _fallback_tune_chain(
        self,
        metadata: KernelMetadata,
        target: str,
        start: float,
        fallback_reason: str | None = None,
    ) -> TuningResult:
        """When real IR isn't available, fall back to the legacy metadata path."""
        cached = None
        if self.enable_cache:
            cached = self._get_cached(metadata.cache_key, target)
            if cached is None:
                vendor, arch = self._vendor_arch_from_target(target)
                cc_cached = self.config_cache.get(metadata.cache_key, vendor, arch)
                if cc_cached is not None:
                    cached = MappedTuningConfig(**cc_cached)
                    self._set_cache(metadata.cache_key, target, cached)
        if cached is not None:
            elapsed = (time.perf_counter() - start) * 1000
            return TuningResult(
                config=cached,
                fallback_tier=FallbackTier.L3_DISK_CACHE,
                total_duration_ms=elapsed,
                cache_hit=True,
                stages=self._stages,
                fallback_used=True,
                fallback_reason=fallback_reason,
            )
        tune_result = self._tuning_chain(metadata, target)
        if isinstance(tune_result, Ok):
            mapped = tune_result.unwrap()
        else:
            mapped = self._fallback_config(
                metadata,
                FallbackTier.L4_TRITON_DEFAULT,
                error=tune_result.error.message,
            )
        if self.enable_cache:
            self._set_cache(metadata.cache_key, target, mapped)
            vendor, arch = self._vendor_arch_from_target(target)
            self.config_cache.set(metadata.cache_key, vendor, arch, mapped.__dict__)
        elapsed = (time.perf_counter() - start) * 1000
        return TuningResult(
            config=mapped,
            fallback_tier=FallbackTier.L0_TVM_DB_HIT,
            total_duration_ms=elapsed,
            stages=self._stages,
            fallback_used=True,
            fallback_reason=fallback_reason,
        )

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _tuning_chain(
        self,
        metadata: KernelMetadata,
        target: str,
    ) -> Result[MappedTuningConfig, TuningError]:
        """Run the TVM tuning chain with fallback tiers.

        Returns a Rust-style :class:`Result`:

          - ``Ok(MappedTuningConfig)`` — the chain produced a real
            TVM-MetaSchedule-tuned config (or a fast cache hit). This
            is the value the caller should prefer.
          - ``Err(TuningError)`` — the chain degraded to a safe
            fallback config. Callers can still extract the fallback
            via :meth:`Result.unwrap_or` to get a usable ``MappedTuningConfig``;
            the ``Err`` is informational and tells you the tuning
            itself was bypassed.

        The chain NEVER raises a ``TuningError`` directly — every
        failure path wraps the underlying error in ``Err`` and
        returns. This lets callers do a single ``match`` and act on
        the result without worrying about an unhandled exception.
        """
        t0 = time.perf_counter()
        try:
            tir_mod = self._build_tir_template(metadata)
            self._stages["build_template"] = (time.perf_counter() - t0) * 1000
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning(
                "TIR template build rejected input (%s): %s",
                type(exc).__name__,
                exc,
            )
            self._stages["build_template"] = -1
            return Err(
                TuningError(
                    f"TIR template build rejected input: {exc}",
                    context={"tier": FallbackTier.L5_SAFE_FALLBACK.name},
                )
            )
        except ImportError as exc:
            logger.warning(
                "TIR template build: missing dependency (%s); using safe fallback",
                exc,
            )
            self._stages["build_template"] = -1
            return Err(
                TuningError(
                    f"TIR template build missing dependency: {exc}",
                    context={"tier": FallbackTier.L5_SAFE_FALLBACK.name},
                )
            )
        except OSError as exc:
            logger.warning(
                "TIR template build: filesystem error (%s); using safe fallback",
                exc,
            )
            self._stages["build_template"] = -1
            return Err(
                TuningError(
                    f"TIR template build filesystem error: {exc}",
                    context={"tier": FallbackTier.L5_SAFE_FALLBACK.name},
                )
            )

        t0 = time.perf_counter()
        if self.tvm_adapter and self.enable_tvm:
            try:
                tune_result: Result[MappedTuningConfig, TuningError] = self.tvm_adapter.tune(
                    tir_mod=tir_mod,
                    target_str=target,
                    max_trials=self.max_trials,
                    cache_key=metadata.cache_key,
                )
            except StageTimeoutError as exc:
                logger.error(
                    "TVM tuning timed out (budget=%.1fs elapsed=%.1fs); "
                    "falling back to Triton default",
                    exc.budget_s,
                    exc.elapsed_s,
                )
                self._stages["tvm_tune"] = -1
                return Err(
                    TuningError(
                        f"TVM tuning timed out: {exc}",
                        context={"tier": FallbackTier.L4_TRITON_DEFAULT.name},
                    )
                )
            except (ValueError, TypeError) as exc:
                logger.error(
                    "TVM tuning rejected input (%s): %s",
                    type(exc).__name__,
                    exc,
                )
                self._stages["tvm_tune"] = -1
                return Err(
                    TuningError(
                        f"TVM tuning rejected input: {exc}",
                        context={"tier": FallbackTier.L4_TRITON_DEFAULT.name},
                    )
                )
            except ImportError as exc:
                logger.error(
                    "TVM tuning: missing dependency (%s); falling back to Triton default",
                    exc,
                )
                self._stages["tvm_tune"] = -1
                return Err(
                    TuningError(
                        f"TVM tuning missing dependency: {exc}",
                        context={"tier": FallbackTier.L4_TRITON_DEFAULT.name},
                    )
                )
            except OSError as exc:
                logger.error(
                    "TVM tuning: filesystem error (%s); falling back to Triton default",
                    exc,
                )
                self._stages["tvm_tune"] = -1
                return Err(
                    TuningError(
                        f"TVM tuning filesystem error: {exc}",
                        context={"tier": FallbackTier.L4_TRITON_DEFAULT.name},
                    )
                )
            except RuntimeError as exc:
                logger.error(
                    "TVM tuning: runtime error (%s); falling back to Triton default",
                    exc,
                )
                self._stages["tvm_tune"] = -1
                return Err(
                    TuningError(
                        f"TVM tuning runtime error: {exc}",
                        context={"tier": FallbackTier.L4_TRITON_DEFAULT.name},
                    )
                )

            if isinstance(tune_result, Ok):
                self._stages["tvm_tune"] = (time.perf_counter() - t0) * 1000
                return tune_result
            logger.error(
                "TVM tuning returned Err (%s); falling back to Triton default",
                tune_result.error.message,
            )
            self._stages["tvm_tune"] = -1
            return tune_result
        else:
            self._stages["tvm_tune"] = 0

        return Err(
            TuningError(
                "TVM tuning bypassed; using Triton default config",
                context={"tier": FallbackTier.L4_TRITON_DEFAULT.name},
            )
        )

    def _fallback_config(
        self,
        metadata: KernelMetadata,
        tier: FallbackTier,
        error: str | None = None,
    ) -> MappedTuningConfig:
        """Compute the safe fallback config for a given tier (helper).

        Use this to extract a usable ``MappedTuningConfig`` from an
        ``Err(_tuning_chain(...))`` Result.
        """
        logger.info("Using fallback %s for %s", tier.name, metadata.kernel_name)
        if metadata.is_matmul:
            return MappedTuningConfig(
                block_m=128,
                block_n=128,
                block_k=32,
                num_warps=min(metadata.num_warps, 8),
                num_stages=metadata.num_stages,
            )
        return MappedTuningConfig.defaults()

    def _fallback(
        self,
        metadata: KernelMetadata,
        tier: FallbackTier,
        error: str | None = None,
    ) -> MappedTuningConfig:
        """Return a safe fallback config."""
        logger.info("Using fallback %s for %s", tier.name, metadata.kernel_name)
        if metadata.is_matmul:
            return MappedTuningConfig(
                block_m=128,
                block_n=128,
                block_k=32,
                num_warps=min(metadata.num_warps, 8),
                num_stages=metadata.num_stages,
            )
        return MappedTuningConfig.defaults()

    # ------------------------------------------------------------------
    # TIR template builder (lazy import)
    # ------------------------------------------------------------------

    def _build_tir_template(self, metadata: KernelMetadata) -> Any:
        """Build a TVM TIR template from kernel metadata."""
        from src.bridges.triton_tvm.tir_template import TIRTemplateBuilder

        builder = TIRTemplateBuilder()
        return builder.build_from_metadata(metadata)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _get_cached(self, cache_key: str, target: str) -> MappedTuningConfig | None:
        """Check in-memory and disk cache."""
        full_key = f"{cache_key}:{target}"

        # In-memory check
        if full_key in self._cache:
            self._lru_order.remove(full_key)
            self._lru_order.append(full_key)
            return self._cache[full_key]

        # Disk cache check
        cache_path = self._disk_cache_path(full_key)
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                config = MappedTuningConfig(**data)
                self._cache[full_key] = config
                self._lru_order.append(full_key)
                return config
            except (json.JSONDecodeError, KeyError):
                cache_path.unlink(missing_ok=True)

        return None

    def _set_cache(self, cache_key: str, target: str, config: MappedTuningConfig) -> None:
        """Store in both in-memory and disk cache."""
        full_key = f"{cache_key}:{target}"

        # In-memory with LRU eviction
        if full_key in self._cache:
            self._lru_order.remove(full_key)
        self._cache[full_key] = config
        self._lru_order.append(full_key)

        if len(self._lru_order) > self._max_cache_entries:
            oldest = self._lru_order.pop(0)
            self._cache.pop(oldest, None)
            # Also remove from disk to prevent re-population on get
            disk_path = self._disk_cache_path(oldest)
            if disk_path.exists():
                disk_path.unlink(missing_ok=True)

        # Disk cache
        cache_path = self._disk_cache_path(full_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "block_m": config.block_m,
            "block_n": config.block_n,
            "block_k": config.block_k,
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
            "num_ctas": config.num_ctas,
        }
        cache_path.write_text(json.dumps(data, indent=2))

    def _disk_cache_path(self, full_key: str) -> Path:
        """Compute disk cache path for a cache key."""
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        return self.cache_dir / "bridge_cache" / f"{key_hash[:32]}.json"

    # ------------------------------------------------------------------
    # Grid resolution
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ConfigCache integration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vendor_arch_from_target(target: str) -> tuple[str, str]:
        """Extract (vendor, arch) from a TVM target string for cache keying.

        Examples::

            "nvidia/nvidia-h100"      → ("nvidia", "sm_90")
            "rocm/gfx942"             → ("amd", "gfx942")
            "intel/gaudi-2"           → ("intel", "gaudi2")
            "cuda"                    → ("nvidia", "generic")
        """
        t = target.lower()
        if any(x in t for x in ("nvidia", "cuda")):
            vendor = "nvidia"
            arch = "generic"
            for a in ("sm_90", "sm_80", "sm_70", "h100", "h200", "a100"):
                if a in t:
                    arch = f"sm_{a.split('_')[-1]}" if a.startswith("sm_") else a
                    break
        elif any(x in t for x in ("amd", "rocm")):
            vendor = "amd"
            # Extract gfxXXX from target
            arch = "generic"
            for part in t.replace("/", " ").split():
                if part.startswith("gfx"):
                    arch = part
                    break
        elif any(x in t for x in ("intel", "gaudi", "spirv", "xe")):
            vendor = "intel"
            if "gaudi2" in t or "gaudi-2" in t:
                arch = "gaudi2"
            elif "gaudi3" in t or "gaudi-3" in t:
                arch = "gaudi3"
            else:
                # Try to extract xe_* or other arch markers
                arch = "generic"
                for part in t.replace("/", " ").split():
                    if part.startswith("xe") or part.startswith("intel"):
                        arch = part
                        break
        elif "apple" in t or "metal" in t:
            vendor = "apple"
            for m in ("m4", "m3", "m2", "m1"):
                if m in t:
                    arch = f"apple_{m}"
                    break
            else:
                arch = "apple_generic"
        else:
            vendor = "unknown"
            arch = "generic"
        return vendor, arch

    @staticmethod
    def _resolve_grid(grid: tuple[int, ...] | Callable) -> tuple[int, int, int]:
        """Convert grid to a tuple, resolving lambdas if possible."""
        if isinstance(grid, tuple):
            grid_tuple = grid
        elif callable(grid):
            try:
                result = grid({"BLOCK_SIZE": 128})
                if isinstance(result, tuple):
                    grid_tuple = result
                else:
                    grid_tuple = (result, 1, 1)
            except (TypeError, ValueError, KeyError, IndexError, RuntimeError, ArithmeticError):
                grid_tuple = (1, 1, 1)
        else:
            grid_tuple = (1, 1, 1)

        if len(grid_tuple) == 1:
            grid_tuple = (*grid_tuple, 1, 1)
        elif len(grid_tuple) == 2:
            grid_tuple = (*grid_tuple, 1)

        return (int(grid_tuple[0]), int(grid_tuple[1]), int(grid_tuple[2]))


# ------------------------------------------------------------------
# Convenience function for @triton.autotune integration
# ------------------------------------------------------------------


def autotune_configs(
    kernel_fn: Any,
    example_args: tuple[Any, ...],
    target: str = "nvidia/nvidia-a100",
    num_warps: int = 4,
    num_stages: int = 3,
    **bridge_kwargs: Any,
) -> list[Any]:
    """Generate @triton.autotune configs using TVM MetaSchedule tuning.

    Usage:
        @triton.autotune(
            configs=autotune_configs(matmul_kernel, (x, y)),
            key=["M", "N", "K"],
        )
        @triton.jit
        def matmul_kernel(...): ...

    Args:
        kernel_fn: The @triton.jit kernel function.
        example_args: Example tensors for shape extraction.
        target: TVM target string.
        num_warps: Initial warps guess.
        num_stages: Initial stages guess.
        **bridge_kwargs: Additional kwargs for TritonTVMBridge.

    Returns:
        List of triton.Config objects.
    """
    bridge = TritonTVMBridge(**bridge_kwargs)
    metadata = bridge.extractor.extract_from_call(
        kernel_fn=kernel_fn,
        grid=(1, 1, 1),
        args=example_args,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    tune_result = bridge._tuning_chain(metadata, target)
    if isinstance(tune_result, Ok):
        mapped = tune_result.unwrap()
    else:
        mapped = bridge._fallback_config(
            metadata,
            FallbackTier.L4_TRITON_DEFAULT,
            error=tune_result.error.message,
        )
    # Generate variants
    results = [mapped]
    if metadata.is_matmul:
        results.append(
            MappedTuningConfig(
                block_m=mapped.block_m * 2, block_n=mapped.block_n, block_k=mapped.block_k
            )
        )
        results.append(
            MappedTuningConfig(
                block_m=mapped.block_m, block_n=mapped.block_n * 2, block_k=mapped.block_k
            )
        )
        results.append(
            MappedTuningConfig(
                block_m=mapped.block_m, block_n=mapped.block_n, block_k=max(mapped.block_k // 2, 16)
            )
        )
    return [r.to_triton_config() for r in results]

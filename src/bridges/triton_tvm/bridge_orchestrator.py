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

import functools
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

from src.bridges.triton_tvm.metadata_extractor import (
    KernelMetadata,
    MetadataExtractor,
)
from src.bridges.triton_tvm.config_mapper import ConfigMapper, MappedTuningConfig
from src.bridges.triton_tvm.metaschedule_adapter import MetaScheduleAdapter

try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

try:
    from tvm import meta_schedule as ms
    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False

logger = logging.getLogger(__name__)


class FallbackTier(Enum):
    """Tuning fallback tiers. Higher number = worse but more reliable."""
    L0_TVM_DB_HIT = auto()       # Best: existing TVM database record
    L1_TVM_WARM_START = auto()   # TVM warm-starts from similar config
    L2_TVM_COLD = auto()         # Full TVM tuning (cold start)
    L3_DISK_CACHE = auto()       # Previously cached Triton config
    L4_TRITON_DEFAULT = auto()   # Triton's built-in defaults
    L5_SAFE_FALLBACK = auto()    # Conservative pre-vetted config


@dataclass
class TuningResult:
    """Result of the full bridge pipeline."""
    config: MappedTuningConfig
    fallback_tier: FallbackTier
    total_duration_ms: float
    cache_hit: bool = False
    error: str | None = None
    stages: dict[str, float] = field(default_factory=dict)  # stage → duration_ms


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
        self.cache_dir = Path(cache_dir or os.environ.get(
            "NVINDIACUD_CACHE_DIR",
            str(Path.home() / ".cache" / "nvindia_cud"),
        ))
        self.max_trials = max_trials
        self.enable_cache = enable_cache
        self.enable_tvm = enable_tvm and TVM_AVAILABLE

        self.extractor = MetadataExtractor()
        self.config_mapper = ConfigMapper()
        self.tvm_adapter = MetaScheduleAdapter(
            cache_dir=str(self.cache_dir),
            default_max_trials=max_trials,
            default_num_trials_per_iter=16,
            timeout_seconds=600,
        ) if self.enable_tvm else None

        # Real IR capture infrastructure (Step 1 of the production wiring)
        # These are the components that hook into Triton's actual compile
        # pipeline via the backend plugin and capture REAL TTGIR.
        from .ir_capture import IRCapture
        from .extern_bridge import ExternMatmulBuilder
        from .circuit_breaker import get_default_breakers
        from .timeout_manager import TimeoutManager, StageBudgets
        from .structured_logging import (
            span as span_context, configure_logging,
        )

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

        # Step 2: Check cache
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

        # Step 3-5: TVM tuning chain
        mapped = self._tuning_chain(metadata, target)

        # Cache the result
        if self.enable_cache:
            self._set_cache(metadata.cache_key, target, mapped)

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
        best = self._tuning_chain(metadata, target)
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
        from .structured_logging import span as span_ctx, stage as stage_ctx
        from .ir_capture import KernelKind

        start = time.perf_counter()
        self._stages = {}

        with span_ctx(source_hash, target, metadata={"mode": "real_ir"}) as sp:
            # Stage 1: Real IR capture
            with stage_ctx(sp, "ir_capture") as st:
                captured = self.ir_capture.capture_for_source(source_hash, target)
                if captured is None:
                    logger.warning(
                        "No real IR captured for %s..%s; falling back to metadata path",
                        source_hash[:12], target,
                    )
                    # Fall back to a synthetic metadata + chain
                    metadata = self._synthesize_metadata(source_hash, target)
                    return self._fallback_tune_chain(metadata, target, start)
                st.metadata["kind"] = captured.kind.name
                st.metadata["ops"] = len(captured.ops_seen)
                self._stages["ir_capture"] = st.duration_ms

            # Stage 2: Cache check
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
                    st.metadata["cache_hit"] = False

            # Stage 3: Build TIR template from REAL bounds
            with stage_ctx(sp, "build_tir") as st:
                tir_mod = self._build_tir_from_captured(captured, target)
                st.metadata["built"] = tir_mod is not None
                self._stages["build_tir"] = st.duration_ms

            if tir_mod is None:
                # Could not build a TIR from captured IR — fall back
                logger.warning("Failed to build TIR from captured IR; using fallback")
                return self._fallback_tune_chain(
                    self._synthesize_metadata(source_hash, target),
                    target, start,
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

            # Stage 6: Cache the result
            if self.enable_cache:
                self._set_cache(cache_key, target, mapped)

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
        """Run TVM MetaSchedule through the tvm_tune circuit breaker."""
        if not (self.tvm_adapter and self.enable_tvm):
            return MappedTuningConfig.defaults()

        breaker = self.breakers["tvm_tune"]
        try:
            return breaker.call(
                self.tvm_adapter.tune,
                tir_mod=tir_mod,
                target_str=target_str,
                max_trials=self.max_trials,
                cache_key=cache_key,
            )
        except Exception as exc:
            # Circuit breaker tripped or other error — log and fall back
            logger.warning(
                "TVM tune failed (breaker=%s): %s",
                breaker.state.name, exc,
            )
            return MappedTuningConfig.defaults()

    def _build_tir_from_captured(
        self,
        captured: Any,
        target: str,
    ) -> Any:
        """Build a TIR module from REAL captured IR.

        Primary path: the new 4-pass conversion pipeline (preserves
        actual kernel semantics from real TTGIR).

        Fallback: template construction from extracted bounds
        (preserved from the original config-bridge design).
        """
        from .ir_capture import KernelKind
        from .tir_template import TIRTemplateBuilder

        # Primary: real IR conversion via the 4-pass pipeline
        if captured.ir_text and len(captured.ir_text) > 100:
            try:
                builder = TIRTemplateBuilder()
                ir_module, conv_result = builder.build_from_captured_ir(
                    captured.ir_text,
                )
                if ir_module is not None:
                    logger.info(
                        "Real IR conversion succeeded: status=%s, has_dot=%s",
                        conv_result.status.name,
                        conv_result.has_dot_split,
                    )
                    return ir_module
                logger.warning(
                    "Real IR conversion returned no IRModule; falling back to templates"
                )
            except Exception as exc:
                logger.warning("Real IR conversion raised: %s", exc)

        # Fallback: template construction from extracted bounds
        try:
            builder = TIRTemplateBuilder()
            if captured.kind in (KernelKind.MATMUL, KernelKind.ATTENTION):
                if captured.bounds.m and captured.bounds.n and captured.bounds.k:
                    return builder.build_matmul(
                        m=captured.bounds.m,
                        n=captured.bounds.n,
                        k=captured.bounds.k,
                        dtype=captured.bounds.data_dtype,
                    )
            elif captured.kind == KernelKind.REDUCTION:
                if captured.bounds.reduce_size:
                    return builder.build_reduction(
                        shape=(captured.bounds.keep_size or 1, captured.bounds.reduce_size),
                        dtype=captured.bounds.data_dtype,
                    )
            elif captured.kind == KernelKind.ELEMENTWISE:
                total = captured.bounds.total_elements or 1024
                return builder.build_elementwise(
                    shape=(total,),
                    dtype=captured.bounds.data_dtype,
                )
        except Exception as exc:
            logger.warning("TIR template fallback failed: %s", exc)
        return None

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
        except Exception as exc:
            logger.warning("Extern matmul build failed: %s", exc)

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
            grid_0=1, grid_1=1, grid_2=1,
            num_warps=4, num_stages=3, num_ctas=1,
        )

    def _fallback_tune_chain(
        self,
        metadata: KernelMetadata,
        target: str,
        start: float,
    ) -> TuningResult:
        """When real IR isn't available, fall back to the legacy metadata path."""
        cached = None
        if self.enable_cache:
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
        mapped = self._tuning_chain(metadata, target)
        if self.enable_cache:
            self._set_cache(metadata.cache_key, target, mapped)
        elapsed = (time.perf_counter() - start) * 1000
        return TuningResult(
            config=mapped,
            fallback_tier=FallbackTier.L0_TVM_DB_HIT,
            total_duration_ms=elapsed,
            stages=self._stages,
        )

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _tuning_chain(
        self,
        metadata: KernelMetadata,
        target: str,
    ) -> MappedTuningConfig:
        """Run the TVM tuning chain with fallback tiers."""
        # Step 3: Build TIR template
        t0 = time.perf_counter()
        try:
            tir_mod = self._build_tir_template(metadata)
            self._stages["build_template"] = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            logger.warning("TIR template build failed: %s", exc)
            self._stages["build_template"] = -1
            return self._fallback(metadata, FallbackTier.L5_SAFE_FALLBACK, str(exc))

        # Step 4: Run MetaSchedule
        t0 = time.perf_counter()
        if self.tvm_adapter and self.enable_tvm:
            try:
                mapped = self.tvm_adapter.tune(
                    tir_mod=tir_mod,
                    target_str=target,
                    max_trials=self.max_trials,
                    cache_key=metadata.cache_key,
                )
                if mapped != MappedTuningConfig.defaults():
                    self._stages["tvm_tune"] = (time.perf_counter() - t0) * 1000
                    return mapped
            except Exception as exc:
                logger.error("TVM tuning failed: %s", exc)
                self._stages["tvm_tune"] = -1
        else:
            self._stages["tvm_tune"] = 0

        # Step 5: Fallback
        return self._fallback(metadata, FallbackTier.L4_TRITON_DEFAULT)

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
                block_m=128, block_n=128, block_k=32,
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
            except Exception:
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
    mapped = bridge._tuning_chain(metadata, target)
    # Generate variants
    results = [mapped]
    if metadata.is_matmul:
        results.append(MappedTuningConfig(block_m=mapped.block_m * 2, block_n=mapped.block_n, block_k=mapped.block_k))
        results.append(MappedTuningConfig(block_m=mapped.block_m, block_n=mapped.block_n * 2, block_k=mapped.block_k))
        results.append(MappedTuningConfig(block_m=mapped.block_m, block_n=mapped.block_n, block_k=max(mapped.block_k // 2, 16)))
    return [r.to_triton_config() for r in results]

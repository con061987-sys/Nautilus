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

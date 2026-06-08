"""Transfer learning engine for cross-vendor tuning configuration.

Given an optimal tuning configuration discovered on one GPU vendor
(e.g. Nvidia H100), the :class:`TransferEngine` predicts a good
configuration for a different vendor (e.g. AMD MI300X) by applying
architectural scaling factors derived from known hardware parameters.

The core insight is that optimal tile sizes, warp counts, and pipeline
stages correlate with measurable hardware properties:

* **Warp/thread ratios** — a config using 8 warps on Nvidia (warp=32)
  maps to roughly 4 wavefronts on AMD (warp=64), so the tile sizes
  scale accordingly.
* **Shared memory capacity** — determines how much data fits per
  block, which constrains tile sizes and pipeline depth.
* **Memory bandwidth** — affects optimal pipeline stages for hiding
  latency.
* **Matrix core availability** — tensor-core vs. MFMA vs. SIMD
  changes the optimal tile shape ratios.

Usage::

    engine = TransferEngine()
    result = engine.transfer(
        source_vendor="nvidia",
        target_vendor="amd",
        source_config={"block_m": 128, "block_n": 128, "block_k": 64,
                       "num_warps": 8, "num_stages": 4},
    )
    if result.confidence >= 0.6:
        use_config(result.config)
    else:
        seed_tuning_with(result.seed_config)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.bridges.triton_tvm.config_cache import ConfigCache
from src.bridges.triton_tvm.config_mapper import MappedTuningConfig

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger: Any

try:
    from src.common.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architecture parameters used for transfer scaling
# ---------------------------------------------------------------------------
# These are the "known hardware properties" from which scaling factors
# are derived.  Each entry is (warp_size, max_shared_memory_bytes,
# max_registers, hbm_bw_gbps, has_matrix_core).

_ARCH_PROPS: dict[str, tuple[int, int, int, float, bool]] = {
    "nvidia": (
        32,       # warp size
        228000,   # max shared memory per SM (bytes) — H100
        65536,    # max registers per SM
        3350.0,   # HBM bandwidth (GB/s)
        True,     # has tensor/matrix cores
    ),
    "amd": (
        64,       # wavefront size
        196608,   # max LDS per CU (bytes) — MI300X: 192 KB usable
        131072,   # max SGPRs per CU
        5300.0,   # HBM bandwidth (GB/s)
        True,     # has MFMA matrix cores
    ),
    "intel": (
        32,       # SIMD width (sub-group size)
        131072,   # max SLM per sub-slice (bytes) — Gaudi: 128 KB
        131072,   # max GRF (bytes)
        1200.0,   # HBM bandwidth (GB/s)
        False,    # no dedicated matrix cores on Gaudi
    ),
    "apple": (
        32,       # SIMD-group size
        65536,    # max threadgroup memory (bytes) — M3: 64 KB
        65536,    # max registers
        800.0,    # unified memory bandwidth (GB/s)
        False,    # no dedicated matrix cores
    ),
}

# ---------------------------------------------------------------------------
# Transfer matrices
# ---------------------------------------------------------------------------
# Each entry maps (source_vendor, target_vendor) to a dict of scaling
# factors.  The factors are pure architecture-derived ratios — no
# kernel-specific heuristics.
#
# Convention:
#   warp_scale          = source_warp_size / target_warp_size
#   (num_warps scales inversely: more target threads per warp → fewer warps)
#
#   shared_memory_ratio = target_shared_memory / source_shared_memory
#   (bigger shared memory on target → larger tile K allowed)
#
#   bw_ratio            = target_bandwidth / source_bandwidth
#   (higher bandwidth → deeper pipelines to hide latency)
#
#   matrix_core_scale   = 1.0 if both have cores, else adjusted for SIMD
#   (absence of matrix cores shifts optimal tile shapes toward
#    memory-bound configurations)

TRANSFER_MATRICES: dict[tuple[str, str], dict[str, float]] = {
    ("nvidia", "amd"): {
        "warp_scale": 32.0 / 64.0,          # Nvidia warp 32 → AMD wavefront 64 → half as many
        "shared_memory_ratio": 196608.0 / 228000.0,  # AMD LDS / Nvidia shared mem
        "bw_ratio": 5300.0 / 3350.0,         # AMD higher bandwidth
        "register_ratio": 131072.0 / 65536.0,
        "matrix_core_scale": 0.9,            # both have cores, slight MFMA vs MMA diff
    },
    ("nvidia", "intel"): {
        "warp_scale": 32.0 / 32.0,           # same SIMD-group width
        "shared_memory_ratio": 131072.0 / 228000.0,  # Intel SLM smaller
        "bw_ratio": 1200.0 / 3350.0,         # Intel bandwidth significantly lower
        "register_ratio": 131072.0 / 65536.0,
        "matrix_core_scale": 0.5,            # Intel Gaudi lacks matrix cores → SIMD fallback
    },
    ("nvidia", "apple"): {
        "warp_scale": 32.0 / 32.0,
        "shared_memory_ratio": 65536.0 / 228000.0,   # Apple threadgroup mem much smaller
        "bw_ratio": 800.0 / 3350.0,                   # lower bandwidth
        "register_ratio": 65536.0 / 65536.0,
        "matrix_core_scale": 0.5,                     # no matrix cores on Apple GPU
    },
    ("amd", "nvidia"): {
        "warp_scale": 64.0 / 32.0,           # AMD wavefront 64 → Nvidia warp 32 → twice as many
        "shared_memory_ratio": 228000.0 / 196608.0,
        "bw_ratio": 3350.0 / 5300.0,
        "register_ratio": 65536.0 / 131072.0,
        "matrix_core_scale": 0.9,
    },
    ("amd", "intel"): {
        "warp_scale": 64.0 / 32.0,
        "shared_memory_ratio": 131072.0 / 196608.0,
        "bw_ratio": 1200.0 / 5300.0,
        "register_ratio": 131072.0 / 131072.0,
        "matrix_core_scale": 0.5,
    },
    ("amd", "apple"): {
        "warp_scale": 64.0 / 32.0,
        "shared_memory_ratio": 65536.0 / 196608.0,
        "bw_ratio": 800.0 / 5300.0,
        "register_ratio": 65536.0 / 131072.0,
        "matrix_core_scale": 0.5,
    },
    ("intel", "nvidia"): {
        "warp_scale": 32.0 / 32.0,
        "shared_memory_ratio": 228000.0 / 131072.0,
        "bw_ratio": 3350.0 / 1200.0,
        "register_ratio": 65536.0 / 131072.0,
        "matrix_core_scale": 0.5,
    },
    ("intel", "amd"): {
        "warp_scale": 32.0 / 64.0,
        "shared_memory_ratio": 196608.0 / 131072.0,
        "bw_ratio": 5300.0 / 1200.0,
        "register_ratio": 131072.0 / 131072.0,
        "matrix_core_scale": 0.5,
    },
    ("apple", "nvidia"): {
        "warp_scale": 32.0 / 32.0,
        "shared_memory_ratio": 228000.0 / 65536.0,
        "bw_ratio": 3350.0 / 800.0,
        "register_ratio": 65536.0 / 65536.0,
        "matrix_core_scale": 0.5,
    },
    ("apple", "amd"): {
        "warp_scale": 32.0 / 64.0,
        "shared_memory_ratio": 196608.0 / 65536.0,
        "bw_ratio": 5300.0 / 800.0,
        "register_ratio": 131072.0 / 65536.0,
        "matrix_core_scale": 0.5,
    },
    ("intel", "apple"): {
        "warp_scale": 32.0 / 32.0,
        "shared_memory_ratio": 65536.0 / 131072.0,
        "bw_ratio": 800.0 / 1200.0,
        "register_ratio": 65536.0 / 131072.0,
        "matrix_core_scale": 1.0,  # neither has matrix cores → similar SIMD path
    },
    ("apple", "intel"): {
        "warp_scale": 32.0 / 32.0,
        "shared_memory_ratio": 131072.0 / 65536.0,
        "bw_ratio": 1200.0 / 800.0,
        "register_ratio": 131072.0 / 65536.0,
        "matrix_core_scale": 1.0,
    },
}


# ---------------------------------------------------------------------------
# Confidence anchors (architecture-level similarity)
# ---------------------------------------------------------------------------

_ARCH_SIMILARITY: dict[tuple[str, str], float] = {
    ("nvidia", "nvidia"): 1.0,
    ("nvidia", "amd"): 0.65,
    ("nvidia", "intel"): 0.45,
    ("nvidia", "apple"): 0.40,
    ("amd", "nvidia"): 0.65,
    ("amd", "amd"): 1.0,
    ("amd", "intel"): 0.50,
    ("amd", "apple"): 0.45,
    ("intel", "nvidia"): 0.45,
    ("intel", "amd"): 0.50,
    ("intel", "intel"): 1.0,
    ("intel", "apple"): 0.55,
    ("apple", "nvidia"): 0.40,
    ("apple", "amd"): 0.45,
    ("apple", "intel"): 0.55,
    ("apple", "apple"): 1.0,
}

# Default target warp sizes for clamping
_TARGET_WARP_SIZES: dict[str, int] = {
    "nvidia": 32,
    "amd": 64,
    "intel": 32,
    "apple": 32,
}

# ---------------------------------------------------------------------------
# PerformanceDB protocol
# ---------------------------------------------------------------------------


class PerformanceDB(Protocol):
    """Interface for a performance history database.

    A concrete implementation can query historical transfer results to
    refine confidence scores.  When no database is provided, confidence
    is computed from architecture data alone.
    """

    def get_historical_accuracy(
        self,
        source_vendor: str,
        target_vendor: str,
        kernel_hash: str,
    ) -> float | None:
        """Return historical transfer accuracy (0.0-1.0) or ``None``.

        Args:
            source_vendor: Source vendor identifier.
            target_vendor: Target vendor identifier.
            kernel_hash: Hash identifying the kernel family.

        Returns:
            Accuracy score from previous transfers, or ``None`` if no
            history exists for this (source, target, kernel) triple.
        """
        ...


# ---------------------------------------------------------------------------
# TransferredConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferredConfig:
    """Result of a cross-vendor config transfer.

    Attributes:
        config: The predicted tuning config for the target vendor,
            expressed as a :class:`MappedTuningConfig`.
        confidence: Confidence score in [0.0, 1.0].  Scores >= 0.6
            indicate the config can be used directly.  Lower scores
            mean the config should be used as a *seed* for auto-tuning.
        seed_config: The same config, provided as a convenience dict
            for seeding MetaSchedule's evolutionary search.  This is
            a ``dict[str, int]`` suitable for passing as an initial
            candidate to ``tune_tir``.
        source_vendor: The vendor the config was transferred from.
        target_vendor: The vendor the config was transferred to.
        transfer_matrix: The matrix key used for the transfer.
        historical_accuracy: Optional historical accuracy from
            ``PerformanceDB``, if available.
    """

    config: MappedTuningConfig
    confidence: float
    seed_config: dict[str, int] = field(default_factory=dict)
    source_vendor: str = ""
    target_vendor: str = ""
    transfer_matrix: str = ""
    historical_accuracy: float | None = None

    def should_tune(self, threshold: float = 0.6) -> bool:
        """Return ``True`` if this config should trigger a tuning run.

        Args:
            threshold: Confidence threshold.  Below this, the config
                is used as a seed rather than applied directly.

        Returns:
            ``True`` if tuning is recommended.
        """
        return self.confidence < threshold

    def as_tune_kwargs(self) -> dict[str, Any]:
        """Return kwargs suitable for ``tune_tir(init_config=...)``.

        Returns:
            A dict that can be passed as an initial candidate to
            MetaSchedule's tuning API.
        """
        return dict(self.seed_config)


# ---------------------------------------------------------------------------
# TransferEngine
# ---------------------------------------------------------------------------


class TransferEngine:
    """Predict optimal tuning configs for a target vendor based on known
    good configs from a source vendor.

    The engine uses architecture-derived scaling factors (see
    :data:`TRANSFER_MATRICES`) rather than per-kernel heuristics,
    so it generalises to any kernel type without modification.
    """

    def __init__(
        self,
        config_cache: ConfigCache | None = None,
        perf_db: PerformanceDB | None = None,
        default_threshold: float = 0.6,
    ) -> None:
        """Initialise the engine.

        Args:
            config_cache: Optional :class:`ConfigCache` for persisting
                transferred configs.
            perf_db: Optional :class:`PerformanceDB` for querying
                historical transfer accuracy.
            default_threshold: Confidence threshold below which the
                engine recommends using the transferred config as a
                seed for full auto-tuning rather than applying it
                directly.  Defaults to 0.6.
        """
        self._config_cache = config_cache
        self._perf_db = perf_db
        self._default_threshold = default_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transfer(
        self,
        source_vendor: str,
        target_vendor: str,
        source_config: MappedTuningConfig | dict[str, Any],
        kernel_hash: str = "",
        *,
        threshold: float | None = None,
    ) -> TransferredConfig:
        """Transfer a config from *source_vendor* to *target_vendor*.

        Args:
            source_vendor: Source vendor (``"nvidia"``, ``"amd"``,
                ``"intel"``, ``"apple"``).
            target_vendor: Target vendor.
            source_config: The known-good config on the source vendor.
                Can be a :class:`MappedTuningConfig` or a dict with
                keys matching the dataclass fields.
            kernel_hash: Optional hash identifying the kernel for
                historical accuracy lookup.
            threshold: Optional override for the confidence threshold
                (defaults to ``self._default_threshold``).

        Returns:
            A :class:`TransferredConfig` with the predicted config
            and confidence score.
        """
        src = source_vendor.lower().strip()
        tgt = target_vendor.lower().strip()

        # Normalise input config to MappedTuningConfig.
        cfg = self._normalise_config(source_config)

        # Fetch the transfer matrix.  If the pair is unknown, we fall
        # back to an identity transfer with very low confidence.
        matrix = TRANSFER_MATRICES.get((src, tgt))
        if matrix is None:
            return self._identity_fallback(cfg, src, tgt, kernel_hash)

        # Apply the transfer matrix to produce the predicted config.
        adapted = self._apply_matrix(cfg, matrix, src, tgt)

        # Compute confidence.
        confidence = self._compute_confidence(
            src, tgt, matrix, kernel_hash=kernel_hash,
        )

        # Build seed_config dict for downstream tuning.
        seed = {
            "BLOCK_SIZE_M": adapted.block_m,
            "BLOCK_SIZE_N": adapted.block_n,
            "BLOCK_SIZE_K": adapted.block_k,
            "num_warps": adapted.num_warps,
            "num_stages": adapted.num_stages,
            "num_ctas": adapted.num_ctas,
        }

        # Write to cache if available.
        if self._config_cache and kernel_hash:
            self._config_cache.set(
                kernel_hash,
                tgt,
                "transferred",
                {
                    "source_vendor": src,
                    "confidence": confidence,
                    "config": {
                        "block_m": adapted.block_m,
                        "block_n": adapted.block_n,
                        "block_k": adapted.block_k,
                        "num_warps": adapted.num_warps,
                        "num_stages": adapted.num_stages,
                        "num_ctas": adapted.num_ctas,
                    },
                },
            )

        matrix_key = f"{src}→{tgt}"

        return TransferredConfig(
            config=adapted,
            confidence=confidence,
            seed_config=seed,
            source_vendor=src,
            target_vendor=tgt,
            transfer_matrix=matrix_key,
        )

    # ------------------------------------------------------------------
    # Tile adaptation
    # ------------------------------------------------------------------

    def adapt_tile(
        self,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        source_vendor: str,
        target_vendor: str,
    ) -> tuple[int, int, int]:
        """Scale tile sizes from *source_vendor* to *target_vendor*.

        This is a convenience method that performs only the tile-
        specific part of a transfer, useful when the caller wants to
        experiment with tile sizes independently of other parameters.

        Args:
            tile_m: Source M-dimension tile size.
            tile_n: Source N-dimension tile size.
            tile_k: Source K-dimension (reduction) tile size.
            source_vendor: Source vendor string.
            target_vendor: Target vendor string.

        Returns:
            A ``(tile_m, tile_n, tile_k)`` tuple scaled for the target.
        """
        src = source_vendor.lower().strip()
        tgt = target_vendor.lower().strip()

        matrix = TRANSFER_MATRICES.get((src, tgt))
        if matrix is None:
            return (tile_m, tile_n, tile_k)

        warp_scale = matrix.get("warp_scale", 1.0)
        smem_ratio = matrix.get("shared_memory_ratio", 1.0)
        mc_scale = matrix.get("matrix_core_scale", 1.0)

        # M and N tiles scale by warp_ratio * matrix_core_factor.
        combined_scale_mn = math.sqrt(warp_scale * mc_scale)
        m = self._round_to_warp_multiple(tile_m * combined_scale_mn, tgt)
        n = self._round_to_warp_multiple(tile_n * combined_scale_mn, tgt)

        # K tile (reduction dim) scales more by shared memory ratio
        # because a larger K tile consumes more shared memory per block.
        k_scale = smem_ratio * math.sqrt(warp_scale)
        k = self._round_to_warp_multiple(tile_k * k_scale, tgt, base=16)

        # Clamp to sane bounds.
        return (
            max(16, min(512, m)),
            max(16, min(512, n)),
            max(8, min(256, k)),
        )

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_pair(source_vendor: str, target_vendor: str) -> bool:
        """Check whether a transfer matrix exists for the pair.

        Args:
            source_vendor: Source vendor string.
            target_vendor: Target vendor string.

        Returns:
            ``True`` if a transfer is possible.
        """
        key = (source_vendor.lower().strip(), target_vendor.lower().strip())
        return key in TRANSFER_MATRICES

    @staticmethod
    def list_available_targets(source_vendor: str) -> list[str]:
        """List all target vendors *source_vendor* can transfer to.

        Args:
            source_vendor: Source vendor string.

        Returns:
            Sorted list of target vendor names.
        """
        src = source_vendor.lower().strip()
        return sorted(
            tgt for (s, tgt) in TRANSFER_MATRICES if s == src
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_matrix(
        self,
        cfg: MappedTuningConfig,
        matrix: dict[str, float],
        src: str,
        tgt: str,
    ) -> MappedTuningConfig:
        """Apply the transfer matrix to produce a new config."""
        # Unpack scaling factors (defaults to 1.0 if missing).
        warp_scale = matrix.get("warp_scale", 1.0)
        smem_ratio = matrix.get("shared_memory_ratio", 1.0)
        bw_ratio = matrix.get("bw_ratio", 1.0)
        reg_ratio = matrix.get("register_ratio", 1.0)
        mc_scale = matrix.get("matrix_core_scale", 1.0)

        # --- Tile sizes ---
        new_m, new_n, new_k = self.adapt_tile(
            cfg.block_m, cfg.block_n, cfg.block_k, src, tgt,
        )

        # --- num_warps ---
        # Scales inversely with warp_scale (larger warps → fewer needed),
        # then adjusted by register pressure.
        warp_warps_raw = cfg.num_warps * warp_scale

        # On matrix-core-less targets, we often need more warps to
        # saturate compute (SIMD units need more thread parallelism).
        reg_factor = math.sqrt(reg_ratio)
        new_warps = max(
            1,
            round(warp_warps_raw * reg_factor / mc_scale),
        )
        # Snap to common warp counts.
        new_warps = self._snap_warps(new_warps, tgt)

        # --- num_stages ---
        # Pipeline depth scales with memory bandwidth ratio (more BW
        # → can feed deeper pipeline) and inversely with shared memory
        # (less shared mem → fewer stages possible).
        stages_raw = cfg.num_stages * bw_ratio * smem_ratio
        new_stages = max(1, min(8, round(stages_raw)))

        # --- num_ctas ---
        # Hopper-specific feature; default to 1 on non-Nvidia targets.
        new_ctas = cfg.num_ctas if tgt == "nvidia" else 1

        return MappedTuningConfig(
            block_m=new_m,
            block_n=new_n,
            block_k=new_k,
            num_warps=new_warps,
            num_stages=new_stages,
            num_ctas=new_ctas,
        )

    def _compute_confidence(
        self,
        src: str,
        tgt: str,
        matrix: dict[str, float],
        kernel_hash: str = "",
    ) -> float:
        """Compute a confidence score for a (src, tgt, matrix) triple.

        The score blends:
        1. Architecture similarity (dominant factor)
        2. Matrix coverage (how many scaling factors are defined)
        3. Historical accuracy from PerformanceDB (if available)
        """
        # Base: architecture similarity.
        base = _ARCH_SIMILARITY.get((src, tgt), 0.3)

        # Matrix coverage bonus: if all five scaling factors are
        # present, confidence is higher.
        defined = sum(
            1 for k in ("warp_scale", "shared_memory_ratio", "bw_ratio",
                        "register_ratio", "matrix_core_scale")
            if k in matrix and matrix[k] != 1.0
        )
        coverage_bonus = min(0.15, defined * 0.03)

        # Historical accuracy from PerformanceDB.
        historical: float | None = None
        if self._perf_db is not None and kernel_hash:
            try:
                historical = self._perf_db.get_historical_accuracy(
                    src, tgt, kernel_hash,
                )
            except Exception as exc:
                logger.warning(
                    "PerformanceDB lookup failed for %s→%s hash=%s: %s",
                    src, tgt, kernel_hash, exc,
                )

        if historical is not None:
            # Blend with base: give history 40% weight.
            confidence = base * 0.6 + historical * 0.4 + coverage_bonus
        else:
            confidence = base + coverage_bonus

        return max(0.0, min(1.0, confidence))

    def _identity_fallback(
        self,
        cfg: MappedTuningConfig,
        src: str,
        tgt: str,
        kernel_hash: str = "",
    ) -> TransferredConfig:
        """Create a low-confidence identity transfer when no matrix exists.

        This allows the caller to still seed a tuning run with the
        source config, which is better than starting from scratch.
        """
        logger.warning(
            "No transfer matrix for %s→%s; falling back to identity "
            "with low confidence",
            src,
            tgt,
        )
        seed = {
            "BLOCK_SIZE_M": cfg.block_m,
            "BLOCK_SIZE_N": cfg.block_n,
            "BLOCK_SIZE_K": cfg.block_k,
            "num_warps": cfg.num_warps,
            "num_stages": cfg.num_stages,
            "num_ctas": cfg.num_ctas,
        }
        return TransferredConfig(
            config=cfg,
            confidence=0.25,
            seed_config=seed,
            source_vendor=src,
            target_vendor=tgt,
            transfer_matrix="identity (unknown pair)",
        )

    @staticmethod
    def _normalise_config(
        source_config: MappedTuningConfig | dict[str, Any],
    ) -> MappedTuningConfig:
        """Normalise a config to ``MappedTuningConfig``."""
        if isinstance(source_config, MappedTuningConfig):
            return source_config
        return MappedTuningConfig(
            block_m=source_config.get("block_m", 128),
            block_n=source_config.get("block_n", 128),
            block_k=source_config.get("block_k", 32),
            num_warps=source_config.get("num_warps", 4),
            num_stages=source_config.get("num_stages", 3),
            num_ctas=source_config.get("num_ctas", 1),
            enable_fp_fusion=source_config.get("enable_fp_fusion", True),
            max_num_imprecise_acc=source_config.get(
                "max_num_imprecise_acc", 0,
            ),
        )

    @staticmethod
    def _round_to_warp_multiple(
        value: float,
        target_vendor: str,
        base: int = 16,
    ) -> int:
        """Round to nearest multiple of *base*, ensuring sane results.

        Args:
            value: Raw scaled tile size.
            target_vendor: Used to determine minimum alignment.
            base: Alignment granularity (default 16).

        Returns:
            Rounded integer at least ``base``.
        """
        rounded = max(base, round(value / base) * base)
        # Tile sizes should be powers of 2 or multiples of 16.
        return int(rounded)

    @staticmethod
    def _snap_warps(raw: int, target_vendor: str) -> int:
        """Snap a raw warp count to a valid value for the target.

        Common warp/wavefront counts across vendors:
            Nvidia: 2, 4, 8, 16, 32, 64
            AMD:    2, 4, 6, 8, 12, 16, 20, 32, 40
            Intel:  2, 4, 8, 16
            Apple:  2, 4, 8, 16
        """
        valid_sets = {
            "nvidia": (2, 4, 8, 16, 32, 64),
            "amd": (2, 4, 6, 8, 12, 16, 20, 32, 40),
            "intel": (2, 4, 8, 16),
            "apple": (2, 4, 8, 16),
        }
        valid = valid_sets.get(target_vendor, (2, 4, 8, 16))
        return min(valid, key=lambda x: abs(x - raw))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def transfer_config(
    source_vendor: str,
    target_vendor: str,
    source_config: MappedTuningConfig | dict[str, Any],
    kernel_hash: str = "",
    *,
    engine: TransferEngine | None = None,
    config_cache: ConfigCache | None = None,
    perf_db: PerformanceDB | None = None,
    threshold: float = 0.6,
) -> TransferredConfig:
    """One-shot config transfer without managing a :class:`TransferEngine`.

    Args:
        source_vendor: Source vendor identifier.
        target_vendor: Target vendor identifier.
        source_config: Known-good config on the source.
        kernel_hash: Optional kernel hash for cache/history.
        engine: Optional pre-built engine (created fresh if ``None``).
        config_cache: Optional config cache (passed to new engine).
        perf_db: Optional performance DB (passed to new engine).
        threshold: Confidence threshold for tuning recommendation.

    Returns:
        A :class:`TransferredConfig`.
    """
    if engine is None:
        engine = TransferEngine(
            config_cache=config_cache,
            perf_db=perf_db,
            default_threshold=threshold,
        )
    return engine.transfer(
        source_vendor=source_vendor,
        target_vendor=target_vendor,
        source_config=source_config,
        kernel_hash=kernel_hash,
        threshold=threshold,
    )


__all__ = [
    "TRANSFER_MATRICES",
    "PerformanceDB",
    "TransferEngine",
    "TransferredConfig",
    "transfer_config",
]

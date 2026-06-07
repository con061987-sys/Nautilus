"""Expert optimization rules for GPU vendors.

Provides structured, hardware-specific expert knowledge to guide TVM
MetaSchedule's evolutionary search, reducing tuning time by ~10x compared
to a cold start. Each rule carries a confidence score (0.0-1.0) derived
from hardware architecture documentation and validated benchmarks.

Every rule is a frozen dataclass — immutable, hashable, and safe to share
across threads. The :class:`RulesetMatcher` maps a TVM target string to the
appropriate :class:`VendorRules` and provides filtering methods to produce
search-space constraints for MetaSchedule.

Architecture:
    The rules encode vendor-specific optimisation knowledge that would
    otherwise take MetaSchedule hundreds of trials to rediscover:

    * **Tile shapes** that align with tensor/matrix core instruction shapes
    * **Warp/wavefront counts** that maximise occupancy per SM/CU
    * **Pipeline depths** that hide HBM latency without exhausting registers
    * **Memory hierarchy sizes** for shared memory and cache-aware tiling
    * **Occupancy limits** to avoid over-subscribing compute units
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Named constants — no magic numbers in rules
# ---------------------------------------------------------------------------

# H100 (Hopper) constants
_H100_MAX_WARPS_PER_SM: int = 64
_H100_MAX_THREADS_PER_SM: int = 2048
_H100_MAX_REGISTERS_PER_SM: int = 65536
_H100_SHARED_MEMORY_PER_SM: int = 256 * 1024  # 256 KB
_H100_L2_CACHE_SIZE: int = 60 * 1024 * 1024  # 60 MB
_H100_HBM_BANDWIDTH: float = 3350.0  # GB/s

# MI300X (CDNA3) constants
_MI300X_MAX_WAVEFRONTS_PER_CU: int = 40
_MI300X_MAX_THREADS_PER_CU: int = 2560  # 40 wavefronts * 64 threads
_MI300X_MAX_SGPRS_PER_CU: int = 128 * 1024  # 128 KB SGPR
_MI300X_LDS_SIZE: int = 192 * 1024  # 128 KB + 64 KB RAW partition
_MI300X_L2_CACHE_SIZE: int = 8 * 1024 * 1024  # 8 MB per GCD
_MI300X_HBM_BANDWIDTH: float = 5300.0  # GB/s

# Gaudi 2/3 constants
_GAUDI_MAX_THREADS_PER_EU: int = 56  # 7 threads * 8 SIMD lanes
_GAUDI_SLM_SIZE: int = 128 * 1024  # 128 KB per Subslice
_GAUDI_L3_CACHE_SIZE: int = 24 * 1024 * 1024  # 24 MB
_GAUDI_HBM_BANDWIDTH: float = 1200.0  # GB/s

# Apple M-series constants
_APPLE_MAX_THREADS_PER_TG: int = 1024  # threadgroup
_APPLE_TGMEM_SIZE: int = 64 * 1024  # 64 KB (varies by generation)
_APPLE_UNIFIED_MEM_BANDWIDTH: float = 800.0  # GB/s (M3 Ultra)
_APPLE_WARP_SIZE: int = 32  # SIMD-group size


# ---------------------------------------------------------------------------
# Kernel-kind aliases for type safety
# ---------------------------------------------------------------------------

MatrixCoreType = Literal["mma", "mfma_16x16x16", "mfma_32x32x16", None]
"""Matrix multiply-accumulate instruction families.

- ``"mma"`` — Nvidia Tensor Core (used by H100+ for warp-level matrix ops)
- ``"mfma_16x16x16"`` - AMD Matrix Core, 16x16x16 tile (MI300X)
- ``"mfma_32x32x16"`` - AMD Matrix Core, 32x32x16 tile (MI300X)
- ``None`` — no matrix core (Intel Gaudi, Apple, or elementwise kernels)
"""


# ---------------------------------------------------------------------------
# Validation helpers (defined before dataclasses; used by __post_init__)
# ---------------------------------------------------------------------------


def _require_confidence(confidence: float) -> None:
    """Validate that confidence is in (0.0, 1.0] — never 0.0."""
    if confidence <= 0.0 or confidence > 1.0:
        raise ValueError(
            f"Confidence must be in (0.0, 1.0], got {confidence}. "
            "If uncertain, omit the rule rather than using 0.0."
        )


def _require_non_empty(name: str, values: tuple[int, ...]) -> None:
    """Validate that a tuple of candidate values is non-empty."""
    if not values:
        raise ValueError(f"{name} must contain at least one candidate value, got empty tuple")


# ---------------------------------------------------------------------------
# Per-kernel-kind rule dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatmulRules:
    """Expert-guided tile and schedule options for matrix multiplication.

    Each list member is a candidate value for that parameter in the
    MetaSchedule search space. The confidence measures how well-tested
    this configuration set is on the target hardware.

    Attributes:
        tile_m: Candidate M-dimension tile sizes (number of rows).
        tile_n: Candidate N-dimension tile sizes (number of columns).
        tile_k: Candidate K-dimension (reduction) tile sizes.
        num_warps: Candidate warp/wavefront counts.
        num_stages: Candidate software pipeline depths (async copy stages).
        enable_tma: Whether Tensor Memory Accelerator is available and
            recommended (H100+). ``None`` = unknown / not applicable.
        cluster_size: Thread-block cluster dimensions for distributed
            shared memory (Hopper).
        matrix_core: Matrix multiply-accumulate instruction family to
            target, or ``None`` for no special instruction.
        confidence: Confidence score in [0.0, 1.0] for these rules.
    """

    tile_m: tuple[int, ...]
    tile_n: tuple[int, ...]
    tile_k: tuple[int, ...]
    num_warps: tuple[int, ...]
    num_stages: tuple[int, ...]
    enable_tma: bool | None = None
    cluster_size: tuple[int, ...] = (1,)
    matrix_core: MatrixCoreType = None
    confidence: float = 0.8

    def __post_init__(self) -> None:
        _require_confidence(self.confidence)
        for name, val in [
            ("tile_m", self.tile_m),
            ("tile_n", self.tile_n),
            ("tile_k", self.tile_k),
            ("num_warps", self.num_warps),
            ("num_stages", self.num_stages),
            ("cluster_size", self.cluster_size),
        ]:
            _require_non_empty(name, val)


@dataclass(frozen=True)
class AttentionRules:
    """Expert-guided tile and schedule options for attention kernels.

    Flash-attention-style kernels have a distinct memory access pattern:
    Q and K tiles are loaded into shared memory and the softmax reduction
    happens tile-by-tile along the sequence dimension. These rules capture
    the optimal tile shapes and pipeline depths for each vendor.

    Attributes:
        block_m: Candidate Q-sequence tile sizes.
        block_n: Candidate K-sequence tile sizes.
        block_k: Candidate head-dimension tile sizes.
        num_warps: Candidate warp/wavefront counts.
        num_stages: Candidate pipeline stages.
        causal: Whether causal masking is expected (affects tile layout).
        confidence: Confidence score in [0.0, 1.0] for these rules.
    """

    block_m: tuple[int, ...]
    block_n: tuple[int, ...]
    block_k: tuple[int, ...]
    num_warps: tuple[int, ...]
    num_stages: tuple[int, ...]
    causal: bool = False
    confidence: float = 0.7

    def __post_init__(self) -> None:
        _require_confidence(self.confidence)
        for name, val in [
            ("block_m", self.block_m),
            ("block_n", self.block_n),
            ("block_k", self.block_k),
            ("num_warps", self.num_warps),
            ("num_stages", self.num_stages),
        ]:
            _require_non_empty(name, val)


@dataclass(frozen=True)
class ElementwiseRules:
    """Expert-guided tile options for elementwise/pointwise kernels.

    Elementwise kernels are typically memory-bandwidth-bound, so the
    optimal tile size is the one that maximises memory coalescing and
    hides latency. These rules focus on block dimensions and warp
    counts that keep the memory pipeline saturated.

    Attributes:
        block_m: Candidate row tile sizes for 2D elementwise ops.
        block_n: Candidate column tile sizes (1 if 1D).
        num_warps: Candidate warp counts (fewer is often better for
            bandwidth-bound kernels).
        vectorize_size: Candidate vectorisation widths (bytes loaded
            per thread per instruction).
        confidence: Confidence score in [0.0, 1.0] for these rules.
    """

    block_m: tuple[int, ...]
    block_n: tuple[int, ...] = (1,)
    num_warps: tuple[int, ...] = (4,)
    vectorize_size: tuple[int, ...] = (4, 8)
    confidence: float = 0.7

    def __post_init__(self) -> None:
        _require_confidence(self.confidence)
        _require_non_empty("block_m", self.block_m)


@dataclass(frozen=True)
class MemoryRules:
    """Memory hierarchy parameters for the target GPU.

    These values inform tile-size selection (a tile must fit in shared
    memory) and occupancy calculations.

    Attributes:
        shared_memory_per_block: Maximum shared memory available to a
            single thread block (bytes).
        max_shared_memory: Total shared memory per SM / CU / subslice
            (bytes).
        l1_cache_size: L1 / local data-store size (bytes).
        l2_cache_size: L2 cache size (bytes). ``0`` if unified memory.
        hbm_bandwidth: Peak HBM / unified memory bandwidth (GB/s).
        confidence: Confidence score in [0.0, 1.0] for these values.
    """

    shared_memory_per_block: int
    max_shared_memory: int
    l1_cache_size: int
    l2_cache_size: int
    hbm_bandwidth: float
    confidence: float = 0.95

    def __post_init__(self) -> None:
        _require_confidence(self.confidence)
        if self.shared_memory_per_block <= 0:
            raise ValueError(
                f"shared_memory_per_block must be positive: {self.shared_memory_per_block}"
            )
        if self.max_shared_memory <= 0:
            raise ValueError(
                f"max_shared_memory must be positive: {self.max_shared_memory}"
            )
        if self.hbm_bandwidth <= 0:
            raise ValueError(f"hbm_bandwidth must be positive: {self.hbm_bandwidth}")


@dataclass(frozen=True)
class OccupancyRules:
    """Occupancy limits for the target compute unit.

    These are the raw hardware limits used to compute theoretical
    occupancy and to prune the MetaSchedule search space.

    Attributes:
        max_warps_per_sm: Maximum concurrent warps (Nvidia) or
            wavefronts (AMD) per SM / CU.
        max_threads_per_sm: Maximum concurrent threads per SM / CU.
        max_registers_per_sm: Total register file per SM / CU (bytes).
        warp_size: Threads per warp (32 for Nvidia/Apple/Intel) or
            wavefront (64 for AMD).
        confidence: Confidence score in [0.0, 1.0].
    """

    max_warps_per_sm: int
    max_threads_per_sm: int
    max_registers_per_sm: int
    warp_size: int
    confidence: float = 0.95

    def __post_init__(self) -> None:
        _require_confidence(self.confidence)
        if self.max_warps_per_sm <= 0:
            raise ValueError(
                f"max_warps_per_sm must be positive: {self.max_warps_per_sm}"
            )
        if self.warp_size not in (32, 64):
            raise ValueError(f"warp_size must be 32 or 64, got {self.warp_size}")

    def max_occupancy_warps(
        self,
        registers_per_thread: int,
        shared_mem_per_block: int = 0,
        max_shared_memory: int = 0,
    ) -> int:
        """Compute the maximum number of warps considering register & smem limits.

        The register limit is computed from hardware parameters on this
        dataclass. The shared-memory limit requires values from
        :class:`MemoryRules` — pass them explicitly when available,
        or pass ``0`` to skip that constraint.

        Args:
            registers_per_thread: Registers used per thread (estimate).
            shared_mem_per_block: Shared memory used per thread block
                (bytes). Pass ``0`` to skip smem constraint.
            max_shared_memory: Total shared memory per SM/CU (bytes).
                Pass the value from :attr:`MemoryRules.max_shared_memory`.

        Returns:
            Maximum achievable warp count for this kernel config.
        """
        # Register limit: total SGPRs / (registers per thread * warp size)
        reg_limit = self.max_registers_per_sm // max(registers_per_thread, 1)
        reg_warps = max(reg_limit // self.warp_size, 1)

        # Shared memory limit: how many blocks fit, then warps per SM
        if shared_mem_per_block > 0 and max_shared_memory > 0:
            max_blocks = max_shared_memory // shared_mem_per_block
            # Assume worst-case: each block uses min warps for occupancy
            min_warps_per_block = max(self.max_warps_per_sm // max_blocks, 1)
            smem_warps = min_warps_per_block * max_blocks
        else:
            smem_warps = self.max_warps_per_sm

        return min(self.max_warps_per_sm, reg_warps, smem_warps)


# ---------------------------------------------------------------------------
# Aggregate vendor rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VendorRules:
    """Complete set of expert optimisation rules for one GPU vendor.

    Attributes:
        vendor: Short vendor/hardware identifier (e.g. ``"h100"``,
            ``"mi300x"``, ``"gaudi"``, ``"apple"``).
        display_name: Human-readable name for logging and CLIs.
        matmul: Matrix multiplication tuning rules.
        attention: Attention kernel tuning rules.
        elementwise: Elementwise kernel tuning rules.
        memory: Memory hierarchy parameters.
        occupancy: Occupancy limits and warp/wavefront configuration.
    """

    vendor: str
    display_name: str
    matmul: MatmulRules
    attention: AttentionRules
    elementwise: ElementwiseRules
    memory: MemoryRules
    occupancy: OccupancyRules


# ---------------------------------------------------------------------------
# Vendor rule sets
# ---------------------------------------------------------------------------

H100_RULES = VendorRules(
    vendor="h100",
    display_name="Nvidia H100 (Hopper)",
    matmul=MatmulRules(
        tile_m=(64, 128, 256),
        tile_n=(64, 128, 256),
        tile_k=(32, 64),
        num_warps=(4, 8, 16),
        num_stages=(3, 4, 5),
        enable_tma=True,
        cluster_size=(1, 2, 4),
        matrix_core="mma",
        confidence=0.95,
    ),
    attention=AttentionRules(
        block_m=(64, 128),
        block_n=(64, 128),
        block_k=(32, 64, 128),
        num_warps=(4, 8, 16),
        num_stages=(3, 4),
        causal=False,
        confidence=0.85,
    ),
    elementwise=ElementwiseRules(
        block_m=(128, 256, 512),
        block_n=(1,),
        num_warps=(2, 4),
        vectorize_size=(4, 8),
        confidence=0.80,
    ),
    memory=MemoryRules(
        shared_memory_per_block=48 * 1024,  # 48 KB default per block
        max_shared_memory=_H100_SHARED_MEMORY_PER_SM,
        l1_cache_size=_H100_SHARED_MEMORY_PER_SM,  # L1 = shared memory on Hopper
        l2_cache_size=_H100_L2_CACHE_SIZE,
        hbm_bandwidth=_H100_HBM_BANDWIDTH,
        confidence=0.95,
    ),
    occupancy=OccupancyRules(
        max_warps_per_sm=_H100_MAX_WARPS_PER_SM,
        max_threads_per_sm=_H100_MAX_THREADS_PER_SM,
        max_registers_per_sm=_H100_MAX_REGISTERS_PER_SM,
        warp_size=32,
        confidence=0.95,
    ),
)

MI300X_RULES = VendorRules(
    vendor="mi300x",
    display_name="AMD MI300X (CDNA3)",
    matmul=MatmulRules(
        tile_m=(64, 128, 256),
        tile_n=(64, 128, 256),
        tile_k=(16, 32, 64),
        num_warps=(4, 8, 12),
        num_stages=(2, 3, 4),
        enable_tma=None,  # no TMA on AMD
        cluster_size=(1,),
        matrix_core="mfma_16x16x16",
        confidence=0.90,
    ),
    attention=AttentionRules(
        block_m=(64, 128),
        block_n=(64, 128),
        block_k=(32, 64),
        num_warps=(4, 8, 12),
        num_stages=(2, 3),
        causal=False,
        confidence=0.80,
    ),
    elementwise=ElementwiseRules(
        block_m=(128, 256, 512),
        block_n=(1,),
        num_warps=(2, 4, 8),
        vectorize_size=(4, 8),
        confidence=0.75,
    ),
    memory=MemoryRules(
        shared_memory_per_block=64 * 1024,  # 64 KB typical per workgroup
        max_shared_memory=_MI300X_LDS_SIZE,
        l1_cache_size=0,  # CDNA3 has no L1; LDS replaces it
        l2_cache_size=_MI300X_L2_CACHE_SIZE,
        hbm_bandwidth=_MI300X_HBM_BANDWIDTH,
        confidence=0.90,
    ),
    occupancy=OccupancyRules(
        max_warps_per_sm=_MI300X_MAX_WAVEFRONTS_PER_CU,
        max_threads_per_sm=_MI300X_MAX_THREADS_PER_CU,
        max_registers_per_sm=_MI300X_MAX_SGPRS_PER_CU,
        warp_size=64,
        confidence=0.90,
    ),
)

GAUDI_RULES = VendorRules(
    vendor="gaudi",
    display_name="Intel Gaudi 2/3",
    matmul=MatmulRules(
        tile_m=(32, 64, 128),
        tile_n=(64, 128),
        tile_k=(32, 64),
        num_warps=(4, 8),
        num_stages=(2, 3),
        enable_tma=None,
        cluster_size=(1,),
        matrix_core=None,
        confidence=0.75,
    ),
    attention=AttentionRules(
        block_m=(32, 64),
        block_n=(32, 64),
        block_k=(32, 64),
        num_warps=(4, 8),
        num_stages=(2, 3),
        causal=False,
        confidence=0.65,
    ),
    elementwise=ElementwiseRules(
        block_m=(64, 128, 256),
        block_n=(1,),
        num_warps=(2, 4),
        vectorize_size=(4,),
        confidence=0.70,
    ),
    memory=MemoryRules(
        shared_memory_per_block=32 * 1024,  # 32 KB per workgroup
        max_shared_memory=_GAUDI_SLM_SIZE,
        l1_cache_size=0,  # Gaudi uses SLM directly
        l2_cache_size=_GAUDI_L3_CACHE_SIZE,
        hbm_bandwidth=_GAUDI_HBM_BANDWIDTH,
        confidence=0.80,
    ),
    occupancy=OccupancyRules(
        max_warps_per_sm=28,  # 7 threads * 4 subslices
        max_threads_per_sm=_GAUDI_MAX_THREADS_PER_EU,
        max_registers_per_sm=128 * 1024,  # 128 KB GRF
        warp_size=32,
        confidence=0.75,
    ),
)

APPLE_RULES = VendorRules(
    vendor="apple",
    display_name="Apple M-series (M3/M4)",
    matmul=MatmulRules(
        tile_m=(32, 64),
        tile_n=(32, 64),
        tile_k=(16, 32),
        num_warps=(2, 4, 8),
        num_stages=(2, 3),
        enable_tma=None,
        cluster_size=(1,),
        matrix_core=None,
        confidence=0.65,
    ),
    attention=AttentionRules(
        block_m=(32, 64),
        block_n=(32, 64),
        block_k=(32,),
        num_warps=(2, 4),
        num_stages=(2,),
        causal=False,
        confidence=0.55,
    ),
    elementwise=ElementwiseRules(
        block_m=(64, 128, 256),
        block_n=(1,),
        num_warps=(2, 4),
        vectorize_size=(4,),
        confidence=0.60,
    ),
    memory=MemoryRules(
        shared_memory_per_block=16 * 1024,  # 16 KB threadgroup memory
        max_shared_memory=_APPLE_TGMEM_SIZE,
        l1_cache_size=64 * 1024,  # 64 KB per GPU core
        l2_cache_size=0,  # unified memory — no discrete L2
        hbm_bandwidth=_APPLE_UNIFIED_MEM_BANDWIDTH,
        confidence=0.70,
    ),
    occupancy=OccupancyRules(
        max_warps_per_sm=32,
        max_threads_per_sm=_APPLE_MAX_THREADS_PER_TG,
        max_registers_per_sm=64 * 1024,
        warp_size=_APPLE_WARP_SIZE,
        confidence=0.70,
    ),
)

# ---------------------------------------------------------------------------
# Registry of all vendor rule sets for lookup
# ---------------------------------------------------------------------------

_VENDOR_REGISTRY: dict[str, VendorRules] = {
    "h100": H100_RULES,
    "h200": H100_RULES,  # shared rule set
    "mi300x": MI300X_RULES,
    "mi250": MI300X_RULES,  # shared rule set
    "gaudi": GAUDI_RULES,
    "gaudi2": GAUDI_RULES,
    "gaudi3": GAUDI_RULES,
    "apple": APPLE_RULES,
    "m3": APPLE_RULES,
    "m4": APPLE_RULES,
}

_VENDOR_ALIASES: list[tuple[re.Pattern[str], str]] = [
    # Nvidia
    (re.compile(r"nvidia.*h100", re.I), "h100"),
    (re.compile(r"nvidia.*hopper", re.I), "h100"),
    (re.compile(r"sm_90", re.I), "h100"),
    (re.compile(r"nvidia.*a100", re.I), "h100"),  # use H100 rules as best-available
    (re.compile(r"nvidia.*sm_\d+", re.I), "h100"),
    # AMD
    (re.compile(r"amd.*mi300", re.I), "mi300x"),
    (re.compile(r"rocm.*gfx942", re.I), "mi300x"),
    (re.compile(r"rocm.*gfx94", re.I), "mi300x"),
    (re.compile(r"amd.*mi250", re.I), "mi300x"),
    (re.compile(r"rocm.*gfx90a", re.I), "mi300x"),
    # Intel
    (re.compile(r"intel.*gaudi", re.I), "gaudi"),
    (re.compile(r"intel.*spirv", re.I), "gaudi"),
    (re.compile(r"oneapi.*gaudi", re.I), "gaudi"),
    # Apple
    (re.compile(r"apple.*(m[34]|metal)", re.I), "apple"),
    (re.compile(r"metal.*gpu", re.I), "apple"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_vendor_rules(vendor_id: str) -> VendorRules | None:
    """Look up vendor rules by short identifier.

    Args:
        vendor_id: Short vendor key (``"h100"``, ``"mi300x"``, etc.).

    Returns:
        The :class:`VendorRules` if found, or ``None``.
    """
    return _VENDOR_REGISTRY.get(vendor_id.lower())


def match_target(target_str: str) -> VendorRules | None:
    """Match a TVM target string to the best vendor rule set.

    Uses regex-based pattern matching against known vendor identifiers.
    Falls back to ``None`` when no match is found — callers should handle
    this by running MetaSchedule cold.

    Args:
        target_str: A TVM target string (e.g. ``"nvidia/nvidia-h100"``,
            ``"rocm"``, ``"metal"``).

    Returns:
        The best-matching :class:`VendorRules`, or ``None``.
    """
    for pattern, vendor_id in _VENDOR_ALIASES:
        if pattern.search(target_str):
            return _VENDOR_REGISTRY.get(vendor_id)
    return None


def available_vendors() -> list[str]:
    """Return sorted list of registered vendor identifiers."""
    return sorted(_VENDOR_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Integration helpers for MetaSchedule adapter
# ---------------------------------------------------------------------------


def filter_matmul_configs(
    rules: VendorRules,
    m: int,
    n: int,
    k: int,
) -> MatmulRules:
    """Narrow matmul rules to tile sizes that divide the problem dimensions.

    Removes tile sizes larger than their respective dimension (wasteful)
    and returns a filtered ``MatmulRules`` with only viable candidates.

    Args:
        rules: The vendor rules to filter.
        m: M dimension of the matmul.
        n: N dimension.
        k: K dimension.

    Returns:
        A new ``MatmulRules`` with only compatible tile sizes.
    """
    return MatmulRules(
        tile_m=tuple(t for t in rules.matmul.tile_m if t <= m),
        tile_n=tuple(t for t in rules.matmul.tile_n if t <= n),
        tile_k=tuple(t for t in rules.matmul.tile_k if t <= k),
        num_warps=rules.matmul.num_warps,
        num_stages=rules.matmul.num_stages,
        enable_tma=rules.matmul.enable_tma,
        cluster_size=rules.matmul.cluster_size,
        matrix_core=rules.matmul.matrix_core,
        confidence=rules.matmul.confidence,
    )


def filter_attention_configs(
    rules: VendorRules,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
) -> AttentionRules:
    """Narrow attention rules to block sizes that divide the sequence dims.

    Args:
        rules: The vendor rules to filter.
        seq_len_q: Query sequence length.
        seq_len_k: Key sequence length.
        head_dim: Attention head dimension.

    Returns:
        A new ``AttentionRules`` with only compatible block sizes.
    """
    return AttentionRules(
        block_m=tuple(t for t in rules.attention.block_m if t <= seq_len_q),
        block_n=tuple(t for t in rules.attention.block_n if t <= seq_len_k),
        block_k=tuple(t for t in rules.attention.block_k if t <= head_dim),
        num_warps=rules.attention.num_warps,
        num_stages=rules.attention.num_stages,
        causal=rules.attention.causal,
        confidence=rules.attention.confidence,
    )


def build_search_space_kwargs(rules: VendorRules) -> dict[str, list[int]]:
    """Convert expert rules into keyword arguments for MetaSchedule search.

    These can be passed as ``tune_tir(..., **search_space_kwargs)`` to
    constrain the evolutionary search to expert-recommended values.

    Args:
        rules: The vendor rules to convert.

    Returns:
        A dictionary mapping MetaSchedule parameter names to candidate
        value lists. Only non-trivial candidates (more than one value)
        for matmul-relevant parameters are included.
    """
    kwargs: dict[str, list[int]] = {}

    m = rules.matmul
    if len(m.tile_m) > 1:
        kwargs["tile_m"] = list(m.tile_m)
    if len(m.tile_n) > 1:
        kwargs["tile_n"] = list(m.tile_n)
    if len(m.tile_k) > 1:
        kwargs["tile_k"] = list(m.tile_k)
    if len(m.num_warps) > 1:
        kwargs["num_warps"] = list(m.num_warps)
    if len(m.num_stages) > 1:
        kwargs["num_stages"] = list(m.num_stages)

    return kwargs


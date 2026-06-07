"""Adaptive search strategies per kernel type x vendor combination.

Each :class:`SearchStrategy` is a frozen dataclass of tuning parameters
that control how TVM MetaSchedule explores the configuration space for
a specific (kernel_type, vendor) pair.  The module provides a catalog of
strategies tuned for each combination's dominant performance
characteristics (tensor-core vs LDS vs SIMD vs compute-bound), plus a
public :func:`get_strategy` function that selects the right one.

The strategy is **pure data** — it contains no control flow.  The caller
interprets the fields (e.g. by passing ``max_trials`` and
``num_trials_per_iter`` to the adapter) without branching on kernel or
vendor inside the adapter itself.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from src.common.logging import get_logger
from src.common.primitives import Vendor

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Kernel type — lightweight mirror of ir_capture.KernelKind so this module
# can be imported without dragging in the full IR capture pipeline.
# ---------------------------------------------------------------------------


class KernelType(Enum):
    """High-level kernel classification for strategy selection.

    This is intentionally a *subset* of ``ir_capture.KernelKind`` —
    strategies only differ for kernel shapes where the performance
    characteristics are meaningfully different.
    """

    MATMUL = auto()
    ATTENTION = auto()
    REDUCTION = auto()
    ELEMENTWISE = auto()
    SCAN = auto()
    PERSISTENT = auto()
    BROADCAST = auto()
    TRANSPOSE = auto()
    UNKNOWN = auto()

    @classmethod
    def from_kernel_kind(cls, kind: Any) -> KernelType:
        """Convert from ``ir_capture.KernelKind`` (or any enum with the
        same member names) to ``KernelType``.

        Falls back to ``UNKNOWN`` when the value can't be mapped so that
        callers never explode at import time.
        """
        try:
            return cls[kind.name]  # type: ignore[union-attr]
        except (AttributeError, KeyError, ValueError, TypeError):
            return cls.UNKNOWN


# ---------------------------------------------------------------------------
# The strategy data type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchStrategy:
    """Search strategy configuration for TVM MetaSchedule.

    Every field is a pure tuning parameter.  The caller (typically
    :class:`~src.bridges.triton_tvm.metaschedule_adapter.MetaScheduleAdapter`)
    maps these to TVM's ``tune_tir`` arguments.

    Attributes:
        population_size: Number of candidates per evolutionary generation.
            Maps to MetaSchedule's ``num_trials_per_iter``.
        mutation_rate: Probability of mutating a candidate's genes.
            Higher values encourage exploration of novel configurations.
        crossover_rate: Probability of recombining two parent candidates.
        elite_ratio: Fraction of top-performing candidates preserved
            unchanged into the next generation.
        max_trials: Total number of tuning trials across all generations.
            Maps to MetaSchedule's ``max_trials_global``.
        early_stop_generations: Number of generations without improvement
            after which tuning stops early.
        memory_bound_heuristic: When ``True`` the strategy emphasises
            memory-bandwidth optimisations (LDS tiling, fewer pipeline
            stages).  When ``False`` it emphasises compute optimisation
            (larger tiles, deeper pipelines, tensor-core usage).
        cache_enabled: Whether intermediate tuning results may be cached
            and reused across kernels of the same family.
        description: Human-readable description of what this strategy
            targets and why.
    """

    population_size: int = 64
    mutation_rate: float = 0.15
    crossover_rate: float = 0.75
    elite_ratio: float = 0.15
    max_trials: int = 500
    early_stop_generations: int = 10
    memory_bound_heuristic: bool = False
    cache_enabled: bool = True
    description: str = "Default strategy"

    def __post_init__(self) -> None:
        """Validate numeric bounds at construction time."""
        if self.population_size < 1:
            raise ValueError(
                f"population_size must be >= 1, got {self.population_size}",
            )
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError(
                f"mutation_rate must be in [0, 1], got {self.mutation_rate}",
            )
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError(
                f"crossover_rate must be in [0, 1], got {self.crossover_rate}",
            )
        if not (0.0 <= self.elite_ratio <= 1.0):
            raise ValueError(
                f"elite_ratio must be in [0, 1], got {self.elite_ratio}",
            )
        if self.max_trials < 1:
            raise ValueError(
                f"max_trials must be >= 1, got {self.max_trials}",
            )

    @property
    def num_trials_per_iter(self) -> int:
        """Convenience alias for MetaSchedule's ``num_trials_per_iter``."""
        return self.population_size

    @property
    def max_trials_global(self) -> int:
        """Convenience alias for MetaSchedule's ``max_trials_global``."""
        return self.max_trials

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat dict for logging / persistence."""
        return {
            "population_size": self.population_size,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elite_ratio": self.elite_ratio,
            "max_trials": self.max_trials,
            "early_stop_generations": self.early_stop_generations,
            "memory_bound_heuristic": self.memory_bound_heuristic,
            "cache_enabled": self.cache_enabled,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Strategy catalog — at least 8 distinct (kernel, vendor) combos
# ---------------------------------------------------------------------------
#
# Naming convention: _S_<kernel>_<vendor>
# Units are chosen so the search budget is appropriate for each target:
#   - Tensor-core targets (NVIDIA) → large pop, low mutation, compute-heavy
#   - LDS-heavy targets (AMD)      → medium pop, higher mutation, mem-heavy
#   - SIMD targets (Intel)          → medium pop, high mutation
#   - Tile-based targets (Apple)    → small pop, fast convergence

# --- Matmul ---------------------------------------------------------------
# NVIDIA H100/A100: tensor-core bound.  Large tiles, deep pipeline, low
# mutation — the optimal is near-deterministic for a given shape.
_S_MATMUL_NVIDIA = SearchStrategy(
    population_size=256,
    mutation_rate=0.10,
    crossover_rate=0.80,
    elite_ratio=0.20,
    max_trials=1000,
    early_stop_generations=20,
    memory_bound_heuristic=False,
    cache_enabled=True,
    description="Matmul x NVIDIA: large population, low mutation, tensor-core focused",
)

# AMD MI300X: LDS + MFMA bound.  Medium population with higher mutation
# to explore the larger LDS-placement search space.
_S_MATMUL_AMD = SearchStrategy(
    population_size=128,
    mutation_rate=0.25,
    crossover_rate=0.80,
    elite_ratio=0.20,
    max_trials=800,
    early_stop_generations=15,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Matmul x AMD: medium population, higher mutation, LDS focused",
)

# Intel Gaudi / Xe: SIMD bound.  Medium population, high mutation because
# the SIMD-width / sub-group trade-offs are less explored.
_S_MATMUL_INTEL = SearchStrategy(
    population_size=96,
    mutation_rate=0.30,
    crossover_rate=0.80,
    elite_ratio=0.15,
    max_trials=600,
    early_stop_generations=15,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Matmul x Intel: medium population, high mutation, SIMD-width emphasis",
)

# Apple M-series: tile-gang bound.  Smaller search space — fewer warp
# configurations to try.
_S_MATMUL_APPLE = SearchStrategy(
    population_size=64,
    mutation_rate=0.20,
    crossover_rate=0.75,
    elite_ratio=0.20,
    max_trials=400,
    early_stop_generations=10,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Matmul x Apple: small population, balanced mutation, tile-gang focused",
)

# --- Attention ------------------------------------------------------------
# NVIDIA: async-copy bound.  Pipeline depth is critical; emphasise
# register-pressure / shared-memory trade-offs.
_S_ATTENTION_NVIDIA = SearchStrategy(
    population_size=192,
    mutation_rate=0.10,
    crossover_rate=0.80,
    elite_ratio=0.15,
    max_trials=800,
    early_stop_generations=18,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Attention x NVIDIA: async-copy focused, pipeline-depth emphasis",
)

# AMD: LDS + async-copy.  Higher mutation to explore LDS bank-conflict
# avoidance patterns.
_S_ATTENTION_AMD = SearchStrategy(
    population_size=128,
    mutation_rate=0.20,
    crossover_rate=0.75,
    elite_ratio=0.15,
    max_trials=600,
    early_stop_generations=15,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Attention x AMD: LDS-balance emphasis, higher mutation",
)

# Intel: SIMD-width emphasis.  Attention on Gaudi benefits from wider
# sub-groups but the trade-off is architecture-specific.
_S_ATTENTION_INTEL = SearchStrategy(
    population_size=96,
    mutation_rate=0.25,
    crossover_rate=0.80,
    elite_ratio=0.15,
    max_trials=500,
    early_stop_generations=12,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Attention x Intel: SIMD-width emphasis, higher mutation",
)

_S_ATTENTION_APPLE = SearchStrategy(
    population_size=64,
    mutation_rate=0.20,
    crossover_rate=0.75,
    elite_ratio=0.15,
    max_trials=400,
    early_stop_generations=10,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Attention x Apple: balanced, small search space",
)

# --- Elementwise ----------------------------------------------------------
# All vendors: compute-bound pointwise ops.  Tiny search space — mostly
# vectorize width and loop unroll factors.  Fast convergence.
_S_ELEMENTWISE_NVIDIA = SearchStrategy(
    population_size=48,
    mutation_rate=0.15,
    crossover_rate=0.75,
    elite_ratio=0.10,
    max_trials=150,
    early_stop_generations=6,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Elementwise x NVIDIA: small search space, fast convergence",
)

_S_ELEMENTWISE_AMD = SearchStrategy(
    population_size=40,
    mutation_rate=0.20,
    crossover_rate=0.70,
    elite_ratio=0.10,
    max_trials=120,
    early_stop_generations=6,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Elementwise x AMD: small search space, fast convergence",
)

_S_ELEMENTWISE_INTEL = SearchStrategy(
    population_size=40,
    mutation_rate=0.25,
    crossover_rate=0.70,
    elite_ratio=0.10,
    max_trials=120,
    early_stop_generations=6,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Elementwise x Intel: small search space, SIMD emphasis",
)

_S_ELEMENTWISE_APPLE = SearchStrategy(
    population_size=32,
    mutation_rate=0.20,
    crossover_rate=0.70,
    elite_ratio=0.10,
    max_trials=100,
    early_stop_generations=5,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Elementwise x Apple: very small search space, fast convergence",
)

# --- Reduction ------------------------------------------------------------
# Memory-bound by definition — limited search space (block size, warps).
_S_REDUCTION_NVIDIA = SearchStrategy(
    population_size=48,
    mutation_rate=0.15,
    crossover_rate=0.70,
    elite_ratio=0.15,
    max_trials=200,
    early_stop_generations=8,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Reduction x NVIDIA: memory-bound, limited search space",
)

_S_REDUCTION_AMD = SearchStrategy(
    population_size=48,
    mutation_rate=0.20,
    crossover_rate=0.70,
    elite_ratio=0.15,
    max_trials=200,
    early_stop_generations=8,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Reduction x AMD: memory-bound, moderate mutation",
)

_S_REDUCTION_INTEL = SearchStrategy(
    population_size=40,
    mutation_rate=0.25,
    crossover_rate=0.70,
    elite_ratio=0.15,
    max_trials=160,
    early_stop_generations=8,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Reduction x Intel: memory-bound, higher mutation",
)

_S_REDUCTION_APPLE = SearchStrategy(
    population_size=32,
    mutation_rate=0.20,
    crossover_rate=0.65,
    elite_ratio=0.15,
    max_trials=120,
    early_stop_generations=6,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Reduction x Apple: small, fast convergence",
)

# --- Default / unknown ----------------------------------------------------
_S_DEFAULT = SearchStrategy(
    population_size=64,
    mutation_rate=0.15,
    crossover_rate=0.75,
    elite_ratio=0.15,
    max_trials=500,
    early_stop_generations=10,
    memory_bound_heuristic=False,
    cache_enabled=True,
    description="Default strategy for unclassified kernel-vendor combinations",
)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------
# A flat mapping from (KernelType, Vendor) → SearchStrategy.  This is the
# single source of truth that get_strategy queries.  Consumers that want to
# override or extend the registry can use register_strategy().

_STRATEGY_REGISTRY: dict[tuple[KernelType, Vendor], SearchStrategy] = {
    # Matmul
    (KernelType.MATMUL, Vendor.NVIDIA): _S_MATMUL_NVIDIA,
    (KernelType.MATMUL, Vendor.AMD): _S_MATMUL_AMD,
    (KernelType.MATMUL, Vendor.INTEL): _S_MATMUL_INTEL,
    (KernelType.MATMUL, Vendor.APPLE): _S_MATMUL_APPLE,
    # Attention
    (KernelType.ATTENTION, Vendor.NVIDIA): _S_ATTENTION_NVIDIA,
    (KernelType.ATTENTION, Vendor.AMD): _S_ATTENTION_AMD,
    (KernelType.ATTENTION, Vendor.INTEL): _S_ATTENTION_INTEL,
    (KernelType.ATTENTION, Vendor.APPLE): _S_ATTENTION_APPLE,
    # Elementwise
    (KernelType.ELEMENTWISE, Vendor.NVIDIA): _S_ELEMENTWISE_NVIDIA,
    (KernelType.ELEMENTWISE, Vendor.AMD): _S_ELEMENTWISE_AMD,
    (KernelType.ELEMENTWISE, Vendor.INTEL): _S_ELEMENTWISE_INTEL,
    (KernelType.ELEMENTWISE, Vendor.APPLE): _S_ELEMENTWISE_APPLE,
    # Reduction
    (KernelType.REDUCTION, Vendor.NVIDIA): _S_REDUCTION_NVIDIA,
    (KernelType.REDUCTION, Vendor.AMD): _S_REDUCTION_AMD,
    (KernelType.REDUCTION, Vendor.INTEL): _S_REDUCTION_INTEL,
    (KernelType.REDUCTION, Vendor.APPLE): _S_REDUCTION_APPLE,
}

# Less-common kernel types get a simpler mapping: per-vendor fallback
# for each, but we only have a few distinct strategies for SCAN,
# PERSISTENT, BROADCAST, TRANSPOSE.

_S_SCAN_COMMON = SearchStrategy(
    population_size=64,
    mutation_rate=0.20,
    crossover_rate=0.75,
    elite_ratio=0.15,
    max_trials=300,
    early_stop_generations=10,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Scan: memory-bound, moderate search space",
)

_S_PERSISTENT_COMMON = SearchStrategy(
    population_size=48,
    mutation_rate=0.15,
    crossover_rate=0.70,
    elite_ratio=0.15,
    max_trials=200,
    early_stop_generations=8,
    memory_bound_heuristic=False,
    cache_enabled=True,
    description="Persistent: compute-bound loops, moderate search space",
)

_S_BROADCAST_COMMON = SearchStrategy(
    population_size=32,
    mutation_rate=0.15,
    crossover_rate=0.70,
    elite_ratio=0.10,
    max_trials=100,
    early_stop_generations=5,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Broadcast: memory-bound, very small search space",
)

_S_TRANSPOSE_COMMON = SearchStrategy(
    population_size=32,
    mutation_rate=0.15,
    crossover_rate=0.70,
    elite_ratio=0.10,
    max_trials=100,
    early_stop_generations=5,
    memory_bound_heuristic=True,
    cache_enabled=True,
    description="Transpose: memory-bound, very small search space",
)

# Register the common strategies for all vendors.
for _vendor in Vendor:
    if _vendor is Vendor.UNKNOWN:
        continue
    _key_scan = (KernelType.SCAN, _vendor)
    if _key_scan not in _STRATEGY_REGISTRY:
        _STRATEGY_REGISTRY[_key_scan] = _S_SCAN_COMMON
    _key_persistent = (KernelType.PERSISTENT, _vendor)
    if _key_persistent not in _STRATEGY_REGISTRY:
        _STRATEGY_REGISTRY[_key_persistent] = _S_PERSISTENT_COMMON
    _key_broadcast = (KernelType.BROADCAST, _vendor)
    if _key_broadcast not in _STRATEGY_REGISTRY:
        _STRATEGY_REGISTRY[_key_broadcast] = _S_BROADCAST_COMMON
    _key_transpose = (KernelType.TRANSPOSE, _vendor)
    if _key_transpose not in _STRATEGY_REGISTRY:
        _STRATEGY_REGISTRY[_key_transpose] = _S_TRANSPOSE_COMMON


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_strategy(
    kernel_type: KernelType | Any,
    vendor: Vendor | str | None = None,
) -> SearchStrategy:
    """Return the best :class:`SearchStrategy` for a (kernel, vendor) pair.

    Args:
        kernel_type: A :class:`KernelType` enum member, or any object
            with a ``.name`` attribute matching a ``KernelType`` member
            (e.g. an ``ir_capture.KernelKind`` value).  When ``None``
            or unrecognised, falls back to the unknown default.
        vendor: A :class:`Vendor` enum member or a vendor string like
            ``"nvidia"``.  When ``None`` or unrecognised, falls back to
            ``Vendor.UNKNOWN``.

    Returns:
        A :class:`SearchStrategy` instance.  This is **always** a
        specific strategy — never ``None``.  Unrecognised inputs receive
        a conservative default strategy.

    Examples:
        >>> from src.common.primitives import Vendor
        >>> strategy = get_strategy(KernelType.MATMUL, Vendor.NVIDIA)
        >>> strategy.population_size
        256

        >>> # With a string vendor:
        >>> strategy = get_strategy(KernelType.ATTENTION, "amd")
        >>> strategy.memory_bound_heuristic
        True
    """
    # Normalise kernel_type.
    kt: KernelType
    if isinstance(kernel_type, KernelType):
        kt = kernel_type
    elif hasattr(kernel_type, "name"):
        kt = KernelType.from_kernel_kind(kernel_type)
    else:
        kt = KernelType.UNKNOWN

    # Normalise vendor.
    v: Vendor
    if isinstance(vendor, Vendor):
        v = vendor
    elif isinstance(vendor, str):
        v = Vendor.from_string(vendor, strict=False)
    else:
        v = Vendor.UNKNOWN

    strategy = _STRATEGY_REGISTRY.get((kt, v))
    if strategy is not None:
        return strategy

    # Vendor-level fallback: try with UNKNOWN vendor.
    fallback = _STRATEGY_REGISTRY.get((KernelType.UNKNOWN, v))
    if fallback is not None:
        return fallback

    # Last-resort global default.
    return _S_DEFAULT


def register_strategy(
    kernel_type: KernelType,
    vendor: Vendor,
    strategy: SearchStrategy,
    *,
    override: bool = False,
) -> None:
    """Register or override a strategy for a (kernel, vendor) pair.

    Args:
        kernel_type: The kernel type to register for.
        vendor: The vendor to register for.
        strategy: The strategy to associate.
        override: When ``False`` (default), registering an existing key
            raises :class:`KeyError`.  Pass ``True`` to silently replace.

    Raises:
        KeyError: If the key already exists and ``override`` is ``False``.
    """
    key = (kernel_type, vendor)
    if key in _STRATEGY_REGISTRY and not override:
        raise KeyError(
            f"Strategy already registered for {kernel_type.name} x {vendor.value}. "
            f"Set override=True to replace.",
        )
    _STRATEGY_REGISTRY[key] = strategy
    logger.info(
        "Registered strategy for %s x %s: %s",
        kernel_type.name,
        vendor.value,
        strategy.description,
    )


def list_strategies() -> dict[tuple[str, str], dict[str, Any]]:
    """Return a human-readable summary of all registered strategies.

    Returns:
        A dict mapping ``(kernel_type_name, vendor_value)`` tuples to
        strategy dicts.
    """
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for (kt, v), strat in sorted(
        _STRATEGY_REGISTRY.items(),
        key=lambda x: (x[0][0].name, x[0][1].value),
    ):
        result[(kt.name, v.value)] = strat.to_dict()
    return result


# ---------------------------------------------------------------------------
# Performance tracking
# ---------------------------------------------------------------------------


@dataclass
class StrategyRecord:
    """Record of a strategy application for performance tracking.

    Attached to a :class:`TuningResult` so that downstream consumers
    (benchmarks, dashboards, drift-detection CI) can analyse whether
    the strategy actually performed well.
    """

    kernel_type: str
    vendor: str
    strategy_name: str
    started_at: float = 0.0
    duration_seconds: float = 0.0
    trials_completed: int = 0
    best_score: float | None = None
    converged: bool = False
    cache_hit: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        return self.duration_seconds * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_type": self.kernel_type,
            "vendor": self.vendor,
            "strategy_name": self.strategy_name,
            "duration_seconds": self.duration_seconds,
            "trials_completed": self.trials_completed,
            "best_score": self.best_score,
            "converged": self.converged,
            "cache_hit": self.cache_hit,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Adapter integration helper
# ---------------------------------------------------------------------------


def strategy_to_tune_kwargs(
    strategy: SearchStrategy,
    *,
    override_population_size: int | None = None,
    override_max_trials: int | None = None,
) -> dict[str, Any]:
    """Convert a :class:`SearchStrategy` to kwargs for ``ms.tune_tir``.

    This is the bridge between the pure-data strategy and the
    MetaSchedule adapter's ``_run_tune_with_timeout`` method.

    Args:
        strategy: The selected strategy.
        override_population_size: Optional override for
            ``num_trials_per_iter``.
        override_max_trials: Optional override for
            ``max_trials_global``.

    Returns:
        A dict suitable for unpacking into ``ms.tune_tir``:
        ``max_trials_global`` and ``num_trials_per_iter``.
    """
    kwargs: dict[str, Any] = {
        "max_trials_global": (
            override_max_trials if override_max_trials is not None else strategy.max_trials
        ),
        "num_trials_per_iter": (
            override_population_size
            if override_population_size is not None
            else strategy.population_size
        ),
    }
    return kwargs


# ---------------------------------------------------------------------------
# Strategy persistence
# ---------------------------------------------------------------------------


def save_strategy_report(
    records: list[StrategyRecord],
    path: str | Path | None = None,
) -> str:
    """Persist a list of strategy application records to a JSON file.

    Args:
        records: One or more :class:`StrategyRecord` instances.
        path: Output path.  Defaults to
            ``~/.cache/nvindia_cud/strategy_report_{timestamp}.json``.

    Returns:
        The path the report was written to.
    """
    cache_dir = Path(
        os.environ.get(
            "NVINDIACUD_CACHE_DIR",
            str(Path.home() / ".cache" / "nvindia_cud"),
        ),
    )
    if path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = cache_dir / "strategy_reports" / f"report_{timestamp}.json"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "records": [r.to_dict() for r in records],
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved strategy report (%d records) to %s", len(records), path)
    return str(path)


__all__ = [
    "KernelType",
    "SearchStrategy",
    "StrategyRecord",
    "get_strategy",
    "list_strategies",
    "register_strategy",
    "save_strategy_report",
    "strategy_to_tune_kwargs",
]

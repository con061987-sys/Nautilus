"""TVMOptions — backend-specific options for the Triton ↔ TVM bridge.

This dataclass is constructed by TVMBackend.parse_options() and passed
through Triton's compilation pipeline. It carries the knobs that control
the bridge's behaviour at compile time, not the kernel parameters
themselves (those come from @triton.jit / @triton.autotune).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TVMOptions:
    """Immutable backend options consumed by TVMBackend.add_stages().

    These options are passed via triton.compile(target=..., options={...})
    and must survive a round-trip through Triton's cache key derivation.
    Therefore they must be hashable and serialisable.
    """

    # Target hardware (TVM target string, e.g. "nvidia/nvidia-h100")
    target: str = "nvidia/nvidia-h100"

    # How many MetaSchedule trials to run when tuning
    max_trials: int = 64

    # Trials per evolutionary search iteration
    num_trials_per_iter: int = 16

    # Search strategy: "evolutionary" | "replay-trace" | "replay-func"
    search_strategy: str = "evolutionary"

    # Cost model: "xgb" | "mlp" | "random"
    cost_model: str = "xgb"

    # Skip TVM entirely and use Triton's native autotuner with TVM's
    # suggested search space (hybrid mode)
    use_hybrid_autotune: bool = False

    # Cache directory for TVM databases
    work_dir: str = "/tmp/nvindia_cud_tuning"

    # Per-stage timeout in seconds
    tune_timeout_s: float = 600.0
    extract_timeout_s: float = 5.0
    build_timeout_s: float = 30.0

    # Whether to use the C++ plugin (if available) for IR capture
    use_native_plugin: bool = True

    # Free-form overrides for downstream consumers
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate options at construction time."""
        if self.max_trials < 1:
            raise ValueError(f"max_trials must be >= 1, got {self.max_trials}")
        if self.search_strategy not in ("evolutionary", "replay-trace", "replay-func"):
            raise ValueError(
                f"search_strategy must be one of 'evolutionary', 'replay-trace', "
                f"'replay-func'; got {self.search_strategy!r}"
            )
        if self.cost_model not in ("xgb", "mlp", "random"):
            raise ValueError(
                f"cost_model must be one of 'xgb', 'mlp', 'random'; "
                f"got {self.cost_model!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for Triton option forwarding."""
        return {
            "target": self.target,
            "max_trials": self.max_trials,
            "num_trials_per_iter": self.num_trials_per_iter,
            "search_strategy": self.search_strategy,
            "cost_model": self.cost_model,
            "use_hybrid_autotune": self.use_hybrid_autotune,
            "work_dir": self.work_dir,
            "tune_timeout_s": self.tune_timeout_s,
            "extract_timeout_s": self.extract_timeout_s,
            "build_timeout_s": self.build_timeout_s,
            "use_native_plugin": self.use_native_plugin,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TVMOptions:
        """Construct from a dict, ignoring unknown keys (with extras)."""
        known = {
            "target", "max_trials", "num_trials_per_iter", "search_strategy",
            "cost_model", "use_hybrid_autotune", "work_dir",
            "tune_timeout_s", "extract_timeout_s", "build_timeout_s",
            "use_native_plugin",
        }
        base = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(**base, extra=extra)

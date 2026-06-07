"""TVM MetaSchedule integration adapter.

Runs TVM MetaSchedule tuning on TIR templates and returns the
best configuration as a mapping compatible with Triton's Config.

This module implements the 'run_metaschedule' step in the config
bridge architecture with production-grade error handling, timeouts,
and caching.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from src.bridges.triton_tvm.config_mapper import ConfigMapper, MappedTuningConfig
from src.bridges.triton_tvm.timeout_manager import StageTimeoutError
from src.common.errors import TuningError
from src.common.logging import get_logger
from src.common.result import Err, Ok, Result

try:
    from tvm import meta_schedule as ms
    from tvm import tir
    from tvm.target import Target

    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False

logger = get_logger(__name__)


class MetaScheduleAdapter:
    """Adapter that runs TVM MetaSchedule and returns mapped configs.

    This is the core integration point between the bridge and TVM.
    It handles:
      - Running tune_tir() on TIR templates
      - Managing the tuning database (persistence, caching)
      - Querying the database for best configs
      - Graceful degradation when TVM is unavailable
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        default_max_trials: int = 64,
        default_num_trials_per_iter: int = 16,
        timeout_seconds: int = 600,
    ) -> None:
        self.cache_dir = cache_dir or os.environ.get(
            "NVINDIACUD_CACHE_DIR",
            str(Path.home() / ".cache" / "nvindia_cud"),
        )
        self.default_max_trials = default_max_trials
        self.default_num_trials_per_iter = default_num_trials_per_iter
        self.timeout_seconds = timeout_seconds
        self.config_mapper = ConfigMapper()

        if not TVM_AVAILABLE:
            logger.warning(
                "TVM not installed. MetaScheduleAdapter will return "
                "default configs. Install with: pip install apache-tvm"
            )

    def tune(
        self,
        tir_mod: Any,
        target_str: str,
        max_trials: int | None = None,
        num_trials_per_iter: int | None = None,
        cache_key: str | None = None,
    ) -> Result[MappedTuningConfig, TuningError]:
        """Run MetaSchedule tuning on a TIR module and return best config.

        Returns a Rust-style :class:`Result` so callers MUST handle
        both arms. On ``Ok(config)`` the config came from either a
        cache hit or a real MetaSchedule search; on ``Err(TuningError)``
        the failure reason is preserved with full context. Callers
        that want graceful degradation should map the error to
        :meth:`MappedTuningConfig.defaults` themselves — this method
        never lies about success.

        Args:
            tir_mod: TVM IRModule containing the TIR PrimFunc to tune.
            target_str: TVM target string (e.g., 'nvidia/nvidia-a100').
            max_trials: Max tuning trials (default: self.default_max_trials).
            num_trials_per_iter: Trials per evolution iteration.
            cache_key: Optional cache key for reusing previous results.

        Returns:
            ``Ok(MappedTuningConfig)`` on success (including cache hit);
            ``Err(TuningError)`` on any failure.
        """
        return self.tune_result(
            tir_mod=tir_mod,
            target_str=target_str,
            max_trials=max_trials,
            num_trials_per_iter=num_trials_per_iter,
            cache_key=cache_key,
        )

    def tune_result(
        self,
        tir_mod: Any,
        target_str: str,
        max_trials: int | None = None,
        num_trials_per_iter: int | None = None,
        cache_key: str | None = None,
    ) -> Result[MappedTuningConfig, TuningError]:
        """Run MetaSchedule tuning on a TIR module and return ``Result``.

        The Rust-style :class:`Result` makes failure modes explicit and
        forces callers to acknowledge errors. Cache hits return
        ``Ok(cached_config)``; TVM unavailability, timeouts, and
        recoverable runtime errors return ``Err(TuningError)``.

        Args:
            tir_mod: TVM IRModule containing the TIR PrimFunc to tune.
            target_str: TVM target string (e.g., 'nvidia/nvidia-a100').
            max_trials: Max tuning trials (default: self.default_max_trials).
            num_trials_per_iter: Trials per evolution iteration.
            cache_key: Optional cache key for reusing previous results.

        Returns:
            ``Ok(MappedTuningConfig)`` on success (including cache hit);
            ``Err(TuningError)`` on any failure.
        """
        if not TVM_AVAILABLE:
            logger.info("TVM unavailable, returning Err")
            return Err(
                TuningError(
                    "TVM is not installed; cannot run MetaSchedule tuning",
                    context={"target": target_str},
                )
            )

        max_trials = max_trials or self.default_max_trials
        num_trials_per_iter = num_trials_per_iter or self.default_num_trials_per_iter

        # Check cache first
        if cache_key:
            cached = self._check_cache(cache_key, target_str)
            if cached is not None:
                logger.info("Cache hit for key=%s target=%s", cache_key[:12], target_str)
                return Ok(cached)

        # Set up target and work directory
        try:
            target = Target(target_str)
        except (ValueError, TypeError) as exc:
            logger.error("Invalid TVM target %r: %s", target_str, exc)
            return Err(
                TuningError(
                    f"Invalid TVM target: {target_str}",
                    context={"target": target_str, "cause": str(exc)},
                )
            )

        work_dir = os.path.join(self.cache_dir, "tuning", _sanitize_target(target_str))

        # Run MetaSchedule with timeout
        try:
            os.makedirs(work_dir, exist_ok=True)
        except OSError as exc:
            logger.error("Cannot create work_dir %s: %s", work_dir, exc)
            return Err(
                TuningError(
                    f"Cannot create tuning work_dir: {work_dir}",
                    context={"work_dir": work_dir, "cause": str(exc)},
                )
            )

        logger.info(
            "Starting MetaSchedule tuning: target=%s, max_trials=%d, work_dir=%s",
            target_str,
            max_trials,
            work_dir,
        )

        # Specific exception handlers: we deliberately do NOT use
        # ``except Exception`` so that programmer errors (TypeError,
        # AttributeError, NameError) and unrelated interrupts propagate
        # to the caller instead of being silently masked.
        try:
            database = self._run_tune_with_timeout(
                tir_mod,
                target,
                work_dir,
                max_trials,
                num_trials_per_iter,
            )

            best = database.query_tuning_record(
                tir_mod, target, tir_mod.get_global_vars()[0].name_hint
            )

            if best is None:
                logger.warning("MetaSchedule returned no tuning records")
                return Err(
                    TuningError(
                        "MetaSchedule produced no tuning records",
                        context={"target": target_str, "max_trials": max_trials},
                    )
                )

            mapped = self.config_mapper.map_record(best)
            logger.info(
                "Tuning complete: block=(%d,%d,%d) warps=%d stages=%d",
                mapped.block_m,
                mapped.block_n,
                mapped.block_k,
                mapped.num_warps,
                mapped.num_stages,
            )

            if cache_key:
                try:
                    self._write_cache(cache_key, target_str, mapped)
                except (OSError, ValueError, TypeError) as cache_exc:
                    logger.warning(
                        "Cache write failed for key=%s: %s",
                        cache_key[:12],
                        cache_exc,
                    )

            return Ok(mapped)

        except StageTimeoutError as exc:
            logger.error(
                "MetaSchedule tuning timed out (budget=%.1fs elapsed=%.1fs)",
                exc.budget_s,
                exc.elapsed_s,
            )
            return Err(
                TuningError(
                    f"MetaSchedule tuning timed out after {exc.elapsed_s:.1f}s",
                    context={
                        "target": target_str,
                        "budget_s": exc.budget_s,
                        "elapsed_s": exc.elapsed_s,
                    },
                )
            )
        except (ValueError, TypeError) as exc:
            logger.error("MetaSchedule tuning rejected input: %s", exc)
            return Err(
                TuningError(
                    f"MetaSchedule tuning rejected input: {exc}",
                    context={"target": target_str, "cause": str(exc)},
                )
            )
        except ImportError as exc:
            logger.error(
                "MetaSchedule tuning: import failed mid-flight (%s)",
                exc,
            )
            return Err(
                TuningError(
                    f"MetaSchedule tuning: missing dependency: {exc}",
                    context={"target": target_str, "cause": str(exc)},
                )
            )
        except OSError as exc:
            logger.error(
                "MetaSchedule tuning: filesystem error (%s)",
                exc,
            )
            return Err(
                TuningError(
                    f"MetaSchedule tuning: filesystem error: {exc}",
                    context={"target": target_str, "cause": str(exc)},
                )
            )
        except RuntimeError as exc:
            logger.error(
                "MetaSchedule tuning: TVM runtime error (%s)",
                exc,
            )
            return Err(
                TuningError(
                    f"MetaSchedule tuning: TVM runtime error: {exc}",
                    context={"target": target_str, "cause": str(exc)},
                )
            )

    # ------------------------------------------------------------------
    # Private implementation
    # ------------------------------------------------------------------

    def _run_tune_with_timeout(
        self,
        tir_mod: Any,
        target: Any,
        work_dir: str,
        max_trials: int,
        num_trials_per_iter: int,
    ) -> Any:
        """Run tune_tir with a wall-clock timeout enforced via ``Thread.join``.

        Raises ``StageTimeoutError`` when the budget is exceeded. The
        tuner thread is left to finish in the background as a daemon
        and is killed when the process exits — TVM's tuning loop has
        no cooperative cancellation hook.
        """
        stage_name = "tvm_tune"
        budget = float(self.timeout_seconds)
        result_holder: list[Any] = []
        error_holder: list[BaseException] = []
        timed_out = threading.Event()

        start_time = time.time()

        def _run_tune() -> None:
            try:
                database = ms.tune_tir(
                    mod=tir_mod,
                    target=target,
                    work_dir=work_dir,
                    max_trials_global=max_trials,
                    num_trials_per_iter=num_trials_per_iter,
                    task_scheduler="gradient",
                    strategy="evolutionary",
                    seed=42,
                )
                if not timed_out.is_set():
                    result_holder.append(database)
            except BaseException as exc:
                if not timed_out.is_set():
                    error_holder.append(exc)

        worker = threading.Thread(
            target=_run_tune,
            name="metaschedule-tune",
            daemon=True,
        )
        worker.start()
        worker.join(timeout=budget)

        elapsed = time.time() - start_time

        if worker.is_alive():
            timed_out.set()
            logger.error(
                "MetaSchedule tuning exceeded budget of %.1fs (elapsed=%.1fs); "
                "aborting. work_dir=%s",
                budget,
                elapsed,
                work_dir,
            )
            raise StageTimeoutError(stage_name, budget, elapsed)

        if error_holder:
            exc = error_holder[0]
            logger.error("MetaSchedule tuning raised: %s", exc, exc_info=exc)
            raise exc

        if not result_holder:
            logger.error(
                "MetaSchedule worker returned no result and no error after %.1fs",
                elapsed,
            )
            raise StageTimeoutError(stage_name, budget, elapsed)

        database = result_holder[0]
        logger.info("MetaSchedule tuning took %.1f seconds", elapsed)
        return database

    def _check_cache(self, cache_key: str, target: str) -> MappedTuningConfig | None:
        """Check disk cache for existing tuning results."""
        cache_path = self._cache_path(cache_key, target)
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text())
            return MappedTuningConfig(**data)
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Cache read failed for %s: %s", cache_path, exc)
            cache_path.unlink(missing_ok=True)
            return None

    def _write_cache(self, cache_key: str, target: str, config: MappedTuningConfig) -> None:
        """Write tuning result to disk cache."""
        cache_path = self._cache_path(cache_key, target)
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
        logger.debug("Cached tuning result to %s", cache_path)

    def _cache_path(self, cache_key: str, target: str) -> Path:
        """Compute file path for cache entry."""
        safe_target = _sanitize_target(target)
        return Path(self.cache_dir) / "tuning_cache" / safe_target / f"{cache_key[:16]}.json"


def _sanitize_target(target: str) -> str:
    """Sanitize target string for use in file paths."""
    return target.replace("/", "_").replace(":", "_").replace(" ", "_")

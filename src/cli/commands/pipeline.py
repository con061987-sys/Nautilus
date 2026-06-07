"""`nautilus pipeline` — End-to-end pipeline orchestrator.

Wires all four Nautilus bridges together in a single command:

    1. Capture       — load the input file (Triton kernel or PyTorch model)
    2. Shard         — auto-shard the computation across a device mesh
    3. Extract       — pull the Triton kernels out of the captured graph
    4. Tune          — run TVM MetaSchedule per kernel
    5. Build         — compile tuned kernels into a per-shard fat binary
    6. Dispatch      — emit a dispatch plan + manifest for the cluster

The command is intentionally thin: it composes the per-bridge
APIs (GraphCapture → GSPMD → DTensor → TritonTVMBridge → FatBinaryBuilder)
and never re-implements the per-bridge logic. That means any
improvement in a single bridge automatically propagates here.

Flags
-----
* ``--resume-from STAGE``  restart from a specific stage (skip the
  previous ones). Stages are referenced by lowercase name
  (``capture``, ``shard``, ``extract``, ``tune``, ``build``,
  ``dispatch``). When used, stages 1..(N-1) are not executed; their
  artifacts are loaded from the output directory instead.
* ``--dry-run``            validate inputs and list what each stage
  *would* do, without performing the expensive work. Useful in CI
  to surface misconfigurations before kicking off a real build.
* ``--target``             one or more ``vendor/arch`` targets.
  Defaults to all vendors.
* ``--mesh``               mesh shape for the sharding stage.
* ``--output-dir``         where to write artifacts (defaults to
  ``./nautilus-out``).

Each stage is wrapped in a structured ``span`` so the per-stage
timing and error context are visible in the logs. If a stage fails,
the pipeline raises a typed ``NautilusError`` whose message
identifies the failing stage explicitly.

Graceful degradation
--------------------
* No torch → Capture/Shard become no-ops with a clear warning,
  pipeline reduces to Extract→Tune→Build→Dispatch for a single
  device.
* No TVM / Triton → Tune falls back to ``TuningConfig.defaults()``
  with a warning.
* No lld / gcc → Build still produces per-vendor object files but
  skips the linking step.
* No AMD AOTriton / Intel oneAPI → Build skips the corresponding
  vendor with a warning; the fat binary is still emitted for the
  vendors that succeeded.

Examples
--------

    # Full pipeline, default targets
    nautilus pipeline path/to/matmul.py

    # Dry run on a kernel for H100
    nautilus pipeline path/to/matmul.py --target nvidia/sm_90 --dry-run

    # Resume from the build stage after fixing a tuning config
    nautilus pipeline path/to/matmul.py --resume-from build
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import click

from src.common.errors import (
    BridgeError,
    ConfigError,
    DependencyMissingError,
    NautilusError,
)
from src.common.logging import (
    get_logger,
)
from src.common.logging import (
    span as span_context,
)
from src.common.logging import (
    stage as stage_context,
)
from src.common.types import HardwareTarget, TuningConfig, Vendor

log = get_logger("nautilus.cli.pipeline")


# ---------------------------------------------------------------------------
# Stage enum
# ---------------------------------------------------------------------------


class PipelineStage(str, Enum):
    """Ordered list of pipeline stages.

    The order is the contract: ``PipelineStage`` values are sorted by
    definition order. ``--resume-from`` parses a string into one of
    these and ``stages_from`` returns the suffix starting at the
    given stage.
    """

    CAPTURE = "capture"
    SHARD = "shard"
    EXTRACT = "extract"
    TUNE = "tune"
    BUILD = "build"
    DISPATCH = "dispatch"

    @classmethod
    def values(cls) -> list[str]:
        return [s.value for s in cls]

    @classmethod
    def from_str(cls, s: str) -> PipelineStage:
        try:
            return cls(s.lower())
        except ValueError as exc:
            raise ConfigError(
                f"Unknown pipeline stage {s!r}; valid stages: {cls.values()}",
                context={"requested": s, "valid": cls.values()},
            ) from exc

    @classmethod
    def stages_from(cls, start: PipelineStage) -> list[PipelineStage]:
        """Return all stages from ``start`` (inclusive) to the end."""
        all_stages = list(cls)
        idx = all_stages.index(start)
        return all_stages[idx:]


# ---------------------------------------------------------------------------
# Pipeline context (artifacts passed between stages)
# ---------------------------------------------------------------------------


@dataclass
class PipelineContext:
    """Mutable carrier for the per-stage artifacts.

    Each stage reads from / writes to this object. Stages that are
    skipped (via ``--resume-from``) populate their slot by *loading*
    the artifact from ``state.json`` rather than executing the
    stage body.
    """

    input_path: Path
    kernel_name: str = ""
    kernel_source: str = ""

    # Capture
    captured_graph_text: str = ""
    captured_graph_hash: str = ""

    # Shard
    mesh_axes: list[int] = field(default_factory=list)
    sharding_strategy: str = "auto"
    shard_count: int = 1
    sharding_cache_key: str = ""

    # Extract
    extracted_kernels: list[dict[str, Any]] = field(default_factory=list)

    # Tune
    tuning_configs: dict[str, dict[str, int]] = field(default_factory=dict)

    # Build
    fat_binary_paths: dict[str, Path] = field(default_factory=dict)
    build_stage_times: dict[str, float] = field(default_factory=dict)
    skipped_vendors: list[str] = field(default_factory=list)

    # Dispatch
    dispatch_plan: dict[str, Any] = field(default_factory=dict)

    # Bookkeeping
    target_strings: list[str] = field(default_factory=list)
    output_dir: Path | None = None
    dry_run: bool = False

    def save_state(self, path: Path) -> None:
        """Serialize the context to ``path`` (JSON)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # Path fields can't be JSON-serialised directly; convert.
        serialisable = {
            "input_path": str(self.input_path),
            "kernel_name": self.kernel_name,
            "kernel_source": self.kernel_source,
            "captured_graph_text": self.captured_graph_text,
            "captured_graph_hash": self.captured_graph_hash,
            "mesh_axes": list(self.mesh_axes),
            "sharding_strategy": self.sharding_strategy,
            "shard_count": self.shard_count,
            "sharding_cache_key": self.sharding_cache_key,
            "extracted_kernels": self.extracted_kernels,
            "tuning_configs": self.tuning_configs,
            "fat_binary_paths": {k: str(v) for k, v in self.fat_binary_paths.items()},
            "build_stage_times": self.build_stage_times,
            "skipped_vendors": self.skipped_vendors,
            "dispatch_plan": self.dispatch_plan,
            "target_strings": self.target_strings,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "dry_run": self.dry_run,
        }
        path.write_text(json.dumps(serialisable, indent=2, default=str))

    @classmethod
    def load_state(cls, path: Path) -> PipelineContext:
        """Inverse of :meth:`save_state`. Reconstructs Path fields."""
        blob = json.loads(path.read_text())
        ctx = cls(input_path=Path(blob["input_path"]))
        ctx.kernel_name = blob.get("kernel_name", "")
        ctx.kernel_source = blob.get("kernel_source", "")
        ctx.captured_graph_text = blob.get("captured_graph_text", "")
        ctx.captured_graph_hash = blob.get("captured_graph_hash", "")
        ctx.mesh_axes = list(blob.get("mesh_axes", []))
        ctx.sharding_strategy = blob.get("sharding_strategy", "auto")
        ctx.shard_count = int(blob.get("shard_count", 1))
        ctx.sharding_cache_key = blob.get("sharding_cache_key", "")
        ctx.extracted_kernels = list(blob.get("extracted_kernels", []))
        ctx.tuning_configs = dict(blob.get("tuning_configs", {}))
        ctx.fat_binary_paths = {k: Path(v) for k, v in blob.get("fat_binary_paths", {}).items()}
        ctx.build_stage_times = dict(blob.get("build_stage_times", {}))
        ctx.skipped_vendors = list(blob.get("skipped_vendors", []))
        ctx.dispatch_plan = dict(blob.get("dispatch_plan", {}))
        ctx.target_strings = list(blob.get("target_strings", []))
        out_dir = blob.get("output_dir")
        ctx.output_dir = Path(out_dir) if out_dir else None
        ctx.dry_run = bool(blob.get("dry_run", False))
        return ctx


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


@dataclass
class StageOutcome:
    """One stage's outcome, reported to the user via stdout + logs."""

    stage: PipelineStage
    success: bool
    duration_ms: float
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    skipped: bool = False
    dry_run: bool = False


class Pipeline:
    """The end-to-end pipeline orchestrator.

    The class is intentionally a thin wrapper: each stage method
    calls into the appropriate bridge and stores the artifact in
    :class:`PipelineContext`. Per-stage timeouts, circuit breakers,
    and graceful degradation are handled by the bridges themselves;
    the pipeline just observes their results.
    """

    def __init__(
        self,
        ctx: PipelineContext,
        targets: list[HardwareTarget],
        trials: int = 64,
        resume_from: PipelineStage | None = None,
        dry_run: bool = False,
        parallel: bool = True,
    ) -> None:
        self.ctx = ctx
        self.targets = targets
        self.trials = trials
        self.resume_from = resume_from
        self.dry_run = dry_run
        self.parallel = parallel
        self.outcomes: list[StageOutcome] = []
        self.state_path = (ctx.output_dir or Path("nautilus-out")) / "state.json"

    # -- public API --------------------------------------------------------

    def run(self) -> list[StageOutcome]:
        """Execute all stages in order. Returns per-stage outcomes."""
        start_stage = self.resume_from or PipelineStage.CAPTURE
        stages = PipelineStage.stages_from(start_stage)

        # If resuming, load state and skip the loadable stages.
        if self.resume_from is not None and self.resume_from != PipelineStage.CAPTURE:
            try:
                self.ctx = PipelineContext.load_state(self.state_path)
                # Preserve run-time overrides
                self.ctx.output_dir = self.ctx.output_dir or Path("nautilus-out")
                log.info(
                    "resumed from previous state",
                    state_path=str(self.state_path),
                    start_stage=start_stage.value,
                )
            except FileNotFoundError as exc:
                raise ConfigError(
                    f"Cannot resume from {start_stage.value!r}: "
                    f"no state file at {self.state_path}. Run the "
                    f"pipeline at least once without --resume-from first.",
                    context={"state_path": str(self.state_path)},
                ) from exc

        log.info(
            "pipeline starting",
            start_stage=start_stage.value,
            stages=[s.value for s in stages],
            targets=[t.to_tvm_target() for t in self.targets],
            dry_run=self.dry_run,
        )

        with span_context(
            "nautilus_pipeline",
            start_stage=start_stage.value,
            dry_run=self.dry_run,
            targets=[t.to_tvm_target() for t in self.targets],
        ) as sp:
            for stage in stages:
                handler = _STAGE_HANDLERS[stage]
                outcome = self._run_one(stage, handler, sp)
                self.outcomes.append(outcome)
                if not outcome.success and not outcome.skipped:
                    log.error(
                        "pipeline failed at stage",
                        stage=stage.value,
                        error=outcome.error,
                        duration_ms=outcome.duration_ms,
                    )
                    # Mark the span with the error and stop.
                    sp.set(failed_stage=stage.value, error=outcome.error)
                    break
                # Persist state after each successful stage so
                # --resume-from can recover from any later failure.
                if not self.dry_run:
                    try:
                        self.ctx.save_state(self.state_path)
                    except Exception as exc:
                        log.warning("could not persist state: %s", exc)

        return self.outcomes

    def _run_one(
        self,
        stage: PipelineStage,
        handler: Callable[[Pipeline], dict[str, Any]],
        sp: Any,
    ) -> StageOutcome:
        """Run a single stage and convert exceptions into StageOutcome."""
        if self.dry_run and stage != PipelineStage.DISPATCH:
            # Dispatch always runs in dry-run; it just emits a plan.
            outcome = StageOutcome(
                stage=stage,
                success=True,
                duration_ms=0.0,
                summary={"dry_run": True, "would_run": handler.__name__},
                dry_run=True,
            )
            self._print_outcome(outcome)
            return outcome
        t0 = time.perf_counter()
        try:
            with stage_context(sp, stage.value) as st:
                summary = handler(self)
                st.set(
                    **{k: v for k, v in summary.items() if isinstance(v, (str, int, float, bool))}
                )
        except NautilusError as exc:
            duration = (time.perf_counter() - t0) * 1000
            outcome = StageOutcome(
                stage=stage,
                success=False,
                duration_ms=duration,
                error=f"[{exc.code.value}] {exc.message}",
            )
            self._print_outcome(outcome)
            return outcome
        except Exception as exc:
            duration = (time.perf_counter() - t0) * 1000
            outcome = StageOutcome(
                stage=stage,
                success=False,
                duration_ms=duration,
                error=f"unexpected: {type(exc).__name__}: {exc}",
            )
            self._print_outcome(outcome)
            return outcome
        duration = (time.perf_counter() - t0) * 1000
        outcome = StageOutcome(
            stage=stage,
            success=True,
            duration_ms=duration,
            summary=summary,
        )
        self._print_outcome(outcome)
        return outcome

    def _print_outcome(self, outcome: StageOutcome) -> None:
        marker = "OK " if outcome.success else "FAIL"
        dry = " (dry-run)" if outcome.dry_run else ""
        summary_str = ""
        if outcome.summary:
            # Render the summary as a short, single-line hint.
            keys = (
                "kernel",
                "kernels",
                "shards",
                "targets",
                "vendors",
                "shard_count",
                "output_path",
                "would_run",
                "skipped",
                "available",
                "unavailable",
            )
            for k in keys:
                if k in outcome.summary:
                    v = outcome.summary[k]
                    summary_str = f" {k}={v}"
                    break
        click.echo(
            f"[{marker}] {outcome.stage.value:8s} {outcome.duration_ms:8.1f} ms{dry}{summary_str}",
            err=False,
        )

    # -- stage implementations --------------------------------------------

    def stage_capture(self) -> dict[str, Any]:
        """Stage 1: Capture — load the input file and extract the kernel.

        For a Triton kernel file: parses the AST to find the
        ``@triton.jit`` function. For a PyTorch model file: tries
        to import the model and capture its FX graph (best effort,
        degrades to skipping if torch is missing).
        """
        path = self.ctx.input_path
        log.info("capture stage", path=str(path))

        if not path.exists():
            raise ConfigError(
                f"Input file does not exist: {path}",
                context={"path": str(path)},
            )

        # Try Triton-kernel first
        try:
            from src.cli.commands.tune import _load_kernel_file

            name, text = _load_kernel_file(path)
            self.ctx.kernel_name = name
            self.ctx.kernel_source = text
            self.ctx.captured_graph_text = text
            self.ctx.captured_graph_hash = hashlib.sha256(
                text.encode("utf-8"),
            ).hexdigest()
            return {
                "kernel": name,
                "kind": "triton",
                "source_hash": self.ctx.captured_graph_hash[:12],
                "lines": text.count("\n") + 1,
            }
        except NautilusError:
            raise
        except Exception:
            # Not a Triton kernel file — fall through to PyTorch capture.
            pass

        # Try PyTorch model capture
        try:
            from src.bridges.pytorch_xla.graph_capture import GraphCapture
            from src.cli.commands.shard import _import_user_module
        except ImportError as exc:
            log.warning(
                "PyTorch capture unavailable; treating input as Triton kernel with no @triton.jit",
                error=str(exc),
            )
            raise KernelNotFoundError(
                f"No @triton.jit function in {path} and torch is not "
                f"available for model capture ({exc}).",
            )

        module = _import_user_module(path)
        model = None
        if hasattr(module, "Model") and isinstance(module.Model, type):
            model = module.Model()
        elif hasattr(module, "model"):
            model = module.model
        elif hasattr(module, "build_model"):
            model = module.build_model()
        if model is None:
            raise KernelNotFoundError(
                f"No @triton.jit function and no Model / model / build_model in {path}.",
            )

        # Capture with no example inputs — best effort; downstream
        # sharding will degrade gracefully.
        try:
            capturer = GraphCapture()
            captured = capturer.capture(
                model=model,
                example_inputs=(),
                model_name=model.__class__.__name__,
            )
            self.ctx.captured_graph_text = (
                captured.fx_graph_text or f"# model={model.__class__.__name__}"
            )
            self.ctx.captured_graph_hash = (
                captured.metadata.source_hash
                or hashlib.sha256(self.ctx.captured_graph_text.encode()).hexdigest()
            )
        except Exception as exc:
            log.warning("graph capture failed: %s; continuing", exc)
            self.ctx.captured_graph_text = f"# model={model.__class__.__name__}"
            self.ctx.captured_graph_hash = hashlib.sha256(
                self.ctx.captured_graph_text.encode(),
            ).hexdigest()

        # We still need a kernel to feed into the rest of the
        # pipeline. The pipeline falls back to a default kernel name
        # for non-Triton models so the build stage can still produce
        # a dispatch plan.
        self.ctx.kernel_name = model.__class__.__name__.lower()
        return {
            "kernel": self.ctx.kernel_name,
            "kind": "pytorch",
            "source_hash": self.ctx.captured_graph_hash[:12],
        }

    def stage_shard(self) -> dict[str, Any]:
        """Stage 2: Shard — auto-shard the captured graph.

        For a single-kernel input, this is essentially a no-op that
        produces a 1x1 mesh. For a PyTorch model with real sharding
        targets, this would call into the AutoShardingBridge.
        """
        log.info("shard stage", mesh_axes=self.ctx.mesh_axes)

        # If the user didn't pass a mesh, default to 1x1.
        if not self.ctx.mesh_axes:
            self.ctx.mesh_axes = [1]
        self.ctx.shard_count = 1
        for a in self.ctx.mesh_axes:
            self.ctx.shard_count *= int(a)

        # Try to use the real AutoShardingBridge; if torch/XLA is
        # missing or any other dependency is unavailable, log a
        # warning and produce a single-shard plan.
        try:
            from src.bridges.pytorch_xla import AutoShardingBridge, DeviceMesh
            from src.bridges.pytorch_xla.device_mesh import (
                DeviceVendor,
                InterconnectType,
                MeshDevice,
            )
        except ImportError as exc:
            log.warning(
                "AutoShardingBridge unavailable; using single-shard plan",
                error=str(exc),
            )
            self.ctx.sharding_cache_key = hashlib.sha256(
                json.dumps(self.ctx.mesh_axes).encode()
            ).hexdigest()
            return {
                "shards": self.ctx.shard_count,
                "mesh": self.ctx.mesh_axes,
                "strategy": self.ctx.sharding_strategy,
                "sharding_cache_key": self.ctx.sharding_cache_key[:12],
                "note": "AutoShardingBridge not importable; single-shard plan only",
            }

        # Build a synthetic device mesh matching the requested shape.
        # The real sharding bridge would accept a model + example
        # inputs; we don't have those here, so we emit a plan that
        # downstream stages can consume without re-running the bridge.
        try:
            total = self.ctx.shard_count
            # First vendor from targets — best effort vendor for the
            # synthetic mesh.
            vendor = self.targets[0].vendor if self.targets else Vendor.NVIDIA
            arch = self.targets[0].arch.value if self.targets else "sm_90"
            devices = [
                MeshDevice(
                    device_id=i,
                    vendor=DeviceVendor(vendor.value),
                    arch=arch,
                    memory_gb=80.0,
                    compute_tflops=989.0,
                    interconnect=InterconnectType.NVLINK,
                )
                for i in range(total)
            ]
            mesh = DeviceMesh(
                devices=devices,
                mesh_shape=list(self.ctx.mesh_axes),
            )
            self.ctx.sharding_cache_key = hashlib.sha256(
                json.dumps(
                    {
                        "mesh": list(self.ctx.mesh_axes),
                        "strategy": self.ctx.sharding_strategy,
                        "kernel": self.ctx.kernel_name,
                    }
                ).encode()
            ).hexdigest()
            # Note: we deliberately do not call AutoShardingBridge.shard()
            # here because that requires a torch model with example
            # inputs, which the single-kernel pipeline doesn't have.
            # The plan object is the artifact the rest of the pipeline
            # consumes.
            return {
                "shards": total,
                "mesh": list(mesh.mesh_shape),
                "strategy": self.ctx.sharding_strategy,
                "sharding_cache_key": self.ctx.sharding_cache_key[:12],
                "devices": [d.vendor.value for d in mesh.devices],
            }
        except Exception as exc:
            log.warning("shard stage failed: %s; falling back to 1x1", exc)
            self.ctx.mesh_axes = [1]
            self.ctx.shard_count = 1
            return {
                "shards": 1,
                "mesh": [1],
                "strategy": self.ctx.sharding_strategy,
                "sharding_cache_key": hashlib.sha256(b"single").hexdigest()[:12],
                "error": str(exc),
            }

    def stage_extract(self) -> dict[str, Any]:
        """Stage 3: Extract — pull kernels out of the captured graph.

        For a Triton-kernel input, this is essentially a confirmation
        that we have one kernel. For a PyTorch model, the real
        kernel extraction would call into the StableHLO→Triton
        translator.
        """
        log.info("extract stage", kernel_name=self.ctx.kernel_name)
        if not self.ctx.kernel_name:
            raise KernelNotFoundError(
                "Cannot extract kernels: no kernel_name in context. Capture stage must run first.",
            )
        if not self.ctx.kernel_source:
            raise KernelNotFoundError(
                "Cannot extract kernels: no kernel_source in context.",
            )
        kernel = {
            "name": self.ctx.kernel_name,
            "source_hash": hashlib.sha256(self.ctx.kernel_source.encode()).hexdigest()[:12],
            "lines": self.ctx.kernel_source.count("\n") + 1,
            "shard_id": 0,
        }
        self.ctx.extracted_kernels = [kernel]
        return {
            "kernels": len(self.ctx.extracted_kernels),
            "names": [k["name"] for k in self.ctx.extracted_kernels],
        }

    def stage_tune(self) -> dict[str, Any]:
        """Stage 4: Tune — run TVM MetaSchedule per kernel.

        For each extracted kernel, runs the bridge. If TVM/Triton
        is unavailable, falls back to the default block config.
        """
        log.info("tune stage", kernels=len(self.ctx.extracted_kernels), trials=self.trials)
        for kernel in self.ctx.extracted_kernels:
            name = kernel["name"]
            source_hash = kernel["source_hash"]
            config = self._tune_one_kernel(name, source_hash)
            self.ctx.tuning_configs[name] = {
                "block_m": config.block_m,
                "block_n": config.block_n,
                "block_k": config.block_k,
                "num_warps": config.num_warps,
                "num_stages": config.num_stages,
                "num_ctas": config.num_ctas,
            }
        return {
            "configs": list(self.ctx.tuning_configs.keys()),
            "trials": self.trials,
        }

    def _tune_one_kernel(
        self,
        kernel_name: str,
        source_hash: str,
    ) -> TuningConfig:
        """Tune a single kernel. Falls back to defaults on missing deps."""
        try:
            from src.bridges.triton_tvm.bridge_orchestrator import (
                TritonTVMBridge,
            )
            from src.bridges.triton_tvm.metadata_extractor import (
                KernelMetadata,
            )
        except ImportError as exc:
            log.warning(
                "Triton/TVM bridge unavailable; using default tuning",
                error=str(exc),
            )
            return TuningConfig.defaults()

        if not self.targets:
            return TuningConfig.defaults()

        target = self.targets[0]
        try:
            bridge = TritonTVMBridge(
                max_trials=self.trials,
                enable_cache=True,
            )
            metadata = KernelMetadata(
                kernel_name=kernel_name,
                source_hash=source_hash,
                grid_0=1,
                grid_1=1,
                grid_2=1,
                num_warps=4,
                num_stages=3,
                num_ctas=1,
            )
            mapped = bridge._tuning_chain(
                metadata,
                target.to_tvm_target(),
            )
            return TuningConfig(
                block_m=mapped.block_m,
                block_n=mapped.block_n,
                block_k=mapped.block_k,
                num_warps=mapped.num_warps,
                num_stages=mapped.num_stages,
                num_ctas=mapped.num_ctas,
            )
        except Exception as exc:
            log.warning(
                "tuning failed for %s; using defaults",
                kernel_name,
                error=str(exc),
            )
            return TuningConfig.defaults()

    def stage_build(self) -> dict[str, Any]:
        """Stage 5: Build — compile tuned kernels into a fat binary.

        Runs the FatBinaryBuilder for the first target only (the
        AOT packager produces a multi-vendor fat binary in a single
        build call). Parallelises the per-vendor compilation by
        issuing all vendor targets to the builder (the builder
        itself does the parallel work; the CLI just passes the
        config).
        """
        log.info("build stage", targets=len(self.targets))

        if not self.ctx.extracted_kernels:
            raise ConfigError(
                "No extracted kernels to build; extract stage must run first.",
            )

        kernel = self.ctx.extracted_kernels[0]
        config_dict = self.ctx.tuning_configs.get(
            kernel["name"],
            {
                f: getattr(TuningConfig.defaults(), f)
                for f in ("block_m", "block_n", "block_k", "num_warps", "num_stages", "num_ctas")
            },
        )

        # Map targets to FatBinaryConfig skip flags. The builder
        # accepts one (vendor, arch) pair at a time but iterates
        # the available backends internally; we use the per-vendor
        # skip flags to control which backends are actually run.
        vendors_present = {t.vendor for t in self.targets}
        skip_amd = Vendor.AMD not in vendors_present
        skip_intel = Vendor.INTEL not in vendors_present
        skip_nvidia = Vendor.NVIDIA not in vendors_present
        skip_apple = Vendor.APPLE not in vendors_present

        try:
            from src.bridges.aot_packager.builder import (
                FatBinaryBuilder,
                FatBinaryConfig,
            )
        except ImportError as exc:
            raise DependencyMissingError(
                f"AOT packager not importable: {exc}. Install with: pip install -e .",
            ) from exc

        out_dir = self.ctx.output_dir or Path("nautilus-out")
        out_dir.mkdir(parents=True, exist_ok=True)
        builder = FatBinaryBuilder(cache_dir=str(out_dir / "cache"))

        # Parallel option: build the kernel for each requested
        # (vendor, arch) in parallel, then merge the per-target
        # FatBinary outputs. The single FatBinaryConfig approach
        # compiles all vendors in one call, so when ``parallel`` is
        # True we run the builder once per target's leading vendor
        # and stitch the output. When ``parallel`` is False we run
        # one call with all backends.
        if self.parallel and len(self.targets) > 1:
            return self._build_parallel(
                builder,
                out_dir,
                config_dict,
                kernel,
                skip_amd,
                skip_intel,
                skip_nvidia,
                skip_apple,
            )

        return self._build_single(
            builder,
            out_dir,
            config_dict,
            kernel,
            skip_amd,
            skip_intel,
            skip_nvidia,
            skip_apple,
        )

    def _build_single(
        self,
        builder: Any,
        out_dir: Path,
        config_dict: dict[str, int],
        kernel: dict[str, Any],
        skip_amd: bool,
        skip_intel: bool,
        skip_nvidia: bool,
        skip_apple: bool,
    ) -> dict[str, Any]:
        """Single FatBinaryBuilder call covering all vendors."""
        from src.bridges.aot_packager.builder import FatBinaryConfig

        cfg = FatBinaryConfig(
            kernel_name=kernel["name"],
            kernel_source=self.ctx.kernel_source,
            block_m=config_dict["block_m"],
            block_n=config_dict["block_n"],
            block_k=config_dict["block_k"],
            num_warps=config_dict["num_warps"],
            num_stages=config_dict["num_stages"],
            output_dir=str(out_dir),
            skip_amd=skip_amd,
            skip_intel=skip_intel,
            skip_nvidia=skip_nvidia,
            skip_apple=skip_apple,
            skip_validation=True,
        )
        t0 = time.perf_counter()
        result = builder.build(cfg)
        elapsed = time.perf_counter() - t0
        self.ctx.build_stage_times["total_s"] = elapsed
        if result.output_path is not None:
            self.ctx.fat_binary_paths["primary"] = result.output_path
        for vendor_str in (
            "amd",
            "intel",
            "nvidia",
            "apple",
        ):
            if not getattr(cfg, f"skip_{vendor_str}", True):
                # The builder result exposes the per-vendor
                # compilation result; if it failed, mark skipped.
                sub = getattr(result, f"{vendor_str}_result", None)
                if sub is not None and not getattr(sub, "success", True):
                    if vendor_str not in self.ctx.skipped_vendors:
                        self.ctx.skipped_vendors.append(vendor_str)
        return {
            "output_path": (str(result.output_path) if result.output_path else None),
            "vendors": ([v.value for v in result.fat_binary.vendors] if result.fat_binary else []),
            "skipped": list(self.ctx.skipped_vendors),
            "elapsed_s": round(elapsed, 3),
        }

    def _build_parallel(
        self,
        builder: Any,
        out_dir: Path,
        config_dict: dict[str, int],
        kernel: dict[str, Any],
        skip_amd: bool,
        skip_intel: bool,
        skip_nvidia: bool,
        skip_apple: bool,
    ) -> dict[str, Any]:
        """Build per-vendor fat binaries in parallel and merge.

        Each future produces a fat binary with a single vendor's
        section; we record the output paths and present a unified
        summary. This trades build-time parallelism for the
        convenience of a single fat binary — useful when the user
        wants to keep each vendor's artifact separate for caching.
        """
        from src.bridges.aot_packager.builder import FatBinaryConfig

        per_vendor: dict[str, dict[str, bool]] = {
            "amd": {"skip_amd": False, "skip_intel": True, "skip_nvidia": True, "skip_apple": True},
            "intel": {
                "skip_amd": True,
                "skip_intel": False,
                "skip_nvidia": True,
                "skip_apple": True,
            },
            "nvidia": {
                "skip_amd": True,
                "skip_intel": True,
                "skip_nvidia": False,
                "skip_apple": True,
            },
            "apple": {
                "skip_amd": True,
                "skip_intel": True,
                "skip_nvidia": True,
                "skip_apple": False,
            },
        }

        # Only run vendors the user actually requested.
        requested = {
            "amd": not skip_amd,
            "intel": not skip_intel,
            "nvidia": not skip_nvidia,
            "apple": not skip_apple,
        }
        vendors_to_build = [v for v, req in requested.items() if req]
        if not vendors_to_build:
            vendors_to_build = ["nvidia"]  # Always build at least one

        vendor_subdir = out_dir / "per_vendor"
        vendor_subdir.mkdir(parents=True, exist_ok=True)

        def _build_one(vendor: str) -> tuple[str, str, float]:
            flags = per_vendor[vendor]
            sub_out = vendor_subdir / vendor
            sub_out.mkdir(parents=True, exist_ok=True)
            cfg = FatBinaryConfig(
                kernel_name=f"{kernel['name']}_{vendor}",
                kernel_source=self.ctx.kernel_source,
                block_m=config_dict["block_m"],
                block_n=config_dict["block_n"],
                block_k=config_dict["block_k"],
                num_warps=config_dict["num_warps"],
                num_stages=config_dict["num_stages"],
                output_dir=str(sub_out),
                skip_amd=flags["skip_amd"],
                skip_intel=flags["skip_intel"],
                skip_nvidia=flags["skip_nvidia"],
                skip_apple=flags["skip_apple"],
                skip_validation=True,
            )
            t0 = time.perf_counter()
            r = builder.build(cfg)
            elapsed = time.perf_counter() - t0
            out = str(r.output_path) if r.output_path else ""
            return vendor, out, elapsed

        paths: dict[str, str] = {}
        skipped: list[str] = []
        t_total = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, len(vendors_to_build))) as ex:
            futures = {ex.submit(_build_one, v): v for v in vendors_to_build}
            for fut in as_completed(futures):
                vendor, out_path, elapsed = fut.result()
                self.ctx.build_stage_times[vendor] = elapsed
                if out_path:
                    paths[vendor] = out_path
                    self.ctx.fat_binary_paths[vendor] = Path(out_path)
                else:
                    skipped.append(vendor)
                    if vendor not in self.ctx.skipped_vendors:
                        self.ctx.skipped_vendors.append(vendor)
        total = time.perf_counter() - t_total
        self.ctx.build_stage_times["total_s"] = total
        return {
            "output_paths": paths,
            "skipped": skipped,
            "elapsed_s": round(total, 3),
        }

    def stage_dispatch(self) -> dict[str, Any]:
        """Stage 6: Dispatch — emit a dispatch plan for the cluster.

        For a real run, the dispatch plan tells the cluster runtime
        which shard gets which fat binary and which device. In
        dry-run, the plan is purely informational.
        """
        log.info("dispatch stage", shards=self.ctx.shard_count)
        plan = {
            "kernel_name": self.ctx.kernel_name,
            "shards": self.ctx.shard_count,
            "mesh_axes": self.ctx.mesh_axes,
            "tuning": self.ctx.tuning_configs,
            "fat_binaries": {k: str(v) for k, v in self.ctx.fat_binary_paths.items()},
            "skipped_vendors": list(self.ctx.skipped_vendors),
            "sharding_cache_key": self.ctx.sharding_cache_key,
            "dry_run": self.dry_run,
        }
        # Write a manifest next to the state file so downstream
        # tooling (e.g. ``nautilus inspect``) can consume it.
        if self.ctx.output_dir is not None and not self.dry_run:
            manifest = self.ctx.output_dir / "dispatch_plan.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(plan, indent=2, default=str))
        self.ctx.dispatch_plan = plan
        return plan


# Re-export for callers that need KernelNotFoundError without an
# extra import (the stage_capture method raises it).
from src.common.errors import KernelNotFoundError  # noqa: E402

# Map stages to handler methods.  Defined after the class so the
# bound method lookup is unambiguous.
_STAGE_HANDLERS: dict[PipelineStage, Callable[[Pipeline], dict[str, Any]]] = {}


def _register_handlers() -> None:
    if _STAGE_HANDLERS:
        return
    _STAGE_HANDLERS[PipelineStage.CAPTURE] = lambda p: p.stage_capture()
    _STAGE_HANDLERS[PipelineStage.SHARD] = lambda p: p.stage_shard()
    _STAGE_HANDLERS[PipelineStage.EXTRACT] = lambda p: p.stage_extract()
    _STAGE_HANDLERS[PipelineStage.TUNE] = lambda p: p.stage_tune()
    _STAGE_HANDLERS[PipelineStage.BUILD] = lambda p: p.stage_build()
    _STAGE_HANDLERS[PipelineStage.DISPATCH] = lambda p: p.stage_dispatch()


_register_handlers()


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


def _parse_targets(target_strs: tuple[str, ...]) -> list[HardwareTarget]:
    """Parse ``--target`` values into HardwareTarget objects.

    Re-uses ``tune._parse_target`` so the syntax is identical to
    ``nautilus tune`` and ``nautilus build``. Empty tuple means
    "all vendors" — returns a default list covering Nvidia, AMD,
    Intel, and Apple.
    """
    if not target_strs:
        # Default to one target per vendor so the fat binary has
        # something to compile.
        from src.cli.commands.tune import _parse_target

        return [
            _parse_target("nvidia/sm_90"),
            _parse_target("amd/gfx942"),
            _parse_target("intel/intel_gpu_xehpg"),
        ]
    from src.cli.commands.tune import _parse_target

    return [_parse_target(t) for t in target_strs]


def _parse_mesh(mesh_str: str | None) -> list[int]:
    """Parse ``--mesh`` into a list of axis sizes. Empty list = 1x1."""
    if not mesh_str:
        return []
    try:
        return [int(x) for x in mesh_str.split(",") if x.strip()]
    except ValueError as exc:
        raise ConfigError(
            f"Invalid --mesh {mesh_str!r}: {exc}",
            context={"mesh": mesh_str},
        ) from exc


@click.command(
    "pipeline",
    short_help="Run the full cross-bridge pipeline (capture → dispatch)",
    help="""
End-to-end pipeline that wires all four Nautilus bridges together:

    1. Capture   — load the input file (Triton kernel or PyTorch model)
    2. Shard     — auto-shard the computation across a device mesh
    3. Extract   — pull the Triton kernels out of the captured graph
    4. Tune      — run TVM MetaSchedule per kernel
    5. Build     — compile tuned kernels into per-vendor fat binaries
    6. Dispatch  — emit a dispatch plan + manifest for the cluster

Each stage is independently observable: structured logs and a one-
line per-stage summary are printed to stdout. If a stage fails,
the pipeline stops, the failing stage is named in the error, and
``--resume-from`` can be used to restart from that stage.

Use ``--dry-run`` to validate inputs and emit a per-stage plan
without doing the expensive work. The pipeline degrades gracefully
when a dependency (TVM, torch, lld) is missing — see the module
docstring for the full degradation matrix.

Examples:

  # Full pipeline with default targets
  nautilus pipeline path/to/matmul.py

  # Dry run on a single target
  nautilus pipeline path/to/matmul.py --target nvidia/sm_90 --dry-run

  # Resume from the build stage after fixing a tuning config
  nautilus pipeline path/to/matmul.py --resume-from build
""",
)
@click.argument(
    "input_file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--target",
    "-t",
    "targets",
    multiple=True,
    help="Target hardware as 'vendor/arch' (e.g. nvidia/sm_90, amd/gfx942). "
    "Can be passed multiple times. Default: one per vendor.",
)
@click.option(
    "--mesh",
    "-m",
    default=None,
    help="Device mesh as comma-separated axes (e.g. '2,2'). Default: 1x1.",
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(
        ["auto", "replicated", "data_parallel", "model_parallel", "tensor_parallel"],
        case_sensitive=False,
    ),
    default="auto",
    show_default=True,
    help="Sharding strategy hint.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    default=Path("./nautilus-out"),
    show_default=True,
    help="Where to write per-stage artifacts and the dispatch plan.",
)
@click.option(
    "--resume-from",
    type=click.Choice(PipelineStage.values(), case_sensitive=False),
    default=None,
    help="Restart from a specific stage, using state.json from a previous run.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate inputs and print a per-stage plan without doing expensive work.",
)
@click.option(
    "--trials",
    "-n",
    type=click.IntRange(min=1, max=10000),
    default=64,
    show_default=True,
    help="Number of MetaSchedule trials per kernel.",
)
@click.option(
    "--no-parallel",
    is_flag=True,
    default=False,
    help="Disable per-vendor build parallelism (default: parallel).",
)
def cli(
    input_file: Path,
    targets: tuple[str, ...],
    mesh: str | None,
    strategy: str,
    output_dir: Path,
    resume_from: str | None,
    dry_run: bool,
    trials: int,
    no_parallel: bool,
) -> None:
    """Run the full cross-bridge pipeline."""
    try:
        _pipeline_impl(
            input_file=input_file,
            target_strs=list(targets),
            mesh_str=mesh,
            strategy=strategy,
            output_dir=output_dir,
            resume_from=resume_from,
            dry_run=dry_run,
            trials=trials,
            parallel=not no_parallel,
        )
    except NautilusError as exc:
        click.echo(f"nautilus: {exc.message}", err=True)
        if exc.context:
            click.echo(f"  context: {exc.context}", err=True)
        sys.exit(2)
    except KeyboardInterrupt:
        click.echo("nautilus: interrupted", err=True)
        sys.exit(130)


def _pipeline_impl(
    input_file: Path,
    target_strs: list[str],
    mesh_str: str | None,
    strategy: str,
    output_dir: Path,
    resume_from: str | None,
    dry_run: bool,
    trials: int,
    parallel: bool,
) -> None:
    """Implementation backing the click command.

    Separated from the click decorator so tests can call it
    directly without going through ``click.testing.CliRunner``.
    """
    hardware_targets = _parse_targets(tuple(target_strs))
    mesh_axes = _parse_mesh(mesh_str)
    resume_stage = PipelineStage.from_str(resume_from) if resume_from else None
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = PipelineContext(
        input_path=input_file,
        mesh_axes=mesh_axes,
        sharding_strategy=strategy,
        target_strings=[t.to_tvm_target() for t in hardware_targets],
        output_dir=output_dir,
        dry_run=dry_run,
    )
    pipeline = Pipeline(
        ctx=ctx,
        targets=hardware_targets,
        trials=trials,
        resume_from=resume_stage,
        dry_run=dry_run,
        parallel=parallel,
    )
    log.info(
        "pipeline invoked",
        input=str(input_file),
        targets=[t.to_tvm_target() for t in hardware_targets],
        mesh=mesh_axes,
        strategy=strategy,
        output_dir=str(output_dir),
        resume_from=resume_stage.value if resume_stage else None,
        dry_run=dry_run,
        parallel=parallel,
    )
    click.echo(
        f"nautilus pipeline: input={input_file.name} "
        f"targets={[t.to_tvm_target() for t in hardware_targets]} "
        f"mesh={mesh_axes} strategy={strategy} "
        f"{'(dry-run)' if dry_run else ''}",
    )
    outcomes = pipeline.run()
    # Final summary
    failed = [o for o in outcomes if not o.success and not o.skipped]
    click.echo("")
    click.echo("=" * 60)
    click.echo("PIPELINE SUMMARY")
    click.echo("=" * 60)
    for o in outcomes:
        marker = "OK  " if o.success else "FAIL"
        dry = " (dry-run)" if o.dry_run else ""
        skip = " (skipped)" if o.skipped else ""
        click.echo(
            f"  [{marker}] {o.stage.value:8s} {o.duration_ms:8.1f} ms{dry}{skip}",
        )
    click.echo("")
    if failed:
        failed_stage = failed[0].stage.value
        raise BridgeError(
            f"Pipeline failed at stage '{failed_stage}': {failed[0].error}",
            context={
                "failed_stage": failed_stage,
                "stage_error": failed[0].error,
            },
        )
    # Write a final summary JSON for tooling to consume.
    summary = {
        "input": str(input_file),
        "output_dir": str(output_dir),
        "dry_run": dry_run,
        "stages": [
            {
                "stage": o.stage.value,
                "success": o.success,
                "duration_ms": o.duration_ms,
                "dry_run": o.dry_run,
                "skipped": o.skipped,
                "summary": o.summary,
            }
            for o in outcomes
        ],
        "kernel_name": ctx.kernel_name,
        "mesh_axes": ctx.mesh_axes,
        "shard_count": ctx.shard_count,
        "fat_binary_paths": {k: str(v) for k, v in ctx.fat_binary_paths.items()},
        "dispatch_plan": ctx.dispatch_plan,
    }
    summary_path = output_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    click.echo(f"Summary written to {summary_path}")


if __name__ == "__main__":
    cli()

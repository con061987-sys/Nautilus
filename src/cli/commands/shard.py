"""`nautilus shard` — Shard a PyTorch model across a device mesh.

Captures a PyTorch model as a TorchFX graph, converts to StableHLO,
runs GSPMD to compute optimal sharding, and emits per-shard fat
binaries built from the per-shard StableHLO translated to Triton
via ``stablehlo_to_triton``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import click

from src.bridges.pytorch_xla import (
    AutoShardingBridge,
    DeviceMesh,
    ShardingConfig,
)
from src.bridges.pytorch_xla.device_mesh import (
    DeviceVendor,
    InterconnectType,
    MeshDevice,
)
from src.bridges.pytorch_xla.gspmd_runner import ShardingStrategy
from src.common.errors import (
    NautilusError,
    ShardingError,
)
from src.common.logging import get_logger
from src.common.logging import span as span_context
from src.common.types import MeshShape

log = get_logger("nautilus.cli.shard")


@click.command(
    "shard",
    short_help="Shard a PyTorch model across a device mesh",
    help="""
Capture a PyTorch model, convert to StableHLO, run GSPMD, translate
the sharded StableHLO to Triton via stablehlo_to_triton, and emit
per-shard fat binaries.

Examples:

  # Shard a 2-layer MLP across 2x2 mesh
  nautilus shard path/to/model.py --mesh 2,2 --output-dir ./shards

  # Use a custom strategy
  nautilus shard path/to/model.py --mesh 4 --strategy data_parallel
""",
)
@click.argument(
    "model_file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--mesh",
    "-m",
    required=True,
    help="Device mesh as comma-separated axes (e.g. '2,2' for 2x2, '4' for 1x4).",
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
    help="Sharding strategy hint to GSPMD.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    default=Path("./shards"),
    show_default=True,
    help="Where to write per-shard artifacts.",
)
@click.option(
    "--example-inputs",
    "-e",
    type=str,
    default=None,
    help="Comma-separated example input shapes (e.g. '1,3,224,224'). "
    "Required if the model file does not define EXAMPLE_INPUTS.",
)
def cli(
    model_file: Path,
    mesh: str,
    strategy: str,
    output_dir: Path,
    example_inputs: str | None,
) -> None:
    """Shard a model."""
    try:
        _shard_impl(model_file, mesh, strategy, output_dir, example_inputs)
    except NautilusError as exc:
        click.echo(f"nautilus: {exc.message}", err=True)
        if exc.context:
            click.echo(f"  context: {exc.context}", err=True)
        sys.exit(2)
    except KeyboardInterrupt:
        click.echo("nautilus: interrupted", err=True)
        sys.exit(130)


def _shard_impl(
    model_file: Path,
    mesh_str: str,
    strategy: str,
    output_dir: Path,
    example_inputs: str | None,
) -> None:
    """Run the auto-sharding pipeline via ``AutoShardingBridge.shard``.

    The CLI is a thin wrapper: it loads the model + example inputs
    from disk, builds a synthetic device mesh that matches the
    requested ``--mesh`` shape, hands everything to the bridge, and
    then writes per-shard artifacts (``stablehlo.mlir``, ``shard_spec.json``,
    ``kernel.fat.o``) from the bridge's ``ShardingResult``.

    The bridge owns the full 6-stage pipeline (graph capture → StableHLO
    export → GSPMD → DTensor → fat-binary per shard → dispatch) — the
    CLI does not re-implement any of those stages.
    """
    axes = tuple(int(x) for x in mesh_str.split(","))
    try:
        mesh_shape = MeshShape(axes=axes)
    except Exception as exc:
        raise NautilusError(
            f"Invalid mesh {mesh_str!r}: {exc}",
            context={"mesh_str": mesh_str},
        ) from exc
    log.info(
        "sharding started",
        model=str(model_file),
        mesh=list(mesh_shape.axes),
        strategy=strategy,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the synthetic device mesh and load the model. Both are
    # best-effort: a failure here produces a clear NautilusError
    # before the bridge is called.
    device_mesh = _build_device_mesh(mesh_shape)
    model, resolved_example_inputs = _load_model_and_inputs(
        model_file=model_file,
        cli_example_inputs=example_inputs,
    )

    strategy_map = {
        "auto": ShardingStrategy.AUTO,
        "replicated": ShardingStrategy.REPLICATED,
        "data_parallel": ShardingStrategy.DATA_PARALLEL,
        "model_parallel": ShardingStrategy.MODEL_PARALLEL,
        "tensor_parallel": ShardingStrategy.TENSOR_PARALLEL,
    }

    config = ShardingConfig(
        model=model,
        example_inputs=resolved_example_inputs,
        device_mesh=device_mesh,
        sharding_strategy=strategy_map[strategy.lower()],
    )

    bridge = AutoShardingBridge()
    with span_context("auto_shard", model=str(model_file), mesh=list(mesh_shape.axes)) as sp:
        result = bridge.shard(
            model=model,
            example_inputs=resolved_example_inputs,
            device_mesh=device_mesh,
            config=config,
        )
        sp.set(
            success=result.success,
            total_duration_ms=result.total_duration_ms,
            error=result.error,
        )

    if not result.success:
        raise ShardingError(
            f"Auto-sharding failed: {result.error or 'unknown error'}",
            context={
                "model": str(model_file),
                "mesh": list(mesh_shape.axes),
                "stage_durations": result.stage_durations,
            },
        )

    # Per-shard artifacts — written from the bridge's output.
    shards_emitted = 0
    for shard_idx, shard_exec in enumerate(result.shard_executions):
        shard_dir = output_dir / f"shard_{shard_idx:04d}"
        shard_dir.mkdir(exist_ok=True)

        # StableHLO + sharding spec are still useful for debugging,
        # even though the actual compute now lives in the fat binary.
        if result.stablehlo_module is not None and result.stablehlo_module.mlir_text:
            (shard_dir / "stablehlo.mlir").write_text(
                result.stablehlo_module.mlir_text,
            )
        if result.gspmd_result is not None and result.gspmd_result.sharding_spec is not None:
            spec = result.gspmd_result.sharding_spec
            (shard_dir / "shard_spec.json").write_text(
                json.dumps(
                    {
                        "shard_id": shard_idx,
                        "mesh": list(mesh_shape.axes),
                        "vendor": shard_exec.vendor,
                        "arch": shard_exec.arch,
                        "device_id": shard_exec.device_id,
                        "tensor_shardings": {
                            n: {
                                "axes": list(s.mesh_axes),
                                "shape": list(s.partition_shape),
                            }
                            for n, s in spec.tensor_shardings.items()
                        },
                        "estimated_comm_volume_bytes": spec.estimated_comm_volume_bytes,
                    },
                    indent=2,
                )
            )

        # The fat binary: per-shard product of StableHLO→Triton
        # translation (Wave 1.1) + per-vendor AOT compilation (Phase 2).
        fat = shard_exec.fat_binary_result
        if fat is not None and fat.output_path is not None and fat.output_path.exists():
            (shard_dir / "kernel.fat.o").write_bytes(
                fat.output_path.read_bytes(),
            )
        elif fat is not None and fat.fat_binary is not None:
            # Fallback: serialise the FatBinary in-memory container
            # (used in tests / environments without a real linker).
            (shard_dir / "kernel.fat.o").write_bytes(
                fat.fat_binary.to_bytes(),
            )
        else:
            log.warning(
                "no fat binary for shard %d (vendor=%s arch=%s); skipped kernel.fat.o",
                shard_idx,
                shard_exec.vendor,
                shard_exec.arch,
            )

        shards_emitted += 1

    log.info(
        "sharding complete",
        shards=shards_emitted,
        output_dir=str(output_dir),
    )
    click.echo(
        json.dumps(
            {
                "model": str(model_file),
                "mesh": list(mesh_shape.axes),
                "strategy": strategy,
                "shards_emitted": shards_emitted,
                "tensor_shardings": (
                    len(result.gspmd_result.sharding_spec.tensor_shardings)
                    if result.gspmd_result is not None
                    and result.gspmd_result.sharding_spec is not None
                    else 0
                ),
                "comm_volume_bytes": (
                    result.gspmd_result.sharding_spec.estimated_comm_volume_bytes
                    if result.gspmd_result is not None
                    and result.gspmd_result.sharding_spec is not None
                    else 0
                ),
                "output_dir": str(output_dir),
                "stage_durations_ms": result.stage_durations,
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_device_mesh(mesh_shape: MeshShape) -> DeviceMesh:
    """Build a synthetic device mesh that matches the requested shape.

    Uses detected local devices (Nvidia / AMD / Intel) when available,
    and falls back to a CPU mesh so the CLI works on dev machines
    without GPU. The sharding decisions are independent of the actual
    hardware topology — GSPMD treats the mesh shape as a logical
    grid — so a synthetic mesh is a valid sharding target.
    """
    try:
        mesh = DeviceMesh.detect_local()
        # Override the detected shape with the user-requested shape.
        mesh.mesh_shape = list(mesh_shape.axes)
        return mesh
    except Exception as exc:
        log.debug("local detection failed: %s; using synthetic mesh", exc)

    # Synthetic fallback: one NVIDIA device per requested slot.
    total = mesh_shape.total_devices
    return DeviceMesh(
        devices=[
            MeshDevice(
                device_id=i,
                vendor=DeviceVendor.NVIDIA,
                arch="sm_90",
                memory_gb=80.0,
                compute_tflops=989.0,
                interconnect=InterconnectType.NVLINK,
            )
            for i in range(total)
        ],
        mesh_shape=list(mesh_shape.axes),
    )


def _load_model_and_inputs(
    model_file: Path,
    cli_example_inputs: str | None,
) -> tuple[Any, tuple[Any, ...]]:
    """Import the model file and resolve its example inputs.

    Recognised conventions in the model file (first match wins):
      - ``Model``  class — instantiated with no arguments
      - ``model``  module-level instance
      - ``build_model()``  factory function
      - ``EXAMPLE_INPUTS``  tuple of shapes — used for the example
        inputs; tensors are constructed from CLI shapes when given,
        else from ``EXAMPLE_INPUTS`` alone.

    If the model file does not expose any of these, returns
    ``(None, ())`` so the bridge reports a clear graph-capture error
    rather than the CLI raising before the pipeline starts.
    """
    try:
        module = _import_user_module(model_file)
    except Exception as exc:
        log.warning(
            "could not import model file %s: %s; "
            "proceeding with model=None (bridge will report a clean error)",
            model_file,
            exc,
        )
        return None, ()

    model: Any = None
    if hasattr(module, "Model") and isinstance(module.Model, type):
        try:
            model = module.Model()
        except Exception as exc:
            log.warning("Model() raised: %s; falling back to model=None", exc)
    elif hasattr(module, "model"):
        model = module.model
    elif hasattr(module, "build_model") and callable(module.build_model):
        try:
            model = module.build_model()
        except Exception as exc:
            log.warning("build_model() raised: %s; falling back to model=None", exc)

    example_inputs_attr = vars(module).get("EXAMPLE_INPUTS")
    inputs = _resolve_example_inputs(
        cli_value=cli_example_inputs,
        module=example_inputs_attr,
    )
    return model, inputs


def _import_user_module(model_file: Path) -> Any:
    """Import a user-supplied model file as a Python module."""
    spec = importlib.util.spec_from_file_location(
        f"nautilus_shard_model_{model_file.stem}",
        str(model_file),
    )
    if spec is None or spec.loader is None:
        raise NautilusError(
            f"Could not load {model_file} as a Python module",
            context={"path": str(model_file)},
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        # Don't keep user modules alive in sys.modules across runs
        sys.modules.pop(spec.name, None)
    return module


def _resolve_example_inputs(
    cli_value: str | None,
    module: Any,
) -> tuple[Any, ...]:
    """Build example input tensors from the CLI value or module attribute.

    The CLI value is a single shape (e.g. ``1,3,224,224``). When a
    module exposes ``EXAMPLE_INPUTS`` as a tuple of shapes, the CLI
    value is used for the first input and the module attribute fills
    the remaining slots.
    """
    try:
        import torch
    except ImportError:
        # No torch → no tensors can be built; the bridge will produce
        # a clear graph-capture error in that case.
        return ()

    shapes: list[tuple[int, ...]] = []
    if cli_value is not None:
        try:
            shapes.append(tuple(int(x) for x in cli_value.split(",")))
        except ValueError as exc:
            raise NautilusError(
                f"Invalid --example-inputs {cli_value!r}",
                context={"value": cli_value},
            ) from exc
    if isinstance(module, (tuple, list)) and module:
        for s in module:
            if isinstance(s, (tuple, list)) and all(isinstance(d, int) for d in s):
                shapes.append(tuple(s))

    if not shapes:
        return ()

    return tuple(torch.randn(*s) for s in shapes)


if __name__ == "__main__":
    cli()

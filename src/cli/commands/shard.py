"""`nautilus shard` — Shard a PyTorch model across a device mesh.

Captures a PyTorch model as a TorchFX graph, converts to StableHLO,
runs GSPMD to compute optimal sharding, and emits per-shard Triton
source plus the fat binaries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from src.common.errors import (
    DependencyMissingError,
    GraphCaptureError,
    GSPMDError,
    NautilusError,
    StableHLOExportError,
)
from src.common.logging import get_logger, span as span_context
from src.common.types import MeshShape

log = get_logger("nautilus.cli.shard")


@click.command(
    "shard",
    short_help="Shard a PyTorch model across a device mesh",
    help="""
Capture a PyTorch model, convert to StableHLO, run GSPMD, and emit
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
    "--mesh", "-m",
    required=True,
    help="Device mesh as comma-separated axes (e.g. '2,2' for 2x2, '4' for 1x4).",
)
@click.option(
    "--strategy", "-s",
    type=click.Choice(["auto", "replicated", "data_parallel", "model_parallel", "tensor_parallel"],
                     case_sensitive=False),
    default="auto",
    show_default=True,
    help="Sharding strategy hint to GSPMD.",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    default=Path("./shards"),
    show_default=True,
    help="Where to write per-shard artifacts.",
)
@click.option(
    "--example-inputs", "-e",
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
    axes = tuple(int(x) for x in mesh_str.split(","))
    try:
        mesh = MeshShape(axes=axes)
    except Exception as exc:
        raise NautilusError(
            f"Invalid mesh {mesh_str!r}: {exc}",
            context={"mesh_str": mesh_str},
        ) from exc
    log.info(
        "sharding started",
        model=str(model_file),
        mesh=list(mesh.axes),
        strategy=strategy,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Capture the model
    with span_context("graph_capture", model=str(model_file)) as sp:
        captured = _capture_model(model_file, example_inputs)
        sp.set(num_ops=captured.metadata.op_count if hasattr(captured.metadata, "op_count") else 0)

    # 2. Export to StableHLO
    with span_context("stablehlo_export") as sp:
        stablehlo = _export_to_stablehlo(captured)
        if not stablehlo.is_usable:
            raise StableHLOExportError(
                "StableHLO export produced no usable module",
                context={"is_real_stablehlo": stablehlo.is_real_stablehlo},
            )
        sp.set(op_count=stablehlo.op_count, real_stablehlo=stablehlo.is_real_stablehlo)

    # 3. Run GSPMD
    with span_context("gspmd", mesh_axes=list(mesh.axes)) as sp:
        spec = _run_gspmd(stablehlo, mesh, strategy)
        sp.set(
            num_tensor_shardings=len(spec.tensor_shardings),
            comm_volume_bytes=spec.estimated_comm_volume_bytes,
        )

    # 4. Emit per-shard artifacts
    shards_emitted = 0
    for shard_idx in range(mesh.total_devices):
        shard_dir = output_dir / f"shard_{shard_idx:04d}"
        shard_dir.mkdir(exist_ok=True)
        # Per-shard StableHLO
        (shard_dir / "stablehlo.mlir").write_text(stablehlo.mlir_text)
        # Per-shard Triton source
        shard_source = _generate_shard_source(stablehlo, spec, shard_idx, mesh)
        (shard_dir / "kernel.py").write_text(shard_source)
        # Per-shard sharding spec
        (shard_dir / "shard_spec.json").write_text(json.dumps({
            "shard_id": shard_idx,
            "mesh": list(mesh.axes),
            "tensor_shardings": {
                n: {
                    "axes": list(s.mesh_axes),
                    "shape": list(s.partition_shape),
                }
                for n, s in spec.tensor_shardings.items()
            },
        }, indent=2))
        shards_emitted += 1

    log.info(
        "sharding complete",
        shards=shards_emitted,
        output_dir=str(output_dir),
    )
    click.echo(json.dumps({
        "model": str(model_file),
        "mesh": list(mesh.axes),
        "strategy": strategy,
        "shards_emitted": shards_emitted,
        "tensor_shardings": len(spec.tensor_shardings),
        "comm_volume_bytes": spec.estimated_comm_volume_bytes,
        "output_dir": str(output_dir),
    }, indent=2))


def _capture_model(model_file: Path, example_inputs: str | None) -> Any:
    """Capture a PyTorch model. Requires torch + the model file to be importable."""
    try:
        from src.bridges.pytorch_xla.graph_capture import GraphCapture
        from src.bridges.pytorch_xla.graph_capture import CapturedGraph
    except ImportError as exc:
        raise DependencyMissingError(
            f"PyTorch/XLA bridge import failed: {exc}. "
            "Install with: pip install -e .[sharding,nvidia]",
        ) from exc

    # Resolve EXAMPLE_INPUTS from the model file if not provided on CLI
    inputs_shapes: list[tuple[int, ...]] | None = None
    if example_inputs:
        try:
            inputs_shapes = [
                tuple(int(x) for x in shape.split(","))
                for shape in example_inputs.split(";")
            ]
        except ValueError as exc:
            raise NautilusError(
                f"Invalid --example-inputs {example_inputs!r}",
                context={"value": example_inputs},
            ) from exc

    capture = GraphCapture()
    try:
        captured = capture.capture(
            model_file=str(model_file),
            example_input_shapes=inputs_shapes,
        )
    except GraphCaptureError:
        raise
    except Exception as exc:
        raise GraphCaptureError(
            f"Failed to capture graph from {model_file}: {exc}",
            cause=exc,
        ) from exc
    return captured


def _export_to_stablehlo(captured: Any) -> Any:
    try:
        from src.bridges.pytorch_xla.stablehlo_export import StableHLOExporter
    except ImportError as exc:
        raise DependencyMissingError(
            f"StableHLO exporter import failed: {exc}",
        ) from exc
    exporter = StableHLOExporter()
    return exporter.export_from_captured(captured)


def _run_gspmd(stablehlo: Any, mesh: Any, strategy: str) -> Any:
    try:
        from src.bridges.pytorch_xla.gspmd_runner import GSPMDRunner, ShardingStrategy
    except ImportError as exc:
        raise DependencyMissingError(
            f"GSPMD runner import failed: {exc}",
        ) from exc
    strategy_map = {
        "auto": ShardingStrategy.AUTO,
        "replicated": ShardingStrategy.REPLICATED,
        "data_parallel": ShardingStrategy.DATA_PARALLEL,
        "model_parallel": ShardingStrategy.MODEL_PARALLEL,
        "tensor_parallel": ShardingStrategy.TENSOR_PARALLEL,
    }
    runner = GSPMDRunner()
    result = runner.run(
        stablehlo_module=stablehlo,
        device_mesh=mesh,
        strategy=strategy_map[strategy.lower()],
    )
    if not result.is_usable:
        raise GSPMDError(
            f"GSPMD failed: {result.error}",
            context={"strategy": strategy},
        )
    return result.sharding_spec


def _generate_shard_source(
    stablehlo: Any,
    spec: Any,
    shard_idx: int,
    mesh: Any,
) -> str:
    """Generate Triton source for a single shard.

    This is a stub of the eventual StableHLO→Triton translator. It
    emits a Triton kernel that acknowledges the shard's position in
    the mesh and the partitioning of its tensors, ready to be filled
    in by the bridge_orchestrator's tuning step.
    """
    return f'''"""Auto-generated Triton source for shard {shard_idx} of {mesh.total_devices}.

Mesh axes: {list(mesh.axes)}
Total devices: {mesh.total_devices}
Sharding spec hash: {spec.cache_key}
"""
import triton
import triton.language as tl

# Mesh-aware constants
_MESH_AXES = {list(mesh.axes)!r}
_SHARD_ID = {shard_idx}
_TOTAL_DEVICES = {mesh.total_devices}

@triton.jit
def shard_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Sharded matmul kernel. The actual computation is per-shard."""
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a = tl.load(A_ptr + rm[:, None] * K + tl.arange(0, BLOCK_K)[None, :])
    b = tl.load(B_ptr + tl.arange(0, BLOCK_K)[:, None] * N + rn[None, :])
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc += tl.dot(a, b)
    tl.store(C_ptr + rm[:, None] * N + rn[None, :], acc)
'''


if __name__ == "__main__":
    cli()  # type: ignore[reportArgumentType]

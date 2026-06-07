"""`nautilus build` — Build a fat binary for a Triton kernel.

Compiles a Triton kernel for all specified vendor targets, then links
the per-vendor artifacts into a single fat binary with a C runtime
stub that dispatches at runtime based on the detected hardware.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from src.cli.commands.tune import _parse_target
from src.common.errors import (
    DependencyMissingError,
    LinkingError,
    NautilusError,
)
from src.common.logging import get_logger
from src.common.logging import span as span_context
from src.common.types import Vendor

log = get_logger("nautilus.cli.build")


@click.command(
    "build",
    short_help="Build a fat binary for a Triton kernel",
    help="""
Compile a Triton kernel for one or more vendor targets, then link
the per-vendor binaries into a single fat binary that dispatches to
the right backend at runtime.

Examples:

  # Build a fat binary for Nvidia + AMD
  nautilus build path/to/matmul.py --target nvidia/sm_90 --target amd/gfx942 -o matmul.fat.o

  # Build for Intel Arc
  nautilus build path/to/matmul.py --target intel/xe_hpg -o matmul.fat.o

  # Skip validation (faster; useful for CI)
  nautilus build path/to/matmul.py --target nvidia/sm_90 --no-validate
""",
)
@click.argument(
    "kernel_file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--target",
    "-t",
    "targets",
    multiple=True,
    required=True,
    help="Target hardware as 'vendor/arch'. Can be passed multiple times.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Output fat binary path. Default: <kernel_name>.fat.o in cwd.",
)
@click.option(
    "--tune/--no-tune",
    default=True,
    help="Run MetaSchedule tuning before compilation. Default: --tune.",
)
@click.option(
    "--trials",
    "-n",
    type=click.IntRange(min=1, max=10000),
    default=64,
    show_default=True,
    help="Number of MetaSchedule trials (if --tune).",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Validate the fat binary after build. Default: --validate.",
)
@click.option(
    "--block-m",
    type=click.IntRange(min=16, max=512),
    default=128,
    help="Manual BLOCK_M (overrides tuned value if --no-tune).",
)
@click.option(
    "--block-n",
    type=click.IntRange(min=16, max=512),
    default=128,
    help="Manual BLOCK_N (overrides tuned value if --no-tune).",
)
@click.option(
    "--block-k",
    type=click.IntRange(min=16, max=128),
    default=32,
    help="Manual BLOCK_K (overrides tuned value if --no-tune).",
)
@click.option(
    "--num-warps",
    type=click.IntRange(min=1, max=64),
    default=8,
    help="Manual num_warps (overrides tuned value if --no-tune).",
)
@click.option(
    "--num-stages",
    type=click.IntRange(min=1, max=10),
    default=3,
    help="Manual num_stages (overrides tuned value if --no-tune).",
)
def cli(
    kernel_file: Path,
    targets: tuple[str, ...],
    output: Path | None,
    tune: bool,
    trials: int,
    validate: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> None:
    """Build a fat binary."""
    try:
        _build_impl(
            kernel_file=kernel_file,
            target_strs=list(targets),
            output=output,
            tune_enabled=tune,
            trials=trials,
            validate=validate,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    except NautilusError as exc:
        click.echo(f"nautilus: {exc.message}", err=True)
        if exc.context:
            click.echo(f"  context: {exc.context}", err=True)
        sys.exit(2)
    except KeyboardInterrupt:
        click.echo("nautilus: interrupted", err=True)
        sys.exit(130)


def _build_impl(
    kernel_file: Path,
    target_strs: list[str],
    output: Path | None,
    tune_enabled: bool,
    trials: int,
    validate: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> None:
    hardware_targets = [_parse_target(t) for t in target_strs]
    log.info(
        "fat binary build started",
        kernel=str(kernel_file),
        targets=[t.to_tvm_target() for t in hardware_targets],
        tune=tune_enabled,
    )

    # 1. Load kernel source
    from src.cli.commands.tune import _hash_source, _load_kernel_file

    kernel_name, kernel_text = _load_kernel_file(kernel_file)
    source_hash = _hash_source(kernel_text)
    log.info("kernel loaded", name=kernel_name, source_hash=source_hash[:12])

    # 2. Optionally tune
    block_m_eff, block_n_eff, block_k_eff = block_m, block_n, block_k
    num_warps_eff, num_stages_eff = num_warps, num_stages
    if tune_enabled:
        with span_context("tune_for_build", kernel=kernel_name) as sp:
            try:
                from src.bridges.triton_tvm.bridge_orchestrator import TritonTVMBridge
                from src.bridges.triton_tvm.metadata_extractor import KernelMetadata

                bridge = TritonTVMBridge(max_trials=trials, enable_cache=True)
                metadata = KernelMetadata(
                    kernel_name=kernel_name,
                    source_hash=source_hash,
                    grid_0=1,
                    grid_1=1,
                    grid_2=1,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    num_ctas=1,
                )
                target_for_tune = hardware_targets[0].to_tvm_target()
                mapped = bridge._tuning_chain(metadata, target_for_tune)
                block_m_eff, block_n_eff, block_k_eff = (
                    mapped.block_m,
                    mapped.block_n,
                    mapped.block_k,
                )
                num_warps_eff, num_stages_eff = mapped.num_warps, mapped.num_stages
                sp.set(block_m=block_m_eff, block_n=block_n_eff, block_k=block_k_eff)
                log.info(
                    "tuning complete",
                    block_m=block_m_eff,
                    block_n=block_n_eff,
                    block_k=block_k_eff,
                    num_warps=num_warps_eff,
                    num_stages=num_stages_eff,
                )
            except DependencyMissingError:
                log.warning("TVM/Triton missing; using manual --block-* values")
            except Exception as exc:
                log.warning("tuning failed; falling back to manual values", error=str(exc))

    # 3. Compile per-vendor
    from src.bridges.aot_packager.builder import (
        FatBinaryBuilder,
        FatBinaryConfig,
    )

    config = FatBinaryConfig(
        kernel_name=kernel_name,
        kernel_source=kernel_text,
        block_m=block_m_eff,
        block_n=block_n_eff,
        block_k=block_k_eff,
        num_warps=num_warps_eff,
        num_stages=num_stages_eff,
        output_dir=str(output.parent) if output else None,
        skip_amd=not any(t.vendor == Vendor.AMD for t in hardware_targets),
        skip_intel=not any(t.vendor == Vendor.INTEL for t in hardware_targets),
        skip_nvidia=not any(t.vendor == Vendor.NVIDIA for t in hardware_targets),
        skip_validation=not validate,
    )
    builder = FatBinaryBuilder()
    result = builder.build(config)

    # 4. Move output to requested path if user specified one
    if output is not None and result.output_path is not None:
        import shutil

        shutil.move(str(result.output_path), str(output))
        result.output_path = output

    # 5. Report
    if not result.is_usable:
        raise LinkingError(
            f"Fat binary build failed: {result.error}",
            context={"stage_times": result.stage_times},
        )
    log.info(
        "fat binary built",
        path=str(result.output_path),
        vendors=result.fat_binary.vendors if result.fat_binary else [],
        total_size=result.fat_binary.total_size if result.fat_binary else 0,
        total_time_s=result.total_time_s,
    )
    summary = result.to_dict()
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    cli()

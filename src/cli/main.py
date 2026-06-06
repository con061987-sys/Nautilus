"""src.cli — Command-line interface for Nautilus.

Provides:
  nautilus              top-level command (lists subcommands)
  nautilus-tune         tune a single Triton kernel via TVM MetaSchedule
  nautilus-build        build a fat binary for a kernel
  nautilus-shard        shard a PyTorch model across a device mesh
  nautilus-verify       print environment / hardware status
"""

from __future__ import annotations

import click

from src.cli.commands.build import cli as build_cmd
from src.cli.commands.inspect import cli as inspect_cmd
from src.cli.commands.shard import cli as shard_cmd
from src.cli.commands.tune import cli as tune_cmd
from src.cli.commands.verify import cli as verify_cmd
from src.common.logging import configure_logging


@click.group(
    name="nautilus",
    help="Cross-vendor AI compilation framework.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--log-level", "-l",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="info",
    help="Logging verbosity.",
)
@click.option(
    "--log-json/--log-human",
    default=True,
    help="JSON-structured logs (default) or human-readable.",
)
@click.version_option(
    version="0.1.0",
    prog_name="nautilus",
    message="%(prog)s %(version)s",
)
@click.pass_context
def cli(ctx: click.Context, log_level: str, log_json: bool) -> None:
    """Nautilus — cross-vendor AI compilation framework."""
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level
    ctx.obj["log_json"] = log_json
    configure_logging(level=log_level, json=log_json)


cli.add_command(tune_cmd, name="tune")
cli.add_command(build_cmd, name="build")
cli.add_command(shard_cmd, name="shard")
cli.add_command(verify_cmd, name="verify")
cli.add_command(inspect_cmd, name="inspect")


__all__ = ["cli"]

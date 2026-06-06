"""`nautilus verify` — Print environment and hardware status.

Wraps scripts/verify_env.py and additionally calls
src.common.hardware.format_device_summary() for a hardware view.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import click

from src.common.logging import get_logger
from src.common.errors import NautilusError

log = get_logger("nautilus.cli.verify")


@click.command(
    "verify",
    short_help="Print environment and hardware status",
    help="""
Report what's installed, what's missing, and what hardware is visible.
Useful as a first diagnostic step when a build or run fails.
""",
)
@click.option(
    "--target", "-t",
    type=click.Choice(["all", "cuda", "rocm", "intel"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Focus on a specific target's dependencies.",
)
@click.option(
    "--json", "as_json",
    is_flag=True,
    default=False,
    help="Machine-readable JSON output.",
)
def cli(target: str, as_json: bool) -> None:
    """Verify environment."""
    try:
        _verify_impl(target, as_json)
    except NautilusError as exc:
        click.echo(f"nautilus: {exc.message}", err=True)
        sys.exit(2)


def _verify_impl(target: str, as_json: bool) -> None:
    # 1. Run scripts/verify_env.py
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "verify_env.py"
    if not script.exists():
        raise NautilusError(
            f"verify_env.py not found at {script}",
        )
    cmd = [sys.executable, str(script), "--target", target]
    if as_json:
        cmd.append("--json")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 2. Add hardware summary
    from src.common.hardware import format_device_summary
    hardware_text = format_device_summary()

    if as_json:
        try:
            env_report = json.loads(result.stdout)
        except json.JSONDecodeError:
            env_report = {"raw_stdout": result.stdout}
        click.echo(json.dumps({
            "env": env_report,
            "hardware": hardware_text,
        }, indent=2))
    else:
        click.echo(result.stdout)
        click.echo()
        click.echo("=" * 60)
        click.echo("HARDWARE")
        click.echo("=" * 60)
        click.echo(hardware_text)

    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    cli()  # type: ignore[reportArgumentType]

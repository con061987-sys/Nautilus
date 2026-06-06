"""`nautilus inspect` — Print JSON metadata for a fat binary file."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.command("inspect")
@click.argument("fat_binary", type=click.Path(exists=True, dir_okay=False))
def cli(fat_binary: Path) -> None:
    """Inspect a fat binary and print its metadata as JSON."""
    from src.bridges.aot_packager.fat_binary import FatBinary

    fb = FatBinary.from_bytes(fat_binary.read_bytes())
    click.echo(json.dumps(
        {
            "kernel_name": fb.kernel_name,
            "sections": [
                {
                    "vendor": s.vendor,
                    "format": s.format.value,
                    "size": len(s.data),
                }
                for s in fb.sections
            ],
            "total_size": fb.total_size,
        },
        indent=2,
    ))

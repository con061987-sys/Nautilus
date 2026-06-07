"""`nautilus inspect` — Print JSON metadata for fat binaries or system topology.

Subcommands:
    nautilus inspect fat-binary <file>   — inspect a fat binary file
    nautilus inspect topology            — inspect the local GPU topology
"""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.group("inspect", context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Inspect fat binaries and system topology (outputs JSON)."""


@cli.command("fat-binary")
@click.argument("fat_binary", type=click.Path(exists=True, dir_okay=False))
def fat_binary_cmd(fat_binary: Path) -> None:
    """Inspect a fat binary file and print its metadata as JSON."""
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


@cli.command("topology")
def topology_cmd() -> None:
    """Inspect the local GPU topology and print it as JSON.

    The output is a JSON object with the following shape:
        {
            "host": { ... host info ... },
            "device_count": N,
            "devices": [ ... per-device details ... ],
            "bandwidth_gbps": [ ... per-pair bandwidth records ... ],
            "links": [ ... detailed link records ... ],
            "numa_nodes": { device_id: numa_node, ... }
        }

    Designed to be consumed by `nautilus shard` and other tools that need
    to know which devices are present and how they are interconnected.
    """
    from src.common.hardware import discover_topology

    topo = discover_topology()
    click.echo(topo.to_json(indent=2))


__all__ = ["cli"]

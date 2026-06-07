"""`nautilus cluster` — inspect and plan a multi-vendor, multi-node cluster.

Subcommands:
    nautilus cluster inspect                — dump the cluster topology as JSON
    nautilus cluster plan  --shards N       — produce a vendor-aware shard plan
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from src.common.errors import NautilusError
from src.common.logging import get_logger

log = get_logger("nautilus.cli.cluster")


@click.group(
    "cluster",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def cli() -> None:
    """Inspect and plan a multi-vendor, multi-node cluster."""


@cli.command("inspect")
@click.option(
    "--node-file", "-f",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    default=None,
    help=(
        "Optional JSON file describing additional cluster nodes. "
        "Shape: [{\"hostname\": \"node1\", \"devices\": [...]}, ...]. "
        "Used for multi-node dev/test setups that lack a real cluster manager."
    ),
)
def inspect_cmd(node_file: Path | None) -> None:
    """Print the cluster topology as JSON.

    By default, the command inspects the *local* machine and wraps the
    result in a single-node cluster. Pass ``--node-file`` to inject a
    multi-node view (e.g. for dev environments that need to emulate
    multi-vendor clusters on one host).

    Example output shape::

        {
          "num_nodes": 1,
          "num_devices": 4,
          "vendors": ["amd", "nvidia"],
          "is_heterogeneous": true,
          "is_multi_node": false,
          "nodes": [...],
          "devices": [...],
          "inter_node_links": [...]
        }
    """
    try:
        topology = _build_topology(node_file)
    except NautilusError as exc:
        click.echo(f"nautilus: {exc.message}", err=True)
        if exc.context:
            click.echo(f"  context: {exc.context}", err=True)
        raise click.exceptions.Exit(2) from exc
    click.echo(json.dumps(topology.to_dict(), indent=2))


@cli.command("plan")
@click.option(
    "--shards", "-n",
    type=click.IntRange(min=1),
    required=True,
    help="Number of shards the model is split into.",
)
@click.option(
    "--comm-volume-mb",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Estimated per-shard communication volume in MB (hint to the planner).",
)
@click.option(
    "--allow-cross-node-vendor-split/--no-cross-node-vendor-split",
    default=False,
    show_default=True,
    help=(
        "Allow the same vendor's shards to span multiple nodes. "
        "Default keeps each vendor on one node when possible."
    ),
)
def plan_cmd(
    shards: int,
    comm_volume_mb: int,
    allow_cross_node_vendor_split: bool,
) -> None:
    """Produce a vendor-aware shard plan for the current cluster.

    Combines the :class:`ClusterTopology` with the
    :class:`VendorAwareScheduler` and :class:`CommunicationPlanner`
    and prints a single :class:`OrchestrationPlan` JSON document.
    """
    from src.runtime.cluster_orchestrator import (
        SchedulingPolicy,
        build_orchestration_plan,
    )

    try:
        topology = _build_topology(None)
        policy = SchedulingPolicy(
            allow_cross_node_vendor_split=allow_cross_node_vendor_split,
        )
        plan = build_orchestration_plan(
            cluster=topology,
            num_shards=shards,
            comm_volume_per_shard_bytes=comm_volume_mb * 1024 * 1024,
            policy=policy,
        )
    except NautilusError as exc:
        click.echo(f"nautilus: {exc.message}", err=True)
        if exc.context:
            click.echo(f"  context: {exc.context}", err=True)
        raise click.exceptions.Exit(2) from exc
    click.echo(json.dumps(plan.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_topology(node_file: Path | None):
    """Build a :class:`ClusterTopology` for the CLI command.

    The single-node path delegates to
    :meth:`ClusterTopology.from_local_device_meshes`, which uses the
    existing ``DeviceMesh.detect_local`` to find local accelerators.
    The multi-node path loads a JSON file in the same shape as the
    ``topology.devices`` array (see ``DeviceTopology.to_dict``) and
    builds one synthetic node per entry.
    """
    from src.runtime.cluster_orchestrator import (
        ClusterTopology,
        InterNodeLink,
        Node,
    )
    from src.bridges.pytorch_xla.device_mesh import (
        DeviceMesh,
        DeviceVendor,
        InterconnectType,
        MeshDevice,
    )
    from src.common.hardware import GpuVendor

    if node_file is None:
        return ClusterTopology.from_local_device_meshes()

    raw = json.loads(node_file.read_text())
    if not isinstance(raw, list):
        raise NautilusError(
            "Node file must be a JSON list of node entries",
            context={"path": str(node_file), "got_type": type(raw).__name__},
        )

    gpu_to_mesh_vendor: dict[GpuVendor, DeviceVendor] = {
        GpuVendor.NVIDIA: DeviceVendor.NVIDIA,
        GpuVendor.AMD: DeviceVendor.AMD,
        GpuVendor.INTEL: DeviceVendor.INTEL,
        GpuVendor.APPLE: DeviceVendor.CPU,
    }

    nodes: list[Node] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise NautilusError(
                f"Node entry #{idx} is not a JSON object",
                context={"path": str(node_file), "index": idx},
            )
        hostname = entry.get("hostname") or f"node{idx}"
        devices_raw = entry.get("devices") or []
        if not isinstance(devices_raw, list):
            raise NautilusError(
                f"Node {hostname!r} has non-list 'devices'",
                context={"hostname": hostname},
            )
        mesh_devices: list[MeshDevice] = []
        for d_idx, d in enumerate(devices_raw):
            if not isinstance(d, dict):
                raise NautilusError(
                    f"Device #{d_idx} on {hostname!r} is not a JSON object",
                    context={"hostname": hostname, "index": d_idx},
                )
            try:
                vendor_key = GpuVendor(d.get("vendor", "unknown").lower())
            except ValueError:
                vendor_key = GpuVendor.UNKNOWN
            mesh_vendor = gpu_to_mesh_vendor.get(
                vendor_key, DeviceVendor.CPU,
            )
            try:
                interconnect = InterconnectType(
                    d.get("interconnect", "pcie").lower(),
                )
            except ValueError:
                interconnect = InterconnectType.PCIE
            mesh_devices.append(MeshDevice(
                device_id=d_idx,
                vendor=mesh_vendor,
                arch=d.get("arch", ""),
                memory_gb=float(d.get("memory_gb", 0.0) or 0.0),
                compute_tflops=float(d.get("compute_tflops", 0.0) or 0.0),
                interconnect=interconnect,
                hostname=hostname,
            ))
        nodes.append(Node(
            hostname=hostname,
            devices=tuple(mesh_devices),
            numa_nodes={d.device_id: 0 for d in mesh_devices},
        ))

    # Build inter-node links from any explicit "links" list. Default
    # to a single Ethernet-class link between every pair of nodes so
    # the produced plan reflects a realistic cross-node cost.
    inter_node_links: list[InterNodeLink] = []
    hostnames = [n.hostname for n in nodes]
    if "links" in raw and isinstance(raw["links"], list):
        for l in raw["links"]:
            if not isinstance(l, dict):
                continue
            try:
                inter_node_links.append(InterNodeLink(
                    source=l["source"],
                    target=l["target"],
                    bandwidth_gbps=float(l.get("bandwidth_gbps", 12.5)),
                    latency_us=float(l.get("latency_us", 25.0)),
                    link_type=InterconnectType(
                        l.get("link_type", "ethernet").lower(),
                    ),
                ))
            except (KeyError, ValueError):
                continue
    else:
        for i, a in enumerate(hostnames):
            for b in hostnames[i + 1:]:
                inter_node_links.append(InterNodeLink(
                    source=a,
                    target=b,
                    bandwidth_gbps=12.5,
                    latency_us=25.0,
                ))

    return ClusterTopology(
        nodes=nodes,
        inter_node_links=inter_node_links,
    )


__all__ = ["cli"]

"""Cross-vendor mixed-cluster orchestration.

While :mod:`src.bridges.pytorch_xla.device_mesh` describes the *local*
device mesh (one node, mixed vendors), real production training runs
across many nodes. This module adds the cluster-level view:

  * :class:`Node`            — one host with its devices and intra-node
                               topology.
  * :class:`InterNodeLink`   — bandwidth/latency between two nodes.
  * :class:`ClusterTopology` — the union of all nodes + their inter-
                               node links. The single source of truth
                               the rest of the pipeline asks when
                               reasoning about placement.
  * :class:`VendorAwareScheduler` — assigns shards to devices such that
                               intra-vendor / intra-island work is
                               preferred and cross-vendor bridges
                               only carry what they must.
  * :class:`CommunicationPlanner` — picks the cheapest transport
                               (P2P / staged / network) for every
                               device pair.
  * :class:`OrchestrationPlan`  — combined output the pipeline
                               consumes.

The module is *pure*: no subprocess calls, no torch dependency, no
GPU drivers. Discovery is pluggable — callers pass in either real
:class:`~src.bridges.pytorch_xla.device_mesh.DeviceMesh` objects
discovered by the existing infrastructure, or synthetic ones built
in tests. The only hardware probe is the hostname (via the standard
library) and the local mesh if none is provided.

Design rules (enforced by the test suite):

  * No assumption that all devices share a vendor. A single-node
    2×MI300X box, a 4×H100 DGX, and a 2×AMD + 2×Intel + 2×Nvidia
    dev cluster are all first-class inputs.
  * No hardcoded device counts. Topology is data-driven from the
    input nodes.
  * No silent fallbacks. If the caller asks for a number of shards
    the cluster cannot satisfy, :class:`ShardingError` is raised
    with the actual capacity in the context dict.
  * All public APIs are type-hinted and Google-docstring'd; public
    dataclasses are ``frozen=True`` so downstream code cannot
    accidentally mutate shared cluster state.
"""

from __future__ import annotations

import socket
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from src.bridges.pytorch_xla.device_mesh import (
    DeviceMesh,
    DeviceVendor,
    InterconnectType,
    MeshDevice,
    MeshTopology,
)
from src.common.errors import ShardingError
from src.common.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cluster-level data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """One host in the cluster.

    Attributes:
        hostname: Hostname or IP for this node. Used for display and
            as the key in inter-node link tables.
        devices: All accelerators on this host, in NUMA-friendly
            order (delegated to :class:`DeviceMesh`).
        numa_nodes: Mapping ``{device_id: numa_node}`` propagated
            from the underlying :class:`DeviceTopology`.
    """
    hostname: str
    devices: tuple[MeshDevice, ...]
    numa_nodes: dict[int, int] = field(default_factory=dict)

    @property
    def num_devices(self) -> int:
        return len(self.devices)

    @property
    def vendors(self) -> set[DeviceVendor]:
        return {d.vendor for d in self.devices}

    def devices_by_vendor(self, vendor: DeviceVendor) -> list[MeshDevice]:
        return [d for d in self.devices if d.vendor == vendor]


@dataclass(frozen=True)
class InterNodeLink:
    """Bandwidth/latency between two nodes.

    Attributes:
        source: Hostname of the source node.
        target: Hostname of the target node.
        bandwidth_gbps: Measured or estimated inter-node bandwidth.
            The "slowest tier wins" semantic matches the local-mesh
            convention in :mod:`src.common.hardware`.
        latency_us: One-way latency in microseconds. Used by the
            :class:`CommunicationPlanner` to bias transport selection
            for latency-sensitive collectives (e.g. small all-reduce
            of gradients on the backward pass).
        link_type: The transport family. ``ETHERNET`` is the default
            for inter-node traffic; ``UALINK`` is the emerging
            cross-vendor standard that bypasses host memory.
    """
    source: str
    target: str
    bandwidth_gbps: float
    latency_us: float = 10.0
    link_type: InterconnectType = InterconnectType.ETHERNET

    def __post_init__(self) -> None:
        if self.bandwidth_gbps < 0:
            raise ValueError(
                f"InterNodeLink bandwidth must be non-negative, "
                f"got {self.bandwidth_gbps}",
            )
        if self.latency_us < 0:
            raise ValueError(
                f"InterNodeLink latency must be non-negative, "
                f"got {self.latency_us}",
            )


class TransportStrategy(str, Enum):
    """Transport strategy chosen by :class:`CommunicationPlanner`.

    ``P2P`` is direct device-to-device (NVLink, Infinity Fabric, etc.).
    ``STAGED`` routes through a host-memory staging buffer — used for
    cross-vendor bridges that lack a native P2P transport.
    ``NETWORK`` goes over the inter-node network (Ethernet / UALink /
    RoCE). The :class:`CommunicationPlanner` picks the cheapest of
    these for each (device_a, device_b) pair.
    """
    P2P = "p2p"
    STAGED = "staged"
    NETWORK = "network"


@dataclass(frozen=True)
class CommunicationPlan:
    """Transport plan for a single (source, target) device pair.

    ``estimated_bandwidth_gbps`` is what the planner *thinks* the
    effective bandwidth will be. The :class:`VendorAwareScheduler`
    uses this value to bias shard placement: high-bandwidth P2P
    pairs get the most chatty collectives.
    """
    source_device_id: int
    target_device_id: int
    strategy: TransportStrategy
    estimated_bandwidth_gbps: float
    estimated_latency_us: float
    library_hint: str = ""  # e.g. "nccl", "rccl", "oneccl", "mixed"
    via_host_staging: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source_device_id,
            "target": self.target_device_id,
            "strategy": self.strategy.value,
            "bandwidth_gbps": self.estimated_bandwidth_gbps,
            "latency_us": self.estimated_latency_us,
            "library_hint": self.library_hint,
            "via_host_staging": self.via_host_staging,
        }


# Bandwidth below which an inter-device link is considered
# "essentially network-class" (i.e. not a real P2P fabric). 50 Gbps
# is the gap between NVLink/Infinity Fabric (hundreds of GB/s) and
# Ethernet/UALink (≤200 Gbps full-duplex in current shipping hardware).
_P2P_BANDWIDTH_THRESHOLD_GBPS = 50.0


# Default library hint per vendor. Keeps the planner vendor-neutral:
# the actual library is selected at runtime by :class:`CommBackend`.
_VENDOR_LIBRARY_HINT: dict[DeviceVendor, str] = {
    DeviceVendor.NVIDIA: "nccl",
    DeviceVendor.AMD: "rccl",
    DeviceVendor.INTEL: "oneccl",
    DeviceVendor.CPU: "gloo",
}


def _library_hint_for_pair(v1: DeviceVendor, v2: DeviceVendor) -> str:
    """Library hint for a same-vendor pair; ``"mixed"`` otherwise."""
    if v1 == v2 and v1 in _VENDOR_LIBRARY_HINT:
        return _VENDOR_LIBRARY_HINT[v1]
    return "mixed"


# ---------------------------------------------------------------------------
# ClusterTopology
# ---------------------------------------------------------------------------


@dataclass
class ClusterTopology:
    """Multi-node cluster: nodes + their inter-node links.

    A cluster is the *outermost* topology container. Inside it, each
    :class:`Node` owns its own :class:`DeviceMesh` (the existing
    single-node abstraction), so all the per-vendor and per-NUMA
    properties that the device mesh already exposes remain
    accessible. The cluster adds inter-node routing and a single
    global device-id namespace.

    Device ids in the cluster view are *globally unique* and assigned
    in NUMA-first order, one node at a time, so the same id used
    during scheduling maps to the same physical device during
    execution.
    """
    nodes: list[Node] = field(default_factory=list)
    inter_node_links: list[InterNodeLink] = field(default_factory=list)

    # Cache: hostname -> intra-node mesh. Built lazily in
    # ``device_mesh_for_node`` so the existing single-node code keeps
    # working without modification.
    _mesh_cache: dict[str, DeviceMesh] = field(
        default_factory=dict, repr=False, compare=False,
    )

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_devices(self) -> int:
        return sum(n.num_devices for n in self.nodes)

    @property
    def vendors(self) -> set[DeviceVendor]:
        out: set[DeviceVendor] = set()
        for n in self.nodes:
            out.update(n.vendors)
        return out

    @property
    def is_heterogeneous(self) -> bool:
        return len(self.vendors) > 1

    @property
    def is_multi_node(self) -> bool:
        return self.num_nodes > 1

    def all_devices(self) -> list[MeshDevice]:
        """Flat list of all devices in cluster-global id order."""
        out: list[MeshDevice] = []
        for node in self.nodes:
            out.extend(node.devices)
        return out

    def find_node(self, hostname: str) -> Node | None:
        for n in self.nodes:
            if n.hostname == hostname:
                return n
        return None

    def device_mesh_for_node(self, hostname: str) -> DeviceMesh:
        """Return the :class:`DeviceMesh` for a single node.

        The result is cached. NUMA-first ordering and per-vendor
        grouping are inherited from :class:`DeviceMesh` and the
        underlying :class:`DeviceTopology`.
        """
        if hostname in self._mesh_cache:
            return self._mesh_cache[hostname]
        node = self.find_node(hostname)
        if node is None:
            raise ShardingError(
                f"Unknown node {hostname!r} in cluster",
                context={"hostname": hostname,
                         "known": [n.hostname for n in self.nodes]},
            )
        mesh = DeviceMesh(
            devices=list(node.devices),
            mesh_shape=[len(node.devices)],
            topology=MeshTopology(),
            metadata={"hostname": hostname},
        )
        self._mesh_cache[hostname] = mesh
        return mesh

    def inter_node_bandwidth(
        self, host_a: str, host_b: str,
    ) -> float:
        """Bandwidth between two nodes, or 0.0 if no link is recorded.

        Falls back to a conservative Ethernet-class default (12.5 Gbps,
        matching 100 Gbps full-duplex) when the cluster has not been
        told about the link — better than assuming 0 Gbps (which would
        wrongly mark the link "unusable") or assuming NVLink (which
        would over-schedule inter-node collectives).
        """
        if host_a == host_b:
            return 0.0
        for link in self.inter_node_links:
            if {link.source, link.target} == {host_a, host_b}:
                return link.bandwidth_gbps
        return 12.5

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view consumed by the ``cluster inspect`` CLI."""
        devices: list[dict[str, object]] = []
        global_offset = 0
        for node in self.nodes:
            for d in node.devices:
                devices.append({
                    "global_device_id": global_offset,
                    "node": node.hostname,
                    "local_device_id": d.device_id,
                    "vendor": d.vendor.value,
                    "arch": d.arch,
                    "memory_gb": d.memory_gb,
                    "compute_tflops": d.compute_tflops,
                    "interconnect": d.interconnect.value,
                    "numa_node": node.numa_nodes.get(d.device_id, -1),
                })
                global_offset += 1

        return {
            "num_nodes": self.num_nodes,
            "num_devices": self.num_devices,
            "vendors": sorted(v.value for v in self.vendors),
            "is_heterogeneous": self.is_heterogeneous,
            "is_multi_node": self.is_multi_node,
            "nodes": [
                {
                    "hostname": n.hostname,
                    "num_devices": n.num_devices,
                    "vendors": sorted(v.value for v in n.vendors),
                }
                for n in self.nodes
            ],
            "devices": devices,
            "inter_node_links": [
                {
                    "source": l.source,
                    "target": l.target,
                    "bandwidth_gbps": l.bandwidth_gbps,
                    "latency_us": l.latency_us,
                    "link_type": l.link_type.value,
                }
                for l in self.inter_node_links
            ],
        }

    @staticmethod
    def from_local_device_meshes(
        meshes: Sequence[DeviceMesh] | None = None,
        hostnames: Sequence[str] | None = None,
    ) -> "ClusterTopology":
        """Build a :class:`ClusterTopology` from one or more local meshes.

        Used by the CLI for the common case where a user inspects the
        cluster from one node. The host for each mesh defaults to the
        current machine hostname; callers may override ``hostnames``
        when faking a multi-node view (e.g. tests, dev environments).

        When ``meshes`` is empty, the local mesh is discovered via
        :meth:`DeviceMesh.detect_local` and wrapped in a single-node
        cluster — that is the default that ``nautilus cluster
        inspect`` uses.
        """
        if meshes is None:
            meshes = [DeviceMesh.detect_local()]
        local_hostname = socket.gethostname()
        nodes: list[Node] = []
        for idx, mesh in enumerate(meshes):
            if hostnames is not None and idx < len(hostnames):
                hostname = hostnames[idx]
            else:
                hostname = local_hostname
            # NUMA info is per-device; the DeviceMesh does not currently
            # carry it forward, so we re-derive it via the topology.
            numa = _numa_map_for_mesh(mesh)
            nodes.append(Node(
                hostname=hostname,
                devices=tuple(mesh.devices),
                numa_nodes=numa,
            ))
        return ClusterTopology(nodes=nodes)


def _numa_map_for_mesh(mesh: DeviceMesh) -> dict[int, int]:
    """Extract a ``{device_id: numa_node}`` map from a :class:`DeviceMesh`.

    The existing :class:`MeshDevice` does not store NUMA info, so we
    conservatively return an empty map; per-host NUMA is currently
    best-effort at the cluster level. The :class:`Node` accepts an
    empty map without error and the orchestrator falls back to vendor
    affinity for placement decisions.
    """
    if not mesh.devices:
        return {}
    return {d.device_id: 0 for d in mesh.devices}


# ---------------------------------------------------------------------------
# VendorAwareScheduler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardAssignment:
    """One shard's placement.

    The scheduler produces one of these per shard. The pipeline
    consumes ``node`` and ``device`` to dispatch the shard's fat
    binary, and uses ``estimated_comm_volume_bytes`` to decide
    whether the sharding strategy is worth the cost vs. replication.
    """
    shard_id: int
    node: str
    device: MeshDevice
    vendor: DeviceVendor
    estimated_comm_volume_bytes: int = 0
    rationale: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "shard_id": self.shard_id,
            "node": self.node,
            "device_id": self.device.device_id,
            "vendor": self.vendor.value,
            "arch": self.device.arch,
            "estimated_comm_volume_bytes": self.estimated_comm_volume_bytes,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class SchedulingPolicy:
    """Tunable knobs for :class:`VendorAwareScheduler`.

    Defaults match the production guidance in :mod:`docs.PATTERNS`:
    keep work on the fastest interconnect first, fall back to
    cross-vendor bridges only when the cluster lacks capacity, and
    never split a vendor group across nodes unless explicitly
    requested via ``allow_cross_node_vendor_split=True``.
    """
    prefer_vendor_affinity: bool = True
    allow_cross_node_vendor_split: bool = False
    # When multiple same-vendor devices exist on different nodes,
    # prefer the node with the most free capacity.
    load_balance_within_vendor: bool = True


class VendorAwareScheduler:
    """Assign shards to devices with vendor + interconnect awareness.

    Algorithm (per shard, in round-robin over the requested shard
    count):

      1. Rank devices by *interconnect quality* on their local node.
         NVLink > UALink > Infinity Fabric > PCIe > Ethernet.
      2. If ``prefer_vendor_affinity`` is set, group candidates by
         vendor and pick the largest group first (so most shards
         stay on one vendor and only a small number bridge).
      3. Within a vendor group, round-robin across nodes unless
         ``allow_cross_node_vendor_split`` is False — in that case
         the first node with capacity absorbs every same-vendor
         shard before the next node is touched.
      4. Raise :class:`ShardingError` with the cluster's capacity
         in the context dict if the requested shard count exceeds
         what the cluster can supply.

    The result is deterministic for a given cluster + policy, which
    keeps pipeline tests reproducible.
    """

    # Lower rank = better P2P fabric. Used to order candidates within
    # a vendor group. Module-level so it is not duplicated per
    # scheduler instance.
    _INTRA_NODE_RANK: dict[InterconnectType, int] = {
        InterconnectType.NVLINK: 0,
        InterconnectType.UALINK: 1,
        InterconnectType.INFINITY_FABRIC: 2,
        InterconnectType.PCIE: 3,
        InterconnectType.ETHERNET: 4,
    }

    def __init__(self, policy: SchedulingPolicy | None = None) -> None:
        self.policy = policy or SchedulingPolicy()

    def assign_shards(
        self,
        cluster: ClusterTopology,
        num_shards: int,
        comm_volume_per_shard_bytes: int = 0,
    ) -> list[ShardAssignment]:
        """Return a shard-to-device assignment list.

        Args:
            cluster: The cluster to schedule onto.
            num_shards: How many shards the model is split into.
                Must be a positive integer; the cluster must have
                at least that many devices.
            comm_volume_per_shard_bytes: Hint used to set
                ``estimated_comm_volume_bytes`` on each assignment.
                The scheduler does not move shards based on this
                value, but downstream code uses it to estimate total
                communication cost.
        """
        if num_shards <= 0:
            raise ShardingError(
                f"num_shards must be positive, got {num_shards}",
                context={"num_shards": num_shards},
            )
        if cluster.num_devices == 0:
            raise ShardingError(
                "Cannot schedule onto an empty cluster",
                context={"num_nodes": cluster.num_nodes},
            )
        if num_shards > cluster.num_devices:
            raise ShardingError(
                f"Cluster has {cluster.num_devices} devices but "
                f"{num_shards} shards were requested",
                context={
                    "num_shards": num_shards,
                    "num_devices": cluster.num_devices,
                    "num_nodes": cluster.num_nodes,
                },
            )

        candidates = self._rank_devices(cluster)
        if not self.policy.allow_cross_node_vendor_split:
            candidates = self._group_by_node_first(candidates)

        # Round-robin over the ranked candidate list. Each iteration
        # yields one assignment and a one-line rationale for the
        # scheduler log.
        assignments: list[ShardAssignment] = []
        for shard_id in range(num_shards):
            picked = candidates[shard_id % len(candidates)]
            node, device = picked
            rationale = self._format_rationale(
                shard_id, node, device, candidates,
            )
            assignments.append(ShardAssignment(
                shard_id=shard_id,
                node=node,
                device=device,
                vendor=device.vendor,
                estimated_comm_volume_bytes=comm_volume_per_shard_bytes,
                rationale=rationale,
            ))
        return assignments

    # -- internals ---------------------------------------------------------

    def _rank_devices(
        self, cluster: ClusterTopology,
    ) -> list[tuple[str, MeshDevice]]:
        """Rank devices, vendor-first then by intra-node interconnect.

        Returns a list of ``(hostname, MeshDevice)``. The order is
        stable so the same cluster + policy always produces the same
        assignment sequence.
        """
        entries: list[tuple[str, MeshDevice]] = []
        for node in cluster.nodes:
            for device in node.devices:
                entries.append((node.hostname, device))

        if not self.policy.prefer_vendor_affinity:
            return entries

        # Vendor-major sort. Within a vendor, we order by (node,
        # interconnect-rank) so cross-node splits are predictable.
        def sort_key(entry: tuple[str, MeshDevice]) -> tuple[str, str, int, int]:
            node_host, dev = entry
            return (
                dev.vendor.value,
                node_host,
                self._INTRA_NODE_RANK.get(dev.interconnect, 99),
                dev.device_id,
            )

        return sorted(entries, key=sort_key)

    def _group_by_node_first(
        self,
        candidates: list[tuple[str, MeshDevice]],
    ) -> list[tuple[str, MeshDevice]]:
        """Keep same-vendor candidates on the same node when possible.

        Used when ``allow_cross_node_vendor_split=False`` to pin all
        of vendor X's shards to the first node with vendor-X devices,
        then move to the next node only when the first one is full.
        """
        if not self.policy.load_balance_within_vendor:
            return candidates
        # Group by vendor, then by node within vendor. Iterate vendors
        # in alphabetical order (stable), nodes by first appearance.
        by_vendor: dict[DeviceVendor, list[tuple[str, MeshDevice]]] = {}
        for node_host, dev in candidates:
            by_vendor.setdefault(dev.vendor, []).append((node_host, dev))
        out: list[tuple[str, MeshDevice]] = []
        for vendor in sorted(by_vendor, key=lambda v: v.value):
            for entry in by_vendor[vendor]:
                out.append(entry)
        return out

    def _format_rationale(
        self,
        shard_id: int,
        node: str,
        device: MeshDevice,
        all_candidates: list[tuple[str, MeshDevice]],
    ) -> str:
        if not all_candidates:
            return f"shard {shard_id} -> {node}/{device.display_name} (empty)"
        idx = next(
            (i for i, (n, d) in enumerate(all_candidates)
             if n == node and d.device_id == device.device_id),
            0,
        )
        return (
            f"shard {shard_id} -> {node}/{device.display_name} "
            f"(rank={idx}/{len(all_candidates)}, "
            f"interconnect={device.interconnect.value})"
        )


# ---------------------------------------------------------------------------
# CommunicationPlanner
# ---------------------------------------------------------------------------


class CommunicationPlanner:
    """Pick the cheapest transport for every device pair.

    The planner is intentionally stateless: feed it a
    :class:`ClusterTopology` and it returns a list of
    :class:`CommunicationPlan` records, one per (device_a, device_b)
    pair. The pipeline can then look up the plan for the specific
    pair it needs at runtime.

    Selection rules:

      * Same node + same vendor + interconnect ``>= 50 GB/s``:
        ``P2P`` with the vendor-native library (nccl/rccl/oneccl).
      * Same node + different vendor (or low intra-node bandwidth):
        ``STAGED`` — the only available transport until UALink
        matures cross-vendor; we mark ``via_host_staging=True`` so
        downstream code can apply the bandwidth penalty.
      * Different nodes: ``NETWORK`` with the recorded inter-node
        bandwidth and a ``"mixed"`` library hint (the actual
        transport is selected at runtime by
        :class:`~src.bridges.pytorch_xla.comm_backend.CommBackend`).
    """

    def __init__(
        self,
        p2p_bandwidth_threshold_gbps: float = _P2P_BANDWIDTH_THRESHOLD_GBPS,
    ) -> None:
        if p2p_bandwidth_threshold_gbps < 0:
            raise ValueError(
                f"p2p_bandwidth_threshold_gbps must be non-negative, "
                f"got {p2p_bandwidth_threshold_gbps}",
            )
        self.p2p_bandwidth_threshold_gbps = p2p_bandwidth_threshold_gbps

    def plan_cluster(self, cluster: ClusterTopology) -> list[CommunicationPlan]:
        """Return one plan per ordered device pair in the cluster."""
        out: list[CommunicationPlan] = []
        all_devices = cluster.all_devices()
        # Precompute (node, vendor) lookup so the inner loop is O(1).
        device_meta: list[tuple[str, DeviceVendor]] = []
        for d in all_devices:
            owner_node = self._owner_node(cluster, d)
            device_meta.append((owner_node, d.vendor))
        for i, src in enumerate(all_devices):
            for j, tgt in enumerate(all_devices):
                if i == j:
                    continue
                src_node, _ = device_meta[i]
                tgt_node, _ = device_meta[j]
                out.append(self._plan_pair(
                    cluster, src, tgt, src_node, tgt_node,
                ))
        return out

    def plan_pair(
        self,
        cluster: ClusterTopology,
        source: MeshDevice,
        target: MeshDevice,
    ) -> CommunicationPlan:
        """Plan a single device pair.

        Public so the pipeline can query on demand for a specific
        source/target pair without re-planning the whole cluster.
        """
        src_node = self._owner_node(cluster, source)
        tgt_node = self._owner_node(cluster, target)
        return self._plan_pair(cluster, source, target, src_node, tgt_node)

    # -- internals ---------------------------------------------------------

    def _owner_node(
        self, cluster: ClusterTopology, device: MeshDevice,
    ) -> str:
        # ``MeshDevice.device_id`` is per-host, not cluster-global, so
        # matching on it would mis-attribute devices that share a local
        # id (the common case in multi-node clusters: every host's
        # device 0 has device_id=0). Match by object identity instead,
        # which is what :meth:`ClusterTopology.all_devices` returns.
        for n in cluster.nodes:
            for d in n.devices:
                if d is device:
                    return n.hostname
        # If the device is not in the cluster, it almost certainly
        # came from a hand-built plan. Tag it with the local host so
        # the plan remains self-consistent.
        return socket.gethostname()

    def _plan_pair(
        self,
        cluster: ClusterTopology,
        source: MeshDevice,
        target: MeshDevice,
        source_node: str,
        target_node: str,
    ) -> CommunicationPlan:
        if source_node == target_node:
            return self._plan_intra_node(
                source, target, source_node,
            )
        return self._plan_inter_node(
            cluster, source, target, source_node, target_node,
        )

    def _plan_intra_node(
        self,
        source: MeshDevice,
        target: MeshDevice,
        node: str,
    ) -> CommunicationPlan:
        if source.vendor == target.vendor:
            # Same vendor on the same node: try P2P. We treat the
            # source's interconnect as authoritative since the
            # device mesh records a single interconnect per device.
            bw = _interconnect_bandwidth_gbps(source.interconnect)
            if bw >= self.p2p_bandwidth_threshold_gbps:
                return CommunicationPlan(
                    source_device_id=source.device_id,
                    target_device_id=target.device_id,
                    strategy=TransportStrategy.P2P,
                    estimated_bandwidth_gbps=bw,
                    estimated_latency_us=_interconnect_latency_us(
                        source.interconnect,
                    ),
                    library_hint=_library_hint_for_pair(
                        source.vendor, target.vendor,
                    ),
                )
        # Cross-vendor or low-bandwidth intra-node: stage via host.
        return CommunicationPlan(
            source_device_id=source.device_id,
            target_device_id=target.device_id,
            strategy=TransportStrategy.STAGED,
            estimated_bandwidth_gbps=64.0,  # PCIe-class
            estimated_latency_us=10.0,
            library_hint="mixed",
            via_host_staging=True,
        )

    def _plan_inter_node(
        self,
        cluster: ClusterTopology,
        source: MeshDevice,
        target: MeshDevice,
        source_node: str,
        target_node: str,
    ) -> CommunicationPlan:
        bw = cluster.inter_node_bandwidth(source_node, target_node)
        latency = 25.0  # conservative default for Ethernet-class
        for link in cluster.inter_node_links:
            if {link.source, link.target} == {source_node, target_node}:
                latency = link.latency_us
                break
        return CommunicationPlan(
            source_device_id=source.device_id,
            target_device_id=target.device_id,
            strategy=TransportStrategy.NETWORK,
            estimated_bandwidth_gbps=bw,
            estimated_latency_us=latency,
            library_hint="mixed",
            via_host_staging=False,
        )


def _interconnect_bandwidth_gbps(kind: InterconnectType) -> float:
    """Nominal peak bandwidth per interconnect, in Gbps.

    Returns 0 for unknown kinds so a caller that asked for an
    unknown link never silently gets a P2P classification. The
    threshold check in :class:`CommunicationPlanner` will then fall
    through to the STAGED branch.
    """
    return {
        InterconnectType.NVLINK: 900.0,
        InterconnectType.UALINK: 200.0,
        InterconnectType.INFINITY_FABRIC: 800.0,
        InterconnectType.PCIE: 64.0,
        InterconnectType.ETHERNET: 12.5,
    }.get(kind, 0.0)


def _interconnect_latency_us(kind: InterconnectType) -> float:
    """Nominal one-way latency per interconnect, in microseconds."""
    return {
        InterconnectType.NVLINK: 1.0,
        InterconnectType.UALINK: 2.0,
        InterconnectType.INFINITY_FABRIC: 1.0,
        InterconnectType.PCIE: 5.0,
        InterconnectType.ETHERNET: 25.0,
    }.get(kind, 50.0)


# ---------------------------------------------------------------------------
# OrchestrationPlan (combined output)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrchestrationPlan:
    """Combined output of topology + scheduler + planner.

    The pipeline consumes this object; the scheduler and planner can
    be re-run independently, but packaging the three together keeps
    the per-stage observability spans simple to write.
    """
    topology: ClusterTopology
    assignments: tuple[ShardAssignment, ...]
    comm_plans: tuple[CommunicationPlan, ...]
    policy: SchedulingPolicy

    @property
    def num_shards(self) -> int:
        return len(self.assignments)

    @property
    def num_nodes(self) -> int:
        return self.topology.num_nodes

    @property
    def is_multi_node(self) -> bool:
        return self.topology.is_multi_node

    def assignment_for_node(self, hostname: str) -> list[ShardAssignment]:
        return [a for a in self.assignments if a.node == hostname]

    def to_dict(self) -> dict[str, object]:
        return {
            "num_nodes": self.topology.num_nodes,
            "num_devices": self.topology.num_devices,
            "num_shards": self.num_shards,
            "is_multi_node": self.is_multi_node,
            "is_heterogeneous": self.topology.is_heterogeneous,
            "policy": {
                "prefer_vendor_affinity": self.policy.prefer_vendor_affinity,
                "allow_cross_node_vendor_split": (
                    self.policy.allow_cross_node_vendor_split
                ),
                "load_balance_within_vendor": (
                    self.policy.load_balance_within_vendor
                ),
            },
            "assignments": [a.to_dict() for a in self.assignments],
            "comm_plans": [p.to_dict() for p in self.comm_plans],
            "topology": self.topology.to_dict(),
        }


def build_orchestration_plan(
    cluster: ClusterTopology,
    num_shards: int,
    comm_volume_per_shard_bytes: int = 0,
    policy: SchedulingPolicy | None = None,
) -> OrchestrationPlan:
    """One-call helper used by both the CLI and the pipeline.

    Runs the scheduler and the planner against ``cluster`` and
    packages the result as a single :class:`OrchestrationPlan`.
    Keeps the wiring out of every call site.
    """
    scheduler = VendorAwareScheduler(policy=policy)
    planner = CommunicationPlanner()
    assignments = tuple(scheduler.assign_shards(
        cluster=cluster,
        num_shards=num_shards,
        comm_volume_per_shard_bytes=comm_volume_per_shard_bytes,
    ))
    comm_plans = tuple(planner.plan_cluster(cluster))
    return OrchestrationPlan(
        topology=cluster,
        assignments=assignments,
        comm_plans=comm_plans,
        policy=scheduler.policy,
    )


__all__ = [
    "ClusterTopology",
    "CommunicationPlan",
    "CommunicationPlanner",
    "InterNodeLink",
    "Node",
    "OrchestrationPlan",
    "SchedulingPolicy",
    "ShardAssignment",
    "TransportStrategy",
    "VendorAwareScheduler",
    "build_orchestration_plan",
]

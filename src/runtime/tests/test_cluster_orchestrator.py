"""Tests for src.runtime.cluster_orchestrator.

Exercises the four building blocks of the cluster view:
  * :class:`Node` / :class:`InterNodeLink` -- cluster-level data model
  * :class:`ClusterTopology` -- discovers, aggregates, and exposes nodes
  * :class:`VendorAwareScheduler` -- assigns shards with vendor + interconnect
    awareness
  * :class:`CommunicationPlanner` -- picks the cheapest transport per pair
  * :func:`build_orchestration_plan` -- bundles the three together

The tests are stdlib-only (no torch, no GPU SDKs) and construct synthetic
topologies so the same code paths that the production CLI uses are
exercised end-to-end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3]))

from src.bridges.pytorch_xla.device_mesh import (  # noqa: E402
    DeviceVendor,
    InterconnectType,
    MeshDevice,
)
from src.common.errors import ShardingError  # noqa: E402
from src.runtime.cluster_orchestrator import (  # noqa: E402
    ClusterTopology,
    CommunicationPlanner,
    InterNodeLink,
    Node,
    OrchestrationPlan,
    SchedulingPolicy,
    TransportStrategy,
    VendorAwareScheduler,
    build_orchestration_plan,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _device(
    device_id: int,
    vendor: DeviceVendor,
    interconnect: InterconnectType = InterconnectType.PCIE,
    arch: str = "sm_80",
) -> MeshDevice:
    return MeshDevice(
        device_id=device_id,
        vendor=vendor,
        arch=arch,
        memory_gb=80.0,
        compute_tflops=300.0,
        interconnect=interconnect,
    )


@pytest.fixture
def single_node_nvidia() -> ClusterTopology:
    """2x H100 on a single host, NVLink-only."""
    return ClusterTopology(
        nodes=[
            Node(
                hostname="node-a",
                devices=(
                    _device(0, DeviceVendor.NVIDIA, InterconnectType.NVLINK, "sm_90"),
                    _device(1, DeviceVendor.NVIDIA, InterconnectType.NVLINK, "sm_90"),
                ),
            ),
        ]
    )


@pytest.fixture
def single_node_amd() -> ClusterTopology:
    """2x MI300X on a single host, Infinity Fabric only."""
    return ClusterTopology(
        nodes=[
            Node(
                hostname="amd-box",
                devices=(
                    _device(0, DeviceVendor.AMD, InterconnectType.INFINITY_FABRIC, "gfx942"),
                    _device(1, DeviceVendor.AMD, InterconnectType.INFINITY_FABRIC, "gfx942"),
                ),
            ),
        ]
    )


@pytest.fixture
def mixed_single_node() -> ClusterTopology:
    """2x AMD + 2x Intel + 2x Nvidia on a single dev box."""
    return ClusterTopology(
        nodes=[
            Node(
                hostname="dev-box",
                devices=(
                    _device(0, DeviceVendor.AMD, InterconnectType.INFINITY_FABRIC, "gfx942"),
                    _device(1, DeviceVendor.AMD, InterconnectType.INFINITY_FABRIC, "gfx942"),
                    _device(2, DeviceVendor.INTEL, InterconnectType.PCIE, "gaudi2"),
                    _device(3, DeviceVendor.INTEL, InterconnectType.PCIE, "gaudi2"),
                    _device(4, DeviceVendor.NVIDIA, InterconnectType.NVLINK, "sm_90"),
                    _device(5, DeviceVendor.NVIDIA, InterconnectType.NVLINK, "sm_90"),
                ),
            ),
        ]
    )


@pytest.fixture
def multi_node_cluster() -> ClusterTopology:
    """4 nodes, mixed vendors, with explicit inter-node links."""
    return ClusterTopology(
        nodes=[
            Node(
                hostname="node-a",
                devices=(
                    _device(0, DeviceVendor.NVIDIA, InterconnectType.NVLINK),
                    _device(1, DeviceVendor.NVIDIA, InterconnectType.NVLINK),
                ),
            ),
            Node(
                hostname="node-b",
                devices=(
                    _device(0, DeviceVendor.AMD, InterconnectType.INFINITY_FABRIC, "gfx942"),
                    _device(1, DeviceVendor.AMD, InterconnectType.INFINITY_FABRIC, "gfx942"),
                ),
            ),
            Node(
                hostname="node-c",
                devices=(
                    _device(0, DeviceVendor.INTEL, InterconnectType.PCIE, "gaudi2"),
                    _device(1, DeviceVendor.INTEL, InterconnectType.PCIE, "gaudi2"),
                ),
            ),
            Node(
                hostname="node-d",
                devices=(_device(0, DeviceVendor.NVIDIA, InterconnectType.NVLINK),),
            ),
        ],
        inter_node_links=[
            InterNodeLink("node-a", "node-b", 25.0, 25.0),
            InterNodeLink("node-a", "node-c", 25.0, 25.0),
            InterNodeLink("node-a", "node-d", 200.0, 5.0, InterconnectType.UALINK),
            InterNodeLink("node-b", "node-c", 25.0, 25.0),
            InterNodeLink("node-b", "node-d", 25.0, 25.0),
            InterNodeLink("node-c", "node-d", 25.0, 25.0),
        ],
    )


# ---------------------------------------------------------------------------
# Node / InterNodeLink
# ---------------------------------------------------------------------------


class TestNodeAndLink:
    """Pure-data model: Node, InterNodeLink, validation."""

    def test_node_num_devices(self) -> None:
        node = Node(
            hostname="h1",
            devices=(
                _device(0, DeviceVendor.NVIDIA),
                _device(1, DeviceVendor.AMD, arch="gfx942"),
            ),
        )
        assert node.num_devices == 2
        assert node.vendors == {DeviceVendor.NVIDIA, DeviceVendor.AMD}

    def test_node_devices_by_vendor(self) -> None:
        node = Node(
            hostname="h",
            devices=(
                _device(0, DeviceVendor.NVIDIA),
                _device(1, DeviceVendor.NVIDIA),
                _device(2, DeviceVendor.AMD, arch="gfx942"),
            ),
        )
        nvidia_devs = node.devices_by_vendor(DeviceVendor.NVIDIA)
        assert len(nvidia_devs) == 2
        assert all(d.vendor == DeviceVendor.NVIDIA for d in nvidia_devs)
        assert node.devices_by_vendor(DeviceVendor.INTEL) == []

    def test_inter_node_link_defaults(self) -> None:
        link = InterNodeLink("a", "b", 100.0)
        assert link.latency_us == 10.0
        assert link.link_type == InterconnectType.ETHERNET

    def test_inter_node_link_rejects_negative_bandwidth(self) -> None:
        with pytest.raises(ValueError, match="bandwidth must be non-negative"):
            InterNodeLink("a", "b", -1.0)

    def test_inter_node_link_rejects_negative_latency(self) -> None:
        with pytest.raises(ValueError, match="latency must be non-negative"):
            InterNodeLink("a", "b", 10.0, latency_us=-2.0)


# ---------------------------------------------------------------------------
# ClusterTopology
# ---------------------------------------------------------------------------


class TestClusterTopology:
    """Discovery / aggregation / serialization."""

    def test_empty_cluster(self) -> None:
        topo = ClusterTopology()
        assert topo.num_nodes == 0
        assert topo.num_devices == 0
        assert topo.vendors == set()
        assert not topo.is_heterogeneous
        assert not topo.is_multi_node

    def test_num_devices_aggregates_across_nodes(
        self,
        multi_node_cluster: ClusterTopology,
    ) -> None:
        assert multi_node_cluster.num_nodes == 4
        assert multi_node_cluster.num_devices == 7

    def test_vendors_aggregated(
        self,
        multi_node_cluster: ClusterTopology,
    ) -> None:
        assert multi_node_cluster.vendors == {
            DeviceVendor.NVIDIA,
            DeviceVendor.AMD,
            DeviceVendor.INTEL,
        }
        assert multi_node_cluster.is_heterogeneous is True
        assert multi_node_cluster.is_multi_node is True

    def test_single_node_is_not_multi_node(
        self,
        mixed_single_node: ClusterTopology,
    ) -> None:
        assert mixed_single_node.is_heterogeneous is True
        assert mixed_single_node.is_multi_node is False

    def test_all_devices_order_preserved(
        self,
        multi_node_cluster: ClusterTopology,
    ) -> None:
        devs = multi_node_cluster.all_devices()
        expected = [dev for n in multi_node_cluster.nodes for dev in n.devices]
        assert devs == expected
        assert len(devs) == multi_node_cluster.num_devices

    def test_find_node(self, multi_node_cluster: ClusterTopology) -> None:
        assert multi_node_cluster.find_node("node-a") is multi_node_cluster.nodes[0]
        assert multi_node_cluster.find_node("node-z") is None

    def test_device_mesh_for_node_caches(
        self,
        multi_node_cluster: ClusterTopology,
    ) -> None:
        m1 = multi_node_cluster.device_mesh_for_node("node-a")
        m2 = multi_node_cluster.device_mesh_for_node("node-a")
        assert m1 is m2
        assert m1.num_devices == 2
        assert m1.metadata["hostname"] == "node-a"

    def test_device_mesh_for_unknown_node_raises(
        self,
        multi_node_cluster: ClusterTopology,
    ) -> None:
        with pytest.raises(ShardingError, match="Unknown node"):
            multi_node_cluster.device_mesh_for_node("node-x")

    def test_inter_node_bandwidth_recorded_link(
        self,
        multi_node_cluster: ClusterTopology,
    ) -> None:
        assert multi_node_cluster.inter_node_bandwidth("node-a", "node-d") == 200.0
        assert multi_node_cluster.inter_node_bandwidth("node-d", "node-a") == 200.0

    def test_inter_node_bandwidth_self_is_zero(
        self,
        multi_node_cluster: ClusterTopology,
    ) -> None:
        assert multi_node_cluster.inter_node_bandwidth("node-a", "node-a") == 0.0

    def test_inter_node_bandwidth_falls_back_to_ethernet_default(
        self,
        single_node_nvidia: ClusterTopology,
    ) -> None:
        # No link recorded between a node and itself in another cluster
        # that does not exist — but for any pair of unknown hosts, we
        # still get a conservative Ethernet-class default. The single
        # node has no recorded inter-node links, so a self-lookup is
        # the only safe call.
        assert single_node_nvidia.inter_node_bandwidth("node-a", "node-a") == 0.0
        # Add a synthetic link-less cluster and probe an unknown pair.
        topo = ClusterTopology(
            nodes=[
                Node(hostname="x", devices=(_device(0, DeviceVendor.NVIDIA),)),
                Node(hostname="y", devices=(_device(0, DeviceVendor.AMD, arch="gfx942"),)),
            ]
        )
        assert topo.inter_node_bandwidth("x", "y") == 12.5

    def test_to_dict_is_json_serializable(
        self,
        multi_node_cluster: ClusterTopology,
    ) -> None:
        d = multi_node_cluster.to_dict()
        # Must round-trip through json.dumps without a TypeError.
        encoded = json.dumps(d)
        decoded = json.loads(encoded)
        assert decoded["num_nodes"] == 4
        assert decoded["num_devices"] == 7
        assert sorted(decoded["vendors"]) == ["amd", "intel", "nvidia"]
        assert decoded["is_heterogeneous"] is True
        assert decoded["is_multi_node"] is True
        # 6 explicit inter-node links.
        assert len(decoded["inter_node_links"]) == 6


# ---------------------------------------------------------------------------
# VendorAwareScheduler
# ---------------------------------------------------------------------------


class TestVendorAwareScheduler:
    """Shard assignment with vendor + interconnect awareness."""

    def test_assigns_to_distinct_devices(
        self,
        mixed_single_node: ClusterTopology,
    ) -> None:
        s = VendorAwareScheduler()
        assignments = s.assign_shards(mixed_single_node, num_shards=4)
        assert len(assignments) == 4
        # All shard ids are 0..3.
        assert sorted(a.shard_id for a in assignments) == [0, 1, 2, 3]
        # All four target a different (node, device) pair.
        keys = {(a.node, a.device.device_id) for a in assignments}
        assert len(keys) == 4

    def test_zero_shards_raises(
        self,
        single_node_nvidia: ClusterTopology,
    ) -> None:
        with pytest.raises(ShardingError, match="must be positive"):
            VendorAwareScheduler().assign_shards(single_node_nvidia, num_shards=0)

    def test_too_many_shards_raises_with_context(
        self,
        single_node_amd: ClusterTopology,
    ) -> None:
        with pytest.raises(ShardingError) as exc_info:
            VendorAwareScheduler().assign_shards(single_node_amd, num_shards=99)
        assert exc_info.value.context["num_shards"] == 99
        assert exc_info.value.context["num_devices"] == 2

    def test_empty_cluster_raises(
        self,
    ) -> None:
        with pytest.raises(ShardingError, match="empty cluster"):
            VendorAwareScheduler().assign_shards(ClusterTopology(), num_shards=1)

    def test_vendor_affinity_groups_majority_first(
        self,
        mixed_single_node: ClusterTopology,
    ) -> None:
        # With 2 AMD + 2 INTEL + 2 NVIDIA and vendor affinity on, the
        # first two shards land on AMD (alphabetically first vendor)
        # before the scheduler moves to INTEL for the third.
        s = VendorAwareScheduler(SchedulingPolicy(prefer_vendor_affinity=True))
        assignments = s.assign_shards(mixed_single_node, num_shards=3)
        vendors = [a.vendor for a in assignments]
        assert vendors[0] == DeviceVendor.AMD
        assert vendors[1] == DeviceVendor.AMD
        assert vendors[2] == DeviceVendor.INTEL

    def test_no_vendor_affinity_keeps_node_order(
        self,
        mixed_single_node: ClusterTopology,
    ) -> None:
        s = VendorAwareScheduler(SchedulingPolicy(prefer_vendor_affinity=False))
        assignments = s.assign_shards(mixed_single_node, num_shards=3)
        for a in assignments:
            assert a.node == "dev-box"

    def test_cross_node_vendor_split_disabled_keeps_vendor_on_one_node(
        self,
        multi_node_cluster: ClusterTopology,
    ) -> None:
        s = VendorAwareScheduler(
            SchedulingPolicy(
                prefer_vendor_affinity=True,
                allow_cross_node_vendor_split=False,
                load_balance_within_vendor=True,
            )
        )
        # 3 nvidia devices live on node-a and node-d; with the policy
        # disabled they should all sit on node-a (the first node that
        # advertises nvidia).
        nvidia_assignments = [
            a
            for a in s.assign_shards(multi_node_cluster, num_shards=3)
            if a.vendor == DeviceVendor.NVIDIA
        ]
        assert all(a.node == "node-a" for a in nvidia_assignments)

    def test_cross_node_vendor_split_enabled_distributes(
        self,
        multi_node_cluster: ClusterTopology,
    ) -> None:
        s = VendorAwareScheduler(
            SchedulingPolicy(
                prefer_vendor_affinity=True,
                allow_cross_node_vendor_split=True,
            )
        )
        assignments = s.assign_shards(multi_node_cluster, num_shards=7)
        nvidia_assignments = [a for a in assignments if a.vendor == DeviceVendor.NVIDIA]
        nodes = {a.node for a in nvidia_assignments}
        assert {"node-a", "node-d"}.issubset(nodes)

    def test_rationale_is_set_and_nonempty(
        self,
        mixed_single_node: ClusterTopology,
    ) -> None:
        s = VendorAwareScheduler()
        assignments = s.assign_shards(mixed_single_node, num_shards=2)
        for a in assignments:
            assert a.rationale
            assert "rank=" in a.rationale
            assert "interconnect=" in a.rationale

    def test_comm_volume_hint_propagated(
        self,
        single_node_nvidia: ClusterTopology,
    ) -> None:
        s = VendorAwareScheduler()
        assignments = s.assign_shards(
            single_node_nvidia,
            num_shards=2,
            comm_volume_per_shard_bytes=4096,
        )
        assert all(a.estimated_comm_volume_bytes == 4096 for a in assignments)

    def test_assignment_is_deterministic(
        self,
        multi_node_cluster: ClusterTopology,
    ) -> None:
        s = VendorAwareScheduler()
        a1 = s.assign_shards(multi_node_cluster, num_shards=5)
        a2 = s.assign_shards(multi_node_cluster, num_shards=5)
        for x, y in zip(a1, a2, strict=True):
            assert x.node == y.node
            assert x.device.device_id == y.device.device_id


# ---------------------------------------------------------------------------
# CommunicationPlanner
# ---------------------------------------------------------------------------


class TestCommunicationPlanner:
    """Per-pair transport selection."""

    def test_same_node_same_vendor_high_bandwidth_is_p2p(
        self,
        single_node_nvidia: ClusterTopology,
    ) -> None:
        p = CommunicationPlanner()
        # Both devices are NVLink (900 Gbps), so same-vendor, same-node.
        plan = p.plan_pair(
            single_node_nvidia,
            source=single_node_nvidia.nodes[0].devices[0],
            target=single_node_nvidia.nodes[0].devices[1],
        )
        assert plan.strategy == TransportStrategy.P2P
        assert plan.estimated_bandwidth_gbps >= 50.0
        assert plan.via_host_staging is False
        assert plan.library_hint == "nccl"

    def test_same_node_cross_vendor_is_staged(
        self,
        mixed_single_node: ClusterTopology,
    ) -> None:
        p = CommunicationPlanner()
        amd = mixed_single_node.nodes[0].devices[0]
        nvidia = mixed_single_node.nodes[0].devices[4]
        plan = p.plan_pair(mixed_single_node, source=amd, target=nvidia)
        assert plan.strategy == TransportStrategy.STAGED
        assert plan.via_host_staging is True
        assert plan.library_hint == "mixed"

    def test_cross_node_uses_network(
        self,
        multi_node_cluster: ClusterTopology,
    ) -> None:
        p = CommunicationPlanner()
        src = multi_node_cluster.nodes[0].devices[0]  # node-a
        tgt = multi_node_cluster.nodes[1].devices[0]  # node-b
        plan = p.plan_pair(multi_node_cluster, source=src, target=tgt)
        assert plan.strategy == TransportStrategy.NETWORK
        assert plan.estimated_bandwidth_gbps == 25.0
        assert plan.library_hint == "mixed"

    def test_cross_node_ualink_reflects_recorded_bandwidth(
        self,
        multi_node_cluster: ClusterTopology,
    ) -> None:
        p = CommunicationPlanner()
        src = multi_node_cluster.nodes[0].devices[0]  # node-a
        tgt = multi_node_cluster.nodes[3].devices[0]  # node-d
        plan = p.plan_pair(multi_node_cluster, source=src, target=tgt)
        assert plan.strategy == TransportStrategy.NETWORK
        assert plan.estimated_bandwidth_gbps == 200.0
        assert plan.estimated_latency_us == 5.0

    def test_p2p_threshold_falls_through_to_staged(
        self,
        single_node_nvidia: ClusterTopology,
    ) -> None:
        # Force an absurdly high threshold so NVLink no longer counts as
        # P2P-class.
        p = CommunicationPlanner(p2p_bandwidth_threshold_gbps=10_000.0)
        src = single_node_nvidia.nodes[0].devices[0]
        tgt = single_node_nvidia.nodes[0].devices[1]
        plan = p.plan_pair(single_node_nvidia, source=src, target=tgt)
        assert plan.strategy == TransportStrategy.STAGED
        assert plan.via_host_staging is True

    def test_negative_p2p_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            CommunicationPlanner(p2p_bandwidth_threshold_gbps=-1.0)

    def test_plan_cluster_count(self, single_node_amd: ClusterTopology) -> None:
        p = CommunicationPlanner()
        plans = p.plan_cluster(single_node_amd)
        # N devices → N*(N-1) ordered pairs.
        n = single_node_amd.num_devices
        assert len(plans) == n * (n - 1)
        # All plans reference real devices.
        device_ids = {d.device_id for d in single_node_amd.all_devices()}
        for plan in plans:
            assert plan.source_device_id in device_ids
            assert plan.target_device_id in device_ids
            assert plan.source_device_id != plan.target_device_id

    def test_to_dict_includes_all_fields(
        self,
        single_node_nvidia: ClusterTopology,
    ) -> None:
        p = CommunicationPlanner()
        plan = p.plan_pair(
            single_node_nvidia,
            single_node_nvidia.nodes[0].devices[0],
            single_node_nvidia.nodes[0].devices[1],
        )
        d = plan.to_dict()
        assert d["strategy"] == "p2p"
        bandwidth = d["bandwidth_gbps"]
        assert isinstance(bandwidth, (int, float)) and bandwidth >= 50.0
        assert d["library_hint"] == "nccl"
        assert d["via_host_staging"] is False


# ---------------------------------------------------------------------------
# OrchestrationPlan
# ---------------------------------------------------------------------------


class TestBuildOrchestrationPlan:
    """The one-call helper used by the CLI and pipeline."""

    def test_combines_scheduler_and_planner(
        self,
        multi_node_cluster: ClusterTopology,
    ) -> None:
        plan = build_orchestration_plan(
            cluster=multi_node_cluster,
            num_shards=4,
            comm_volume_per_shard_bytes=2 * 1024 * 1024,
        )
        assert isinstance(plan, OrchestrationPlan)
        assert plan.num_shards == 4
        assert plan.num_nodes == 4
        assert plan.is_multi_node is True
        assert len(plan.assignments) == 4
        assert len(plan.comm_plans) == 7 * 6  # all ordered pairs

    def test_to_dict_is_json_serializable(
        self,
        mixed_single_node: ClusterTopology,
    ) -> None:
        plan = build_orchestration_plan(
            cluster=mixed_single_node,
            num_shards=3,
        )
        encoded = json.dumps(plan.to_dict())
        decoded = json.loads(encoded)
        assert decoded["num_shards"] == 3
        assert decoded["is_heterogeneous"] is True
        assert decoded["is_multi_node"] is False
        assert decoded["policy"]["prefer_vendor_affinity"] is True
        assert len(decoded["assignments"]) == 3
        assert len(decoded["comm_plans"]) == 6 * 5

    def test_assignment_for_node(
        self,
        multi_node_cluster: ClusterTopology,
    ) -> None:
        plan = build_orchestration_plan(
            cluster=multi_node_cluster,
            num_shards=7,
        )
        for node in multi_node_cluster.nodes:
            shards = plan.assignment_for_node(node.hostname)
            for a in shards:
                assert a.node == node.hostname

    def test_policy_is_propagated(
        self,
        single_node_nvidia: ClusterTopology,
    ) -> None:
        policy = SchedulingPolicy(
            prefer_vendor_affinity=False,
            allow_cross_node_vendor_split=True,
        )
        plan = build_orchestration_plan(
            cluster=single_node_nvidia,
            num_shards=1,
            policy=policy,
        )
        assert plan.policy.prefer_vendor_affinity is False
        assert plan.policy.allow_cross_node_vendor_split is True


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestClusterCLI:
    """The CLI subcommand, exercised via click's CliRunner."""

    def test_cluster_inspect_runs(self) -> None:
        from click.testing import CliRunner

        from src.cli.commands.cluster import cli as cluster_cli

        runner = CliRunner()
        result = runner.invoke(cluster_cli, ["inspect"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        # Local-machine fallback produces at least one node and one device.
        assert payload["num_nodes"] >= 1
        assert payload["num_devices"] >= 1
        assert "nodes" in payload
        assert "devices" in payload

    def test_cluster_plan_runs(self) -> None:
        from click.testing import CliRunner

        from src.cli.commands.cluster import cli as cluster_cli

        runner = CliRunner()
        result = runner.invoke(cluster_cli, ["plan", "--shards", "1"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["num_shards"] == 1
        assert "assignments" in payload
        assert "comm_plans" in payload

    def test_cluster_plan_rejects_zero_shards(self) -> None:
        from click.testing import CliRunner

        from src.cli.commands.cluster import cli as cluster_cli

        runner = CliRunner()
        result = runner.invoke(cluster_cli, ["plan", "--shards", "0"])
        # click.IntRange(min=1) rejects this with exit code 2.
        assert result.exit_code != 0

    def test_nautilus_cluster_subcommand_is_registered(self) -> None:
        # The cluster group must be wired into the top-level ``nautilus``
        # CLI so ``nautilus cluster ...`` is a real command. We check
        # the click command map without invoking the full main module
        # (which may have unrelated broken imports in this branch).
        from src.cli.main import cli as nautilus_cli

        assert "cluster" in nautilus_cli.commands

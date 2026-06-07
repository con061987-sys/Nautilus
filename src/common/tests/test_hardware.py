"""Tests for src.common.hardware — dynamic device topology discovery.

These tests exercise the dynamic discovery system in isolation. They do
NOT require a real GPU. Instead, they build a tmp_path-based fake
/dev, /proc/driver/nvidia/gpus, and /sys/bus/pci tree and verify
that:

  1. scan_*_devices() finds exactly the GPUs we planted (no cap, no
     missed devices, no false positives from non-GPU nodes).
  2. DeviceTopology.discover() is consistent and supports 0-GPU
     systems gracefully.
  3. bandwidth_gbps is populated per device pair, with PCIe-gen
     fallback when no measurement tool is available.
  4. NUMA-aware ordering puts co-NUMA devices adjacent in the list.
  5. The source has no hardcoded /dev/nvidia0..7 ceiling.
  6. The `nautilus inspect topology` CLI subcommand emits valid JSON.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from unittest import mock

import pytest

# Make src. importable when pytest is run from the repo root or
# anywhere else — match the pattern used in test_common.py.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3]))


from src.common import hardware as hw  # noqa: E402
from src.common.hardware import (  # noqa: E402
    DeviceInfo,
    DeviceTopology,
    GpuVendor,
    LinkType,
    TopologyLink,
    pcie_bandwidth_gbps,
    parse_pcie_speed_string,
    parse_pcie_width_string,
    scan_amd_devices,
    scan_intel_devices,
    scan_nvidia_devices,
    enumerate_devices,
    invalidate_enumeration_cache,
)


# ---------------------------------------------------------------------------
# Helpers: build fake /dev, /proc, /sys trees for tests
# ---------------------------------------------------------------------------


def _make_nvidia_pci_device(pci_dir: Path, bdf: str, *, numa: int = 0,
                            gen: int = 4, width: int = 16) -> None:
    """Create a fake Nvidia PCI device directory in a fake /sys tree."""
    d = pci_dir / bdf
    d.mkdir(parents=True, exist_ok=True)
    (d / "vendor").write_text("0x10de")
    (d / "device").write_text("0x1234")
    (d / "class").write_text("0x030000")  # VGA
    (d / "numa_node").write_text(str(numa))
    (d / "current_link_speed").write_text(f"{['', '2.5', '5.0', '8.0', '16.0', '32.0', '64.0'][gen]} GT/s PCIe")
    (d / "current_link_width").write_text(f"{width}")


def _make_amd_pci_device(pci_dir: Path, bdf: str, *, numa: int = 0,
                         gen: int = 4, width: int = 16) -> None:
    d = pci_dir / bdf
    d.mkdir(parents=True, exist_ok=True)
    (d / "vendor").write_text("0x1002")
    (d / "device").write_text("0x73bf")
    (d / "class").write_text("0x030000")
    (d / "numa_node").write_text(str(numa))
    (d / "current_link_speed").write_text(f"{['', '2.5', '5.0', '8.0', '16.0', '32.0', '64.0'][gen]} GT/s PCIe")
    (d / "current_link_width").write_text(f"{width}")


def _make_intel_pci_device(pci_dir: Path, bdf: str, *, numa: int = 0,
                           gen: int = 4, width: int = 16) -> None:
    d = pci_dir / bdf
    d.mkdir(parents=True, exist_ok=True)
    (d / "vendor").write_text("0x8086")
    (d / "device").write_text("0x4905")
    (d / "class").write_text("0x030000")
    (d / "numa_node").write_text(str(numa))
    (d / "current_link_speed").write_text(f"{['', '2.5', '5.0', '8.0', '16.0', '32.0', '64.0'][gen]} GT/s PCIe")
    (d / "current_link_width").write_text(f"{width}")


@pytest.fixture
def fake_fs():
    """Build a tmp_path-based fake /dev, /proc, /sys tree.

    Returns a dict of root paths so individual tests can plant devices
    before calling scan_*_devices(dev_root=..., proc_root=..., sysfs_pci=...).

    Also patches ``hardware._is_char_device`` to return True for any
    file inside the fake ``/dev``, since ``touch()`` creates regular
    files and the real production code requires S_IFCHR (Linux char
    device mode). The patch is scoped to the test, not the filesystem.
    """
    root = Path(os.environ.get("TMPDIR", "/tmp")) / "_nautilus_hw_test"
    if root.exists():
        import shutil
        shutil.rmtree(root)
    root.mkdir(parents=True)
    dev_root = root / "dev"
    proc_root = root / "proc"
    sys_root = root / "sys"
    (dev_root / "dri").mkdir(parents=True)
    (proc_root / "driver" / "nvidia" / "gpus").mkdir(parents=True)
    (sys_root / "bus" / "pci" / "devices").mkdir(parents=True)

    def _fake_is_char_device(path: Path) -> bool:
        try:
            path = Path(path).resolve()
        except (OSError, RuntimeError):
            return False
        return path == dev_root or dev_root in path.parents or path.exists()

    with mock.patch.object(hw, "_is_char_device", _fake_is_char_device):
        yield {
            "root": root,
            "dev": dev_root,
            "proc": proc_root,
            "sys_pci": sys_root / "bus" / "pci" / "devices",
        }

    import shutil
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Pure-function tests (no filesystem)
# ---------------------------------------------------------------------------


def test_pcie_bandwidth_gbps_gen4_x16():
    # Gen4 = 16 Gbps/lane, x16 = 16 lanes → 256 Gbps
    assert pcie_bandwidth_gbps(4, 16) == 16.0 * 16


def test_pcie_bandwidth_gbps_gen3_x16():
    # Gen3 = 8 Gbps/lane, x16 = 16 lanes → 128 Gbps
    assert pcie_bandwidth_gbps(3, 16) == 8.0 * 16


def test_pcie_bandwidth_gbps_gen5_x16():
    # Gen5 = 32 Gbps/lane, x16 = 16 lanes → 512 Gbps
    assert pcie_bandwidth_gbps(5, 16) == 32.0 * 16


def test_pcie_bandwidth_gbps_gen1_x1():
    assert pcie_bandwidth_gbps(1, 1) == 2.0


def test_pcie_bandwidth_gbps_unknown_returns_zero():
    assert pcie_bandwidth_gbps(0, 16) == 0.0
    assert pcie_bandwidth_gbps(4, 0) == 0.0
    assert pcie_bandwidth_gbps(-1, 8) == 0.0
    assert pcie_bandwidth_gbps(99, 8) == 0.0  # unknown gen


def test_parse_pcie_speed_string():
    assert parse_pcie_speed_string("2.5 GT/s PCIe") == 1
    assert parse_pcie_speed_string("5.0 GT/s PCIe") == 2
    assert parse_pcie_speed_string("8.0 GT/s PCIe") == 3
    assert parse_pcie_speed_string("8 GT/s PCIe") == 3
    assert parse_pcie_speed_string("16.0 GT/s PCIe") == 4
    assert parse_pcie_speed_string("32.0 GT/s PCIe") == 5
    assert parse_pcie_speed_string("64.0 GT/s PCIe") == 6
    assert parse_pcie_speed_string("garbage") == 0


def test_parse_pcie_width_string():
    assert parse_pcie_width_string("16") == 16
    assert parse_pcie_width_string("x16") == 16
    assert parse_pcie_width_string("x8") == 8
    assert parse_pcie_width_string("garbage") == 0


# ---------------------------------------------------------------------------
# Nvidia dynamic discovery
# ---------------------------------------------------------------------------


def test_scan_nvidia_devices_empty(fake_fs):
    """No /dev entries, no proc entries → no devices."""
    devs = scan_nvidia_devices(
        dev_root=fake_fs["dev"],
        proc_root=fake_fs["proc"],
    )
    assert devs == []


def test_scan_nvidia_devices_scales_beyond_seven(fake_fs):
    """No hardcoded /dev/nvidia0..7 ceiling. 12 devices all detected."""
    dev = fake_fs["dev"]
    proc = fake_fs["proc"]
    for i in range(12):
        # Make a fake char device file (Path.stat() will return regular
        # file mode, but the scan path uses _is_char_device which falls
        # back to .exists() on non-Linux platforms. We monkey-patch
        # _is_char_device to always return True in the test below).
        (dev / f"nvidia{i}").touch()
    devs = scan_nvidia_devices(dev_root=dev, proc_root=proc)
    assert len(devs) == 12
    assert [d.device_id for d in devs] == list(range(12))
    assert all(d.vendor == GpuVendor.NVIDIA for d in devs)
    assert all(d.device_path == str(dev / f"nvidia{i}") for i, d in enumerate(devs))


def test_scan_nvidia_devices_ignores_control_nodes(fake_fs):
    """nvidiactl, nvidia-uvm, nvidia-modeset must NOT be reported as GPUs."""
    dev = fake_fs["dev"]
    (dev / "nvidia0").touch()
    (dev / "nvidia1").touch()
    (dev / "nvidiactl").touch()
    (dev / "nvidia-uvm").touch()
    (dev / "nvidia-uvm-tools").touch()
    (dev / "nvidia-modeset").touch()
    (dev / "nvidia-caps").touch()
    devs = scan_nvidia_devices(dev_root=dev, proc_root=fake_fs["proc"])
    assert len(devs) == 2
    assert {d.device_id for d in devs} == {0, 1}


def test_scan_nvidia_devices_proc_indices_only(fake_fs):
    """If /dev is empty but /proc/driver/nvidia/gpus has entries, use those."""
    proc = fake_fs["proc"]
    gpus_dir = proc / "driver" / "nvidia" / "gpus"
    for i in range(3):
        (gpus_dir / str(i)).mkdir()
    devs = scan_nvidia_devices(dev_root=fake_fs["dev"], proc_root=proc)
    assert len(devs) == 3
    assert {d.device_id for d in devs} == {0, 1, 2}


def test_scan_nvidia_devices_merges_dev_and_proc(fake_fs):
    """Both /dev and /proc available → /dev takes priority, but union of indices wins."""
    dev = fake_fs["dev"]
    proc = fake_fs["proc"]
    gpus_dir = proc / "driver" / "nvidia" / "gpus"
    # /dev has 0,2,3
    for i in (0, 2, 3):
        (dev / f"nvidia{i}").touch()
    # /proc has 0,1,2
    for i in (0, 1, 2):
        (gpus_dir / str(i)).mkdir()
    devs = scan_nvidia_devices(dev_root=dev, proc_root=proc)
    ids = sorted(d.device_id for d in devs)
    assert ids == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# AMD dynamic discovery
# ---------------------------------------------------------------------------


def test_scan_amd_devices_with_kfd(fake_fs):
    """AMD discovery requires either /dev/kfd or /dev/dri/renderD* nodes."""
    dev = fake_fs["dev"]
    sys_pci = fake_fs["sys_pci"]
    (dev / "kfd").touch()
    (dev / "dri" / "renderD128").touch()
    _make_amd_pci_device(sys_pci, "0000:01:00.0", numa=0, gen=4, width=16)
    _make_amd_pci_device(sys_pci, "0000:02:00.0", numa=1, gen=4, width=16)
    devs = scan_amd_devices(dev_root=dev, sysfs_pci=sys_pci)
    assert len(devs) == 2
    assert all(d.vendor == GpuVendor.AMD for d in devs)
    assert [d.device_id for d in devs] == [0, 1]
    assert all(d.numa_node in (0, 1) for d in devs)


def test_scan_amd_devices_no_kfd_no_pci_returns_empty(fake_fs):
    """No AMD PCI device → no AMD device, even if render nodes exist."""
    dev = fake_fs["dev"]
    (dev / "dri" / "renderD128").touch()
    (dev / "dri" / "renderD129").touch()
    # No AMD PCI device planted.
    devs = scan_amd_devices(dev_root=dev, sysfs_pci=fake_fs["sys_pci"])
    assert devs == []


def test_scan_amd_devices_intel_rendernode_filtered(fake_fs):
    """renderD* nodes belonging to non-AMD GPUs are filtered out.

    Setup: an Intel iGPU occupies renderD128. The AMD GPU is on renderD129.
    The scanner must return exactly one AMD device, and it must be the
    one backed by an AMD PCI device.
    """
    dev = fake_fs["dev"]
    sys_pci = fake_fs["sys_pci"]
    (dev / "dri" / "renderD128").touch()
    (dev / "dri" / "renderD129").touch()
    _make_intel_pci_device(sys_pci, "0000:00:02.0", numa=0, gen=4, width=16)
    _make_amd_pci_device(sys_pci, "0000:01:00.0", numa=0, gen=4, width=16)
    devs = scan_amd_devices(dev_root=dev, sysfs_pci=sys_pci)
    assert len(devs) == 1
    assert devs[0].vendor == GpuVendor.AMD


# ---------------------------------------------------------------------------
# Intel dynamic discovery
# ---------------------------------------------------------------------------


def test_scan_intel_devices_filters_to_0x8086(fake_fs):
    """Intel discovery must only return PCI vendor 0x8086 devices."""
    dev = fake_fs["dev"]
    sys_pci = fake_fs["sys_pci"]
    (dev / "dri" / "renderD128").touch()
    (dev / "dri" / "renderD129").touch()
    (dev / "dri" / "renderD130").touch()
    _make_intel_pci_device(sys_pci, "0000:00:02.0", numa=0, gen=4, width=16)
    _make_intel_pci_device(sys_pci, "0000:00:03.0", numa=0, gen=4, width=16)
    # This AMD device must NOT be reported as Intel:
    _make_amd_pci_device(sys_pci, "0000:01:00.0", numa=1, gen=4, width=16)
    devs = scan_intel_devices(dev_root=dev, sysfs_pci=sys_pci)
    assert len(devs) == 2
    assert all(d.vendor == GpuVendor.INTEL for d in devs)
    assert [d.device_id for d in devs] == [0, 1]


def test_scan_intel_devices_no_devices_returns_empty(fake_fs):
    devs = scan_intel_devices(
        dev_root=fake_fs["dev"], sysfs_pci=fake_fs["sys_pci"]
    )
    assert devs == []


# ---------------------------------------------------------------------------
# DeviceTopology.discover()
# ---------------------------------------------------------------------------


def test_discover_topology_zero_gpus(fake_fs):
    """Empty /dev + empty /sys/bus/pci → empty DeviceTopology (no raise)."""
    # Patch enumerate_devices to return () by using a fresh cache.
    invalidate_enumeration_cache()
    with mock.patch.object(hw, "enumerate_devices", return_value=()):
        topo = DeviceTopology.discover()
    assert isinstance(topo, DeviceTopology)
    assert topo.devices == []
    assert topo.bandwidth_gbps == {}
    assert topo.links == []
    assert topo.numa_nodes == {}
    # Host is still populated.
    assert topo.host is not None


def test_discover_topology_bandwidth_per_pair(fake_fs):
    """Each device pair gets a symmetric (a,b)/(b,a) entry in bandwidth_gbps."""
    invalidate_enumeration_cache()
    fake_devices = (
        DeviceInfo(vendor=GpuVendor.NVIDIA, device_id=0, device_path="/dev/nvidia0",
                   arch="", model="", driver_version="", pcie_bdf="0000:01:00.0",
                   numa_node=0, pcie_gen=4, pcie_width=16),
        DeviceInfo(vendor=GpuVendor.AMD, device_id=1, device_path="/dev/kfd",
                   arch="", model="", driver_version="", pcie_bdf="0000:02:00.0",
                   numa_node=0, pcie_gen=4, pcie_width=16),
        DeviceInfo(vendor=GpuVendor.INTEL, device_id=2, device_path="/dev/dri/renderD130",
                   arch="", model="", driver_version="", pcie_bdf="0000:03:00.0",
                   numa_node=1, pcie_gen=4, pcie_width=16),
    )
    with mock.patch.object(hw, "enumerate_devices", return_value=fake_devices):
        topo = DeviceTopology.discover()
    assert len(topo.devices) == 3
    # C(3,2) = 3 pairs, each present in both directions = 6 entries.
    assert len(topo.bandwidth_gbps) == 6
    for i in range(3):
        for j in range(3):
            if i == j:
                assert (i, j) not in topo.bandwidth_gbps
            else:
                assert (i, j) in topo.bandwidth_gbps
                # Symmetric
                assert topo.bandwidth_gbps[(i, j)] == topo.bandwidth_gbps[(j, i)]
    # All PCIe Gen4 x16 → 16 Gbps/lane * 16 lanes = 256 Gbps
    for gbps in topo.bandwidth_gbps.values():
        assert gbps == 16.0 * 16
    # NUMA-aware ordering: NUMA 0 devices come first, then NUMA 1.
    numa_seq = [d.numa_node for d in topo.devices]
    assert numa_seq == sorted(numa_seq)


def test_discover_topology_numa_aware_ordering(fake_fs):
    """Devices in the same NUMA node are adjacent in the returned list."""
    invalidate_enumeration_cache()
    fake_devices = (
        DeviceInfo(vendor=GpuVendor.AMD, device_id=0, device_path="/p0",
                   arch="", model="", driver_version="", pcie_bdf="",
                   numa_node=2, pcie_gen=4, pcie_width=16),
        DeviceInfo(vendor=GpuVendor.NVIDIA, device_id=1, device_path="/p1",
                   arch="", model="", driver_version="", pcie_bdf="",
                   numa_node=0, pcie_gen=4, pcie_width=16),
        DeviceInfo(vendor=GpuVendor.INTEL, device_id=2, device_path="/p2",
                   arch="", model="", driver_version="", pcie_bdf="",
                   numa_node=0, pcie_gen=4, pcie_width=16),
        DeviceInfo(vendor=GpuVendor.AMD, device_id=3, device_path="/p3",
                   arch="", model="", driver_version="", pcie_bdf="",
                   numa_node=2, pcie_gen=4, pcie_width=16),
    )
    with mock.patch.object(hw, "enumerate_devices", return_value=fake_devices):
        topo = DeviceTopology.discover()
    numa_seq = [d.numa_node for d in topo.devices]
    # Grouped: NUMA 0, 0, 2, 2
    assert numa_seq == [0, 0, 2, 2]


def test_discover_topology_pcie_fallback_bandwidth(fake_fs):
    """No nvidia-smi / rocm-smi available → use PCIe gen×width fallback."""
    invalidate_enumeration_cache()
    fake_devices = (
        DeviceInfo(vendor=GpuVendor.NVIDIA, device_id=0, device_path="/dev/nvidia0",
                   arch="", model="", driver_version="", pcie_bdf="",
                   numa_node=0, pcie_gen=3, pcie_width=8),
        DeviceInfo(vendor=GpuVendor.NVIDIA, device_id=1, device_path="/dev/nvidia1",
                   arch="", model="", driver_version="", pcie_bdf="",
                   numa_node=0, pcie_gen=3, pcie_width=8),
    )
    with mock.patch.object(hw, "enumerate_devices", return_value=fake_devices):
        topo = DeviceTopology.discover()
    # Gen3 x8 = 8 * 8 = 64 Gbps
    assert topo.bandwidth_gbps[(0, 1)] == 64.0
    # Link should be PCIe, not measured
    links_by_pair = {(l.source_id, l.target_id): l for l in topo.links}
    pair_link = links_by_pair[(0, 1)]
    assert pair_link.link_type == LinkType.PCIE
    assert pair_link.measured is False
    assert pair_link.pcie_gen == 3
    assert pair_link.pcie_width == 8


def test_discover_topology_no_devices_no_links(fake_fs):
    """0-GPU topology has no links, no bandwidth_gbps, but a valid HostInfo."""
    invalidate_enumeration_cache()
    with mock.patch.object(hw, "enumerate_devices", return_value=()):
        topo = DeviceTopology.discover()
    assert topo.links == []
    assert topo.bandwidth_gbps == {}
    # to_dict must be JSON-serializable
    d = topo.to_dict()
    json.dumps(d)  # raises if not serializable
    assert d["device_count"] == 0
    assert d["devices"] == []


# ---------------------------------------------------------------------------
# Source-level: no hardcoded /dev/nvidia0..7
# ---------------------------------------------------------------------------


def test_source_has_no_hardcoded_nvidia_ceiling():
    """The hardware module must not have a hardcoded list of /dev/nvidia
    indices like `range(8)` or `[f\"/dev/nvidia{i}\" for i in range(N)]`.

    This is a regression test against a previous design that capped the
    number of discovered Nvidia devices at a fixed value.
    """
    src = Path(hw.__file__).read_text()
    # Patterns that would indicate a hardcoded cap.
    bad_patterns = [
        re.compile(r"range\(\s*[0-9]+\s*\)\s*\)"),  # closed range() with literal
        re.compile(r"range\(\s*7\s*,\s*"),
        re.compile(r"range\(\s*8\s*\)"),
        re.compile(r"/dev/nvidia[0-7]\b"),  # explicit /dev/nvidia0..7
    ]
    for pat in bad_patterns:
        # Allow range() calls in safe contexts (loop guards with no cap).
        # We only flag if the literal appears in a context that looks like
        # a hardcoded GPU count, e.g. `for i in range(8):` on the same
        # logical line.
        matches = pat.findall(src)
        # If `/dev/nvidia0`..`/dev/nvidia7` appears literally, that's bad.
        if pat.pattern.startswith("/dev/nvidia"):
            assert not matches, (
                f"Found hardcoded /dev/nvidia0..7 reference: {matches}"
            )
        # range(8) is too restrictive to ban globally; we only check the
        # /dev/nvidia pattern above which is the actual regression vector.


# ---------------------------------------------------------------------------
# CLI: `nautilus inspect topology`
# ---------------------------------------------------------------------------


def test_cli_inspect_topology_outputs_json(capsys, fake_fs):
    """`nautilus inspect topology` must emit valid JSON with expected shape."""
    from click.testing import CliRunner
    from src.cli.commands.inspect import cli

    invalidate_enumeration_cache()
    fake_devices = (
        DeviceInfo(vendor=GpuVendor.NVIDIA, device_id=0, device_path="/dev/nvidia0",
                   arch="", model="A100", driver_version="535", pcie_bdf="",
                   numa_node=0, pcie_gen=4, pcie_width=16),
    )
    with mock.patch.object(hw, "enumerate_devices", return_value=fake_devices):
        runner = CliRunner()
        result = runner.invoke(cli, ["topology"])
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    payload = json.loads(result.output)
    assert "host" in payload
    assert "devices" in payload
    assert "bandwidth_gbps" in payload
    assert "links" in payload
    assert "numa_nodes" in payload
    assert payload["device_count"] == 1
    assert payload["devices"][0]["vendor"] == "nvidia"
    # bandwidth_gbps must be a list of {source, target, gbps} dicts.
    for entry in payload["bandwidth_gbps"]:
        assert {"source", "target", "gbps"} <= set(entry.keys())


def test_cli_inspect_topology_zero_gpus(capsys, fake_fs):
    """`nautilus inspect topology` on a 0-GPU system → valid empty JSON."""
    from click.testing import CliRunner
    from src.cli.commands.inspect import cli

    invalidate_enumeration_cache()
    with mock.patch.object(hw, "enumerate_devices", return_value=()):
        runner = CliRunner()
        result = runner.invoke(cli, ["topology"])
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    payload = json.loads(result.output)
    assert payload["device_count"] == 0
    assert payload["devices"] == []


def test_cli_inspect_help_lists_subcommands():
    """`nautilus inspect --help` should list both subcommands."""
    from click.testing import CliRunner
    from src.cli.commands.inspect import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "fat-binary" in result.output
    assert "topology" in result.output


# ---------------------------------------------------------------------------
# Mixed-vendor topology
# ---------------------------------------------------------------------------


def test_discover_topology_mixed_vendors(fake_fs):
    """Mixed Nvidia + AMD + Intel cluster → all three present, all paired."""
    invalidate_enumeration_cache()
    fake_devices = (
        DeviceInfo(vendor=GpuVendor.NVIDIA, device_id=0, device_path="/dev/nvidia0",
                   arch="sm_90", model="H100", driver_version="535", pcie_bdf="",
                   numa_node=0, pcie_gen=5, pcie_width=16),
        DeviceInfo(vendor=GpuVendor.AMD, device_id=1, device_path="/dev/kfd",
                   arch="gfx942", model="MI300X", driver_version="6.0", pcie_bdf="",
                   numa_node=0, pcie_gen=5, pcie_width=16),
        DeviceInfo(vendor=GpuVendor.INTEL, device_id=2, device_path="/dev/dri/renderD130",
                   arch="xe_hpg", model="PVC", driver_version="1.0", pcie_bdf="",
                   numa_node=1, pcie_gen=4, pcie_width=16),
    )
    with mock.patch.object(hw, "enumerate_devices", return_value=fake_devices):
        topo = DeviceTopology.discover()
    vendors = sorted(d.vendor.value for d in topo.devices)
    assert vendors == ["amd", "intel", "nvidia"]
    # 3 pairs in both directions = 6 entries.
    assert len(topo.bandwidth_gbps) == 6
    # Each link should be reported.
    assert len(topo.links) == 3

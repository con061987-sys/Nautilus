"""
Hardware detection and dynamic device topology discovery.

This module provides three layers of information:

  1. HostInfo       — CPU vendor, model, OS, kernel, architecture.
  2. DeviceInfo     — A single GPU/accelerator detected on the system.
  3. DeviceTopology — All devices + interconnect bandwidths + NUMA layout.

Critical rules (enforced by tests):

  * NO hardcoded device counts. We discover dynamically via /dev globbing
    and lspci enumeration. There is no fixed upper bound on the number
    of GPUs the scanner will find — if the system has 16 GPUs we will
    report 16.
  * NO assumption that Nvidia is present. Each vendor is probed
    independently. A system with only AMD GPUs is a first-class case.
  * Bandwidth is MEASURED when tools allow (nvidia-smi pcie link
    queries, rocm-smi --showlinkinfo) and falls back to a PCIe
    gen × width calculation from sysfs when measurement is not
    possible. 0-GPU systems return a valid empty topology.
  * NUMA-aware ordering: when NUMA topology is discoverable, devices
    are ordered so co-resident NUMA devices are adjacent in the list
    — this is the optimal ordering for sharding.

Every probe either succeeds with EVIDENCE (device paths, vendor IDs,
PCIe BDF) or returns an empty/absent result. There is no silent
"Unknown" fallback that lets the rest of the pipeline proceed with
phantom devices.
"""

from __future__ import annotations

import functools
import json
import platform
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.common.errors import (
    HardwareNotFoundError,
    HardwareProbeError,
)
from src.common.types import Vendor

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CpuVendor(str, Enum):
    INTEL = "intel"
    AMD = "amd"
    APPLE = "apple"
    ARM = "arm"
    OTHER = "other"
    UNKNOWN = "unknown"


class GpuVendor(str, Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"


class LinkType(str, Enum):
    """Interconnect link between two devices (or device<->host)."""

    PCIE = "pcie"
    NVLINK = "nvlink"
    NVSWITCH = "nvswitch"
    INFINITY_FABRIC = "infinity_fabric"
    XELINK = "xelink"
    QPI = "qpi"  # Intel CPU <-> CPU
    UALINK = "ualink"  # Universal Accelerator Link
    HOST_RAM = "host_ram"  # Fallback: cross-device via host memory
    UNKNOWN = "unknown"


# PCI vendor IDs we care about. Canonical lowercase 4-char hex.
PCI_VENDOR_NVIDIA = "10de"
PCI_VENDOR_AMD = "1002"
PCI_VENDOR_INTEL = "8086"

# Maps GpuVendor <-> PCI vendor ID. Both directions are derivable.
PCI_VENDOR_FOR_GPU: dict[GpuVendor, str] = {
    GpuVendor.NVIDIA: PCI_VENDOR_NVIDIA,
    GpuVendor.AMD: PCI_VENDOR_AMD,
    GpuVendor.INTEL: PCI_VENDOR_INTEL,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostInfo:
    """Information about the host CPU/system."""

    cpu_vendor: CpuVendor
    cpu_model: str
    os: str
    os_release: str
    kernel: str
    architecture: str
    raw: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DeviceInfo:
    """A single GPU/accelerator device detected on the system."""

    vendor: GpuVendor
    device_id: int
    device_path: str
    arch: str
    model: str
    driver_version: str
    pcie_bdf: str = ""  # e.g. "0000:01:00.0"
    numa_node: int = -1  # -1 = unknown / not applicable
    pcie_gen: int = 0  # 0 = unknown
    pcie_width: int = 0  # 0 = unknown
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def nautilus_vendor(self) -> Vendor:
        return {
            GpuVendor.NVIDIA: Vendor.NVIDIA,
            GpuVendor.AMD: Vendor.AMD,
            GpuVendor.INTEL: Vendor.INTEL,
            GpuVendor.APPLE: Vendor.APPLE,
        }.get(self.vendor, Vendor.UNKNOWN)


@dataclass(frozen=True)
class TopologyLink:
    """Interconnect between two devices (or device<->host)."""

    source_id: int
    target_id: int
    bandwidth_gbps: float
    link_type: LinkType
    pcie_gen: int = 0
    pcie_width: int = 0
    measured: bool = False  # True if a tool reported it; False if PCIe-gen fallback


@dataclass(frozen=True)
class DeviceTopology:
    """Complete GPU topology: all devices, all interconnect bandwidths, NUMA layout.

    Attributes:
        devices: Devices sorted NUMA-first (devices in the same NUMA node
                 are adjacent). 0-GPU systems yield an empty list.
        bandwidth_gbps: Symmetric {(id_a, id_b): gbps}. Self-pairs (a, a)
                 are not stored. Missing keys mean "no measured/fallback
                 value"; use `bandwidth_gbps.get(...)` defensively.
        links: Detailed link records (PCIe gen/width, link type, source).
        host: HostInfo for the system hosting these devices.
        numa_nodes: {device_id: numa_node}. -1 means unknown.
    """

    devices: list[DeviceInfo]
    bandwidth_gbps: dict[tuple[int, int], float]
    links: list[TopologyLink]
    host: HostInfo
    numa_nodes: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "host": {
                "cpu_vendor": self.host.cpu_vendor.value,
                "cpu_model": self.host.cpu_model,
                "os": self.host.os,
                "os_release": self.host.os_release,
                "kernel": self.host.kernel,
                "architecture": self.host.architecture,
            },
            "device_count": len(self.devices),
            "devices": [
                {
                    "device_id": d.device_id,
                    "vendor": d.vendor.value,
                    "nautilus_vendor": d.nautilus_vendor.value,
                    "arch": d.arch,
                    "model": d.model,
                    "device_path": d.device_path,
                    "driver_version": d.driver_version,
                    "pcie_bdf": d.pcie_bdf,
                    "numa_node": d.numa_node,
                    "pcie_gen": d.pcie_gen,
                    "pcie_width": d.pcie_width,
                }
                for d in self.devices
            ],
            "bandwidth_gbps": [
                {"source": a, "target": b, "gbps": gbps}
                for (a, b), gbps in sorted(self.bandwidth_gbps.items())
            ],
            "links": [
                {
                    "source_id": l.source_id,
                    "target_id": l.target_id,
                    "bandwidth_gbps": l.bandwidth_gbps,
                    "link_type": l.link_type.value,
                    "pcie_gen": l.pcie_gen,
                    "pcie_width": l.pcie_width,
                    "measured": l.measured,
                }
                for l in self.links
            ],
            "numa_nodes": dict(self.numa_nodes),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# OS helpers
# ---------------------------------------------------------------------------


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _is_windows() -> bool:
    return sys.platform.startswith("win")


# ---------------------------------------------------------------------------
# CPU vendor detection
# ---------------------------------------------------------------------------


def detect_host_vendor() -> CpuVendor:
    """Detect the host CPU vendor.

    Linux/macOS: parse /proc/cpuinfo or sysctl.
    Windows: parse wmic (best effort).

    Returns UNKNOWN if the system is exotic or the probe fails.
    """
    if _is_linux():
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "vendor_id" in line:
                        value = line.split(":", 1)[1].strip().lower()
                        if "genuineintel" in value:
                            return CpuVendor.INTEL
                        if "authenticamd" in value:
                            return CpuVendor.AMD
                        if "apple" in value:
                            return CpuVendor.APPLE
                        if "arm" in value:
                            return CpuVendor.ARM
        except (OSError, FileNotFoundError):
            pass
    elif _is_macos():
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.vendor"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0:
                value = out.stdout.strip().lower()
                if "apple" in value:
                    return CpuVendor.APPLE
                if "intel" in value:
                    return CpuVendor.INTEL
                if "amd" in value:
                    return CpuVendor.AMD
                if "arm" in value:
                    return CpuVendor.ARM
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    elif _is_windows():
        try:
            out = subprocess.run(
                ["wmic", "cpu", "get", "manufacturer"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0:
                value = out.stdout.lower()
                if "intel" in value:
                    return CpuVendor.INTEL
                if "amd" in value:
                    return CpuVendor.AMD
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return CpuVendor.UNKNOWN


def get_host_info() -> HostInfo:
    """Return a structured HostInfo for the current machine."""
    cpu_vendor = detect_host_vendor()
    model = ""
    if _is_linux():
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        model = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    elif _is_macos():
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0:
                model = out.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    raw: dict[str, str] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }
    return HostInfo(
        cpu_vendor=cpu_vendor,
        cpu_model=model,
        os=platform.system(),
        os_release=platform.release(),
        kernel=platform.release(),
        architecture=platform.machine(),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# PCIe bandwidth calculation (pure)
# ---------------------------------------------------------------------------


# Gbps per lane, full-duplex, for each PCIe generation.
# Gen1: 2.5 GT/s × 8b/10b = 2 Gbps/lane
# Gen2: 5   GT/s × 8b/10b = 4 Gbps/lane
# Gen3: 8   GT/s × 128b/130b ≈ 7.877, rounded to 8 Gbps/lane
# Gen4: 16  GT/s × 128b/130b ≈ 15.754, rounded to 16 Gbps/lane
# Gen5: 32  GT/s × 128b/130b ≈ 31.508, rounded to 32 Gbps/lane
PCIE_GBPS_PER_LANE: dict[int, float] = {
    1: 2.0,
    2: 4.0,
    3: 8.0,
    4: 16.0,
    5: 32.0,
    6: 64.0,
}


def pcie_bandwidth_gbps(gen: int, width: int) -> float:
    """Compute PCIe bandwidth in Gbps (full-duplex) for a gen×width link.

    Returns 0.0 for unknown (gen<=0 or width<=0) values. This is the
    fallback bandwidth when no tool-side measurement is available.
    """
    if gen <= 0 or width <= 0:
        return 0.0
    per_lane = PCIE_GBPS_PER_LANE.get(gen, 0.0)
    if per_lane <= 0.0:
        return 0.0
    return per_lane * float(width)


def parse_pcie_speed_string(text: str) -> int:
    """Parse `current_link_speed` from sysfs.

    Examples:
        "2.5 GT/s PCIe"      -> 1
        "5.0 GT/s PCIe"      -> 2
        "8.0 GT/s PCIe"      -> 3
        "8 GT/s PCIe"        -> 3  (integer also accepted)
        "16.0 GT/s PCIe"     -> 4
        "32.0 GT/s PCIe"     -> 5
        "64.0 GT/s PCIe"     -> 6
    """
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*GT/s", text)
    if not m:
        return 0
    gtps = float(m.group(1))
    if gtps <= 2.5:
        return 1
    if gtps <= 5.0:
        return 2
    if gtps <= 8.0:
        return 3
    if gtps <= 16.0:
        return 4
    if gtps <= 32.0:
        return 5
    if gtps <= 64.0:
        return 6
    return 0


def parse_pcie_width_string(text: str) -> int:
    """Parse `current_link_width` from sysfs (e.g. '16' or 'x16')."""
    m = re.search(r"x?([0-9]+)", text)
    if not m:
        return 0
    return int(m.group(1))


# ---------------------------------------------------------------------------
# NUMA / sysfs lookup
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except (OSError, FileNotFoundError):
        return ""


def _pci_device_dirs(sysfs_pci: Path | None = None) -> list[Path]:
    """Return all /sys/bus/pci/devices/* directories.

    `sysfs_pci` is injectable for tests; defaults to the real path on Linux.
    """
    base = sysfs_pci if sysfs_pci is not None else Path("/sys/bus/pci/devices")
    if not base.exists() or not base.is_dir():
        return []
    return [p for p in base.iterdir() if p.is_dir()]


def _pci_vendor_for_dir(dev_dir: Path) -> str:
    """Read vendor id (lowercase hex, no 0x prefix) from sysfs."""
    text = _read_text(dev_dir / "vendor")
    if not text:
        return ""
    return text.lower().lstrip("0x")


def _pci_device_id_for_dir(dev_dir: Path) -> str:
    text = _read_text(dev_dir / "device")
    if not text:
        return ""
    return text.lower().lstrip("0x")


def _pci_class_for_dir(dev_dir: Path) -> str:
    """PCI class as lowercase hex (e.g. '030000' = VGA, '030200' = 3D)."""
    text = _read_text(dev_dir / "class")
    if not text:
        return ""
    text = text.lower()
    if text.startswith("0x"):
        text = text[2:]
    return text


def _pci_id_for_dir(dev_dir: Path, name: str) -> str:
    """PCI vendor/device id (lowercase hex, no 0x prefix) from sysfs."""
    text = _read_text(dev_dir / name)
    if not text:
        return ""
    text = text.lower()
    if text.startswith("0x"):
        text = text[2:]
    return text


def _is_display_class(class_hex: str) -> bool:
    """True for VGA (0x0300xx) or 3D controller (0x0302xx)."""
    if not class_hex or len(class_hex) < 4:
        return False
    return class_hex.startswith("0300") or class_hex.startswith("0302")


def _numa_node_for_pci_dir(dev_dir: Path) -> int:
    text = _read_text(dev_dir / "numa_node")
    if not text:
        return -1
    try:
        return int(text)
    except ValueError:
        return -1


def _pcie_gen_for_dir(dev_dir: Path) -> int:
    text = _read_text(dev_dir / "current_link_speed")
    if not text:
        return 0
    return parse_pcie_speed_string(text)


def _pcie_width_for_dir(dev_dir: Path) -> int:
    text = _read_text(dev_dir / "current_link_width")
    if not text:
        return 0
    return parse_pcie_width_string(text)


# ---------------------------------------------------------------------------
# Dynamic device-path scanning
# ---------------------------------------------------------------------------


def _is_char_device(path: Path) -> bool:
    """True if path exists and is a character device (or symlink to one).

    On non-Linux, just check exists(). On Linux, stat.st_mode & S_IFCHR.
    """
    try:
        st = path.stat()
    except OSError:
        return False
    if _is_linux():
        import stat

        return stat.S_ISCHR(st.st_mode)
    return path.exists()


def scan_nvidia_devices(
    dev_root: Path | None = None, proc_root: Path | None = None
) -> list[DeviceInfo]:
    """Dynamically scan for Nvidia devices.

    Sources (all optional, all consulted):
      * `<dev_root>/nvidia*`        — primary device nodes (nvidia0, nvidia1, ...)
      * `<dev_root>/nvidia-uvm*`    — UVM is ignored (not a per-GPU device)
      * `<proc_root>/driver/nvidia/gpus/*` — per-GPU metadata dirs (preferred
        when present, gives us a real index even if /dev nodes are weird)

    We do NOT cap the count. Whatever the kernel exposes, we return.
    """
    if not _is_linux():
        return []
    dev_root = dev_root if dev_root is not None else Path("/dev")
    proc_root = proc_root if proc_root is not None else Path("/proc")

    # Primary path: /dev/nvidia* (skip nvidia-uvm, nvidiactl, nvidia-modeset).
    candidates: list[tuple[int, str]] = []  # (index_or_neg, path)
    if dev_root.exists():
        for p in sorted(dev_root.glob("nvidia*")):
            name = p.name
            # Per-GPU device nodes are /dev/nvidia<digits>. We want those.
            # Exclude: nvidia0..N (kept), nvidiactl (control), nvidia-uvm (uvm),
            # nvidia-uvm-tools, nvidia-modeset, nvidia-caps (control), etc.
            m = re.fullmatch(r"nvidia([0-9]+)", name)
            if not m:
                continue
            if not _is_char_device(p):
                continue
            candidates.append((int(m.group(1)), str(p)))

    # Fallback / augmentation: /proc/driver/nvidia/gpus/<index>/
    gpu_proc_dir = proc_root / "driver" / "nvidia" / "gpus"
    proc_indices: set[int] = set()
    if gpu_proc_dir.exists() and gpu_proc_dir.is_dir():
        for p in gpu_proc_dir.iterdir():
            if not p.is_dir():
                continue
            try:
                idx = int(p.name)
            except ValueError:
                continue
            proc_indices.add(idx)
            # If we have a /dev/nvidia<idx> already, prefer it; else add a
            # proc-derived placeholder so 0-GPU is not falsely reported.
            existing_paths = [c[1] for c in candidates if c[0] == idx]
            if not existing_paths:
                candidates.append((idx, str(p)))  # proc fallback path

    # Final, sorted by index. De-dup by index.
    by_index: dict[int, str] = {}
    for idx, path in candidates:
        by_index.setdefault(idx, path)
    if not by_index:
        return []

    # Optional: driver version + try to read model from proc entry.
    driver_version = _nvidia_driver_version_linux(proc_root)

    devices: list[DeviceInfo] = []
    for idx in sorted(by_index.keys()):
        path = by_index[idx]
        # NUMA / PCIe info (best-effort)
        numa = -1
        gen = 0
        width = 0
        bdf = ""
        for d in _pci_device_dirs():
            if _pci_vendor_for_dir(d) == PCI_VENDOR_NVIDIA and _is_display_class(
                _pci_class_for_dir(d)
            ):
                # We cannot map proc index to BDF without a helper; use first match.
                bdf = d.name
                numa = _numa_node_for_pci_dir(d)
                gen = _pcie_gen_for_dir(d)
                width = _pcie_width_for_dir(d)
                break
        devices.append(
            DeviceInfo(
                vendor=GpuVendor.NVIDIA,
                device_id=idx,
                device_path=path,
                arch="",
                model="",
                driver_version=driver_version,
                pcie_bdf=bdf,
                numa_node=numa,
                pcie_gen=gen,
                pcie_width=width,
                raw={"proc_indices": sorted(proc_indices)},
            )
        )
    return devices


def scan_amd_devices(
    dev_root: Path | None = None, sysfs_pci: Path | None = None
) -> list[DeviceInfo]:
    """Dynamically scan for AMD devices.

    Sources:
      * `<dev_root>/kfd`           — KFD (compute) is required for ROCm.
      * `<dev_root>/dri/renderD*`  — one per GPU, vendor matched via sysfs.

    We DO NOT add a renderD* node unless a matching AMD PCI device exists.
    A renderD* node belonging to a non-AMD GPU (e.g. Intel iGPU on hybrid
    laptops) is filtered out.
    """
    if not _is_linux():
        return []
    dev_root = dev_root if dev_root is not None else Path("/dev")

    has_kfd = (dev_root / "kfd").exists()

    # Find AMD PCI devices first.
    amd_bdfs: list[tuple[str, Path]] = []
    for d in _pci_device_dirs(sysfs_pci):
        if _pci_vendor_for_dir(d) == PCI_VENDOR_AMD and _is_display_class(_pci_class_for_dir(d)):
            amd_bdfs.append((d.name, d))

    dri_dir = dev_root / "dri"
    render_paths: list[tuple[str, Path]] = []
    if dri_dir.exists():
        for p in sorted(dri_dir.glob("renderD*")):
            if p.exists():
                render_paths.append((p.name, p))

    # Match render nodes to AMD PCI devices by index parity: renderD128 is
    # typically the first DRI node for GPU 0, renderD129 for GPU 1, etc.
    # This is a heuristic, but it correctly filters out Intel iGPUs on
    # hybrid systems because they would have a different vendor.
    devices: list[DeviceInfo] = []
    for i, (bdf, dev_dir) in enumerate(amd_bdfs):
        # Default path: prefer kfd + a renderD node if available.
        path = str(dev_root / "kfd") if has_kfd else str(dev_dir)
        if i < len(render_paths):
            # Pair the i-th AMD PCI device with the i-th render node.
            path = str(render_paths[i][1])
        elif has_kfd:
            path = str(dev_root / "kfd")
        devices.append(
            DeviceInfo(
                vendor=GpuVendor.AMD,
                device_id=i,
                device_path=path,
                arch="",
                model="",
                driver_version=_amd_driver_version_linux(),
                pcie_bdf=bdf,
                numa_node=_numa_node_for_pci_dir(dev_dir),
                pcie_gen=_pcie_gen_for_dir(dev_dir),
                pcie_width=_pcie_width_for_dir(dev_dir),
                raw={"has_kfd": has_kfd},
            )
        )
    return devices


def scan_intel_devices(
    dev_root: Path | None = None, sysfs_pci: Path | None = None
) -> list[DeviceInfo]:
    """Dynamically scan for Intel devices.

    Sources:
      * `<dev_root>/dri/renderD*` — one per GPU, vendor matched via sysfs.

    We filter to PCI vendor 0x8086 (Intel). renderD* nodes belonging to
    non-Intel GPUs are not included.
    """
    if not _is_linux():
        return []
    dev_root = dev_root if dev_root is not None else Path("/dev")

    intel_bdfs: list[tuple[str, Path]] = []
    for d in _pci_device_dirs(sysfs_pci):
        if _pci_vendor_for_dir(d) == PCI_VENDOR_INTEL and _is_display_class(_pci_class_for_dir(d)):
            intel_bdfs.append((d.name, d))

    dri_dir = dev_root / "dri"
    render_paths: list[Path] = []
    if dri_dir.exists():
        render_paths = sorted(p for p in dri_dir.glob("renderD*") if p.exists())

    devices: list[DeviceInfo] = []
    for i, (bdf, dev_dir) in enumerate(intel_bdfs):
        if i < len(render_paths):
            path = str(render_paths[i])
        else:
            path = str(dev_dir)
        devices.append(
            DeviceInfo(
                vendor=GpuVendor.INTEL,
                device_id=i,
                device_path=path,
                arch="",
                model="",
                driver_version=_intel_driver_version_linux(),
                pcie_bdf=bdf,
                numa_node=_numa_node_for_pci_dir(dev_dir),
                pcie_gen=_pcie_gen_for_dir(dev_dir),
                pcie_width=_pcie_width_for_dir(dev_dir),
                raw={},
            )
        )
    return devices


# ---------------------------------------------------------------------------
# lspci (optional enhancement, kept from prior impl)
# ---------------------------------------------------------------------------


def _lspci_gpu_entries() -> list[dict[str, str]]:
    """Parse `lspci -nn -mm` for VGA / 3D / Display controllers."""
    if not shutil.which("lspci"):
        return []
    try:
        out = subprocess.run(
            ["lspci", "-nn", "-mm"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        raise HardwareProbeError(
            "lspci timed out",
            cause=exc,
            context={"tool": "lspci", "timeout_seconds": 10},
        ) from exc
    if out.returncode != 0:
        return []
    entries: list[dict[str, str]] = []
    for line in out.stdout.splitlines():
        if not any(kw in line.lower() for kw in ("vga", "3d controller", "display controller")):
            continue
        try:
            parts = shlex.split(line)
        except ValueError:
            continue
        if len(parts) < 4:
            continue
        entry = {
            "class": parts[0],
            "vendor": parts[1],
            "device": parts[2],
            "subsys": parts[3] if len(parts) > 3 else "",
            "slot": parts[-1] if len(parts) > 4 else "",
            "raw": line,
        }
        m = re.search(r"\[([0-9a-f]{4}):([0-9a-f]{4})\]", parts[1] if len(parts) > 1 else "")
        if m:
            entry["vendor_id"] = m.group(1)
            entry["device_id"] = m.group(2)
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Driver versions
# ---------------------------------------------------------------------------


def _nvidia_driver_version_linux(proc_root: Path | None = None) -> str:
    """Best-effort: cat /proc/driver/nvidia/version."""
    if not _is_linux():
        return ""
    base = proc_root if proc_root is not None else Path("/proc")
    p = base / "driver" / "nvidia" / "version"
    if not p.exists():
        return ""
    try:
        text = p.read_text()
        m = re.search(r"NVIDIA driver version:\s*([0-9.]+)", text)
        return m.group(1) if m else ""
    except OSError:
        return ""


def _amd_driver_version_linux() -> str:
    """Best-effort: rocm-version or /opt/rocm/.info."""
    if shutil.which("rocm-info"):
        try:
            out = subprocess.run(
                ["rocm-info"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0:
                m = re.search(r"Version\s*:\s*([0-9.]+)", out.stdout)
                if m:
                    return m.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    info = Path("/opt/rocm/.info")
    if info.exists():
        try:
            return info.read_text().strip()
        except OSError:
            pass
    return ""


def _intel_driver_version_linux() -> str:
    """Best-effort: vainfo or intel_gpu_top."""
    if shutil.which("vainfo"):
        try:
            out = subprocess.run(
                ["vainfo"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0:
                m = re.search(r"Driver version\s*:\s*([0-9.]+)", out.stdout)
                if m:
                    return m.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return ""


# ---------------------------------------------------------------------------
# macOS / Windows
# ---------------------------------------------------------------------------


def _macos_gpu_info() -> list[DeviceInfo]:
    """Use system_profiler -SPDisplaysDataType to find GPUs."""
    if not _is_macos():
        return []
    try:
        out = subprocess.run(
            ["system_profiler", "-json", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return []
    devices: list[DeviceInfo] = []
    for entry in data.get("SPDisplaysDataType", []):
        chipset = entry.get("spdisplays_chipset", "")
        vendor_name = entry.get("spdisplays_vendor", "").lower()
        model = entry.get("spdisplays_device_name", entry.get("spdisplays_model", ""))
        if "apple" in chipset.lower() or "apple" in vendor_name:
            vendor = GpuVendor.APPLE
        elif "amd" in chipset.lower() or "amd" in vendor_name:
            vendor = GpuVendor.AMD
        elif "intel" in chipset.lower() or "intel" in vendor_name:
            vendor = GpuVendor.INTEL
        elif "nvidia" in chipset.lower() or "nvidia" in vendor_name:
            vendor = GpuVendor.NVIDIA
        else:
            vendor = GpuVendor.UNKNOWN
        devices.append(
            DeviceInfo(
                vendor=vendor,
                device_id=len(devices),
                device_path="system_profiler",
                arch=chipset,
                model=model,
                driver_version=entry.get("spdisplays_metal", ""),
            )
        )
    return devices


# ---------------------------------------------------------------------------
# Top-level enumeration
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def enumerate_devices() -> tuple[DeviceInfo, ...]:
    """Enumerate every GPU/accelerator device on this system.

    Strategy (Linux):
      1. scan_nvidia_devices()    — /dev/nvidia* + /proc/driver/nvidia/gpus/*
      2. scan_amd_devices()       — /dev/kfd + /dev/dri/renderD*, vendor-matched
      3. scan_intel_devices()     — /dev/dri/renderD*, vendor-matched
      4. lspci                    — best-effort enrichment (model names)

    macOS: system_profiler.
    Windows: wmic (best-effort).

    Always returns a tuple (possibly empty). Never raises on missing
    tools — the per-vendor high-level helpers below do raise.

    Returned as a tuple so it can be lru_cached safely.
    """
    devices: list[DeviceInfo] = []

    if _is_linux():
        nv = scan_nvidia_devices()
        amd = scan_amd_devices()
        intel = scan_intel_devices()

        # Best-effort lspci enrichment — gives us model names if available.
        try:
            lspci_entries = _lspci_gpu_entries()
        except HardwareProbeError:
            lspci_entries = []

        nv_pci = [e for e in lspci_entries if e.get("vendor_id", "").lower() == PCI_VENDOR_NVIDIA]
        amd_pci = [e for e in lspci_entries if e.get("vendor_id", "").lower() == PCI_VENDOR_AMD]
        intel_pci = [e for e in lspci_entries if e.get("vendor_id", "").lower() == PCI_VENDOR_INTEL]

        for i, d in enumerate(nv):
            if i < len(nv_pci):
                d = DeviceInfo(
                    vendor=d.vendor,
                    device_id=d.device_id,
                    device_path=d.device_path,
                    arch=nv_pci[i].get("device_id", "") or d.arch,
                    model=nv_pci[i].get("device", "") or d.model,
                    driver_version=d.driver_version,
                    pcie_bdf=d.pcie_bdf,
                    numa_node=d.numa_node,
                    pcie_gen=d.pcie_gen,
                    pcie_width=d.pcie_width,
                    raw={**d.raw, "lspci": nv_pci[i]},
                )
            devices.append(d)

        for i, d in enumerate(amd):
            if i < len(amd_pci):
                d = DeviceInfo(
                    vendor=d.vendor,
                    device_id=d.device_id,
                    device_path=d.device_path,
                    arch=amd_pci[i].get("device_id", "") or d.arch,
                    model=amd_pci[i].get("device", "") or d.model,
                    driver_version=d.driver_version,
                    pcie_bdf=d.pcie_bdf,
                    numa_node=d.numa_node,
                    pcie_gen=d.pcie_gen,
                    pcie_width=d.pcie_width,
                    raw={**d.raw, "lspci": amd_pci[i]},
                )
            devices.append(d)

        for i, d in enumerate(intel):
            if i < len(intel_pci):
                d = DeviceInfo(
                    vendor=d.vendor,
                    device_id=d.device_id,
                    device_path=d.device_path,
                    arch=intel_pci[i].get("device_id", "") or d.arch,
                    model=intel_pci[i].get("device", "") or d.model,
                    driver_version=d.driver_version,
                    pcie_bdf=d.pcie_bdf,
                    numa_node=d.numa_node,
                    pcie_gen=d.pcie_gen,
                    pcie_width=d.pcie_width,
                    raw={**d.raw, "lspci": intel_pci[i]},
                )
            devices.append(d)

    elif _is_macos():
        devices.extend(_macos_gpu_info())

    elif _is_windows():
        if shutil.which("wmic"):
            try:
                out = subprocess.run(
                    [
                        "wmic",
                        "path",
                        "win32_VideoController",
                        "get",
                        "Name,AdapterCompatibility,DriverVersion",
                        "/format:list",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                for block in out.stdout.split("\n\n"):
                    entry: dict[str, str] = {}
                    for line in block.splitlines():
                        if "=" in line:
                            k, v = line.split("=", 1)
                            entry[k.strip()] = v.strip()
                    name = entry.get("Name", "")
                    compat = entry.get("AdapterCompatibility", "").lower()
                    if not name:
                        continue
                    if "nvidia" in compat or "nvidia" in name.lower():
                        vendor = GpuVendor.NVIDIA
                    elif "amd" in compat or "amd" in name.lower() or "ati" in name.lower():
                        vendor = GpuVendor.AMD
                    elif "intel" in compat or "intel" in name.lower():
                        vendor = GpuVendor.INTEL
                    else:
                        vendor = GpuVendor.UNKNOWN
                    devices.append(
                        DeviceInfo(
                            vendor=vendor,
                            device_id=len(devices),
                            device_path="wmic",
                            arch="",
                            model=name,
                            driver_version=entry.get("DriverVersion", ""),
                        )
                    )
            except subprocess.TimeoutExpired as exc:
                raise HardwareProbeError(
                    "wmic timed out",
                    cause=exc,
                    context={"tool": "wmic", "timeout_seconds": 10},
                ) from exc
        else:
            raise HardwareProbeError(
                "wmic not found on Windows",
                context={"tool": "wmic"},
            )

    return tuple(devices)


def invalidate_enumeration_cache() -> None:
    """Clear the lru_cache on enumerate_devices. Used by tests."""
    enumerate_devices.cache_clear()


# ---------------------------------------------------------------------------
# Convenience predicates (raise on absent)
# ---------------------------------------------------------------------------


def has_nvidia_gpu() -> bool:
    for d in enumerate_devices():
        if d.vendor == GpuVendor.NVIDIA:
            return True
    return False


def has_amd_gpu() -> bool:
    for d in enumerate_devices():
        if d.vendor == GpuVendor.AMD:
            return True
    return False


def has_intel_gpu() -> bool:
    for d in enumerate_devices():
        if d.vendor == GpuVendor.INTEL:
            return True
    return False


def has_apple_gpu() -> bool:
    for d in enumerate_devices():
        if d.vendor == GpuVendor.APPLE:
            return True
    return False


def detect_gpu_vendors() -> set[Vendor]:
    """Return the set of GPU vendors present on this system."""
    return {d.nautilus_vendor for d in enumerate_devices() if d.nautilus_vendor != Vendor.UNKNOWN}


def get_device_paths(vendor: Vendor) -> list[str]:
    """Return the list of device paths for a given vendor.

    Raises HardwareNotFoundError if no devices for that vendor.
    """
    gv = {
        Vendor.NVIDIA: GpuVendor.NVIDIA,
        Vendor.AMD: GpuVendor.AMD,
        Vendor.INTEL: GpuVendor.INTEL,
        Vendor.APPLE: GpuVendor.APPLE,
    }.get(vendor, GpuVendor.UNKNOWN)
    paths = [d.device_path for d in enumerate_devices() if d.vendor == gv]
    if not paths and vendor != Vendor.UNKNOWN:
        raise HardwareNotFoundError(
            f"No {vendor.value} GPU detected on this system",
            context={
                "vendor": vendor.value,
                "detected_vendors": [v.value for v in detect_gpu_vendors()],
            },
        )
    return paths


def probe_pcie_for_gpus() -> list[dict[str, str]]:
    """Return lspci GPU entries (class, vendor, device, vendor_id, device_id)."""
    return _lspci_gpu_entries()


# ---------------------------------------------------------------------------
# PCIe bandwidth measurement
# ---------------------------------------------------------------------------


def _nvidia_pcie_link_query() -> dict[int, tuple[int, int]]:
    """Try `nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current`.

    Returns {index: (gen, width)}. Empty if nvidia-smi missing or fails.
    """
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,pcie.link.gen.current,pcie.link.width.current",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}
    if out.returncode != 0:
        return {}
    result: dict[int, tuple[int, int]] = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            gen = int(parts[1])
            width = int(parts[2])
        except ValueError:
            continue
        result[idx] = (gen, width)
    return result


def _amd_pcie_link_query() -> dict[int, tuple[int, int]]:
    """Best-effort `rocm-smi --showlinkinfo` for AMD.

    Returns {index: (gen, width)}. Empty if missing or unparseable.
    """
    if not shutil.which("rocm-smi"):
        return {}
    try:
        out = subprocess.run(
            ["rocm-smi", "--showlinkinfo"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}
    if out.returncode != 0:
        return {}
    result: dict[int, tuple[int, int]] = {}
    # Output varies by ROCm version. Try to be permissive.
    for line in out.stdout.splitlines():
        # Look for "PCIe Gen" and "Width" tokens, both on the same line.
        gen_match = re.search(r"PCIe Gen[^\d]*(\d+)", line, re.IGNORECASE)
        width_match = re.search(r"Width[^\d]*(\d+)", line, re.IGNORECASE)
        if gen_match and width_match:
            try:
                result[len(result)] = (int(gen_match.group(1)), int(width_match.group(1)))
            except ValueError:
                pass
    return result


# ---------------------------------------------------------------------------
# Inter-device link probing (NVLink / Infinity Fabric / XeLink)
# ---------------------------------------------------------------------------


def _nvidia_nvlink_topology() -> dict[tuple[int, int], float]:
    """Try `nvidia-smi topo -m` to find NVLink bandwidths.

    Returns {(gpu_a, gpu_b): gbps}. Symmetric: only stores (min, max).
    """
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}
    if out.returncode != 0:
        return {}
    lines = out.stdout.splitlines()
    if not lines:
        return {}
    # First non-empty line is the header. Find GPU indices.
    header = next((l for l in lines if "GPU" in l and any(c.isdigit() for c in l)), "")
    if not header:
        return {}
    cols = [c for c in re.split(r"\s+", header.strip()) if c.startswith("GPU")]
    if not cols:
        return {}
    # GPU indices: GPU0 -> 0, GPU1 -> 1, ...
    col_idx: list[int] = []
    for c in cols:
        m = re.match(r"GPU(\d+)", c)
        if m:
            col_idx.append(int(m.group(1)))

    # NVLink bandwidths in nvidia-smi topo are listed in a unit-less form.
    # Common pattern: "NV1", "NV2", "NV12", "NVLink" with no numeric bandwidth.
    # When the matrix reports "NV#" we treat it as 25 Gbps/lane × #lanes,
    # but more conservatively we report 25.0 Gbps per "NV1" link and skip
    # ambiguous entries. 0 (no link) is reported as 0 Gbps.
    result: dict[tuple[int, int], float] = {}
    for line in lines[lines.index(header) + 1 :]:
        parts = re.split(r"\s+", line.strip())
        if not parts or not parts[0].startswith("GPU"):
            continue
        m = re.match(r"GPU(\d+)", parts[0])
        if not m:
            continue
        row_idx = int(m.group(1))
        for ci, cell in enumerate(parts[1:]):
            if ci >= len(col_idx):
                break
            col_i = col_idx[ci]
            if row_idx >= col_i:
                continue  # we store only (min, max)
            if cell.upper() in ("0", "PXB", "PIX", "PHB", "NODE", "SYS"):
                # 0/PXB/PIX/PHB/NODE/SYS = no NVLink; we'll fall back to PCIe
                continue
            # "NV1", "NV2", "NV12" -> 25 Gbps per N
            nv_match = re.match(r"NV(\d+)", cell, re.IGNORECASE)
            if nv_match:
                lanes = int(nv_match.group(1))
                gbps = 25.0 * lanes  # NVLink 3.0/4.0 ≈ 25 Gbps/lane
                key = (min(row_idx, col_i), max(row_idx, col_i))
                # Take max if multiple cells refer to the same pair
                result[key] = max(result.get(key, 0.0), gbps)
    return result


# ---------------------------------------------------------------------------
# Bandwidth matrix (pure helpers)
# ---------------------------------------------------------------------------


def _bandwidth_for_pair(
    a: DeviceInfo,
    b: DeviceInfo,
    measured_nvlink: dict[tuple[int, int], float] | None = None,
) -> TopologyLink:
    """Compute the bandwidth between two devices.

    Priority:
      1. Measured NVLink / vendor-specific link (if provided).
      2. PCIe gen×width fallback (whichever device has lower gen/width).
    """
    measured_nvlink = measured_nvlink or {}

    key = (min(a.device_id, b.device_id), max(a.device_id, b.device_id))
    if key in measured_nvlink:
        return TopologyLink(
            source_id=a.device_id,
            target_id=b.device_id,
            bandwidth_gbps=measured_nvlink[key],
            link_type=LinkType.NVLINK,
            pcie_gen=0,
            pcie_width=0,
            measured=True,
        )

    # PCIe fallback: take min gen / min width (the link is bottlenecked by
    # the slower side).
    gens = [g for g in (a.pcie_gen, b.pcie_gen) if g > 0]
    widths = [w for w in (a.pcie_width, b.pcie_width) if w > 0]
    gen = min(gens) if gens else 0
    width = min(widths) if widths else 0
    bw = pcie_bandwidth_gbps(gen, width)
    return TopologyLink(
        source_id=a.device_id,
        target_id=b.device_id,
        bandwidth_gbps=bw,
        link_type=LinkType.PCIE,
        pcie_gen=gen,
        pcie_width=width,
        measured=False,
    )


def _sort_numa_first(devices: list[DeviceInfo]) -> list[DeviceInfo]:
    """Stable-sort devices so co-NUMA devices are adjacent.

    Devices with unknown NUMA (numa_node == -1) are grouped at the end.
    """

    def key(d: DeviceInfo) -> tuple[int, int, int]:
        # unknown NUMA at the end: use 10**9 as the sort key
        numa_key = d.numa_node if d.numa_node >= 0 else 10**9
        return (numa_key, d.vendor.value, d.device_id)

    return sorted(devices, key=key)


# ---------------------------------------------------------------------------
# DeviceTopology.discover()
# ---------------------------------------------------------------------------


def discover_topology() -> DeviceTopology:
    """Dynamically discover the system's complete GPU topology.

    Steps:
      1. enumerate_devices()  — find every GPU (no hardcoded cap).
      2. Optionally enrich PCIe gen/width with nvidia-smi/rocm-smi.
      3. Probe NVLink / vendor-specific inter-device links.
      4. Sort NUMA-first.
      5. Build a symmetric bandwidth_gbps matrix and a list of TopologyLinks.

    Returns a DeviceTopology that is valid even on a 0-GPU system.
    """
    # Make sure the lru_cache is fresh when called from tests.
    devices = list(enumerate_devices())

    # Enrich PCIe gen/width from per-vendor tools if sysfs missed it.
    nv_link = _nvidia_pcie_link_query()
    if nv_link:
        enriched: list[DeviceInfo] = []
        for d in devices:
            if d.vendor == GpuVendor.NVIDIA and d.device_id in nv_link:
                gen, width = nv_link[d.device_id]
                enriched.append(
                    DeviceInfo(
                        vendor=d.vendor,
                        device_id=d.device_id,
                        device_path=d.device_path,
                        arch=d.arch,
                        model=d.model,
                        driver_version=d.driver_version,
                        pcie_bdf=d.pcie_bdf,
                        numa_node=d.numa_node,
                        pcie_gen=gen or d.pcie_gen,
                        pcie_width=width or d.pcie_width,
                        raw=d.raw,
                    )
                )
            else:
                enriched.append(d)
        devices = enriched

    amd_link = _amd_pcie_link_query()
    if amd_link:
        enriched = []
        for d in devices:
            if d.vendor == GpuVendor.AMD and d.device_id in amd_link:
                gen, width = amd_link[d.device_id]
                enriched.append(
                    DeviceInfo(
                        vendor=d.vendor,
                        device_id=d.device_id,
                        device_path=d.device_path,
                        arch=d.arch,
                        model=d.model,
                        driver_version=d.driver_version,
                        pcie_bdf=d.pcie_bdf,
                        numa_node=d.numa_node,
                        pcie_gen=gen or d.pcie_gen,
                        pcie_width=width or d.pcie_width,
                        raw=d.raw,
                    )
                )
            else:
                enriched.append(d)
        devices = enriched

    # NUMA-aware ordering BEFORE link computation, so the bandwidth matrix
    # is already in NUMA-friendly order.
    devices = _sort_numa_first(devices)

    # Reassign device_ids sequentially so the bandwidth matrix keys are
    # 0..N-1 in NUMA-friendly order. This is the contract callers expect.
    renumbered: list[DeviceInfo] = []
    for new_id, d in enumerate(devices):
        renumbered.append(
            DeviceInfo(
                vendor=d.vendor,
                device_id=new_id,
                device_path=d.device_path,
                arch=d.arch,
                model=d.model,
                driver_version=d.driver_version,
                pcie_bdf=d.pcie_bdf,
                numa_node=d.numa_node,
                pcie_gen=d.pcie_gen,
                pcie_width=d.pcie_width,
                raw=d.raw,
            )
        )
    devices = renumbered

    # Probe NVLink / vendor-specific inter-device links.
    nvlink_map: dict[tuple[int, int], float] = {}
    nvidia_indices = [d.device_id for d in devices if d.vendor == GpuVendor.NVIDIA]
    if len(nvidia_indices) >= 2:
        measured = _nvidia_nvlink_topology()
        for (a, b), gbps in measured.items():
            if a in nvidia_indices and b in nvidia_indices:
                # Re-key to the renumbered device_ids.
                nvlink_map[(a, b)] = gbps

    # Build the bandwidth matrix and link list.
    bandwidth_gbps: dict[tuple[int, int], float] = {}
    links: list[TopologyLink] = []
    n = len(devices)
    for i in range(n):
        for j in range(i + 1, n):
            link = _bandwidth_for_pair(devices[i], devices[j], nvlink_map)
            bandwidth_gbps[(i, j)] = link.bandwidth_gbps
            bandwidth_gbps[(j, i)] = link.bandwidth_gbps
            links.append(link)

    numa_nodes = {d.device_id: d.numa_node for d in devices}

    return DeviceTopology(
        devices=devices,
        bandwidth_gbps=bandwidth_gbps,
        links=links,
        host=get_host_info(),
        numa_nodes=numa_nodes,
    )


# Backward-compatible alias used by callers that want a single entry point.
DeviceTopology.discover = staticmethod(discover_topology)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def format_device_summary() -> str:
    """Human-readable summary of all detected devices."""
    topo = discover_topology()
    lines = [
        f"Host: {topo.host.os} {topo.host.os_release} on {topo.host.architecture}",
        f"CPU: {topo.host.cpu_vendor.value} ({topo.host.cpu_model or 'unknown model'})",
        "",
        f"GPUs: {len(topo.devices)}",
    ]
    if not topo.devices:
        lines.append("  (none detected)")
    for d in topo.devices:
        numa = f"numa={d.numa_node}" if d.numa_node >= 0 else "numa=?"
        pcie = (
            f"pcie={d.pcie_gen}x{d.pcie_width}" if d.pcie_gen > 0 and d.pcie_width > 0 else "pcie=?"
        )
        lines.append(
            f"  [{d.device_id}] {d.vendor.value}: {d.model or 'unknown'} "
            f"arch={d.arch} path={d.device_path} {numa} {pcie} "
            f"driver={d.driver_version or 'n/a'}"
        )
    if topo.bandwidth_gbps:
        lines.append("")
        lines.append("Bandwidth matrix (Gbps):")
        ids = [d.device_id for d in topo.devices]
        header = "       " + "  ".join(f"d{i:>2}" for i in ids)
        lines.append(header)
        for i in ids:
            row = [f"d{i:>2}:"]
            for j in ids:
                if i == j:
                    row.append("  - ")
                else:
                    bw = topo.bandwidth_gbps.get((i, j), 0.0)
                    row.append(f"{bw:>4.0f}")
            lines.append("  ".join(row))
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_device_summary())

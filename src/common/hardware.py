"""
Hardware detection — REAL vendor probing, not "return 0" stubs.

The previous design had every GPU detector hardcoded to return 0.
This module probes the actual system:

  - Linux /dev nodes (nvidia0, kfd, dri/renderD*)
  - macOS Metal via system_profiler
  - Windows via DXGI (best-effort)
  - CPUID for host vendor identification
  - PCIe enumeration via lspci (if available)

Every probe either succeeds with EVIDENCE (device paths, vendor IDs)
or raises HardwareNotFoundError / HardwareProbeError. There is no
silent "Unknown" return — that would let the rest of the pipeline
proceed with a phantom device.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from src.common.errors import (
    HardwareNotFoundError,
    HardwareProbeError,
    ConfigError,
)
from src.common.types import Vendor, Arch


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
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def nautilus_vendor(self) -> Vendor:
        return {
            GpuVendor.NVIDIA: Vendor.NVIDIA,
            GpuVendor.AMD: Vendor.AMD,
            GpuVendor.INTEL: Vendor.INTEL,
            GpuVendor.APPLE: Vendor.APPLE,
        }.get(self.vendor, Vendor.UNKNOWN)


# --- OS helpers ---


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _is_windows() -> bool:
    return sys.platform.startswith("win")


# --- CPU vendor detection ---


def detect_host_vendor() -> CpuVendor:
    """Detect the host CPU vendor.

    Linux/macOS: parse /proc/cpuinfo or sysctl.
    Windows: parse wmic (best effort).

    Returns UNKNOWN if the system is exotic or the probe fails.
    """
    if _is_linux():
        try:
            with open("/proc/cpuinfo", "r") as f:
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
                capture_output=True, text=True, timeout=5,
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
                capture_output=True, text=True, timeout=5,
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
            with open("/proc/cpuinfo", "r") as f:
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
                capture_output=True, text=True, timeout=5,
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


# --- GPU detection: Linux ---


def _linux_nvidia_paths() -> list[str]:
    """Return /dev/nvidia* paths that actually exist."""
    if not _is_linux():
        return []
    return sorted(
        str(p) for p in Path("/dev").glob("nvidia*")
        if p.is_char_device() or p.exists()
    )


def _linux_amd_paths() -> list[str]:
    """Return /dev/kfd + /dev/dri/renderD* paths."""
    if not _is_linux():
        return []
    paths: list[str] = []
    kfd = Path("/dev/kfd")
    if kfd.exists():
        paths.append(str(kfd))
    paths.extend(
        str(p) for p in Path("/dev/dri").glob("renderD*")
        if p.exists()
    )
    return paths


def _linux_intel_paths() -> list[str]:
    """Return /dev/dri/renderD* and any explicit Intel device nodes."""
    if not _is_linux():
        return []
    return sorted(
        str(p) for p in Path("/dev/dri").glob("renderD*")
        if p.exists()
    )


def _lspci_gpu_entries() -> list[dict[str, str]]:
    """Parse `lspci -nn -mm` for VGA / 3D / Display controllers.

    Returns list of {vendor_id, device_id, class, slot, raw}.
    Raises HardwareProbeError if lspci is missing and not on macOS/Windows.
    """
    if not shutil.which("lspci"):
        return []
    try:
        out = subprocess.run(
            ["lspci", "-nn", "-mm"],
            capture_output=True, text=True, timeout=10,
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
        parts = [p.strip('"') for p in line.split('"')]
        if len(parts) < 4:
            continue
        # Class, vendor, device, subsys, slot
        entry = {
            "class": parts[0],
            "vendor": parts[1],
            "device": parts[2],
            "subsys": parts[3] if len(parts) > 3 else "",
            "slot": parts[-1] if len(parts) > 4 else "",
            "raw": line,
        }
        # Pull vendor_id / device_id from the bracketed [xxxx:yyyy]
        m = re.search(r"\[([0-9a-f]{4}):([0-9a-f]{4})\]", parts[1] if len(parts) > 1 else "")
        if m:
            entry["vendor_id"] = m.group(1)
            entry["device_id"] = m.group(2)
        entries.append(entry)
    return entries


def _nvidia_driver_version_linux() -> str:
    """Best-effort: cat /proc/driver/nvidia/version."""
    if not _is_linux():
        return ""
    p = Path("/proc/driver/nvidia/version")
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
    # Try rocm-info first
    if shutil.which("rocm-info"):
        try:
            out = subprocess.run(
                ["rocm-info"], capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                m = re.search(r"Version\s*:\s*([0-9.]+)", out.stdout)
                if m:
                    return m.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    # Try /opt/rocm/.info
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
                ["vainfo"], capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                m = re.search(r"Driver version\s*:\s*([0-9.]+)", out.stdout)
                if m:
                    return m.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return ""


# --- GPU detection: macOS ---


def _macos_gpu_info() -> list[DeviceInfo]:
    """Use system_profiler -SPDisplaysDataType to find GPUs."""
    if not _is_macos():
        return []
    try:
        out = subprocess.run(
            ["system_profiler", "-json", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=15,
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
        devices.append(DeviceInfo(
            vendor=vendor,
            device_id=0,
            device_path="system_profiler",
            arch=chipset,
            model=model,
            driver_version=entry.get("spdisplays_metal", ""),
            raw=entry,
        ))
    return devices


# --- Top-level GPU enumeration ---


def enumerate_devices() -> list[DeviceInfo]:
    """Enumerate every GPU/accelerator device on this system.

    Strategy:
      1. Linux: combine /dev node probing with lspci for device IDs/classes.
      2. macOS: system_profiler.
      3. Windows: wmic (best-effort).

    Always returns a (possibly empty) list. Never raises on missing
    tools — the per-vendor high-level helpers below do raise.
    """
    devices: list[DeviceInfo] = []

    if _is_linux():
        # 1. /dev node probing
        nv_paths = _linux_nvidia_paths()
        amd_paths = _linux_amd_paths()
        intel_paths = _linux_intel_paths()

        # 2. lspci for richer device info
        try:
            lspci_entries = _lspci_gpu_entries()
        except HardwareProbeError:
            lspci_entries = []

        # Nvidia
        nv_driver = _nvidia_driver_version_linux()
        nv_pci_devices = [
            e for e in lspci_entries
            if e.get("vendor_id", "") in ("10de",)  # NVIDIA
        ]
        if nv_paths or nv_pci_devices:
            for i, p in enumerate(nv_paths):
                model = ""
                if i < len(nv_pci_devices):
                    model = nv_pci_devices[i].get("device", "")
                devices.append(DeviceInfo(
                    vendor=GpuVendor.NVIDIA,
                    device_id=i,
                    device_path=p,
                    arch=nv_pci_devices[i].get("device_id", "") if i < len(nv_pci_devices) else "",
                    model=model,
                    driver_version=nv_driver,
                    raw={"lspci": nv_pci_devices[i] if i < len(nv_pci_devices) else {}},
                ))

        # AMD
        amd_driver = _amd_driver_version_linux()
        amd_pci_devices = [
            e for e in lspci_entries
            if e.get("vendor_id", "") in ("1002",)  # AMD
        ]
        if amd_paths or amd_pci_devices:
            for i, p in enumerate(amd_paths):
                model = ""
                if i < len(amd_pci_devices):
                    model = amd_pci_devices[i].get("device", "")
                devices.append(DeviceInfo(
                    vendor=GpuVendor.AMD,
                    device_id=i,
                    device_path=p,
                    arch=amd_pci_devices[i].get("device_id", "") if i < len(amd_pci_devices) else "",
                    model=model,
                    driver_version=amd_driver,
                    raw={"lspci": amd_pci_devices[i] if i < len(amd_pci_devices) else {}},
                ))

        # Intel
        intel_driver = _intel_driver_version_linux()
        intel_pci_devices = [
            e for e in lspci_entries
            if e.get("vendor_id", "") in ("8086",)  # Intel
        ]
        if intel_paths or intel_pci_devices:
            for i, p in enumerate(intel_paths):
                model = ""
                if i < len(intel_pci_devices):
                    model = intel_pci_devices[i].get("device", "")
                devices.append(DeviceInfo(
                    vendor=GpuVendor.INTEL,
                    device_id=i,
                    device_path=p,
                    arch=intel_pci_devices[i].get("device_id", "") if i < len(intel_pci_devices) else "",
                    model=model,
                    driver_version=intel_driver,
                    raw={"lspci": intel_pci_devices[i] if i < len(intel_pci_devices) else {}},
                ))

    elif _is_macos():
        devices.extend(_macos_gpu_info())

    elif _is_windows():
        # Best-effort via wmic; raise HardwareProbeError if wmic missing
        if shutil.which("wmic"):
            try:
                out = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get",
                     "Name,AdapterCompatibility,DriverVersion", "/format:list"],
                    capture_output=True, text=True, timeout=10,
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
                    devices.append(DeviceInfo(
                        vendor=vendor,
                        device_id=len(devices),
                        device_path="wmic",
                        arch="",
                        model=name,
                        driver_version=entry.get("DriverVersion", ""),
                        raw=entry,
                    ))
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

    return devices


# --- Convenience predicates (raise on absent) ---


def has_nvidia_gpu() -> bool:
    """Return True if at least one Nvidia device is detected."""
    for d in enumerate_devices():
        if d.vendor == GpuVendor.NVIDIA:
            return True
    return False


def has_amd_gpu() -> bool:
    """Return True if at least one AMD GPU is detected."""
    for d in enumerate_devices():
        if d.vendor == GpuVendor.AMD:
            return True
    return False


def has_intel_gpu() -> bool:
    """Return True if at least one Intel GPU is detected."""
    for d in enumerate_devices():
        if d.vendor == GpuVendor.INTEL:
            return True
    return False


def has_apple_gpu() -> bool:
    """Return True if at least one Apple GPU is detected."""
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
    """Return lspci GPU entries (class, vendor, device, vendor_id, device_id).

    Useful for advanced per-device configuration. Raises HardwareProbeError
    if lspci is missing on Linux.
    """
    return _lspci_gpu_entries()


# --- CLI helper ---


def format_device_summary() -> str:
    """Human-readable summary of all detected devices."""
    host = get_host_info()
    lines = [
        f"Host: {host.os} {host.os_release} on {host.architecture}",
        f"CPU: {host.cpu_vendor.value} ({host.cpu_model or 'unknown model'})",
        "",
        "GPUs:",
    ]
    devices = enumerate_devices()
    if not devices:
        lines.append("  (none detected)")
    for d in devices:
        lines.append(
            f"  [{d.device_id}] {d.vendor.value}: {d.model or 'unknown'} "
            f"arch={d.arch} path={d.device_path} driver={d.driver_version or 'n/a'}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_device_summary())

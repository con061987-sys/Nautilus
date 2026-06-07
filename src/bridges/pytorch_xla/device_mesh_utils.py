"""Utilities for inferring TVM target strings from device meshes.

Used by the GSPMD runner to choose a real GPU target for TVM
MetaSchedule tuning, instead of always falling back to
``Target("llvm")`` (which compiles for CPU and discards
GPU-specific scheduling decisions).

The mapping follows the ``nvidia/<arch>`` / ``rocm/<gfx>`` /
``intel/<target>`` convention used by TVM's upstream target registry.

Public API:
    infer_target_from_mesh(mesh) -> str
"""

from __future__ import annotations

from typing import Any

# (vendor, arch) -> TVM target string.
# The (vendor, arch) keys match the values produced by
# ``MeshDevice.vendor`` (``DeviceVendor``) and ``MeshDevice.arch``
# (the GPU arch string from lspci / nvidia-smi / rocm-smi).
_VENDOR_ARCH_TO_TVM: dict[tuple[str, str], str] = {
    # NVIDIA Hopper / Blackwell
    ("nvidia", "sm_90"): "nvidia/nvidia-h100",
    ("nvidia", "sm_100"): "nvidia/nvidia-b100",
    ("nvidia", "sm_120"): "nvidia/nvidia-b200",
    # NVIDIA Ampere
    ("nvidia", "sm_80"): "nvidia/nvidia-a100",
    ("nvidia", "sm_86"): "nvidia/nvidia-a100",
    # NVIDIA Turing / consumer
    ("nvidia", "sm_75"): "nvidia/nvidia-turing",
    ("nvidia", "sm_89"): "nvidia/nvidia-rtx-4090",
    # AMD CDNA
    ("amd", "gfx942"): "rocm/gfx942",
    ("amd", "gfx950"): "rocm/gfx950",
    ("amd", "gfx90a"): "rocm/gfx90a",
    ("amd", "gfx908"): "rocm/gfx908",
    # AMD RDNA
    ("amd", "gfx1100"): "rocm/gfx1100",
    # Intel
    ("intel", "intel_gaudi2"): "intel/gaudi-2",
    ("intel", "intel_gaudi3"): "intel/gaudi-3",
    ("intel", "intel_gpu_xehpc"): "intel/intel-xe-hpc",
    # Apple
    ("apple", "apple_m1"): "metal",
    ("apple", "apple_m2"): "metal",
    ("apple", "apple_m3"): "metal",
    ("apple", "apple_m4"): "metal",
}

# Vendor-only fallback when the (vendor, arch) pair is not in the table.
_VENDOR_FALLBACK: dict[str, str] = {
    "nvidia": "nvidia/nvidia-h100",
    "amd": "rocm/gfx942",
    "intel": "intel/gaudi-2",
    "apple": "metal",
    "cpu": "llvm",
}

# CPU fallback (no vendor info, or unknown vendor).
_CPU_FALLBACK = "llvm"


def infer_target_from_mesh(mesh: Any) -> str:
    """Infer a TVM target string from a device mesh or mesh-shape list.

    Accepts:
        * ``DeviceMesh`` (with ``devices`` list of ``MeshDevice``) — uses
          the first device's vendor + arch to pick a real GPU target.
        * ``list[int]`` / ``tuple[int, ...]`` (mesh_shape only) — no vendor
          information is available, so the function falls back to the
          LLVM (CPU) target. Callers should pass the full device mesh
          when possible to get a real GPU target.
        * anything else — falls back to the LLVM target.

    Returns:
        A TVM target string, e.g. ``"nvidia/nvidia-h100"``,
        ``"rocm/gfx942"``, ``"intel/gaudi-2"``, or ``"llvm"``.
    """
    # mesh_shape list: no vendor info -> CPU fallback.
    if isinstance(mesh, (list, tuple)):
        if not mesh or all(isinstance(x, int) for x in mesh):
            return _CPU_FALLBACK

    # DeviceMesh: use first device's vendor + arch.
    devices = getattr(mesh, "devices", None)
    if devices:
        first = devices[0]
        vendor = getattr(first, "vendor", None)
        arch = getattr(first, "arch", None)
        if vendor is not None:
            vendor_str = vendor.value if hasattr(vendor, "value") else str(vendor)
            if arch is not None:
                key = (str(vendor_str), str(arch))
                if key in _VENDOR_ARCH_TO_TVM:
                    return _VENDOR_ARCH_TO_TVM[key]
            if vendor_str in _VENDOR_FALLBACK:
                return _VENDOR_FALLBACK[vendor_str]

    return _CPU_FALLBACK


__all__ = ["infer_target_from_mesh"]

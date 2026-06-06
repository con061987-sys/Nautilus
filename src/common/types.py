"""
Vendor-neutral type definitions shared across all bridges.

These types are the *contracts* between bridges. The previous design
had bridges passing untyped `Any` around; this module is the
restoration of a real type system.
"""

from __future__ import annotations

import functools
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from src.common.errors import (
    ConfigError,
    DependencyMissingError,
    NautilusError,
)

# --- Re-export Result from result module ---
from src.common.result import Err, Ok, Result  # noqa: E402

# --- Vendors and architectures ---


class Vendor(str, Enum):
    """Hardware vendor. Single source of truth for all bridges."""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, s: str, strict: bool = True) -> "Vendor":
        """Parse a vendor name into a Vendor.

        Args:
            s: Input string. Case-insensitive.
            strict: If True, raise ConfigError on unknown input.
                If False, return Vendor.UNKNOWN (legacy behavior).

        Defaults to strict=True so that typos surface at config-load
        time rather than as silent Vendor.UNKNOWN at runtime.
        """
        try:
            return cls(s.lower())
        except ValueError:
            if strict:
                raise ConfigError(
                    f"Unknown vendor: {s!r}",
                    context={"input": s, "valid": [v.value for v in cls]},
                ) from None
            return cls.UNKNOWN


class Arch(str, Enum):
    """GPU architecture. Vendor-agnostic identifier (e.g. sm_90, gfx942, xe_hpg)."""
    # NVIDIA
    SM_70 = "sm_70"      # V100
    SM_75 = "sm_75"      # Turing
    SM_80 = "sm_80"      # A100
    SM_86 = "sm_86"      # A100
    SM_89 = "sm_89"      # RTX 4090
    SM_90 = "sm_90"      # H100 Hopper
    SM_100 = "sm_100"    # B100 Blackwell
    SM_120 = "sm_120"    # B200 Blackwell
    # AMD
    GFX900 = "gfx900"    # MI50
    GFX906 = "gfx906"    # MI60
    GFX908 = "gfx908"    # MI100
    GFX90A = "gfx90a"    # MI200 / MI250
    GFX942 = "gfx942"    # MI300X
    GFX950 = "gfx950"    # MI325X
    # Intel
    XE = "intel_gpu_xe"
    XE_LP = "intel_gpu_xelp"
    XE_HPG = "intel_gpu_xehpg"  # Arc
    XE_HPC = "intel_gpu_xehpc"  # Ponte Vecchio
    XE2 = "intel_gpu_xe2"        # Lunar Lake / Battlemage
    GAUDI2 = "intel_gaudi2"
    GAUDI3 = "intel_gaudi3"
    # Apple
    APPLE_M1 = "apple_m1"
    APPLE_M2 = "apple_m2"
    APPLE_M3 = "apple_m3"
    APPLE_M4 = "apple_m4"
    # Generic
    GENERIC = "generic"

    @property
    def vendor(self) -> Vendor:
        v = self.value
        if v.startswith("sm_"):
            return Vendor.NVIDIA
        if v.startswith("gfx"):
            return Vendor.AMD
        if v.startswith("intel") or v.startswith("xe") or v.startswith("gaudi"):
            return Vendor.INTEL
        if v.startswith("apple"):
            return Vendor.APPLE
        return Vendor.UNKNOWN


# --- Targets ---


# (vendor, arch) → TVM target alias. Entries here override the
# default "<prefix>/<arch>" rule in HardwareTarget.to_tvm_target.
TVM_TARGET_ALIASES: dict[tuple[Vendor, Arch], str] = {
    (Vendor.NVIDIA, Arch.SM_90): "nvidia/nvidia-h100",
    (Vendor.NVIDIA, Arch.SM_80): "nvidia/nvidia-a100",
    (Vendor.AMD, Arch.GFX942): "rocm/gfx942",
    (Vendor.INTEL, Arch.GAUDI2): "intel/gaudi-2",
}

_TVM_VENDOR_PREFIX: dict[Vendor, str] = {
    Vendor.NVIDIA: "nvidia",
    Vendor.AMD: "rocm",
    Vendor.INTEL: "intel",
}


@dataclass(frozen=True)
class HardwareTarget:
    """A (vendor, arch) pair, possibly with an alias for downstream tools."""
    vendor: Vendor
    arch: Arch
    alias: str = ""  # e.g. "nvidia/nvidia-h100" for TVM

    def to_tvm_target(self) -> str:
        if self.alias:
            return self.alias
        aliased = TVM_TARGET_ALIASES.get((self.vendor, self.arch))
        if aliased is not None:
            return aliased
        prefix = _TVM_VENDOR_PREFIX.get(self.vendor)
        if prefix is None:
            return "cuda"
        return f"{prefix}/{self.arch.value}"

    def to_triton_target(self) -> str:
        if self.vendor == Vendor.NVIDIA:
            return "cuda"
        if self.vendor == Vendor.AMD:
            return "rocm"
        if self.vendor == Vendor.INTEL:
            return "xpu"
        if self.vendor == Vendor.APPLE:
            return "metal"
        return "cuda"


# --- Fat binary ---


class SectionFormat(str, Enum):
    """Binary format of a kernel section inside a fat binary."""
    PTX = "ptx"
    CUBIN = "cubin"
    HSACO = "hsaco"
    SPV = "spv"
    METALLIB = "metallib"
    METAL_AIR = "metal_air"
    STUB = "stub"


@dataclass(frozen=True)
class KernelSection:
    """One vendor's compiled kernel inside a fat binary.

    `data` may be text (PTX) or binary (HSACO/SPV/CUBIN). Format
    determines interpretation.
    """
    vendor: Vendor
    arch: Arch
    format: SectionFormat
    data: bytes
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.data)

    @functools.cached_property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass
class FatBinary:
    """Complete fat binary with one KernelSection per compiled vendor."""
    kernel_name: str
    sections: list[KernelSection] = field(default_factory=list)
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_size(self) -> int:
        return sum(s.size for s in self.sections)

    @property
    def vendors(self) -> list[Vendor]:
        seen: set[Vendor] = set()
        out: list[Vendor] = []
        for s in self.sections:
            if s.vendor not in seen:
                seen.add(s.vendor)
                out.append(s.vendor)
        return out

    def get_section(self, vendor: Vendor) -> KernelSection | None:
        for s in self.sections:
            if s.vendor == vendor:
                return s
        return None

    def add_section(self, section: KernelSection) -> None:
        """Add or replace a section. Dedupes by (vendor, arch)."""
        self.sections = [
            s for s in self.sections
            if not (s.vendor == section.vendor and s.arch == section.arch)
        ]
        self.sections.append(section)

    def to_bytes(self) -> bytes:
        """Serialize the fat binary to a compact binary format.

        Format:
          [magic: 4 bytes "NFAT"]
          [version: 1 byte = 1]
          [kernel_name_len: 2 bytes LE]
          [kernel_name: utf-8]
          [num_sections: 2 bytes LE]
          For each section:
            [vendor_len: 1 byte][vendor: utf-8]
            [arch_len: 1 byte][arch: utf-8]
            [format: 1 byte index into SectionFormat]
            [data_len: 4 bytes LE]
            [data: bytes]
        """
        import struct
        out = bytearray()
        out += b"NFAT"
        out += struct.pack("<B", 1)
        name_bytes = self.kernel_name.encode("utf-8")
        out += struct.pack("<H", len(name_bytes))
        out += name_bytes
        out += struct.pack("<H", len(self.sections))
        format_list = list(SectionFormat)
        for section in self.sections:
            vendor_bytes = section.vendor.value.encode("utf-8")
            arch_bytes = section.arch.value.encode("utf-8")
            out += struct.pack("<B", len(vendor_bytes))
            out += vendor_bytes
            out += struct.pack("<B", len(arch_bytes))
            out += arch_bytes
            try:
                fmt_idx = format_list.index(section.format)
            except ValueError:
                fmt_idx = 0
            out += struct.pack("<B", fmt_idx)
            out += struct.pack("<I", len(section.data))
            out += section.data
        return bytes(out)

    @classmethod
    def from_bytes(cls, blob: bytes) -> "FatBinary":
        """Deserialize from the compact binary format produced by `to_bytes`."""
        import struct
        if blob[:4] != b"NFAT":
            raise ValueError(f"Invalid fat binary magic: {blob[:4]!r}")
        version = blob[4]
        if version != 1:
            raise ValueError(f"Unsupported fat binary version: {version}")
        offset = 5
        name_len = struct.unpack_from("<H", blob, offset)[0]
        offset += 2
        kernel_name = blob[offset:offset + name_len].decode("utf-8")
        offset += name_len
        num_sections = struct.unpack_from("<H", blob, offset)[0]
        offset += 2
        format_list = list(SectionFormat)
        sections: list[KernelSection] = []
        for _ in range(num_sections):
            vendor_len = blob[offset]
            offset += 1
            vendor = Vendor(blob[offset:offset + vendor_len].decode("utf-8"))
            offset += vendor_len
            arch_len = blob[offset]
            offset += 1
            arch = Arch(blob[offset:offset + arch_len].decode("utf-8"))
            offset += arch_len
            fmt_idx = blob[offset]
            offset += 1
            data_len = struct.unpack_from("<I", blob, offset)[0]
            offset += 4
            data = blob[offset:offset + data_len]
            offset += data_len
            sections.append(KernelSection(
                vendor=vendor,
                arch=arch,
                format=format_list[fmt_idx] if fmt_idx < len(format_list) else SectionFormat.STUB,
                data=data,
            ))
        return cls(kernel_name=kernel_name, sections=sections)


# --- Kernels ---


@dataclass(frozen=True)
class KernelHandle:
    """A handle to a compiled kernel in some vendor's binary.

    Used by the runtime loader to dispatch a fat binary to a specific
    device backend.
    """
    kernel_name: str
    vendor: Vendor
    arch: Arch
    binary_sha256: str
    entry_point: str = "nautilus_kernel_default"

    def vendor_section_name(self) -> str:
        return f".{self.vendor.value}_kernel"

    def dispatch_symbol(self) -> str:
        return f"nautilus_kernel_{self.vendor.value}"


# --- Tuning ---


@dataclass
class TuningConfig:
    """Block configuration for a matmul-like kernel. Vendor-neutral."""
    block_m: int = 128
    block_n: int = 128
    block_k: int = 32
    num_warps: int = 8
    num_stages: int = 3
    num_ctas: int = 1
    vendor_overrides: dict[Vendor, dict[str, int]] = field(default_factory=dict)

    def to_triton_config(self) -> "object":
        """Lazily build a triton.Config without importing triton at module load."""
        return _build_triton_config(
            self.block_m,
            self.block_n,
            self.block_k,
            self.num_warps,
            self.num_stages,
            self.num_ctas,
        )

    def overrides_for(self, vendor: Vendor) -> dict[str, int]:
        return dict(self.vendor_overrides.get(vendor, {}))

    @classmethod
    def defaults(cls) -> "TuningConfig":
        return cls()


@functools.lru_cache(maxsize=128)
def _build_triton_config(
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    num_ctas: int,
) -> "object":
    try:
        import triton
    except ImportError as exc:
        raise DependencyMissingError(
            "Triton is not installed; cannot build triton.Config",
        ) from exc
    return triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
        },
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )


# --- Mesh / sharding ---


@dataclass(frozen=True)
class MeshShape:
    """Logical device mesh, e.g. (2, 4) for an 8-device cluster."""
    axes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.axes:
            raise ConfigError(
                "MeshShape must have at least one axis",
                context={"axes": list(self.axes)},
            )
        if any(a <= 0 for a in self.axes):
            raise ConfigError(
                f"MeshShape axes must be positive, got {self.axes}",
                context={"axes": list(self.axes)},
            )

    @property
    def total_devices(self) -> int:
        n = 1
        for a in self.axes:
            n *= a
        return n

    def __iter__(self):
        return iter(self.axes)

    def __len__(self) -> int:
        return len(self.axes)


@dataclass(frozen=True)
class TensorShardingLite:
    """How a single tensor is partitioned across the mesh."""
    tensor_name: str
    mesh_axes: tuple[int, ...]
    partition_shape: tuple[int, ...]
    replicate_on_other_axes: bool = True

    def __post_init__(self) -> None:
        if not self.tensor_name:
            raise ConfigError("TensorShardingLite.tensor_name is required")
        if len(self.mesh_axes) != len(self.partition_shape):
            raise ConfigError(
                f"mesh_axes and partition_shape must have same length "
                f"(got {len(self.mesh_axes)} vs {len(self.partition_shape)})",
                context={
                    "tensor": self.tensor_name,
                    "axes": list(self.mesh_axes),
                    "shape": list(self.partition_shape),
                },
            )


@dataclass(frozen=True)
class ShardingSpecLite:
    """Vendor-neutral sharding spec. Output of GSPMD-like analysis."""
    mesh: MeshShape
    tensor_shardings: dict[str, TensorShardingLite]
    inserted_collectives: tuple[dict, ...] = ()
    estimated_comm_volume_bytes: int = 0
    estimated_compute_time_s: float = 0.0

    def __post_init__(self) -> None:
        for name, sharding in self.tensor_shardings.items():
            for axis in sharding.mesh_axes:
                if axis < 0 or axis >= len(self.mesh.axes):
                    raise ConfigError(
                        f"Tensor {name!r} references mesh axis {axis} "
                        f"but mesh has only {len(self.mesh.axes)} axes",
                        context={"mesh": list(self.mesh.axes)},
                    )

    @property
    def cache_key(self) -> str:
        import hashlib
        import json
        payload = {
            "mesh": list(self.mesh.axes),
            "tensors": {
                n: {
                    "axes": list(s.mesh_axes),
                    "shape": list(s.partition_shape),
                    "replicate": s.replicate_on_other_axes,
                }
                for n, s in sorted(self.tensor_shardings.items())
            },
            "collectives": list(self.inserted_collectives),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()


# --- IR / StableHLO ---


@dataclass
class IRModule:
    """Vendor-neutral intermediate representation module.

    The `text` field is the textual IR (MLIR, TIR, TTGIR, etc.) and
    `dialect` identifies which. Production code should normalize
    through the MLIR Vector dialect before this is filled.
    """
    text: str
    dialect: str
    source: str = ""  # Origin (e.g. "triton-capture", "tvmscript-emit")
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StableHLOModule:
    """A StableHLO module ready for GSPMD sharding."""
    mlir_text: str
    function_name: str
    input_specs: list[dict[str, Any]] = field(default_factory=list)
    output_specs: list[dict[str, Any]] = field(default_factory=list)
    op_count: int = 0
    conversion_time_ms: float = 0.0
    export_method: str = "torch_xla"
    is_usable: bool = True
    is_real_stablehlo: bool = False  # False = fallback (not real StableHLO)

    def __post_init__(self) -> None:
        if self.is_usable and not self.is_real_stablehlo:
            # Soft-warn: fallback is allowed in tests but should be loud in prod
            from src.common.logging import get_logger
            get_logger(__name__).warning(
                "StableHLOModule is_usable=True but is_real_stablehlo=False; "
                "this is a fallback representation, not actual StableHLO. "
                "GSPMD will not produce real sharding."
            )


# --- Logging / spans ---


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SourceLocation:
    """File:line:col for diagnostics."""
    file: str
    line: int
    col: int = 0

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


@dataclass
class StageRecord:
    """A single stage within a span."""
    name: str
    start_ms: float
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class SpanRecord:
    """A top-level span wrapping a multi-stage operation."""
    span_id: str
    operation: str  # e.g. "tune_kernel", "build_fat_binary"
    start_ms: float
    end_ms: float = 0.0
    duration_ms: float = 0.0
    stages: list[StageRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    parent_span_id: str | None = None

    def add_stage(self, stage: StageRecord) -> None:
        self.stages.append(stage)

    def finish(self, error: str | None = None) -> None:
        self.end_ms = time.perf_counter() * 1000
        self.duration_ms = self.end_ms - self.start_ms
        if error is not None:
            self.error = error


__all__ = [
    # Result
    "Result", "Ok", "Err",
    # Vendors / archs
    "Vendor", "Arch", "HardwareTarget",
    # Fat binary
    "SectionFormat", "KernelSection", "FatBinary",
    "KernelHandle",
    # Tuning
    "TuningConfig",
    # Sharding
    "MeshShape", "TensorShardingLite", "ShardingSpecLite",
    # IR
    "IRModule", "StableHLOModule",
    # Logging
    "LogLevel", "SourceLocation", "StageRecord", "SpanRecord",
]

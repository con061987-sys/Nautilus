"""Fat binary format — the in-memory representation of a linked fat binary.

Provides:
  - KernelSection: represents one vendor's kernel binary within a fat binary
  - FatBinary: the complete fat binary with multiple sections
  - Serialization/deserialization for persistence

The fat binary format is designed to be:
  1. Parseable without external tools (for runtime loading)
  2. Linkable into executables (when lld is available)
  3. Cacheable on disk (for persistent storage)
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SectionFormat(Enum):
    """Binary format of a kernel section.

    The enum value is the on-the-wire string used in
    ``.nautilus.index`` records and JSON serialisations. New entries
    must be **appended** — :meth:`FatBinary.to_bytes` encodes the
    format as the index in the enum, so reordering breaks previously
    linked fat binaries.
    """

    PTX = "ptx"  # Nvidia PTX text
    CUBIN = "cubin"  # Nvidia cubin
    HSACO = "hsaco"  # AMD HSA Code Object
    SPV = "spv"  # Intel SPIR-V
    METALLIB = "metallib"  # Apple Metal library (MTLB magic)
    AIR = "air"  # Apple Intermediate Representation (AIRI magic)
    STUB = "stub"  # C runtime stub object


@dataclass
class KernelSection:
    """One vendor's kernel binary within a fat binary."""

    vendor: str  # "nvidia" / "amd" / "intel"
    arch: str  # "sm_90" / "gfx942" / "intel_gpu_xehpg"
    format: SectionFormat
    data: bytes
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "vendor": self.vendor,
            "arch": self.arch,
            "format": self.format.value,
            "size": self.size,
            "sha256": self.sha256,
            "metadata": self.metadata,
        }


@dataclass
class FatBinary:
    """A complete fat binary containing kernel sections for multiple vendors."""

    kernel_name: str
    sections: list[KernelSection] = field(default_factory=list)
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_size(self) -> int:
        return sum(s.size for s in self.sections)

    @property
    def vendors(self) -> list[str]:
        return list({s.vendor for s in self.sections})

    def get_section(self, vendor: str) -> KernelSection | None:
        for section in self.sections:
            if section.vendor == vendor:
                return section
        return None

    def add_section(self, section: KernelSection) -> None:
        # Replace existing section for the same vendor+arch
        self.sections = [
            s for s in self.sections if not (s.vendor == section.vendor and s.arch == section.arch)
        ]
        self.sections.append(section)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "created_at": self.created_at,
            "total_size": self.total_size,
            "vendors": self.vendors,
            "sections": [s.to_dict() for s in self.sections],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize to JSON (sections as base64)."""
        data = self.to_dict()
        for section in data["sections"]:
            section["data"] = self._section_data_for(section).hex()
        return json.dumps(data, indent=2)

    def _section_data_for(self, section_dict: dict[str, Any]) -> bytes:
        for s in self.sections:
            if s.to_dict() == section_dict:
                return s.data
        return b""

    @classmethod
    def from_json(cls, json_text: str) -> FatBinary:
        """Deserialize from JSON."""
        data = json.loads(json_text)
        sections = []
        for s_data in data.get("sections", []):
            section = KernelSection(
                vendor=s_data["vendor"],
                arch=s_data["arch"],
                format=SectionFormat(s_data["format"]),
                data=bytes.fromhex(s_data["data"]),
                metadata=s_data.get("metadata", {}),
            )
            sections.append(section)
        return cls(
            kernel_name=data["kernel_name"],
            sections=sections,
            created_at=data.get("created_at", ""),
            metadata=data.get("metadata", {}),
        )

    def to_bytes(self) -> bytes:
        """Serialize to a compact binary format for on-disk storage.

        Format:
          [magic: 4 bytes "NFAT"]
          [version: 1 byte]
          [kernel_name_len: 2 bytes]
          [kernel_name: utf-8]
          [num_sections: 2 bytes]
          For each section:
            [vendor_len: 1 byte][vendor: utf-8]
            [arch_len: 1 byte][arch: utf-8]
            [format: 1 byte]
            [data_len: 4 bytes]
            [data: bytes]
        """
        out = bytearray()
        out += b"NFAT"
        out += struct.pack("<B", 1)  # version
        name_bytes = self.kernel_name.encode("utf-8")
        out += struct.pack("<H", len(name_bytes))
        out += name_bytes
        out += struct.pack("<H", len(self.sections))
        for section in self.sections:
            vendor_bytes = section.vendor.encode("utf-8")
            arch_bytes = section.arch.encode("utf-8")
            out += struct.pack("<B", len(vendor_bytes))
            out += vendor_bytes
            out += struct.pack("<B", len(arch_bytes))
            out += arch_bytes
            out += struct.pack("<B", list(SectionFormat).index(section.format))
            out += struct.pack("<I", len(section.data))
            out += section.data
        return bytes(out)

    @classmethod
    def from_bytes(cls, data: bytes) -> FatBinary:
        """Deserialize from the compact binary format."""
        if data[:4] != b"NFAT":
            raise ValueError(f"Invalid fat binary magic: {data[:4]!r}")
        version = data[4]
        if version != 1:
            raise ValueError(f"Unsupported fat binary version: {version}")
        offset = 5
        name_len = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        kernel_name = data[offset : offset + name_len].decode("utf-8")
        offset += name_len
        num_sections = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        sections: list[KernelSection] = []
        for _ in range(num_sections):
            vendor_len = data[offset]
            offset += 1
            vendor = data[offset : offset + vendor_len].decode("utf-8")
            offset += vendor_len
            arch_len = data[offset]
            offset += 1
            arch = data[offset : offset + arch_len].decode("utf-8")
            offset += arch_len
            format_idx = data[offset]
            offset += 1
            data_len = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            section_data = data[offset : offset + data_len]
            offset += data_len
            sections.append(
                KernelSection(
                    vendor=vendor,
                    arch=arch,
                    format=list(SectionFormat)[format_idx],
                    data=section_data,
                )
            )
        return cls(kernel_name=kernel_name, sections=sections)

    def save(self, path: Path) -> None:
        """Save the fat binary to a file in the compact binary format."""
        path.write_bytes(self.to_bytes())

    @classmethod
    def load(cls, path: Path) -> FatBinary:
        """Load a fat binary from a file."""
        return cls.from_bytes(path.read_bytes())

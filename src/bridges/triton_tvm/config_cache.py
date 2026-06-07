"""Persistent disk cache for auto-tuning results.

Cache entries are keyed by ``sha256(kernel_ir_hash + vendor + arch)``
and stored as JSON files under ``~/.cache/nautilus/tuning/``.  A kernel
source change produces a different IR hash → automatic cache miss
without explicit invalidation.

Typical usage::

    cache = ConfigCache()
    cached = cache.get(metadata.cache_key, "nvidia", "sm_90")
    if cached is not None:
        config = MappedTuningConfig(**cached)
    else:
        config = run_tuning(...)
        cache.set(metadata.cache_key, "nvidia", "sm_90", config.__dict__)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

CACHE_DIR = Path.home() / ".cache" / "nautilus" / "tuning"

__all__ = ["CACHE_DIR", "ConfigCache"]


class ConfigCache:
    """Persistent, content-addressable cache for auto-tuning results.

    The cache is a directory of JSON files, one per (kernel_ir, vendor,
    arch) triple.  The filename is a hash of the triple so that:

    * Different kernel source → different IR hash → different key →
      automatic cache miss (no manual invalidation needed when a kernel
      is edited).

    * Different target hardware → different vendor/arch → different
      key → the same kernel tuned for two GPUs lives in two entries.

    * Same everything → cache hit → no re-tuning.

    Writes are atomic (write to tempfile, rename) to prevent corruption
    from concurrent processes or power loss.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        """Create/reuse the cache directory.

        Args:
            cache_dir: Directory for cache files.  Created if missing.
                Defaults to ``~/.cache/nautilus/tuning/``.
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, kernel_ir_hash: str, vendor: str, arch: str) -> dict[str, Any] | None:
        """Look up a cached tuning config.

        Args:
            kernel_ir_hash: Hash of the kernel IR text (any stable hash
                string; typically ``KernelMetadata.cache_key`` or
                ``CapturedKernelIR.source_hash``).
            vendor: Hardware vendor, e.g. ``"nvidia"``, ``"amd"``.
            arch: Architecture identifier, e.g. ``"sm_90"``, ``"gfx942"``.

        Returns:
            Deserialized JSON dict (the ``MappedTuningConfig`` fields)
            or ``None`` on cache miss or corruption.
        """
        key = self._make_key(kernel_ir_hash, vendor, arch)
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return self._read_json(path)
        except (json.JSONDecodeError, OSError):
            path.unlink(missing_ok=True)
            return None

    def set(self, kernel_ir_hash: str, vendor: str, arch: str, config: dict[str, Any]) -> None:
        """Store a tuning config in the cache.

        Args:
            kernel_ir_hash: Hash of the kernel IR text.
            vendor: Hardware vendor string.
            arch: Architecture identifier.
            config: Serializable dict of tuning parameters (typically
                a ``MappedTuningConfig`` converted via ``__dict__`` or
                ``dataclasses.asdict()``).
        """
        key = self._make_key(kernel_ir_hash, vendor, arch)
        path = self.cache_dir / f"{key}.json"
        self._write_json(path, config)

    def invalidate(self, kernel_ir_hash: str, vendor: str, arch: str) -> None:
        """Remove a single cache entry, if it exists.

        Args:
            kernel_ir_hash: Hash of the kernel IR text.
            vendor: Hardware vendor string.
            arch: Architecture identifier.
        """
        key = self._make_key(kernel_ir_hash, vendor, arch)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            path.unlink()

    def clear_all(self) -> int:
        """Remove **all** cache entries.

        Returns:
            Number of files removed.
        """
        removed = 0
        for p in self.cache_dir.glob("*.json"):
            p.unlink()
            removed += 1
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(kernel_ir_hash: str, vendor: str, arch: str) -> str:
        """Deterministic 32-char hex key.

        Format: ``sha256(kernel_ir_hash + vendor + arch)[:32]``.
        """
        h = hashlib.sha256()
        h.update(kernel_ir_hash.encode("utf-8"))
        h.update(vendor.encode("utf-8"))
        h.update(arch.encode("utf-8"))
        return h.hexdigest()[:32]

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        """Read and parse a JSON cache file."""
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw)

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        """Atomically write JSON data to *path* via tempfile + rename."""
        fd, tmp_name = tempfile.mkstemp(
            suffix=".tmp",
            prefix=f"{path.stem}_",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp_name, str(path))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise

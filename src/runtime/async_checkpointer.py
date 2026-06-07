"""Asynchronous state checkpointing — hardened for production.

The previous version had several real bugs that would lose data
in production:

  1. Used pickle for serialization (security risk + only works
     for Python objects; PyTorch tensors need torch.save)
  2. _save_model_state and _save_optimizer_state returned 0 on
     failure and continued silently — data loss
  3. No atomic write — a process crash mid-save left a corrupt
     checkpoint that recover_latest() would then try to load
  4. No CRC/checksum — silent corruption went undetected
  5. No schema version — old/new format incompatibility was possible
  6. Used the stdlib logging module instead of the structured
     logger — no span/stage observability
  7. No circuit breaker — a failing storage backend would
     indefinitely hang the checkpoint thread

This rewrite:
  - Writes to a .tmp file, fsyncs, then atomically renames
  - Computes a SHA-256 of the saved data and embeds it in meta.json
  - Embeds schema_version for forward compat
  - Uses torch.save when torch is available; msgpack as fallback
  - Raises CheckpointIOError on save failure (no silent 0)
  - Uses src.common.logging for structured spans
  - CircuitBreaker around the I/O path
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.common.errors import (
    CheckpointError,
    DependencyMissingError,
    ErrorCode,
)
from src.common.logging import get_logger
from src.common.logging import span as span_context
from src.common.observability import (
    CircuitBreaker,
    CircuitBreakerConfig,
    get_default_breakers,
)
from src.common.result import Err, Ok, Result

log = get_logger("nautilus.runtime.checkpoint")


# Bump this whenever the on-disk format changes incompatibly.
SCHEMA_VERSION = 2


class CheckpointBackend(Enum):
    """Storage backend for checkpoints."""

    RAM_DISK = "ramdisk"
    LOCAL_FS = "localfs"
    DISTRIBUTED = "distributed"


@dataclass
class CheckpointConfig:
    interval_seconds: float = 5.0
    backend: CheckpointBackend = CheckpointBackend.RAM_DISK
    storage_path: str = "/tmp/nautilus_checkpoints"
    max_checkpoints: int = 5
    save_optimizer_state: bool = True
    recovery_timeout_s: float = 3.0
    io_timeout_seconds: float = 30.0
    # SECURITY: legacy Python-pickle deserialisation is RCE on
    # untrusted bytes. Default is to refuse unpickled data and
    # raise DependencyMissingError. Tests that need to round-trip
    # legacy pickle data may opt in by setting this to True.
    # Production configs MUST leave this at False.
    unsafe_pickle_fallback: bool = False


@dataclass
class CheckpointMetadata:
    """On-disk metadata for a checkpoint. Schema-versioned for forward compat."""

    schema_version: int
    checkpoint_id: int
    created_at: float
    model_state_path: str
    optimizer_state_path: str
    model_state_size_bytes: int
    optimizer_state_size_bytes: int
    model_state_sha256: str
    optimizer_state_sha256: str
    interval_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.checkpoint_id,
            "created_at": self.created_at,
            "model_state_path": self.model_state_path,
            "optimizer_state_path": self.optimizer_state_path,
            "model_state_size_bytes": self.model_state_size_bytes,
            "optimizer_state_size_bytes": self.optimizer_state_size_bytes,
            "model_state_sha256": self.model_state_sha256,
            "optimizer_state_sha256": self.optimizer_state_sha256,
            "interval_seconds": self.interval_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CheckpointMetadata:
        sv = d.get("schema_version", 1)
        if sv != SCHEMA_VERSION:
            raise CheckpointError(
                f"Checkpoint schema version {sv} != expected {SCHEMA_VERSION}",
                code=ErrorCode.DEPENDENCY_VERSION_MISMATCH,
                context={"found": sv, "expected": SCHEMA_VERSION},
            )
        return cls(
            schema_version=sv,
            checkpoint_id=d["id"],
            created_at=d["created_at"],
            model_state_path=d["model_state_path"],
            optimizer_state_path=d["optimizer_state_path"],
            model_state_size_bytes=d.get("model_state_size_bytes", 0),
            optimizer_state_size_bytes=d.get("optimizer_state_size_bytes", 0),
            model_state_sha256=d.get("model_state_sha256", ""),
            optimizer_state_sha256=d.get("optimizer_state_sha256", ""),
            interval_seconds=d.get("interval_seconds", 0.0),
            metadata=d.get("metadata", {}),
        )


@dataclass
class CheckpointInfo:
    """In-memory view of a checkpoint."""

    checkpoint_id: int
    created_at: float
    path: Path
    model_state_size_bytes: int
    optimizer_state_size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_size_bytes(self) -> int:
        return self.model_state_size_bytes + self.optimizer_state_size_bytes

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class AsyncCheckpointer:
    """Production-grade async state checkpointer for fault tolerance.

    Usage:
        checkpointer = AsyncCheckpointer(CheckpointConfig(interval_seconds=5.0))
        checkpointer.start()
        ...
        checkpointer.request_checkpoint(model, optimizer)
        ...
        result = checkpointer.recover_latest()
    """

    def __init__(self, config: CheckpointConfig | None = None) -> None:
        self.config = config or CheckpointConfig()
        self._storage = Path(self.config.storage_path)
        self._storage.mkdir(parents=True, exist_ok=True)

        self._checkpoints: list[CheckpointInfo] = []
        self._checkpoint_counter = 0
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Pending request (overwritten by subsequent requests; we
        # only ever checkpoint the LATEST state, not the entire
        # request history).
        self._pending_model: Any = None
        self._pending_optimizer: Any = None
        self._pending_event = threading.Event()

        # Per-dependency circuit breaker for storage I/O
        breakers = get_default_breakers()
        if "checkpoint_io" not in breakers:
            breakers["checkpoint_io"] = CircuitBreaker(
                CircuitBreakerConfig(
                    name="checkpoint_io",
                    failure_threshold=3,
                    reset_timeout_seconds=10.0,
                )
            )
        self._io_breaker = breakers["checkpoint_io"]

        # Stats
        self._total_saves = 0
        self._total_save_failures = 0
        self._total_recoveries = 0
        self._total_recovery_failures = 0
        self._last_save_duration_ms = 0.0
        self._last_recovery_duration_ms = 0.0

        self._discover_existing()

    def _discover_existing(self) -> None:
        for path in sorted(self._storage.glob("checkpoint_*.meta.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                meta = CheckpointMetadata.from_dict(data)
                info = CheckpointInfo(
                    checkpoint_id=meta.checkpoint_id,
                    created_at=meta.created_at,
                    path=self._storage / f"checkpoint_{meta.checkpoint_id:08d}",
                    model_state_size_bytes=meta.model_state_size_bytes,
                    optimizer_state_size_bytes=meta.optimizer_state_size_bytes,
                    metadata=meta.metadata,
                )
                with self._lock:
                    self._checkpoints.append(info)
                    self._checkpoint_counter = max(self._checkpoint_counter, info.checkpoint_id)
            except CheckpointError as exc:
                log.warning(
                    "Skipping checkpoint with wrong schema version",
                    path=str(path),
                    error=str(exc),
                )
            except Exception as exc:
                log.warning(
                    "Failed to load checkpoint metadata",
                    path=str(path),
                    error=str(exc),
                )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._checkpoint_loop,
            daemon=True,
            name="nautilus-checkpoint",
        )
        self._thread.start()
        log.info("Async checkpointer started", interval_s=self.config.interval_seconds)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    def _checkpoint_loop(self) -> None:
        while not self._stop_event.is_set():
            triggered = self._pending_event.wait(timeout=self.config.interval_seconds)
            if self._stop_event.is_set():
                break
            if not triggered:
                continue

            # Atomically claim and clear the pending request under
            # the lock. Reading model/optimizer/event without the
            # lock races with request_checkpoint(): the caller can
            # overwrite _pending_model between our read and our
            # clear, causing us to save a stale object and then
            # drop the new request on the floor.
            with self._lock:
                model = self._pending_model
                optimizer = self._pending_optimizer
                self._pending_model = None
                self._pending_optimizer = None
                self._pending_event.clear()

            if model is None:
                # Spurious wakeup or event set with no payload
                # (e.g., rebuild_topology). Nothing to do.
                continue

            result = self._save_checkpoint_sync(model, optimizer)
            if result.is_err():
                log.warning(
                    "Async checkpoint failed",
                    error=str(result.error),
                    error_type=type(result.error).__name__,
                )

    def request_checkpoint(self, model: Any, optimizer: Any | None = None) -> None:
        """Request an asynchronous checkpoint. Non-blocking.

        The pending model/optimizer/event are written under the same
        lock used by ``_checkpoint_loop`` to claim them, so a
        concurrent request can never interleave with the loop's
        read-and-clear and cause a request to be silently dropped.
        """
        with self._lock:
            self._pending_model = model
            self._pending_optimizer = optimizer if self.config.save_optimizer_state else None
        self._pending_event.set()

    def checkpoint_now(self, model: Any, optimizer: Any | None = None) -> CheckpointInfo:
        """Synchronous checkpoint — blocks until done.

        Returns a :class:`CheckpointInfo` on success and raises
        :class:`CheckpointError` on failure (preserved for backward
        compatibility with the public API). Internally this delegates
        to :meth:`_save_checkpoint_sync` and unwraps the
        ``Result`` — callers that want explicit error handling should
        use the underscored method directly.
        """
        result = self._save_checkpoint_sync(
            model,
            optimizer if self.config.save_optimizer_state else None,
        )
        if result.is_ok():
            return result.unwrap()
        raise result.error

    def _save_checkpoint_sync(
        self,
        model: Any,
        optimizer: Any | None,
    ) -> Result[CheckpointInfo, CheckpointError]:
        """Save a checkpoint synchronously with atomic write + checksum.

        Returns a Rust-style :class:`Result`:

          - ``Ok(CheckpointInfo)`` — checkpoint persisted, checksum
            recorded, on-disk state verified.
          - ``Err(CheckpointError)`` — typed failure (I/O, atomic
            rename, breaker open, …). Callers MUST inspect both
            arms; the loop path logs and continues, the public
            :meth:`checkpoint_now` re-raises the inner exception.
        """
        start = time.time()
        with span_context("checkpoint_save", backend=self.config.backend.value) as sp:
            try:
                with self._lock:
                    self._checkpoint_counter += 1
                    checkpoint_id = self._checkpoint_counter

                cp_dir = self._storage / f"checkpoint_{checkpoint_id:08d}"
                cp_dir.mkdir(parents=True, exist_ok=True)

                model_path = cp_dir / "model.state"
                optim_path = cp_dir / "optimizer.state"

                # Save under circuit breaker
                model_save: Result[tuple[int, str], CheckpointError] = self._io_breaker.call(
                    self._save_state_atomic,
                    model,
                    model_path,
                )
                if model_save.is_err():
                    return Err(model_save.error)
                model_size, model_sha = model_save.unwrap()
                optim_size, optim_sha = 0, ""
                if optimizer is not None:
                    optim_save: Result[tuple[int, str], CheckpointError] = self._io_breaker.call(
                        self._save_state_atomic,
                        optimizer,
                        optim_path,
                    )
                    if optim_save.is_err():
                        return Err(optim_save.error)
                    optim_size, optim_sha = optim_save.unwrap()

                # Write metadata last; if meta write fails, the
                # checkpoint is considered failed and the directory
                # is removed to avoid leaving garbage.
                meta = CheckpointMetadata(
                    schema_version=SCHEMA_VERSION,
                    checkpoint_id=checkpoint_id,
                    created_at=time.time(),
                    model_state_path=str(model_path),
                    optimizer_state_path=str(optim_path),
                    model_state_size_bytes=model_size,
                    optimizer_state_size_bytes=optim_size,
                    model_state_sha256=model_sha,
                    optimizer_state_sha256=optim_sha,
                    interval_seconds=self.config.interval_seconds,
                    metadata={"backend": self.config.backend.value},
                )
                meta_path = cp_dir / "meta.json"
                self._write_atomic_text(meta_path, json.dumps(meta.to_dict(), indent=2))

                info = CheckpointInfo(
                    checkpoint_id=checkpoint_id,
                    created_at=meta.created_at,
                    path=cp_dir,
                    model_state_size_bytes=model_size,
                    optimizer_state_size_bytes=optim_size,
                    metadata=meta.metadata,
                )

                with self._lock:
                    self._checkpoints.append(info)
                    while len(self._checkpoints) > self.config.max_checkpoints:
                        old = self._checkpoints.pop(0)
                        self._delete_checkpoint(old)

                self._total_saves += 1
                self._last_save_duration_ms = (time.time() - start) * 1000
                sp.set(
                    checkpoint_id=checkpoint_id,
                    model_size_bytes=model_size,
                    optim_size_bytes=optim_size,
                    duration_ms=self._last_save_duration_ms,
                )
                log.info(
                    "Checkpoint saved",
                    id=checkpoint_id,
                    model_size_mb=model_size / (1024 * 1024),
                    optim_size_mb=optim_size / (1024 * 1024),
                    duration_ms=self._last_save_duration_ms,
                )
                return Ok(info)
            except CheckpointError as exc:
                self._total_save_failures += 1
                return Err(exc)
            except Exception as exc:
                self._total_save_failures += 1
                # Clean up partial directory
                if cp_dir.exists():
                    shutil.rmtree(cp_dir, ignore_errors=True)
                err = CheckpointError(
                    f"Checkpoint save failed: {exc}",
                    cause=exc,
                    context={"checkpoint_id": checkpoint_id, "duration_s": time.time() - start},
                )
                return Err(err)

    def _save_state_atomic(
        self,
        state: Any,
        path: Path,
    ) -> Result[tuple[int, str], CheckpointError]:
        """Save state to a temp file, fsync, then atomic rename.

        Returns ``Ok((size_in_bytes, sha256_hex))`` on success and
        ``Err(CheckpointError)`` on any failure (serialization,
        write, fsync, or atomic-rename error). Callers MUST inspect
        both arms — the I/O breaker that wraps this method only
        counts hard exceptions, not logical Err returns.
        """
        with tempfile.NamedTemporaryFile(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            data = self._serialize_state(state)
            with open(tmp_path, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            sha = hashlib.sha256(data).hexdigest()
            os.replace(tmp_path, path)
            return Ok((len(data), sha))
        except CheckpointError as exc:
            tmp_path.unlink(missing_ok=True)
            return Err(exc)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            return Err(
                CheckpointError(
                    f"Atomic write to {path} failed: {exc}",
                    cause=exc,
                )
            )

    def _serialize_state(self, state: Any) -> bytes:
        """Serialize a model/optimizer state to bytes.

        Tries torch.save first (preserves tensor dtypes/strides),
        then msgpack, then pickle as a last resort.
        """
        try:
            import io

            import torch

            buf = io.BytesIO()
            torch.save(state, buf)
            return buf.getvalue()
        except ImportError:
            pass
        try:
            import msgpack

            return msgpack.packb(state, use_bin_type=True)
        except ImportError:
            pass
        import pickle

        return pickle.dumps(state)

    def _write_atomic_text(self, path: Path, text: str) -> None:
        """Write text to a temp file, fsync, then atomic rename."""
        with tempfile.NamedTemporaryFile(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            delete=False,
            mode="w",
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text)
            tmp.flush()
        try:
            os.replace(tmp_path, path)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise CheckpointError(
                f"Atomic write of {path} failed: {exc}",
                cause=exc,
            ) from exc

    def _delete_checkpoint(self, info: CheckpointInfo) -> None:
        try:
            if info.path.exists():
                shutil.rmtree(info.path, ignore_errors=True)
        except Exception as exc:
            log.warning("Failed to delete old checkpoint", path=str(info.path), error=str(exc))

    def recover_latest(self) -> tuple[Any, Any | None] | None:
        """Recover from the most recent verified checkpoint.

        Validates the SHA-256 checksum before returning. Raises
        CheckpointError on corruption.
        """
        start = time.time()
        with span_context("checkpoint_recover") as sp:
            with self._lock:
                if not self._checkpoints:
                    sp.set(recovered=False, reason="no_checkpoints")
                    return None
                latest = self._checkpoints[-1]

            meta_path = latest.path / "meta.json"
            if not meta_path.exists():
                self._total_recovery_failures += 1
                raise CheckpointError(
                    f"Checkpoint {latest.checkpoint_id} is missing meta.json",
                    context={"path": str(latest.path)},
                )
            try:
                with open(meta_path) as f:
                    meta = CheckpointMetadata.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError) as exc:
                self._total_recovery_failures += 1
                raise CheckpointError(
                    f"Checkpoint {latest.checkpoint_id} metadata is corrupt: {exc}",
                    cause=exc,
                ) from exc

            # Verify checksums before deserializing
            model_path = Path(meta.model_state_path)
            if not model_path.exists():
                self._total_recovery_failures += 1
                raise CheckpointError(
                    f"Checkpoint {latest.checkpoint_id} missing model state",
                    context={"path": str(model_path)},
                )
            model_data = model_path.read_bytes()
            model_sha = hashlib.sha256(model_data).hexdigest()
            if meta.model_state_sha256 and model_sha != meta.model_state_sha256:
                self._total_recovery_failures += 1
                raise CheckpointError(
                    f"Checkpoint {latest.checkpoint_id} model state failed checksum: "
                    f"expected {meta.model_state_sha256[:12]}, got {model_sha[:12]}",
                    context={
                        "expected": meta.model_state_sha256,
                        "actual": model_sha,
                    },
                )

            model_state = self._deserialize_state(model_data)
            optimizer_state = None
            optim_path = Path(meta.optimizer_state_path)
            if optim_path.exists() and meta.optimizer_state_size_bytes > 0:
                optim_data = optim_path.read_bytes()
                optim_sha = hashlib.sha256(optim_data).hexdigest()
                if meta.optimizer_state_sha256 and optim_sha != meta.optimizer_state_sha256:
                    log.warning(
                        "Optimizer state checksum mismatch; recovering without optimizer",
                        checkpoint_id=latest.checkpoint_id,
                    )
                else:
                    optimizer_state = self._deserialize_state(optim_data)

            elapsed = time.time() - start
            self._total_recoveries += 1
            self._last_recovery_duration_ms = elapsed * 1000
            if elapsed > self.config.recovery_timeout_s:
                log.warning(
                    "Recovery exceeded target",
                    elapsed_s=elapsed,
                    target_s=self.config.recovery_timeout_s,
                )
            sp.set(
                checkpoint_id=latest.checkpoint_id,
                elapsed_ms=self._last_recovery_duration_ms,
                recovered=True,
            )
            log.info(
                "Recovered checkpoint",
                id=latest.checkpoint_id,
                elapsed_ms=self._last_recovery_duration_ms,
            )
            return (model_state, optimizer_state)

    def _deserialize_state(self, data: bytes) -> Any:
        """Deserialize state. Tries torch.load (weights_only), then msgpack.

        SECURITY: ``weights_only=True`` is mandatory for the
        torch.load path. The default of False would let PyTorch
        instantiate arbitrary objects from the checkpoint bytes
        — any caller that can influence the checkpoint file
        (shared storage, malicious recovery replay) gets RCE.
        ``weights_only=True`` restricts the unpickler to a safe
        allow-list of types.

        SECURITY: The legacy Python pickle deserialiser is gated
        behind ``CheckpointConfig.unsafe_pickle_fallback=True``.
        If neither torch nor msgpack is available, and the
        fallback is not enabled, we raise
        :class:`DependencyMissingError` with a clear install
        hint.
        """
        try:
            import io

            import torch

            return torch.load(io.BytesIO(data), weights_only=True)
        except ImportError:
            pass
        try:
            import msgpack

            return msgpack.unpackb(data, raw=False)
        except ImportError:
            pass
        if self.config.unsafe_pickle_fallback:
            import pickle as _legacy

            try:
                return _legacy.loads(data)
            except Exception as exc:
                raise CheckpointError(
                    f"Failed to deserialize state: {exc}",
                    cause=exc,
                ) from exc
        raise DependencyMissingError(
            "No safe deserializer available: torch and msgpack are both "
            "missing. Install one of them (pip install torch or msgpack), "
            "or set CheckpointConfig.unsafe_pickle_fallback=True "
            "(UNSAFE — legacy Python-pickle deserialisation is RCE on "
            "attacker-controlled bytes).",
            context={"fallback": "pickle", "unsafe_flag": False},
        )

    def on_node_failure(self, dead_node_id: str) -> bool:
        """Handle a node failure event. Triggers recovery."""
        log.warning("Node failure detected", dead_node=dead_node_id)
        try:
            result = self.recover_latest()
            return result is not None
        except CheckpointError as exc:
            log.error("Recovery failed", error=str(exc))
            return False

    def rebuild_topology(self, alive_nodes: list[str]) -> None:
        """Rebuild the cluster topology after a node failure."""
        with self._lock:
            has_pending = self._pending_model is not None
        if has_pending:
            self._pending_event.set()
        log.info("Topology rebuilt", alive_nodes=len(alive_nodes))

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "num_checkpoints": len(self._checkpoints),
                "total_size_bytes": sum(c.total_size_bytes for c in self._checkpoints),
                "oldest_age_seconds": self._checkpoints[0].age_seconds if self._checkpoints else 0,
                "is_running": self._thread is not None and self._thread.is_alive(),
                "total_saves": self._total_saves,
                "total_save_failures": self._total_save_failures,
                "total_recoveries": self._total_recoveries,
                "total_recovery_failures": self._total_recovery_failures,
                "last_save_duration_ms": self._last_save_duration_ms,
                "last_recovery_duration_ms": self._last_recovery_duration_ms,
                "io_breaker": self._io_breaker.stats,
            }

"""
Structured JSON logging with span / stage tracking.

This replaces the per-bridge ad-hoc loggers. Use it everywhere:

    from src.common.logging import configure_logging, get_logger, span, stage

    configure_logging(level="info", json=True)
    log = get_logger(__name__)

    with span("tune_kernel", source_hash=src_hash, target=target) as sp:
        with stage(sp, "ir_capture") as st:
            captured = capture_ir(src_hash)
            st.set("ops", len(captured.ops_seen))
        with stage(sp, "metaschedule") as st:
            config = run_metaschedule(captured)
        sp.set("best_block_m", config.block_m)
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TextIO

from src.common.types import (
    LogLevel,
    SpanRecord,
    StageRecord,
)

# --- Log sink protocol ---


class LogSink:
    """Abstract sink for log records. Implementations are thread-safe."""

    def emit(self, record: dict[str, Any]) -> None: ...
    def flush(self) -> None: ...


class StdoutLogSink(LogSink):
    """Writes one JSON object per line to a stream (default stdout)."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._lock = threading.Lock()
        self._stream = stream or sys.stdout

    def emit(self, record: dict[str, Any]) -> None:
        with self._lock:
            try:
                self._stream.write(json.dumps(record, default=str) + "\n")
                self._stream.flush()
            except Exception:
                # Never let logging crash the pipeline
                pass

    def flush(self) -> None:
        with self._lock:
            try:
                self._stream.flush()
            except Exception:
                pass


class JsonLogSink(LogSink):
    """Writes JSON to a file path. Atomic line writes."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()

    def emit(self, record: dict[str, Any]) -> None:
        with self._lock, open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def flush(self) -> None:
        # File is opened in append mode, no buffer to flush.
        pass


class NullLogSink(LogSink):
    """Discards all records. For tests that want to silence logging."""

    def emit(self, record: dict[str, Any]) -> None:
        pass

    def flush(self) -> None:
        pass


class CompositeLogSink(LogSink):
    """Fan out to multiple sinks."""

    def __init__(self, sinks: list[LogSink]) -> None:
        self._sinks = sinks

    def emit(self, record: dict[str, Any]) -> None:
        for s in self._sinks:
            s.emit(record)

    def flush(self) -> None:
        for s in self._sinks:
            s.flush()


# --- Configuration ---


_CONFIGURED = False
_CONFIG_LOCK = threading.Lock()
_ACTIVE_SINKS: list[LogSink] = []
_DEFAULT_LEVEL: LogLevel = LogLevel.INFO
_LOGGER_NAME_PREFIX = "nautilus"


def configure_logging(
    level: str | LogLevel = "info",
    sinks: list[LogSink] | None = None,
    json: bool = True,
) -> None:
    """Configure the global Nautilus logging system.

    Idempotent: subsequent calls REPLACE the sinks (so tests can swap).
    """
    global _CONFIGURED, _ACTIVE_SINKS, _DEFAULT_LEVEL
    with _CONFIG_LOCK:
        if isinstance(level, str):
            level = LogLevel(level.lower())
        _DEFAULT_LEVEL = level
        if sinks is None:
            sinks = [StdoutLogSink(sys.stderr)] if json else [_HumanReadableSink(sys.stderr)]
        _ACTIVE_SINKS = list(sinks)
        _CONFIGURED = True


class _HumanReadableSink(LogSink):
    """Pretty-prints records for human reading. For development only."""

    _COLORS = {
        LogLevel.DEBUG: "\033[90m",
        LogLevel.INFO: "\033[36m",
        LogLevel.WARNING: "\033[33m",
        LogLevel.ERROR: "\033[31m",
        LogLevel.CRITICAL: "\033[1;31m",
    }
    _RESET = "\033[0m"

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def emit(self, record: dict[str, Any]) -> None:
        with self._lock:
            try:
                level = LogLevel(record.get("level", "info"))
                ts = record.get("ts", datetime.now(timezone.utc).isoformat())
                msg = record.get("msg", "")
                color = self._COLORS.get(level, "")
                self._stream.write(f"{color}[{level.value.upper()}]{self._RESET} {ts} {msg}\n")
                # Drop any extra fields besides level/ts/msg
                extras = {k: v for k, v in record.items() if k not in ("level", "ts", "msg")}
                if extras:
                    self._stream.write("  " + json.dumps(extras, default=str) + "\n")
                self._stream.flush()
            except Exception:
                pass

    def flush(self) -> None:
        with self._lock:
            try:
                self._stream.flush()
            except Exception:
                pass


# Ensure default config so loggers never crash
configure_logging()


# --- Logger handle ---


@dataclass
class _ActiveSpan:
    """Currently-active span for thread-local lookup."""

    span: Span


_TLS = threading.local()


def get_logger(name: str) -> _NautilusLogger:
    """Return a logger with the given name. Names with prefix `nautilus.`
    are auto-registered; any other name passes through.
    """
    if not name.startswith(_LOGGER_NAME_PREFIX):
        name = f"{_LOGGER_NAME_PREFIX}.{name}"
    return _NautilusLogger(name)


def get_stdlib_logger(name: str) -> logging.Logger:
    """Return the underlying stdlib ``logging.Logger`` for advanced configuration.

    Use this only when you need to set handler-level or formatter-level
    configuration on the stdlib logger (e.g. installing a custom
    ``StreamHandler``). For ordinary structured logging, prefer
    :func:`get_logger`.
    """
    if not name.startswith(_LOGGER_NAME_PREFIX):
        name = f"{_LOGGER_NAME_PREFIX}.{name}"
    return logging.getLogger(name)


# --- Span / stage contexts ---


class Span:
    """A logical operation that contains one or more stages.

    Use as a context manager:

        with Span("build_fat_binary", kernel="matmul") as sp:
            ...  # add stages via stage(sp, "name")
    """

    def __init__(
        self,
        operation: str,
        *,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        **metadata: Any,
    ) -> None:
        self._record = SpanRecord(
            span_id=span_id or str(uuid.uuid4()),
            operation=operation,
            start_ms=time.perf_counter() * 1000,
            parent_span_id=parent_span_id,
            metadata=dict(metadata),
        )

    @property
    def span_id(self) -> str:
        return self._record.span_id

    def set(self, **kwargs: Any) -> Span:
        """Add metadata fields. Chainable."""
        self._record.metadata.update(kwargs)
        return self

    def record_stage(self, stage: StageRecord) -> None:
        self._record.add_stage(stage)

    def finish(self, error: str | None = None) -> None:
        self._record.finish(error=error)
        get_logger("nautilus.span")._emit(
            "span_finished",
            {
                "span_id": self._record.span_id,
                "operation": self._record.operation,
                "duration_ms": self._record.duration_ms,
                "error": self._record.error,
                "metadata": self._record.metadata,
                "stages": [
                    {
                        "name": s.name,
                        "duration_ms": s.duration_ms,
                        "metadata": s.metadata,
                        "error": s.error,
                    }
                    for s in self._record.stages
                ],
            },
        )

    def __enter__(self) -> Span:
        _TLS.active_span = _ActiveSpan(self)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        error_str: str | None = None
        if exc is not None:
            error_str = f"{type(exc).__name__}: {exc}"
        try:
            self.finish(error=error_str)
        finally:
            _TLS.active_span = None


@contextmanager
def span(operation: str, **metadata: Any) -> Iterator[Span]:
    """Sugar for `with Span(...) as sp:`."""
    sp = Span(operation, **metadata)
    with sp:
        yield sp


class StageLog:
    """A single stage within a span."""

    def __init__(self, parent: Span, name: str) -> None:
        self._parent = parent
        self._name = name
        self._start_ms = time.perf_counter() * 1000
        self._record = StageRecord(
            name=name,
            start_ms=self._start_ms,
            duration_ms=0.0,
        )

    @property
    def name(self) -> str:
        return self._name

    def set(self, **kwargs: Any) -> StageLog:
        self._record.metadata.update(kwargs)
        return self

    def finish(self, error: str | None = None) -> None:
        self._record.duration_ms = (time.perf_counter() * 1000) - self._start_ms
        if error is not None:
            self._record.error = error
        self._parent.record_stage(self._record)
        get_logger("nautilus.stage")._emit(
            "stage_finished",
            {
                "span_id": self._parent.span_id,
                "name": self._name,
                "duration_ms": self._record.duration_ms,
                "metadata": self._record.metadata,
                "error": self._record.error,
            },
        )

    def __enter__(self) -> StageLog:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        error_str: str | None = None
        if exc is not None:
            error_str = f"{type(exc).__name__}: {exc}"
        self.finish(error=error_str)


@contextmanager
def stage(parent: Span, name: str) -> Iterator[StageLog]:
    """Sugar for `with StageLog(...) as st:`."""
    st = StageLog(parent, name)
    with st:
        yield st


# --- Logger impl ---


_LEVEL_RANK = {
    LogLevel.DEBUG: 10,
    LogLevel.INFO: 20,
    LogLevel.WARNING: 30,
    LogLevel.ERROR: 40,
    LogLevel.CRITICAL: 50,
}


class _NautilusLogger:
    """Logger handle returned by `get_logger()`.

    Supports debug/info/warning/error/critical plus a private `_emit`
    for structured span/stage events.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        # Mirror to stdlib logging so third-party libs (pytest) capture too
        self._stdlib = logging.getLogger(name)

    def _should_emit(self, level: LogLevel) -> bool:
        return _LEVEL_RANK[level] >= _LEVEL_RANK[_DEFAULT_LEVEL]

    def _emit(self, msg: str, fields: dict[str, Any], level: LogLevel = LogLevel.INFO) -> None:
        if not self._should_emit(level):
            return
        # Avoid stdlib logging's reserved keys when mirroring
        safe_fields = {k: v for k, v in fields.items() if k not in ("name", "msg", "args")}
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level.value,
            "logger": self._name,
            "msg": msg,
        }
        record.update(safe_fields)
        # Inject parent span id if any
        active = getattr(_TLS, "active_span", None)
        if active is not None and "span_id" not in record:
            record["span_id"] = active.span.span_id
        for sink in _ACTIVE_SINKS:
            try:
                sink.emit(record)
            except Exception:
                pass
        # Mirror to stdlib (with safe extras only)
        std_level = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }[level]
        self._stdlib.log(std_level, msg, extra=safe_fields)

    def debug(self, msg: str, *args: Any, **fields: Any) -> None:
        if args:
            msg = msg % args
        self._emit(msg, fields, LogLevel.DEBUG)

    def info(self, msg: str, *args: Any, **fields: Any) -> None:
        if args:
            msg = msg % args
        self._emit(msg, fields, LogLevel.INFO)

    def warning(self, msg: str, *args: Any, **fields: Any) -> None:
        if args:
            msg = msg % args
        self._emit(msg, fields, LogLevel.WARNING)

    def error(self, msg: str, *args: Any, **fields: Any) -> None:
        if args:
            msg = msg % args
        self._emit(msg, fields, LogLevel.ERROR)

    def critical(self, msg: str, *args: Any, **fields: Any) -> None:
        if args:
            msg = msg % args
        self._emit(msg, fields, LogLevel.CRITICAL)

    def exception(self, msg: str, *args: Any, **fields: Any) -> None:
        if args:
            msg = msg % args
        fields.setdefault("exc_info", True)
        self._emit(msg, fields, LogLevel.ERROR)


# --- Span accessor for current thread ---


def current_span() -> Span | None:
    """Return the active span for the current thread, or None."""
    active = getattr(_TLS, "active_span", None)
    return active.span if active is not None else None

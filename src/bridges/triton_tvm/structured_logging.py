"""Structured logging for the bridge with stage and span tracking.

Production-grade logging requirements:
  - Each stage logged with explicit name, duration, status
  - Each tuning run gets a unique span ID
  - IR dumps go to a ring buffer (last 8 only) — never to log flood
  - Machine-readable (JSON-compatible) format option
  - Compatible with OpenTelemetry's log conventions

Span concept: a span is a single end-to-end bridge execution
(extract → build TIR → tune → map → recompile). Spans can contain
sub-spans for each stage.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from src.common.logging import get_logger, get_stdlib_logger

logger = get_logger(__name__)


@dataclass
class StageLog:
    """A single stage execution record within a span."""

    stage_name: str
    status: str  # "started" | "completed" | "failed" | "skipped"
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Span:
    """A single bridge execution span — tracks all stages."""

    span_id: str
    kernel_hash: str
    target: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = "in_progress"  # "in_progress" | "completed" | "failed"
    stages: list[StageLog] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_stage(self, stage: StageLog) -> None:
        self.stages.append(stage)


# Module-level state
_active_spans: dict[str, Span] = {}
_completed_spans: deque[Span] = deque(maxlen=64)  # ring buffer
_ir_dump_ring: deque[tuple[str, str, str]] = deque(maxlen=8)  # (stage, source_hash, ir_text)


def get_active_span(span_id: str) -> Span | None:
    """Return an in-progress span by ID."""
    return _active_spans.get(span_id)


def get_completed_spans() -> list[Span]:
    """Return all completed spans from the ring buffer."""
    return list(_completed_spans)


def get_ir_dumps() -> list[tuple[str, str, str]]:
    """Return IR dumps from the ring buffer (stage, source_hash, ir_text)."""
    return list(_ir_dump_ring)


def dump_ir(stage_name: str, source_hash: str, ir_text: str) -> None:
    """Store an IR dump in the ring buffer (capped at 8)."""
    _ir_dump_ring.append((stage_name, source_hash[:16], ir_text))


def clear_ir_dumps() -> None:
    """Clear the IR dump ring buffer (useful after a successful tune)."""
    _ir_dump_ring.clear()


@contextmanager
def span(
    kernel_hash: str,
    target: str,
    metadata: dict[str, Any] | None = None,
    span_id: str | None = None,
) -> Iterator[Span]:
    """Context manager that creates a span and tracks its duration.

    Usage:
        with span(kernel_hash, target) as s:
            with stage(s, "extract") as st:
                ...
            with stage(s, "tune") as st:
                ...
    """
    sid = span_id or uuid.uuid4().hex[:12]
    s = Span(
        span_id=sid,
        kernel_hash=kernel_hash,
        target=target,
        start_time=time.time(),
        metadata=metadata or {},
    )
    _active_spans[sid] = s
    logger.info(
        "Span[%s] start: kernel=%s..%s target=%s",
        sid,
        kernel_hash[:12],
        "",
        target,
    )
    try:
        yield s
        s.status = "completed"
    except Exception as exc:
        s.status = "failed"
        s.metadata["error"] = str(exc)
        raise
    finally:
        s.end_time = time.time()
        s.duration_ms = (s.end_time - s.start_time) * 1000
        _active_spans.pop(sid, None)
        _completed_spans.append(s)
        logger.info(
            "Span[%s] %s in %.1fms (%d stages)",
            sid,
            s.status,
            s.duration_ms,
            len(s.stages),
        )


@contextmanager
def stage(
    parent_span: Span,
    stage_name: str,
    metadata: dict[str, Any] | None = None,
) -> Iterator[StageLog]:
    """Context manager that records a stage within a span.

    Automatically measures duration and records errors.
    """
    stage_log = StageLog(
        stage_name=stage_name,
        status="started",
        start_time=time.time(),
        metadata=metadata or {},
    )
    parent_span.add_stage(stage_log)
    logger.debug(
        "Span[%s] stage[%s] start",
        parent_span.span_id,
        stage_name,
    )
    try:
        yield stage_log
        stage_log.status = "completed"
    except Exception as exc:
        stage_log.status = "failed"
        stage_log.error = str(exc)
        raise
    finally:
        stage_log.end_time = time.time()
        stage_log.duration_ms = (stage_log.end_time - stage_log.start_time) * 1000
        logger.debug(
            "Span[%s] stage[%s] %s in %.1fms",
            parent_span.span_id,
            stage_name,
            stage_log.status,
            stage_log.duration_ms,
        )


def emit_span_json(span_obj: Span) -> str:
    """Serialize a span to JSON for observability backends."""
    return json.dumps(
        {
            "span_id": span_obj.span_id,
            "kernel_hash": span_obj.kernel_hash,
            "target": span_obj.target,
            "start_time": span_obj.start_time,
            "end_time": span_obj.end_time,
            "duration_ms": span_obj.duration_ms,
            "status": span_obj.status,
            "stages": [
                {
                    "stage_name": s.stage_name,
                    "status": s.status,
                    "duration_ms": s.duration_ms,
                    "metadata": s.metadata,
                    "error": s.error,
                }
                for s in span_obj.stages
            ],
            "metadata": span_obj.metadata,
        },
        indent=2,
    )


def configure_logging(
    level: str | None = None,
    json_format: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure the bridge's logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, emit JSON-formatted log lines (machine-readable).
        log_file: Optional path to also write logs to.
    """
    log_level = level or os.environ.get("NVINDIACUD_LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Get the bridge logger
    bridge_logger = get_stdlib_logger("nvindia_cud")
    bridge_logger.setLevel(numeric_level)
    bridge_logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler()
    if json_format:
        console.setFormatter(JsonFormatter())
    else:
        console.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            )
        )
    bridge_logger.addHandler(console)

    # File handler
    if log_file:
        from pathlib import Path

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        if json_format:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s")
            )
        bridge_logger.addHandler(file_handler)


class JsonFormatter(logging.Formatter):
    """JSON log formatter for observability backends."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


# Public exports
__all__ = [
    "Span",
    "StageLog",
    "clear_ir_dumps",
    "configure_logging",
    "dump_ir",
    "emit_span_json",
    "get_ir_dumps",
    "span",
    "stage",
]

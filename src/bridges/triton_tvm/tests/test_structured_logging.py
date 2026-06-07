"""Tests for the structured logging and span tracking."""

from __future__ import annotations

import json

import pytest

from src.bridges.triton_tvm.structured_logging import (
    Span,
    clear_ir_dumps,
    dump_ir,
    emit_span_json,
    get_ir_dumps,
)
from src.bridges.triton_tvm.structured_logging import (
    span as span_ctx,
)
from src.bridges.triton_tvm.structured_logging import (
    stage as stage_ctx,
)


class TestSpanContext:
    """Tests for the span() context manager."""

    def test_span_creation(self) -> None:
        """A new span should have valid id, hash, target."""
        with span_ctx("hash123", "nvidia/nvidia-h100") as sp:
            assert sp.span_id is not None
            assert sp.kernel_hash == "hash123"
            assert sp.target == "nvidia/nvidia-h100"
            assert sp.status == "in_progress"

    def test_span_completes_on_exit(self) -> None:
        """A span that completes normally should be marked completed."""
        with span_ctx("hash1", "target") as sp:
            pass
        assert sp.status == "completed"
        assert sp.duration_ms > 0

    def test_span_marks_failed_on_exception(self) -> None:
        """A span with an exception should be marked failed."""
        with pytest.raises(RuntimeError), span_ctx("hash2", "target") as sp:
            raise RuntimeError("test error")
        assert sp.status == "failed"
        assert "error" in sp.metadata

    def test_span_stages_tracked(self) -> None:
        """Stages added to a span should appear in the span's stages list."""
        with span_ctx("hash3", "target") as sp:
            with stage_ctx(sp, "extract"):
                pass
            with stage_ctx(sp, "tune"):
                pass
        assert len(sp.stages) == 2
        assert sp.stages[0].stage_name == "extract"
        assert sp.stages[1].stage_name == "tune"
        assert all(s.duration_ms > 0 for s in sp.stages)

    def test_completed_spans_in_ring_buffer(self) -> None:
        """Completed spans should be in the ring buffer."""
        with span_ctx("hash4", "target"):
            pass
        completed = get_completed_spans_func()
        assert len(completed) >= 1

    def test_custom_span_id(self) -> None:
        """A custom span_id should be respected."""
        with span_ctx("hash5", "target", span_id="custom123") as sp:
            assert sp.span_id == "custom123"


class TestStageContext:
    """Tests for the stage() context manager."""

    def test_stage_records_metadata(self) -> None:
        """Stages should record their metadata."""
        sp = Span(
            span_id="test",
            kernel_hash="k",
            target="t",
            start_time=0.0,
        )
        with stage_ctx(sp, "extract", metadata={"kernel": "matmul"}) as st:
            pass
        assert st.status == "completed"
        assert st.metadata["kernel"] == "matmul"

    def test_stage_records_error(self) -> None:
        """Stages with exceptions should record the error."""
        sp = Span(span_id="test", kernel_hash="k", target="t", start_time=0.0)
        with pytest.raises(ValueError), stage_ctx(sp, "extract") as st:
            raise ValueError("test")
        assert st.status == "failed"
        assert st.error == "test"


class TestIRDumpRingBuffer:
    """Tests for the IR dump ring buffer."""

    def test_dump_and_retrieve(self) -> None:
        """Dumped IR should be retrievable from the ring buffer."""
        clear_ir_dumps()
        dump_ir("ttgir", "kernel_abc", "module { ... }")
        dumps = get_ir_dumps()
        assert len(dumps) == 1
        assert dumps[0][0] == "ttgir"
        assert dumps[0][1] == "kernel_abc"
        assert "module" in dumps[0][2]

    def test_ring_buffer_capped(self) -> None:
        """The ring buffer should cap at 8 entries."""
        clear_ir_dumps()
        for i in range(20):
            dump_ir("ttgir", f"k{i}", f"ir_{i}")
        dumps = get_ir_dumps()
        assert len(dumps) == 8  # Capped
        # Most recent should be there
        assert "ir_19" in [d[2] for d in dumps]

    def test_clear(self) -> None:
        """Clear should empty the ring buffer."""
        dump_ir("ttgir", "k1", "ir1")
        assert len(get_ir_dumps()) > 0
        clear_ir_dumps()
        assert len(get_ir_dumps()) == 0


class TestEmitSpanJSON:
    """Tests for JSON span serialization."""

    def test_json_structure(self) -> None:
        """The JSON should contain all the expected fields."""
        with span_ctx("hash_json", "target_json") as sp, stage_ctx(sp, "extract"):
            pass
        json_str = emit_span_json(sp)
        data = json.loads(json_str)
        assert "span_id" in data
        assert "kernel_hash" in data
        assert data["kernel_hash"] == "hash_json"
        assert data["target"] == "target_json"
        assert len(data["stages"]) == 1
        assert data["stages"][0]["stage_name"] == "extract"


def get_completed_spans_func():
    """Helper to import the module-level function."""
    from src.bridges.triton_tvm.structured_logging import get_completed_spans

    return get_completed_spans()

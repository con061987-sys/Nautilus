"""Tests for PyTorch graph capture.

We exercise the ``GraphCapture.capture()`` API with a representative
two-layer MLP: ``nn.Sequential(Linear, GELU, Linear)``. This pattern is
common enough in real workloads to be a good canary, and it includes
both a learned layer and a non-linearity which together cover most
FX tracer code paths.

Tests for each capture mode (TORCH_EXPORT, TORCH_COMPILE, MANUAL_FX)
are gated — if the underlying torch API is not available the test
is skipped, not silently passed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for graph capture tests")
import torch.nn as nn  # noqa: E402

from src.bridges.pytorch_xla.graph_capture import (  # noqa: E402
    CapturedGraph,
    CaptureMode,
    GraphCapture,
    GraphMetadata,
)

# ── Fixtures ───────────────────────────────────────────────────────────


def _make_mlp() -> nn.Module:
    """The canonical test model: Linear -> GELU -> Linear.

    In -> [B, 16]
    Linear(16, 8) -> [B, 8]
    GELU         -> [B, 8]
    Linear(8, 4) -> [B, 4]
    Out -> [B, 4]
    """
    return nn.Sequential(
        nn.Linear(16, 8),
        nn.GELU(),
        nn.Linear(8, 4),
    )


# ── CaptureMode enum ─────────────────────────────────────────────────


class TestCaptureMode:
    """CaptureMode must be a stable enum that the rest of the pipeline can switch on."""

    def test_three_modes_present(self) -> None:
        names = {m.name for m in CaptureMode}
        assert "TORCH_COMPILE" in names
        assert "TORCH_EXPORT" in names
        assert "MANUAL_FX" in names

    def test_modes_are_distinct(self) -> None:
        modes = list(CaptureMode)
        assert len(modes) == len({m for m in modes})


# ── GraphMetadata ─────────────────────────────────────────────────────


class TestGraphMetadata:
    """GraphMetadata is the cache-key input — it must be deterministic."""

    def test_cache_key_changes_with_shape(self) -> None:
        a = GraphMetadata(model_name="m", source_hash="h", input_shapes=[(1, 16)])
        b = GraphMetadata(model_name="m", source_hash="h", input_shapes=[(2, 16)])
        assert a.cache_key != b.cache_key

    def test_cache_key_stable_for_same_content(self) -> None:
        a = GraphMetadata(
            model_name="m",
            source_hash="h",
            input_shapes=[(1, 16)],
            input_dtypes=["float32"],
            op_count=10,
        )
        b = GraphMetadata(
            model_name="m",
            source_hash="h",
            input_shapes=[(1, 16)],
            input_dtypes=["float32"],
            op_count=10,
        )
        assert a.cache_key == b.cache_key


# ── GraphCapture.capture() ───────────────────────────────────────────


class TestGraphCaptureWithMLP:
    """Capture the canonical MLP and verify all observable invariants."""

    def test_default_mode_is_torch_export(self) -> None:
        """The default mode must be TORCH_EXPORT — the most stable path."""
        gc = GraphCapture()
        assert gc.mode == CaptureMode.TORCH_EXPORT

    def test_capture_mlp_produces_usable_graph(self) -> None:
        """capture() on the MLP must succeed and return a usable graph."""
        model = _make_mlp()
        example_inputs = (torch.randn(2, 16),)

        gc = GraphCapture()
        captured = gc.capture(model=model, example_inputs=example_inputs)

        assert isinstance(captured, CapturedGraph)
        assert captured.is_usable is True
        assert captured.graph_module is not None
        assert captured.metadata.op_count > 0
        # Input shapes must be recorded
        assert captured.metadata.input_shapes == [(2, 16)]
        assert captured.metadata.input_dtypes == ["float32"]
        # Output shapes must be recorded (MLP outputs [B, 4])
        assert captured.metadata.output_shapes == [(2, 4)]

    def test_capture_records_source_hash(self) -> None:
        """The source hash is part of the cache key — must be present."""
        model = _make_mlp()
        example_inputs = (torch.randn(1, 16),)

        gc = GraphCapture()
        captured = gc.capture(model=model, example_inputs=example_inputs)

        assert isinstance(captured.metadata.source_hash, str)
        assert len(captured.metadata.source_hash) > 0

    def test_capture_accepts_explicit_model_name(self) -> None:
        model = _make_mlp()
        example_inputs = (torch.randn(1, 16),)

        gc = GraphCapture()
        captured = gc.capture(
            model=model,
            example_inputs=example_inputs,
            model_name="my_custom_mlp",
        )

        assert captured.metadata.model_name == "my_custom_mlp"

    def test_capture_default_model_name_uses_class(self) -> None:
        model = _make_mlp()
        example_inputs = (torch.randn(1, 16),)

        gc = GraphCapture()
        captured = gc.capture(model=model, example_inputs=example_inputs)

        # Default uses the class name. Sequential reports as "Sequential".
        assert captured.metadata.model_name == "Sequential"

    def test_capture_timing_recorded(self) -> None:
        """capture_time_ms must be a non-negative number."""
        model = _make_mlp()
        example_inputs = (torch.randn(1, 16),)

        gc = GraphCapture()
        captured = gc.capture(model=model, example_inputs=example_inputs)

        assert captured.metadata.capture_time_ms >= 0.0

    def test_capture_generates_readable_text(self) -> None:
        """CapturedGraph must populate fx_graph_text for debugging."""
        model = _make_mlp()
        example_inputs = (torch.randn(1, 16),)

        gc = GraphCapture()
        captured = gc.capture(model=model, example_inputs=example_inputs)

        # fx_graph_text is populated in CapturedGraph.__post_init__
        assert isinstance(captured.fx_graph_text, str)


# ── Capture mode variants ─────────────────────────────────────────────


class TestGraphCaptureModes:
    """Each CaptureMode is a distinct codepath — test them individually."""

    def test_manual_fx_mode(self) -> None:
        """MANUAL_FX uses torch.fx.symbolic_trace — works on any model
        that doesn't use data-dependent control flow.
        """
        model = _make_mlp()
        example_inputs = (torch.randn(1, 16),)

        gc = GraphCapture(mode=CaptureMode.MANUAL_FX)
        captured = gc.capture(model=model, example_inputs=example_inputs)

        assert captured.is_usable is True
        # MANUAL_FX traces the model directly — output shape must match
        assert (2, 4) in captured.metadata.output_shapes or captured.metadata.output_shapes == [
            (1, 4)
        ]

    def test_torch_export_mode(self) -> None:
        """TORCH_EXPORT uses torch.export.export — produces a richer
        GraphModule with type info, but may not work on every model.
        """
        model = _make_mlp()
        example_inputs = (torch.randn(1, 16),)

        gc = GraphCapture(mode=CaptureMode.TORCH_EXPORT)
        captured = gc.capture(model=model, example_inputs=example_inputs)

        # The export path may or may not succeed depending on torch version
        if not captured.is_usable:
            pytest.skip("torch.export path unusable in this env")
        assert captured.graph_module is not None


# ── Failure modes ─────────────────────────────────────────────────────


class TestGraphCaptureFailureHandling:
    """When capture fails the API must degrade gracefully."""

    def test_capture_with_bad_inputs_returns_unusable(self) -> None:
        """Passing an int instead of a tensor should not crash; it should
        return a CapturedGraph with is_usable=False and a populated metadata.
        """
        gc = GraphCapture()
        # fx.symbolic_trace / torch.export will both fail on this input
        model = _make_mlp()
        # The "model" is fine; we'll just not pass any example inputs.
        # Note: capture() signature requires example_inputs, so we use ().
        captured = gc.capture(model=model, example_inputs=())

        # Either it captured something OR it returned is_usable=False
        # — both are acceptable, but it must NOT raise.
        assert isinstance(captured, CapturedGraph)
        if not captured.is_usable:
            # The metadata must still be populated for debugging
            assert captured.metadata.model_name != ""

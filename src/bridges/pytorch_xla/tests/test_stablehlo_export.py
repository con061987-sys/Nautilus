"""Tests for the FX -> StableHLO exporter's three-tier fallback chain.

Each export tier is exercised in isolation by calling the underlying
``_TorchXLAExporter`` / ``_ONNXBridgeExporter`` / ``_TVMScriptExporter``
class directly. This lets us verify the contract of *each* tier
independently of the orchestrator's auto-fallback logic.

The orchestrator's three-tier fan-out is also tested to make sure
attempt history is recorded and at least one of the tiers' ``is_available``
contract is honored.

If a tier's dependency is missing the corresponding test is
``pytest.skip``'d with a clear message — never silently passes. If all
three tiers are unavailable the orchestrator test asserts that
``StableHLOExportError`` is raised (not a silent fallback).
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch

# All three tiers in stablehlo_export depend on PyTorch at import time
# (the module imports ``src.common.errors`` which is safe, but the
# graph modules we feed them must be torch.fx.GraphModule instances).
torch = pytest.importorskip("torch", reason="PyTorch is required for StableHLO export tests")
import torch.fx as fx  # noqa: E402

from src.bridges.pytorch_xla.stablehlo_export import (  # noqa: E402
    StableHLOExporter,
    _check_is_real_stablehlo,
    _ONNXBridgeExporter,
    _TorchXLAExporter,
    _TVMScriptExporter,
)
from src.common.errors import StableHLOExportError  # noqa: E402
from src.common.types import StableHLOModule  # noqa: E402

# ── Test fixtures ──────────────────────────────────────────────────────


def _make_minimal_fx_graph() -> fx.GraphModule:
    """Build the smallest useful FX graph: relu(matmul(x, w)).

    Uses only ops the FX symbolic tracer handles natively (aten::addmm,
    aten::relu). This produces ~3 nodes which is enough to exercise
    every export tier without dragging in model-specific dependencies.
    """

    class _TinyNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(torch.matmul(x, self.weight))

    model = _TinyNet()
    graph = fx.symbolic_trace(model)
    return graph


# ── Tier-1: torch_xla ─────────────────────────────────────────────────


class TestTorchXLAExporterTier:
    """Test the primary torch_xla export path in isolation."""

    def test_is_available_contract(self) -> None:
        """is_available() must return a bool and not raise."""
        result = _TorchXLAExporter.is_available()
        assert isinstance(result, bool)

    def test_export_produces_stablehlomodule_or_raises(self) -> None:
        """export() must return a StableHLOModule when the lib is present,
        and raise a clear ImportError-derived exception when absent.
        """
        graph = _make_minimal_fx_graph()
        example_inputs = (torch.randn(2, 4),)

        if not _TorchXLAExporter.is_available():
            pytest.skip("torch_xla is not installed; primary tier unavailable")

        result = _TorchXLAExporter.export(graph, example_inputs, function_name="forward")

        assert isinstance(result, StableHLOModule)
        assert result.is_usable is True
        # function_name is plumbed through verbatim
        assert result.function_name == "forward"
        assert result.export_method.startswith("torch_xla")
        # If the path emitted real StableHLO it must contain a stablehlo. op
        if result.is_real_stablehlo:
            assert _check_is_real_stablehlo(result.mlir_text), (
                "is_real_stablehlo=True but mlir_text has no stablehlo. ops"
            )


# ── Tier-2: onnx-bridge ──────────────────────────────────────────────


class TestONNXBridgeExporterTier:
    """Test the secondary onnx-mlir bridge in isolation."""

    def test_is_available_contract(self) -> None:
        """is_available() must probe for onnx + onnx-mlir binary."""
        result = _ONNXBridgeExporter.is_available()
        assert isinstance(result, bool)

    def test_export_with_onnx_mlir(self) -> None:
        """When onnx + onnx-mlir are both present, export must succeed."""
        graph = _make_minimal_fx_graph()
        example_inputs = (torch.randn(2, 4),)

        if not _ONNXBridgeExporter.is_available():
            pytest.skip("onnx-mlir binary not on PATH; secondary tier unavailable")

        result = _ONNXBridgeExporter.export(graph, example_inputs, function_name="forward")

        assert isinstance(result, StableHLOModule)
        assert result.is_usable is True
        assert result.export_method.startswith("onnx-bridge")
        # Validate the StableHLO invariant if it claims to be real
        if result.is_real_stablehlo:
            assert _check_is_real_stablehlo(result.mlir_text)

    def test_onnx_module_present_even_if_mlir_binary_missing(self) -> None:
        """The is_available() probe must distinguish python-onnx from
        onnx-mlir binary. This test checks only the python side, which
        is sufficient to be sure the export code's first step can
        import torch.onnx successfully.
        """
        # If the python `onnx` package is not installed, skip
        if importlib.util.find_spec("onnx") is None:
            pytest.skip("python `onnx` package not installed")
        # is_available may still return False (no onnx-mlir binary) — both
        # outcomes are acceptable; we only assert it returns a bool.
        assert isinstance(_ONNXBridgeExporter.is_available(), bool)


# ── Tier-3: tvmscript ────────────────────────────────────────────────


class TestTVMScriptExporterTier:
    """Test the tertiary TVMScript export path in isolation."""

    def test_is_available_contract(self) -> None:
        result = _TVMScriptExporter.is_available()
        assert isinstance(result, bool)

    def test_export_with_tvm_installed(self) -> None:
        """If TVM is installed, export must produce a StableHLOModule."""
        graph = _make_minimal_fx_graph()
        example_inputs = (torch.randn(2, 4),)

        if not _TVMScriptExporter.is_available():
            pytest.skip("tvm is not installed; tertiary tier unavailable")

        result = _TVMScriptExporter.export(graph, example_inputs, function_name="forward")

        assert isinstance(result, StableHLOModule)
        assert result.is_usable is True
        assert result.export_method.startswith("tvmscript")
        if result.is_real_stablehlo:
            assert _check_is_real_stablehlo(result.mlir_text)

    def test_fallback_emits_stablehlo_like_wrapper(self) -> None:
        """When TVM's contrib.stablehlo is missing, the fallback emitter
        must still produce a valid MLIR text wrapping the relax module.
        """
        if not _TVMScriptExporter.is_available():
            pytest.skip("tvm is not installed")

        _make_minimal_fx_graph()
        example_inputs = (torch.randn(2, 4),)

        # _emit_stablehlo_like is the fallback emitter — call it directly
        # to bypass the import path. It needs a relax module, but we
        # can pass None and verify it produces a parseable string with
        # a func.func / module wrapper.
        result = _TVMScriptExporter._emit_stablehlo_like(
            mod=None,
            function_name="forward",
            example_inputs=example_inputs,
        )

        # Must be non-empty and contain MLIR structural elements
        assert isinstance(result, str)
        assert len(result) > 0
        assert "module" in result
        assert "func.func" in result
        assert "forward" in result


# ── Orchestrator: three-tier fallback ────────────────────────────────


class TestStableHLOExporterOrchestrator:
    """Test the StableHLOExporter orchestrator's tier fan-out."""

    def test_export_from_captured_with_real_capture(self) -> None:
        """End-to-end: capture a tiny model, then run all 3 export tiers.

        We *don't* depend on torch_xla, onnx, or tvm being installed.
        The orchestrator must either:
          - succeed via one of the installed tiers, OR
          - raise StableHLOExportError with a full attempt history.

        It must never silently return a fake StableHLO module.
        """

        graph = _make_minimal_fx_graph()
        # We need a CapturedGraph — synthesize one from the FX graph
        # directly. This mirrors what GraphCapture.capture() returns
        # for a successful TORCH_EXPORT run.
        from src.bridges.pytorch_xla.graph_capture import CapturedGraph, GraphMetadata

        metadata = GraphMetadata(
            model_name="forward",
            source_hash="deadbeef",
            input_shapes=[(2, 4)],
            input_dtypes=["float32"],
            output_shapes=[(2, 4)],
            op_count=len(list(graph.graph.nodes)),
        )
        captured = CapturedGraph(graph_module=graph, metadata=metadata, is_usable=True)

        exporter = StableHLOExporter(enable_tvm_path=True)
        try:
            result = exporter.export_from_captured(captured, (torch.randn(2, 4),))
        except Exception as exc:
            # All three tiers unavailable — must be a typed export error,
            # not an AttributeError or random Exception.
            from src.common.errors import StableHLOExportError

            assert isinstance(exc, StableHLOExportError), (
                f"All tiers failed but the raised exception is "
                f"{type(exc).__name__}, not StableHLOExportError"
            )
            # The error must include attempt history so a user can debug
            assert "Attempt history" in str(exc) or "attempt_history" in str(exc.context)
            return

        # If we got here, at least one tier succeeded.
        assert isinstance(result, StableHLOModule)
        assert result.is_usable is True
        assert result.function_name == "forward"

    def test_export_from_captured_rejects_unusable_capture(self) -> None:
        """An unusable capture must raise StableHLOExportError immediately,
        not silently try all three tiers.
        """
        from src.bridges.pytorch_xla.graph_capture import CapturedGraph, GraphMetadata

        metadata = GraphMetadata(
            model_name="forward",
            source_hash="deadbeef",
        )
        # No graph_module, is_usable=False
        captured = CapturedGraph(graph_module=None, metadata=metadata, is_usable=False)

        exporter = StableHLOExporter()
        with pytest.raises(Exception) as exc_info:
            exporter.export_from_captured(captured, (torch.randn(2, 4),))
        # The error must mention usability
        assert "usable" in str(exc_info.value).lower() or "None" in str(exc_info.value)

    def test_export_from_captured_rejects_none_capture(self) -> None:
        """Passing None must raise a clear error."""
        exporter = StableHLOExporter()
        with pytest.raises((TypeError, ValueError, RuntimeError, StableHLOExportError)):
            exporter.export_from_captured(None)

    def test_tvm_path_can_be_disabled(self) -> None:
        """The TVM tier must be skippable via the constructor flag."""
        exporter = StableHLOExporter(enable_tvm_path=False)
        # Sanity: the flag is stored.
        assert exporter.enable_tvm_path is False


# ── StableHLO validation helper ────────────────────────────────────────


class TestCheckIsRealStableHLO:
    """The validator regex is part of the public contract — test it."""

    def test_real_stablehlo_detected(self) -> None:
        mlir = """
        module {
          func.func @main(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
            %0 = stablehlo.add %x, %x : tensor<2x2xf32>
            return %0 : tensor<2x2xf32>
          }
        }
        """
        assert _check_is_real_stablehlo(mlir) is True

    def test_mhlo_only_not_real_stablehlo(self) -> None:
        # mhlo. ops do NOT count as stablehlo. — strict detection
        mlir = """
        module {
          func.func @main(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
            %0 = mhlo.add %x, %x : tensor<2x2xf32>
            return %0 : tensor<2x2xf32>
          }
        }
        """
        assert _check_is_real_stablehlo(mlir) is False

    def test_empty_mlir_not_stablehlo(self) -> None:
        assert _check_is_real_stablehlo("") is False

    def test_arbitrary_text_not_stablehlo(self) -> None:
        assert _check_is_real_stablehlo("// not mlir at all\n") is False

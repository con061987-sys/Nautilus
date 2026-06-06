"""PyTorch graph capture for the auto-sharding pipeline.

Captures a PyTorch model's forward pass as an FX (torch.fx) graph.
The FX graph is the canonical representation we feed into the
StableHLO conversion → GSPMD auto-sharding pipeline.

We support two capture modes:
  1. torch.compile() + dynamo.export() — modern, recommended
  2. torch.export() — stable, lower-level

Both produce an FX GraphModule that we can introspect and translate.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from src.common.logging import get_logger

try:
    import torch
    import torch.fx as fx
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = get_logger(__name__)


class CaptureMode(Enum):
    """How to capture the graph."""
    TORCH_COMPILE = auto()      # torch.compile() + dynamo.export()
    TORCH_EXPORT = auto()        # torch.export() direct
    MANUAL_FX = auto()           # Manual FX symbolic_trace


@dataclass
class GraphMetadata:
    """Metadata about a captured graph."""
    model_name: str
    source_hash: str
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    input_dtypes: list[str] = field(default_factory=list)
    output_shapes: list[tuple[int, ...]] = field(default_factory=list)
    op_count: int = 0
    param_count: int = 0
    capture_time_ms: float = 0.0

    @property
    def cache_key(self) -> str:
        parts = [
            self.model_name,
            self.source_hash,
            str(self.input_shapes),
            str(self.input_dtypes),
            str(self.op_count),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()


@dataclass
class CapturedGraph:
    """A captured PyTorch FX graph ready for StableHLO conversion."""
    graph_module: Any  # torch.fx.GraphModule
    metadata: GraphMetadata
    fx_graph_text: str = ""
    is_usable: bool = True

    def __post_init__(self) -> None:
        # If we have a graph module but no text, try to generate it
        if self.graph_module is not None and not self.fx_graph_text:
            try:
                self.fx_graph_text = self.graph_module.print_readable()
            except Exception:
                self.fx_graph_text = ""


class GraphCapture:
    """Captures PyTorch model graphs for the auto-sharding pipeline.

    Usage:
        capture = GraphCapture()
        result = capture.capture(
            model=my_model,
            example_inputs=(torch.randn(1, 3, 224, 224),),
        )
        if result.is_usable:
            # Pass result.graph_module to StableHLOExporter
            ...
    """

    def __init__(self, mode: CaptureMode = CaptureMode.TORCH_EXPORT) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for graph capture. "
                "Install with: pip install torch"
            )
        self.mode = mode

    def capture(
        self,
        model: Any,
        example_inputs: tuple[Any, ...],
        model_name: str | None = None,
    ) -> CapturedGraph:
        """Capture the model's forward pass as an FX graph.

        Args:
            model: The PyTorch model to capture.
            example_inputs: Example inputs for tracing.
            model_name: Optional name (defaults to model.__class__.__name__).

        Returns:
            CapturedGraph with the FX graph module and metadata.
        """
        import time
        start = time.perf_counter()

        if model_name is None:
            model_name = model.__class__.__name__

        source_hash = self._compute_source_hash(model)
        metadata = GraphMetadata(
            model_name=model_name,
            source_hash=source_hash,
            input_shapes=[self._shape_of(a) for a in example_inputs],
            input_dtypes=[self._dtype_of(a) for a in example_inputs],
        )

        try:
            if self.mode == CaptureMode.TORCH_EXPORT:
                graph_module, output_shapes = self._capture_via_export(
                    model, example_inputs,
                )
            elif self.mode == CaptureMode.TORCH_COMPILE:
                graph_module, output_shapes = self._capture_via_compile(
                    model, example_inputs,
                )
            else:
                graph_module, output_shapes = self._capture_via_fx(
                    model, example_inputs,
                )
        except Exception as exc:
            logger.warning("Graph capture failed: %s", exc)
            return CapturedGraph(
                graph_module=None,
                metadata=metadata,
                is_usable=False,
            )

        metadata.output_shapes = output_shapes
        metadata.op_count = sum(1 for _ in graph_module.graph.nodes)
        param_count = 0
        for node in graph_module.graph.nodes:
            if node.op != "placeholder":
                continue
            if not hasattr(node, "meta"):
                continue
            type_str = node.meta.get("type", "")
            if hasattr(type_str, "__name__") and "param" in str(type_str).lower():
                param_count += 1
        metadata.param_count = param_count
        metadata.capture_time_ms = (time.perf_counter() - start) * 1000

        return CapturedGraph(
            graph_module=graph_module,
            metadata=metadata,
        )

    def _capture_via_export(
        self,
        model: Any,
        example_inputs: tuple[Any, ...],
    ) -> tuple[Any, list[tuple[int, ...]]]:
        """Capture via torch.export (the modern stable API)."""
        # torch.export returns an ExportedProgram
        exported = torch.export.export(model, example_inputs)
        # Get the graph module from the exported program
        graph_module = exported.graph_module
        # Run the model to get output shapes
        with torch.no_grad():
            outputs = model(*example_inputs)
        output_shapes = self._collect_output_shapes(outputs)
        return graph_module, output_shapes

    def _capture_via_compile(
        self,
        model: Any,
        example_inputs: tuple[Any, ...],
    ) -> tuple[Any, list[tuple[int, ...]]]:
        """Capture via torch.compile + export.

        On PyTorch 2.5+ we use the stable ``torch.export.export`` API
        on the compiled model. On 2.4 and earlier we fall back to
        ``torch._dynamo.export``.
        """
        import torch._dynamo as dynamo

        # Compile the model first
        compiled = torch.compile(model, fullgraph=True, backend="aot_eager")

        # Then export the compiled model
        with torch.no_grad():
            compiled_output = compiled(*example_inputs)

        if hasattr(torch.export, "export"):
            # PyTorch 2.5+ stable export API
            ep = torch.export.export(compiled, example_inputs)
            graph_module = ep.graph_module
        else:
            # PyTorch 2.4 fallback: dynamo.export returns a tuple
            # (graph_module, guards)
            exported = dynamo.export(compiled, *example_inputs)
            graph_module = (
                exported[0]
                if isinstance(exported, (list, tuple))
                else exported
            )

        output_shapes = self._collect_output_shapes(compiled_output)
        return graph_module, output_shapes

    def _capture_via_fx(
        self,
        model: Any,
        example_inputs: tuple[Any, ...],
    ) -> tuple[Any, list[tuple[int, ...]]]:
        """Capture via manual torch.fx symbolic_trace."""
        graph_module = fx.symbolic_trace(model)
        with torch.no_grad():
            outputs = graph_module(*example_inputs)
        output_shapes = self._collect_output_shapes(outputs)
        return graph_module, output_shapes

    def _compute_source_hash(self, model: Any) -> str:
        """Compute a stable hash of the model's source code."""
        try:
            source = inspect.getsource(model.__class__)
            return hashlib.sha256(source.encode()).hexdigest()
        except (OSError, TypeError):
            return hashlib.sha256(
                model.__class__.__module__.encode()
            ).hexdigest()

    def _shape_of(self, tensor: Any) -> tuple[int, ...]:
        """Get the shape of a tensor (returns () for non-tensors)."""
        if isinstance(tensor, torch.Tensor):
            return tuple(tensor.shape)
        return ()

    def _dtype_of(self, tensor: Any) -> str:
        """Get the dtype of a tensor as a string."""
        if isinstance(tensor, torch.Tensor):
            return str(tensor.dtype).replace("torch.", "")
        return "unknown"

    def _collect_output_shapes(self, outputs: Any) -> list[tuple[int, ...]]:
        """Collect shapes from model outputs (handles tensor and tuple outputs)."""
        if isinstance(outputs, torch.Tensor):
            return [tuple(outputs.shape)]
        if isinstance(outputs, (tuple, list)):
            return [
                tuple(o.shape) if isinstance(o, torch.Tensor) else ()
                for o in outputs
            ]
        return []

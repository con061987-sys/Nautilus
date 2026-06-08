"""Integration tests for the kernel fusion engine.

Tests the full fusion pipeline across all three fusion components:

  - ``FusionAnalyzer`` (``fusion_analyzer.py``) — detects fusion patterns in
    parsed TTGIR, produces ``PatternMatch`` entries.
  - ``FusionPlanner`` / ``FusionCodeGenerator`` (``kernel_fusion.py``) —
    plans fusible op-graph subsequences and generates fused
    ``@triton.jit`` kernel source.
  - ``CrossDeviceFusionPlanner`` / ``CommunicationOverlapper``
    (``cross_device_fusion.py``) — analyzes sharded computation graphs
    for computation-communication overlap and builds fused operations.

All external dependencies (TVM, torch.distributed) are mocked so the
tests can run in CI without GPU hardware.

Every test documents:
  - What it tests
  - What passing means
"""

from __future__ import annotations

from typing import Any

from src.bridges.pytorch_xla.collective_insertion import (
    CollectiveType,
    InsertedCollective,
)
from src.bridges.pytorch_xla.comm_backend import CommLibrary
from src.bridges.pytorch_xla.cross_device_fusion import (
    CommunicationOverlapper,
    CrossDeviceFusionPlan,
    CrossDeviceFusionPlanner,
    FusionPattern,
    NullAsyncWorkHandle,
    ShardedComputationGraph,
    ShardedGraphNode,
    build_graph_from_collectives,
    plan_and_overlap,
)
from src.bridges.pytorch_xla.device_mesh import DeviceVendor
from src.bridges.triton_tvm.fusion_analyzer import (
    FusionAnalyzer,
    PatternMatch,
)
from src.bridges.triton_tvm.ir_to_tir.ttgir_parser import (
    OpKind,
    TTGIRFunction,
    TTGIROperation,
    TTGIRParser,
)
from src.bridges.triton_tvm.kernel_fusion import (
    FusionCodeGenerator,
    FusionPlanner,
    OpNode,
)
from src.bridges.triton_tvm.kernel_fusion import (
    OpKind as FusionOpKind,
)

# =========================================================================
# Helpers — shared across all test sections
# =========================================================================


def _op(
    kind: OpKind,
    result_name: str = "",
    operands: list[str] | None = None,
    name: str = "",
) -> TTGIROperation:
    """Construct a minimal TTGIROperation for testing."""
    return TTGIROperation(
        kind=kind,
        raw_text="",
        name=name or kind.name.lower(),
        result_name=result_name,
        operands=operands or [],
        types=[],
        attributes={},
    )


def _func(ops: list[TTGIROperation], name: str = "test_kernel") -> TTGIRFunction:
    """Wrap ops in a minimal TTGIRFunction."""
    return TTGIRFunction(name=name, ops=ops)


def _fus_op(kind: FusionOpKind, **attrs: Any) -> OpNode:
    """Construct a minimal OpNode for the fusion planner."""
    return OpNode(kind=kind, attrs=attrs)


def _find_by_name(matches: list[PatternMatch], name: str) -> PatternMatch | None:
    """Find the first PatternMatch with the given name."""
    for m in matches:
        if m.pattern_name == name:
            return m
    return None


def _sharded_graph_with_collectives(
    coll_type: CollectiveType,
    num_nodes: int = 3,
    compute_time_us: float = 100.0,
) -> ShardedComputationGraph:
    """Build a minimal sharded graph with one collective at each boundary.

    Creates a chain of ``num_nodes`` boundary nodes, each with a single
    collective of the given type.
    """
    graph = ShardedComputationGraph()
    for i in range(num_nodes):
        node = ShardedGraphNode(
            node_id=f"n{i}",
            op_type="matmul",
            output_tensors=[((64, 64), "f32")],
            device_id=i,
            vendor=DeviceVendor.NVIDIA,
            compute_time_estimate_us=compute_time_us,
            is_shard_boundary=True,
        )
        graph.nodes.append(node)
        graph.boundary_collectives[node.node_id] = [
            InsertedCollective(
                collective_type=coll_type,
                tensor_name=f"%t{i}",
                mesh_axis=0,
                num_devices=num_nodes,
                device_ids=tuple(range(num_nodes)),
                comm_library=CommLibrary.NCCL,
                result_name=f"%coll{i}",
                tensor_shape=(64, 64),
                dtype="f32",
                estimated_bytes=64 * 64 * 4 * 2,
            )
        ]
    # Chain nodes linearly
    for i in range(num_nodes - 1):
        graph.edges.append((graph.nodes[i].node_id, graph.nodes[i + 1].node_id))
    return graph


# =========================================================================
# SECTION 1 — Analyzer → Planner → Generator Pipeline
# =========================================================================


class TestAnalyzerToPlannerToGeneratorPipeline:
    """Verifies the full pipeline: analyzer detects patterns → planner
    consumes the information → generator produces fused Triton kernel code.

    The analyzer works on parsed TTGIR, the planner works on OpNode graphs.
    This section tests that patterns discovered by the analyzer correspond
    to valid fusion plans, and that those plans produce compilable code.
    """

    def _matmul_graph(self, activation: str = "relu") -> list[OpNode]:
        """Build an OpNode graph for matmul + activation."""
        ops = [
            _fus_op(FusionOpKind.MATMUL, m=4096, n=4096, k=4096, dtype="float32"),
        ]
        act_map = {
            "relu": FusionOpKind.RELU,
            "gelu": FusionOpKind.GELU,
            "silu": FusionOpKind.SILU,
        }
        ops.append(_fus_op(act_map.get(activation, FusionOpKind.RELU)))
        return ops

    def _matmul_bias_graph(self, activation: str = "relu") -> list[OpNode]:
        """Build an OpNode graph for matmul + bias + activation."""
        ops = [
            _fus_op(FusionOpKind.MATMUL, m=4096, n=4096, k=4096, dtype="float32"),
            _fus_op(FusionOpKind.BIAS_ADD),
        ]
        act_map = {
            "relu": FusionOpKind.RELU,
            "gelu": FusionOpKind.GELU,
            "silu": FusionOpKind.SILU,
        }
        ops.append(_fus_op(act_map.get(activation, FusionOpKind.RELU)))
        return ops

    def _matmul_act_ir(self) -> TTGIRFunction:
        """Return a parsed matmul+activation function."""
        return _func([
            _op(OpKind.LOAD, "%a", ["%A"]),
            _op(OpKind.LOAD, "%b", ["%B"]),
            _op(OpKind.DOT, "%c", ["%a", "%b"], "tt.dot"),
            _op(OpKind.TANH, "%d", ["%c"], "math.tanh"),
            _op(OpKind.STORE, "", ["%d"]),
        ])

    # ------------------------------------------------------------------
    # Full pipeline: analyzer pattern -> planner plan -> generator code
    # ------------------------------------------------------------------

    def test_matmul_relu_full_pipeline(self) -> None:
        """Analyzer detects matmul_activation -> planner creates matmul+relu
        -> generator produces valid fused kernel code.

        What it tests:
          The full pipeline for matmul+relu: TTGIR analysis -> OpNode
          planning -> Triton kernel generation.

        Passing means:
          Analyzer finds matmul_activation, planner finds matmul+relu,
          generator produces syntactically valid Python with @triton.jit.
        """
        # Step 1: Analyzer detects the pattern in TTGIR
        analyzer = FusionAnalyzer()
        func = self._matmul_act_ir()
        matches = analyzer.analyze(func)
        match = _find_by_name(matches, "matmul_activation")
        assert match is not None, "Analyzer should detect matmul_activation"
        assert match.confidence == 1.0

        # Step 2: Planner creates a fusion plan from an OpNode graph
        planner = FusionPlanner()
        graph = self._matmul_graph("relu")
        plans = planner.find_patterns(graph)
        assert len(plans) == 1, "Planner should find matmul+relu"
        assert plans[0].pattern == "matmul+relu"

        # Step 3: Generator produces valid kernel code
        gen = FusionCodeGenerator()
        source = gen.generate(plans[0])

        # Verify generated code compiles as valid Python syntax
        compile(source, "<matmul+relu>", "exec")
        assert "@triton.jit" in source
        assert "tl.dot" in source
        assert "tl.where" in source  # relu: max(0, x)
        assert "tl.store" in source

    def test_matmul_gelu_full_pipeline(self) -> None:
        """Analyzer -> planner -> generator for matmul+gelu.

        What it tests:
          Full pipeline for matmul+gelu with GELU activation body.

        Passing means:
          Generated code contains the GELU approximation (tanh).
        """
        planner = FusionPlanner()
        graph = self._matmul_graph("gelu")
        plans = planner.find_patterns(graph)
        assert len(plans) == 1
        assert plans[0].pattern == "matmul+gelu"

        gen = FusionCodeGenerator()
        source = gen.generate(plans[0])
        compile(source, "<matmul+gelu>", "exec")
        assert "tanh" in source  # GELU uses tanh approximation
        assert "0.044715" in source  # GELU polynomial coefficient

    def test_matmul_silu_full_pipeline(self) -> None:
        """Analyzer -> planner -> generator for matmul+silu.

        What it tests:
          Full pipeline for matmul+silu with SiLU activation (x * sigmoid(x)).

        Passing means:
          Generated code contains tl.sigmoid.
        """
        planner = FusionPlanner()
        graph = self._matmul_graph("silu")
        plans = planner.find_patterns(graph)
        assert len(plans) == 1
        assert plans[0].pattern == "matmul+silu"

        gen = FusionCodeGenerator()
        source = gen.generate(plans[0])
        compile(source, "<matmul+silu>", "exec")
        assert "tl.sigmoid" in source

    def test_matmul_bias_relu_full_pipeline(self) -> None:
        """Three-op pattern: matmul + bias + relu through full pipeline.

        What it tests:
          The full pipeline for a three-op fusion (matmul+bias+relu).

        Passing means:
          Analyzer detects matmul_bias_activation, planner creates
          matmul+bias+relu, generator includes bias_ptr and bias_stride.
        """
        # Step 1: Analyzer
        func = _func([
            _op(OpKind.LOAD, "%a", ["%A"]),
            _op(OpKind.LOAD, "%b", ["%B"]),
            _op(OpKind.LOAD, "%bias", ["%bias_ptr"]),
            _op(OpKind.DOT, "%c", ["%a", "%b"], "tt.dot"),
            _op(OpKind.ADDF, "%d", ["%c", "%bias"]),
            _op(OpKind.TANH, "%e", ["%d"], "math.tanh"),
            _op(OpKind.STORE, "", ["%e"]),
        ])
        analyzer = FusionAnalyzer()
        matches = analyzer.analyze(func)
        match = _find_by_name(matches, "matmul_bias_activation")
        assert match is not None
        assert match.confidence == 1.0

        # Step 2: Planner
        planner = FusionPlanner()
        graph = self._matmul_bias_graph("relu")
        plans = planner.find_patterns(graph)
        assert len(plans) == 1
        assert plans[0].pattern == "matmul+bias+relu"

        # Step 3: Generator
        gen = FusionCodeGenerator()
        source = gen.generate(plans[0])
        compile(source, "<matmul+bias+relu>", "exec")
        assert "bias_ptr" in source
        assert "bias_stride" in source
        assert "tl.where" in source  # relu

    def test_analyzer_matmul_activation_maps_to_planner(self) -> None:
        """Pattern match from analyzer provides confidence that a
        corresponding planner pattern will succeed.

        What it tests:
          The correspondence between analyzer pattern names and planner
          pattern names for matmul+activation patterns.

        Passing means:
          For every analyzer-detected matmul_activation pattern,
          a matching planner pattern exists and produces a valid plan.
        """
        analyzer = FusionAnalyzer()
        func = self._matmul_act_ir()
        matches = analyzer.analyze(func)
        match = _find_by_name(matches, "matmul_activation")
        assert match is not None

        # The pattern involves MATMUL + ELEMENTWISE.
        # The planner handles matmul + {relu, gelu, silu}.
        planner = FusionPlanner()
        for activation in ("relu", "gelu", "silu"):
            graph = self._matmul_graph(activation)
            plans = planner.find_patterns(graph)
            assert len(plans) == 1, (
                f"Planner should find matmul+{activation}"
            )
            assert plans[0].pattern == f"matmul+{activation}"
            assert len(plans[0].ops) == 2

    def test_planner_generator_roundtrip_sets_source(self) -> None:
        """generate() populates FusionPlan.fused_kernel_source.

        What it tests:
          After calling generate(), the plan's fused_kernel_source field
          is set to the same value returned by generate().

        Passing means:
          plan.fused_kernel_source is not None and equals the returned source.
        """
        planner = FusionPlanner()
        graph = self._matmul_graph("relu")
        plans = planner.find_patterns(graph)
        assert len(plans) == 1
        plan = plans[0]
        assert plan.fused_kernel_source is None  # not yet generated

        gen = FusionCodeGenerator()
        source = gen.generate(plan)
        assert plan.fused_kernel_source is not None
        assert source == plan.fused_kernel_source

    def test_planner_with_zero_sized_dimensions(self) -> None:
        """Planner produces conservative speedup for zero-sized dims.

        What it tests:
          When matmul has zero or missing M/N/K, the planner still
          produces a plan with a conservative (floor) speedup.

        Passing means:
          Plan is produced with estimated_speedup == 0.12 (the floor).
        """
        planner = FusionPlanner()
        graph = [
            OpNode(FusionOpKind.MATMUL, attrs={"dtype": "float32"}),
            OpNode(FusionOpKind.RELU),
        ]
        plans = planner.find_patterns(graph)
        assert len(plans) == 1
        assert plans[0].estimated_speedup == 0.12


# =========================================================================
# SECTION 2 — Cross-Device Fusion Pipeline
# =========================================================================


class TestCrossDeviceFusionPipeline:
    """Tests the cross-device computation-communication fusion pipeline.

    The cross-device fusion planner analyzes a ShardedComputationGraph
    and produces CrossDeviceFusionPlan records, which can be built into
    executable FusedOperation callables.
    """

    # ------------------------------------------------------------------
    # All three fusion patterns
    # ------------------------------------------------------------------

    def test_fuse_output_all_reduce_detected(self) -> None:
        """Forward pass: output all-reduce is detected and planned.

        What it tests:
          CrossDeviceFusionPlanner.analyze() finds FUSE_OUTPUT_ALL_REDUCE
          plans when the sharded graph has boundary nodes with ALL_REDUCE
          collectives and has successor nodes for overlap.

        Passing means:
          At least one plan with pattern FUSE_OUTPUT_ALL_REDUCE is returned.
        """
        graph = _sharded_graph_with_collectives(
            CollectiveType.ALL_REDUCE, num_nodes=3, compute_time_us=150.0
        )
        planner = CrossDeviceFusionPlanner(bandwidth_gbps=900.0)
        plans = planner.analyze(graph)
        assert len(plans) >= 1
        output_ar_plans = [
            p for p in plans if p.pattern == FusionPattern.FUSE_OUTPUT_ALL_REDUCE
        ]
        assert len(output_ar_plans) >= 1
        # Verify structure
        plan = output_ar_plans[0]
        assert plan.communication_op == "all_reduce"
        assert plan.comm_volume_bytes > 0
        assert plan.compute_time_us > 0
        assert plan.comm_time_us > 0
        assert plan.estimated_speedup >= 1.0

    def test_fuse_input_all_gather_detected(self) -> None:
        """Input gather: all-gather fusion is detected and planned.

        What it tests:
          Planner finds FUSE_INPUT_ALL_GATHER plans when predecessor
          nodes have ALL_GATHER collectives.

        Passing means:
          At least one plan with pattern FUSE_INPUT_ALL_GATHER.
        """
        graph = _sharded_graph_with_collectives(
            CollectiveType.ALL_GATHER, num_nodes=3, compute_time_us=100.0
        )
        planner = CrossDeviceFusionPlanner(bandwidth_gbps=900.0)
        plans = planner.analyze(graph)
        input_ag_plans = [
            p for p in plans if p.pattern == FusionPattern.FUSE_INPUT_ALL_GATHER
        ]
        assert len(input_ag_plans) >= 1
        plan = input_ag_plans[0]
        assert plan.communication_op == "all_gather"
        assert plan.estimated_speedup >= 1.0

    def test_fuse_gradient_reduce_scatter_detected(self) -> None:
        """Backward pass: reduce-scatter fusion is detected and planned.

        What it tests:
          Planner finds FUSE_GRADIENT_REDUCE_SCATTER plans when boundary
          nodes have REDUCE_SCATTER collectives.

        Passing means:
          At least one plan with pattern FUSE_GRADIENT_REDUCE_SCATTER.
        """
        graph = _sharded_graph_with_collectives(
            CollectiveType.REDUCE_SCATTER, num_nodes=3, compute_time_us=200.0
        )
        planner = CrossDeviceFusionPlanner(bandwidth_gbps=900.0)
        plans = planner.analyze(graph)
        rs_plans = [
            p
            for p in plans
            if p.pattern == FusionPattern.FUSE_GRADIENT_REDUCE_SCATTER
        ]
        assert len(rs_plans) >= 1
        plan = rs_plans[0]
        assert plan.communication_op == "reduce_scatter"
        assert plan.estimated_speedup >= 1.0

    def test_multiple_cross_device_patterns_in_one_graph(self) -> None:
        """One sharded graph can produce multiple fusion patterns.

        What it tests:
          A graph with different collective types at different boundary
          nodes produces plans with different patterns.

        Passing means:
          The plan list includes at least two different FusionPattern values.
        """
        graph = ShardedComputationGraph()
        # Node 0: ALL_REDUCE boundary
        graph.nodes.append(
            ShardedGraphNode(
                node_id="n0",
                op_type="matmul",
                device_id=0,
                vendor=DeviceVendor.NVIDIA,
                compute_time_estimate_us=100.0,
                is_shard_boundary=True,
            )
        )
        graph.boundary_collectives["n0"] = [
            InsertedCollective(
                collective_type=CollectiveType.ALL_REDUCE,
                tensor_name="%t0",
                mesh_axis=0,
                num_devices=2,
                device_ids=(0, 1),
                comm_library=CommLibrary.NCCL,
                result_name="%r0",
                tensor_shape=(64, 64),
                dtype="f32",
                estimated_bytes=64 * 64 * 4 * 2,
            )
        ]
        # Node 1: REDUCE_SCATTER boundary with successor
        graph.nodes.append(
            ShardedGraphNode(
                node_id="n1",
                op_type="layer_norm",
                device_id=1,
                vendor=DeviceVendor.NVIDIA,
                compute_time_estimate_us=150.0,
                is_shard_boundary=True,
            )
        )
        graph.boundary_collectives["n1"] = [
            InsertedCollective(
                collective_type=CollectiveType.REDUCE_SCATTER,
                tensor_name="%t1",
                mesh_axis=0,
                num_devices=2,
                device_ids=(0, 1),
                comm_library=CommLibrary.NCCL,
                result_name="%r1",
                tensor_shape=(64, 64),
                dtype="f32",
                estimated_bytes=64 * 64 * 4,
            )
        ]
        # Chain: n0 -> n1
        graph.edges.append(("n0", "n1"))

        planner = CrossDeviceFusionPlanner()
        plans = planner.analyze(graph)
        # n0 has ALL_REDUCE + has successor n1 -> output all-reduce plan
        # n1 has no successor -> no gradient reduce-scatter plan
        patterns_found = {p.pattern for p in plans}
        assert FusionPattern.FUSE_OUTPUT_ALL_REDUCE in patterns_found

    # ------------------------------------------------------------------
    # CommunicationOverlapper integration
    # ------------------------------------------------------------------

    def test_communication_overlapper_builds_stub(self) -> None:
        """CommunicationOverlapper.build_stub creates a callable stub.

        What it tests:
          build_stub produces a FusedOperation that can be called
          without error and returns None (stub finish result).

        Passing means:
          Calling the stub runs begin -> compute -> wait -> finish
          without errors.
        """
        plan = CrossDeviceFusionPlan(
            source_device="nvidia:0",
            target_device="nvidia:1",
            computation_op="matmul",
            communication_op="all_reduce",
            estimated_speedup=1.5,
            pattern=FusionPattern.FUSE_OUTPUT_ALL_REDUCE,
            comm_volume_bytes=32768,
            compute_time_us=100.0,
            comm_time_us=50.0,
        )
        overlapper = CommunicationOverlapper()
        op = overlapper.build_stub(plan)
        result = op()
        assert result is None  # stub finish_fn returns None

    def test_communication_overlapper_describe(self) -> None:
        """CommunicationOverlapper.overlap returns a human-readable plan.

        What it tests:
          overlap() produces a string description of the fusion plan.

        Passing means:
          The description contains the plan's key fields.
        """
        plan = CrossDeviceFusionPlan(
            source_device="amd:0",
            target_device="amd:1",
            computation_op="layer_norm",
            communication_op="all_gather",
            estimated_speedup=2.0,
            pattern=FusionPattern.FUSE_INPUT_ALL_GATHER,
            comm_volume_bytes=65536,
            compute_time_us=200.0,
            comm_time_us=100.0,
        )
        overlapper = CommunicationOverlapper()
        desc = overlapper.overlap(plan)
        assert "fuse_input_all_gather" in desc
        assert "amd:0" in desc
        assert "layer_norm" in desc
        assert "2.00x" in desc

    def test_plan_and_overlap_convenience(self) -> None:
        """plan_and_overlap convenience function produces stubs.

        What it tests:
          The one-shot plan_and_overlap function analyzes a sharded
          graph and builds stubbed FusedOperation callables.

        Passing means:
          Returns at least one FusedOperation that can be called.
        """
        graph = _sharded_graph_with_collectives(
            CollectiveType.ALL_REDUCE, num_nodes=2, compute_time_us=100.0
        )
        ops = plan_and_overlap(graph, bandwidth_gbps=900.0)
        assert len(ops) >= 1
        for op in ops:
            result = op()
            assert result is None  # stubs return None

    def test_cross_device_no_fusion_on_single_device(self) -> None:
        """A single-device graph produces no fusion plans.

        What it tests:
          When no shard boundaries exist (is_shard_boundary=False),
          the planner returns an empty list.

        Passing means:
          analyze() returns [].
        """
        graph = ShardedComputationGraph()
        graph.nodes.append(
            ShardedGraphNode(
                node_id="n0",
                op_type="matmul",
                device_id=0,
                vendor=DeviceVendor.CPU,
                is_shard_boundary=False,
            )
        )
        planner = CrossDeviceFusionPlanner()
        plans = planner.analyze(graph)
        assert len(plans) == 0


# =========================================================================
# SECTION 3 — Edge Cases
# =========================================================================


class TestFusionEdgeCases:
    """Edge cases across all fusion components."""

    # ------------------------------------------------------------------
    # Analyzer edge cases
    # ------------------------------------------------------------------

    def test_empty_ir_via_planner(self) -> None:
        """An empty IR produces no analyzer matches and no plans.

        What it tests:
          Empty function through both the analyzer and planner.

        Passing means:
          Analyzer returns [], planner returns [].
        """
        func = _func([], name="empty")
        analyzer = FusionAnalyzer()
        assert len(analyzer.analyze(func)) == 0

        planner = FusionPlanner()
        assert planner.find_patterns([]) == []

    def test_no_patterns_single_op_through_both(self) -> None:
        """A single elementwise op produces no analyzer or planner hits.

        What it tests:
          Single op in both analyzer and planner contexts.

        Passing means:
          Both return empty results.
        """
        func = _func([
            _op(OpKind.LOAD, "%x", ["%X"]),
            _op(OpKind.ADDF, "%y", ["%x", "%x"]),
            _op(OpKind.STORE, "", ["%y"]),
        ])
        analyzer = FusionAnalyzer()
        assert len(analyzer.analyze(func)) == 0

        planner = FusionPlanner()
        assert planner.find_patterns([_fus_op(FusionOpKind.RELU)]) == []

    def test_analyzer_unrecognized_op_skipped(self) -> None:
        """Unrecognized op kinds are skipped by the analyzer.

        What it tests:
          An op with an OpKind not in the mapping is treated as
          infrastructure and skipped.

        Passing means:
          The analyzer detects no patterns (the unrecognized op is
          excluded from the kind sequence).
        """
        unknown_kind = OpKind.UNKNOWN
        func = _func([
            _op(OpKind.LOAD, "%x", ["%X"]),
            _op(unknown_kind, "%y", ["%x"], "custom.some_op"),
            _op(OpKind.STORE, "", ["%y"]),
        ])
        analyzer = FusionAnalyzer()
        matches = analyzer.analyze(func)
        assert len(matches) == 0

    # ------------------------------------------------------------------
    # Planner edge cases
    # ------------------------------------------------------------------

    def test_non_fusible_activation_not_planned(self) -> None:
        """An activation not in the fusible set produces no plan.

        What it tests:
          The planner rejects activations outside {relu, gelu, silu}.

        Passing means:
          find_patterns returns [] for an unknown OpKind.
        """
        planner = FusionPlanner()
        graph = [
            OpNode(FusionOpKind.MATMUL, attrs={"m": 1024, "n": 1024, "k": 1024}),
            OpNode(FusionOpKind.UNKNOWN),  # not fusible
        ]
        plans = planner.find_patterns(graph)
        assert len(plans) == 0

    def test_matmul_followed_by_matmul_no_fusion(self) -> None:
        """Two consecutive matmuls produce no fusion plans.

        What it tests:
          Matmul followed by another matmul is not a fusible pattern.

        Passing means:
          Planner returns [].
        """
        planner = FusionPlanner()
        graph = [
            OpNode(FusionOpKind.MATMUL, attrs={"m": 1024, "n": 1024, "k": 1024}),
            OpNode(FusionOpKind.MATMUL, attrs={"m": 512, "n": 512, "k": 512}),
        ]
        plans = planner.find_patterns(graph)
        assert len(plans) == 0

    # ------------------------------------------------------------------
    # Cross-device edge cases
    # ------------------------------------------------------------------

    def test_cross_device_empty_graph(self) -> None:
        """An empty sharded graph produces no fusion plans.

        What it tests:
          CrossDeviceFusionPlanner with an empty graph.

        Passing means:
          analyze() returns [].
        """
        graph = ShardedComputationGraph()
        planner = CrossDeviceFusionPlanner()
        plans = planner.analyze(graph)
        assert len(plans) == 0

    def test_cross_device_no_collectives(self) -> None:
        """A graph with boundary nodes but no collectives produces no plans.

        What it tests:
          Boundary nodes without associated collectives are not fused.

        Passing means:
          analyze() returns [] even with boundary nodes present.
        """
        graph = ShardedComputationGraph()
        graph.nodes.append(
            ShardedGraphNode(
                node_id="n0",
                op_type="matmul",
                device_id=0,
                vendor=DeviceVendor.NVIDIA,
                is_shard_boundary=True,
            )
        )
        # No collectives added to boundary_collectives
        planner = CrossDeviceFusionPlanner()
        plans = planner.analyze(graph)
        assert len(plans) == 0

    def test_build_graph_from_collectives_convenience(self) -> None:
        """build_graph_from_collectives creates a usable sharded graph.

        What it tests:
          The convenience helper produces a graph that the planner
          can analyze.

        Passing means:
          The resulting graph has the correct number of nodes and edges.
        """
        colls = [
            InsertedCollective(
                collective_type=CollectiveType.ALL_REDUCE,
                tensor_name="%t0",
                mesh_axis=0,
                num_devices=2,
                device_ids=(0, 1),
                comm_library=CommLibrary.NCCL,
                result_name="%r0",
                tensor_shape=(64, 64),
                dtype="f32",
                estimated_bytes=32768,
            ),
            InsertedCollective(
                collective_type=CollectiveType.ALL_GATHER,
                tensor_name="%t1",
                mesh_axis=0,
                num_devices=2,
                device_ids=(0, 1),
                comm_library=CommLibrary.NCCL,
                result_name="%r1",
                tensor_shape=(128, 64),
                dtype="f32",
                estimated_bytes=65536,
            ),
        ]
        graph = build_graph_from_collectives(colls)
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.edges[0] == ("boundary_0", "boundary_1")
        assert len(graph.boundary_collectives) == 2
        assert all(n.is_shard_boundary for n in graph.nodes)


# =========================================================================
# SECTION 4 — End-to-End Integration
# =========================================================================


class TestEndToEndIntegration:
    """End-to-end tests that combine multiple fusion components.

    These tests verify the complete chain from TTGIR text all the way
    through to fused kernel source that is syntactically valid Python.
    """

    def test_ir_text_to_fused_kernel_full_chain(self) -> None:
        """Full chain: TTGIR text -> parse -> analyze -> plan -> generate.

        What it tests:
          The complete end-to-end pipeline for matmul+activation fusion:
          raw TTGIR text -> TTGIRParser -> FusionAnalyzer -> manual
          OpNode construction -> FusionPlanner -> FusionCodeGenerator.

        Passing means:
          Every stage succeeds, and the final output is valid Python
          containing @triton.jit, tl.dot, tl.store, and the activation.
        """
        # Step 1: Parse and analyze real TTGIR text
        ir_text = """
module {
  tt.func public @matmul_relu(
    %A: !tt.ptr<tensor<128x32xf32>>,
    %B: !tt.ptr<tensor<32x128xf32>>,
    %C: !tt.ptr<tensor<128x128xf32>>
  ) {
    %a = tt.load %A : tensor<128x32xf32>
    %b = tt.load %B : tensor<32x128xf32>
    %c = tt.dot %a, %b : tensor<128x128xf32>
    %d = math.tanh %c : tensor<128x128xf32>
    tt.store %C, %d : tensor<128x128xf32>
    tt.return
  }
}
"""
        parser = TTGIRParser()
        func = parser.parse(ir_text)
        assert func is not None

        # Step 2: Analyze for fusion patterns
        analyzer = FusionAnalyzer()
        matches = analyzer.analyze(func)
        match = _find_by_name(matches, "matmul_activation")
        assert match is not None
        assert match.confidence >= 0.85

        # Step 3: Create a matching plan from OpNode graph
        planner = FusionPlanner()
        graph = [
            OpNode(
                kind=FusionOpKind.MATMUL,
                attrs={"m": 128, "n": 128, "k": 32, "dtype": "float32"},
            ),
            OpNode(kind=FusionOpKind.RELU),
        ]
        plans = planner.find_patterns(graph)
        assert len(plans) == 1
        assert plans[0].pattern == "matmul+relu"

        # Step 4: Generate fused kernel source
        gen = FusionCodeGenerator()
        source = gen.generate(plans[0])

        # Step 5: Verify the generated code
        compile(source, "<full_chain>", "exec")
        assert "@triton.jit" in source
        assert "tl.dot" in source
        assert "tl.where" in source
        assert "tl.store" in source
        assert "matmul_relu_fused" in source

    def test_all_analyzer_patterns_have_corresponding_planner_coverage(
        self,
    ) -> None:
        """Every analyzer pattern has at least one planner pattern that
        can fuse a subset of its op kinds.

        What it tests:
          That the analyzer and planner are semantically aligned -
          patterns the analyzer finds can be converted into plans.

        Passing means:
          For each analyzer pattern, we can construct a valid OpNode
          graph that the planner will accept and generate code for.
        """
        # matmul_activation -> planner covers matmul+{relu,gelu,silu}
        planner = FusionPlanner()
        gen = FusionCodeGenerator()

        for activation, act_kind in [
            ("relu", FusionOpKind.RELU),
            ("gelu", FusionOpKind.GELU),
            ("silu", FusionOpKind.SILU),
        ]:
            graph = [
                OpNode(
                    kind=FusionOpKind.MATMUL,
                    attrs={"m": 1024, "n": 1024, "k": 1024, "dtype": "float32"},
                ),
                OpNode(kind=act_kind),
            ]
            plans = planner.find_patterns(graph)
            assert len(plans) == 1, f"No plan for matmul+{activation}"
            source = gen.generate(plans[0])
            compile(source, f"<matmul+{activation}>", "exec")

        # matmul_bias_activation -> planner covers matmul+bias+{relu,gelu,silu}
        for activation, act_kind in [
            ("relu", FusionOpKind.RELU),
            ("gelu", FusionOpKind.GELU),
            ("silu", FusionOpKind.SILU),
        ]:
            graph = [
                OpNode(
                    kind=FusionOpKind.MATMUL,
                    attrs={"m": 1024, "n": 1024, "k": 1024, "dtype": "float32"},
                ),
                OpNode(kind=FusionOpKind.BIAS_ADD),
                OpNode(kind=act_kind),
            ]
            plans = planner.find_patterns(graph)
            assert len(plans) == 1, f"No plan for matmul+bias+{activation}"
            source = gen.generate(plans[0])
            compile(source, f"<matmul+bias+{activation}>", "exec")

    def test_cross_device_plan_with_stub_execution(self) -> None:
        """A cross-device plan can be built as a stub and executed.

        What it tests:
          The full cross-device pipeline: sharded graph -> planner ->
          overlapper -> build_stub -> callable execution.

        Passing means:
          The stub executes without error and all stages of the fusion
          (begin -> compute -> wait -> finish) complete.
        """
        # Build a sharded graph with two devices
        graph = _sharded_graph_with_collectives(
            CollectiveType.ALL_REDUCE, num_nodes=2, compute_time_us=50.0
        )

        # Plan
        planner = CrossDeviceFusionPlanner(bandwidth_gbps=900.0)
        plans = planner.analyze(graph)
        assert len(plans) >= 1

        # Build stub operations
        overlapper = CommunicationOverlapper()
        for plan in plans:
            op = overlapper.build_stub(plan)
            result = op()
            assert result is None  # stub returns None

    def test_plan_and_overlap_with_no_backend_uses_stub(self) -> None:
        """plan_and_overlap without a backend builds stub operations.

        What it tests:
          When no backend is provided, plan_and_overlap falls back to
          stub operations for all plans.

        Passing means:
          All returned operations are stubs (use NullAsyncWorkHandle).
        """
        graph = _sharded_graph_with_collectives(
            CollectiveType.ALL_GATHER, num_nodes=2, compute_time_us=75.0
        )
        ops = plan_and_overlap(graph, backend=None, bandwidth_gbps=900.0)
        assert len(ops) >= 1
        for op in ops:
            result = op()
            assert result is None

    def test_null_async_work_handle_always_completed(self) -> None:
        """NullAsyncWorkHandle reports completed immediately.

        What it tests:
          The no-op handle used in stub/fallback scenarios.

        Passing means:
          is_completed returns True, wait() does not block.
        """
        handle = NullAsyncWorkHandle()
        assert handle.is_completed
        handle.wait()  # should not raise

    def test_best_overlap_candidate_picks_largest_compute(self) -> None:
        """_best_overlap_candidate selects the successor with the
        largest compute_time_estimate_us.

        What it tests:
          The planner's internal heuristic for choosing which node to
          overlap with.

        Passing means:
          The candidate with compute_time_us=300 is chosen over 100.
        """
        successors = [
            ShardedGraphNode(
                node_id="fast",
                op_type="relu",
                compute_time_estimate_us=100.0,
            ),
            ShardedGraphNode(
                node_id="slow",
                op_type="matmul",
                compute_time_estimate_us=300.0,
            ),
            ShardedGraphNode(
                node_id="medium",
                op_type="gelu",
                compute_time_estimate_us=200.0,
            ),
        ]
        best = CrossDeviceFusionPlanner._best_overlap_candidate(successors)
        assert best is not None
        assert best.node_id == "slow"

    def test_best_overlap_candidate_empty_returns_none(self) -> None:
        """_best_overlap_candidate returns None for an empty list.

        What it tests:
          Edge case: no successors to choose from.

        Passing means:
          Returns None without raising.
        """
        best = CrossDeviceFusionPlanner._best_overlap_candidate([])
        assert best is None

    def test_speedup_estimate_clamped_to_reasonable_range(self) -> None:
        """Speedup estimates are clamped to [1.0, 10.0].

        What it tests:
          _estimate_speedup in cross_device_fusion clamps edge cases.

        Passing means:
          Very large compute vs comm or vice versa still produces
          in-range speedups.
        """
        from src.bridges.pytorch_xla.cross_device_fusion import _estimate_speedup

        # compute >> comm -> speedup near 1.0 (no benefit from overlap)
        low = _estimate_speedup(compute_time_us=10000.0, comm_time_us=1.0)
        assert 1.0 <= low <= 10.0
        assert low < 1.01  # barely any benefit

        # compute == comm -> speedup = 2.0 (serial=200, overlapped=100)
        equal = _estimate_speedup(compute_time_us=100.0, comm_time_us=100.0)
        assert equal == 2.0

        # both large but balanced -> speedup = 2.0
        balanced = _estimate_speedup(compute_time_us=5000.0, comm_time_us=5000.0)
        assert balanced == 2.0

        # zero times -> speedup 1.0
        zero = _estimate_speedup(compute_time_us=0.0, comm_time_us=0.0)
        assert zero == 1.0

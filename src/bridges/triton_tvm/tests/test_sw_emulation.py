"""Tests for software emulation of Nvidia-specific features.

Covers FP4, FP8, and Transformer Engine emulation across detection,
rewriting, and performance estimation.  All tests are pure (no GPU
required).
"""

from __future__ import annotations

import math

import pytest

from src.bridges.triton_tvm.sw_emulation import (
    FP4Emulation,
    FP8Emulation,
    EmulationPlan,
    ModelGraph,
    OpCategory,
    OpNode,
    SWEmulationEngine,
    TransformerEngineEmulation,
    build_graph_from_ops,
)


# ===================================================================
# FP4 Emulation
# ===================================================================


class TestFP4Emulation:
    """FP4 quantization simulation tests."""

    def test_simulate_fp4_quantize_zero(self) -> None:
        """Zero should remain zero."""
        result = FP4Emulation.simulate_fp4_quantize(0.0, 1.0)
        assert result == 0.0

    def test_simulate_fp4_quantize_basic(self) -> None:
        """A value of 1.0 with scale 1.0 should round to nearest FP4 value."""
        result = FP4Emulation.simulate_fp4_quantize(1.0, 1.0)
        # 1.0 is exactly representable in E2M1 (exp=10, mantissa=0)
        assert result == 1.0

    def test_simulate_fp4_quantize_clamp_max(self) -> None:
        """Values above FP4 max should clamp to max."""
        result = FP4Emulation.simulate_fp4_quantize(100.0, 1.0)
        assert result == 12.0

    def test_simulate_fp4_quantize_clamp_min(self) -> None:
        """Values below -FP4 max should clamp to -max."""
        result = FP4Emulation.simulate_fp4_quantize(-100.0, 1.0)
        assert result == -12.0

    def test_simulate_fp4_quantize_with_scale(self) -> None:
        """Scale factor should bring value into representable range."""
        # 10.0 / 2.0 = 5.0 → nearest FP4 value is 4.0 (E2M1: exp=11, mant=0)
        # → 4.0 * 2.0 = 8.0
        result = FP4Emulation.simulate_fp4_quantize(10.0, 2.0)
        assert result == pytest.approx(8.0)

    def test_emulate_fp4_linear_small(self) -> None:
        """A small 2x3 @ 3x2 matmul should produce valid output."""
        a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        b = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

        out = FP4Emulation.emulate_fp4_linear(a, b)
        assert len(out) == 2
        assert len(out[0]) == 2
        # All values should be finite
        for row in out:
            for val in row:
                assert math.isfinite(val)

    def test_emulate_fp4_linear_empty(self) -> None:
        """Empty inputs should return empty output."""
        assert FP4Emulation.emulate_fp4_linear([], [[1.0]]) == []
        assert FP4Emulation.emulate_fp4_linear([[1.0]], []) == []

    def test_estimate_performance_impact(self) -> None:
        """FP4 performance impact should be reasonable."""
        impact = FP4Emulation.estimate_performance_impact()
        assert 2.0 <= impact <= 8.0


# ===================================================================
# FP8 Emulation
# ===================================================================


class TestFP8Emulation:
    """FP8 quantization and matmul emulation tests."""

    def test_simulate_fp8_e4m3_zero(self) -> None:
        """Zero should remain zero."""
        result = FP8Emulation.simulate_fp8_quantize(0.0, 1.0, e5m2=False)
        assert result == 0.0

    def test_simulate_fp8_e4m3_basic(self) -> None:
        """1.0 is representable in E4M3 (exp=7, mantissa=0)."""
        result = FP8Emulation.simulate_fp8_quantize(1.0, 1.0, e5m2=False)
        assert result == 1.0

    def test_simulate_fp8_e4m3_clamp(self) -> None:
        """Values above E4M3 max should clamp."""
        result = FP8Emulation.simulate_fp8_quantize(1000.0, 1.0, e5m2=False)
        assert result == 448.0

    def test_simulate_fp8_e5m2_zero(self) -> None:
        """Zero should remain zero in E5M2."""
        result = FP8Emulation.simulate_fp8_quantize(0.0, 1.0, e5m2=True)
        assert result == 0.0

    def test_simulate_fp8_e5m2_scale(self) -> None:
        """Scale factor should bring value into representable range."""
        # 1000.0 / 50.0 = 20.0 → well within E5M2 range (max 57344)
        result = FP8Emulation.simulate_fp8_quantize(1000.0, 50.0, e5m2=True)
        assert result == pytest.approx(1000.0, rel=0.02)

    def test_emulate_fp8_matmul_small(self) -> None:
        """2x3 @ 3x2 matmul with FP8 quantization."""
        a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        b = [[1.0, 0.0], [0.0, 1.0], [1.0, 2.0]]

        out = FP8Emulation.emulate_fp8_matmul(a, b)
        assert len(out) == 2
        assert len(out[0]) == 2
        for row in out:
            for val in row:
                assert math.isfinite(val)

    def test_emulate_fp8_matmul_e5m2(self) -> None:
        """E5M2 format matmul should work."""
        a = [[1.0, 0.0], [0.0, 1.0]]
        b = [[1.0, 2.0], [3.0, 4.0]]

        out = FP8Emulation.emulate_fp8_matmul(a, b, e5m2=True)
        assert len(out) == 2
        assert len(out[0]) == 2

    def test_emulate_fp8_matmul_empty(self) -> None:
        """Empty inputs should return empty."""
        assert FP8Emulation.emulate_fp8_matmul([], [[1.0]]) == []
        assert FP8Emulation.emulate_fp8_matmul([[1.0]], []) == []

    def test_emulate_fp8_softmax_basic(self) -> None:
        """Softmax should produce valid probabilities."""
        logits = [1.0, 2.0, 3.0, 4.0]
        out = FP8Emulation.emulate_fp8_softmax(logits)
        assert len(out) == 4
        # Sum should be approximately 1.0
        assert sum(out) == pytest.approx(1.0, abs=0.02)

    def test_emulate_fp8_softmax_single(self) -> None:
        """Single-element softmax should return 1.0."""
        out = FP8Emulation.emulate_fp8_softmax([42.0])
        assert out[0] == pytest.approx(1.0, abs=0.01)

    def test_estimate_performance_impact(self) -> None:
        """FP8 performance impact should be reasonable."""
        impact_e4m3 = FP8Emulation.estimate_performance_impact(e5m2=False)
        impact_e5m2 = FP8Emulation.estimate_performance_impact(e5m2=True)
        assert 1.5 <= impact_e4m3 <= 5.0
        assert 1.5 <= impact_e5m2 <= 5.0


# ===================================================================
# Transformer Engine Emulation
# ===================================================================


class TestTransformerEngineEmulation:
    """Transformer Engine operation emulation tests."""

    def test_emulate_fp8_gemm_with_scaling_basic(self) -> None:
        """FP8 GEMM with delayed scaling should produce finite output."""
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[1.0, 0.0], [0.0, 1.0]]

        out, amax = TransformerEngineEmulation.emulate_fp8_gemm_with_scaling(a, b)
        assert len(out) == 2
        assert len(out[0]) == 2
        assert amax > 0.0
        for row in out:
            for val in row:
                assert math.isfinite(val)

    def test_emulate_fp8_gemm_with_scaling_empty(self) -> None:
        """Empty inputs should return empty output and zero amax."""
        out, amax = TransformerEngineEmulation.emulate_fp8_gemm_with_scaling([], [])
        assert out == []
        assert amax == 0.0

    def test_emulate_fp8_layer_norm_basic(self) -> None:
        """LayerNorm with FP8 output should work."""
        x = [1.0, 2.0, 3.0, 4.0]
        gamma = [1.0, 1.0, 1.0, 1.0]
        beta = [0.0, 0.0, 0.0, 0.0]

        out = TransformerEngineEmulation.emulate_fp8_layer_norm(x, gamma, beta)
        assert len(out) == 4
        for val in out:
            assert math.isfinite(val)

    def test_emulate_fp8_layer_norm_affine(self) -> None:
        """LayerNorm with non-trivial affine params."""
        x = [1.0, 2.0, 3.0]
        gamma = [2.0, 2.0, 2.0]
        beta = [1.0, 1.0, 1.0]

        out = TransformerEngineEmulation.emulate_fp8_layer_norm(x, gamma, beta)
        assert len(out) == 3

    def test_emulate_fp8_layer_norm_single(self) -> None:
        """Single-element LayerNorm should not crash."""
        out = TransformerEngineEmulation.emulate_fp8_layer_norm(
            [5.0], [1.0], [0.0]
        )
        assert len(out) == 1

    def test_emulate_delayed_scale_update_basic(self) -> None:
        """Delayed scaling from AMAX history."""
        amax_history = [0.1, 0.5, 1.0, 2.0]
        scale = TransformerEngineEmulation.emulate_delayed_scale_update(
            amax_history, fp8_max=448.0
        )
        assert scale > 0.0
        # max=2.0, safety=1.1 → divisor=2.2, scale≈203.6
        assert scale == pytest.approx(448.0 / (2.0 * 1.1), rel=0.01)

    def test_emulate_delayed_scale_update_empty(self) -> None:
        """Empty history should return scale=1.0."""
        scale = TransformerEngineEmulation.emulate_delayed_scale_update([])
        assert scale == 1.0

    def test_emulate_delayed_scale_update_all_zero(self) -> None:
        """All-zero history should return scale=1.0."""
        scale = TransformerEngineEmulation.emulate_delayed_scale_update(
            [0.0, 0.0, 0.0]
        )
        assert scale == 1.0

    def test_estimate_performance_impact(self) -> None:
        """TE performance impact should be reasonable."""
        impact = TransformerEngineEmulation.estimate_performance_impact()
        assert 2.0 <= impact <= 6.0


# ===================================================================
# ModelGraph
# ===================================================================


class TestModelGraph:
    """Computation graph construction and query tests."""

    def test_empty_graph(self) -> None:
        """Empty graph should have no nodes."""
        g = ModelGraph()
        assert len(g.nodes) == 0
        assert len(g.topological_order) == 0

    def test_add_node(self) -> None:
        """Adding a node should update nodes and topological_order."""
        g = ModelGraph()
        node = OpNode(name="test", category=OpCategory.MATMUL, dtype="fp32")
        g.add_node(node)
        assert "test" in g.nodes
        assert "test" in g.topological_order

    def test_find_by_category(self) -> None:
        """find_nodes_by_category should return matching nodes."""
        g = ModelGraph()
        g.add_node(OpNode(name="a", category=OpCategory.MATMUL))
        g.add_node(OpNode(name="b", category=OpCategory.ELEMENTWISE))
        g.add_node(OpNode(name="c", category=OpCategory.MATMUL))

        matmuls = g.find_nodes_by_category(OpCategory.MATMUL)
        assert len(matmuls) == 2
        assert {n.name for n in matmuls} == {"a", "c"}

    def test_find_by_dtype(self) -> None:
        """find_nodes_by_dtype should return matching nodes."""
        g = ModelGraph()
        g.add_node(OpNode(name="a", category=OpCategory.CAST, dtype="fp4"))
        g.add_node(OpNode(name="b", category=OpCategory.CAST, dtype="fp8_e4m3"))
        g.add_node(OpNode(name="c", category=OpCategory.ELEMENTWISE, dtype="fp32"))

        fp4_nodes = g.find_nodes_by_dtype("fp4")
        assert len(fp4_nodes) == 1
        assert fp4_nodes[0].name == "a"

        fp8_nodes = g.find_nodes_by_dtype("fp8_e4m3")
        assert len(fp8_nodes) == 1

    def test_find_by_attr(self) -> None:
        """find_nodes_by_attr should return matching nodes."""
        g = ModelGraph()
        g.add_node(
            OpNode(name="a", category=OpCategory.MATMUL, attributes={"delayed_scaling": True})
        )
        g.add_node(
            OpNode(name="b", category=OpCategory.MATMUL, attributes={"delayed_scaling": False})
        )

        matches = g.find_nodes_by_attr("delayed_scaling", True)
        assert len(matches) == 1
        assert matches[0].name == "a"


# ===================================================================
# Detection
# ===================================================================


class TestFeatureDetection:
    """Detection of Nvidia-specific features in computation graphs."""

    def test_no_nvidia_features(self) -> None:
        """Graph with only FP32 ops should detect nothing."""
        graph = build_graph_from_ops([
            {"name": "matmul_fp32", "category": "MATMUL", "dtype": "fp32"},
            {"name": "add", "category": "ELEMENTWISE", "dtype": "fp32"},
            {"name": "relu", "category": "ELEMENTWISE", "dtype": "fp32"},
        ])

        engine = SWEmulationEngine(auto_emulate=True)
        plans = engine.detect_nvidia_features(graph)

        assert all(not p.detected for p in plans)

    def test_detect_fp4_by_op_name(self) -> None:
        """FP4 operation names should be detected."""
        graph = build_graph_from_ops([
            {"name": "fp4_cast", "category": "CAST", "dtype": "fp4",
             "attributes": {"target_dtype": "fp4"}},
            {"name": "matmul", "category": "MATMUL", "dtype": "fp16"},
        ])

        engine = SWEmulationEngine(auto_emulate=True)
        plans = engine.detect_nvidia_features(graph)

        fp4_plan = next(p for p in plans if p.feature == "fp4")
        assert fp4_plan.detected

    def test_detect_fp4_by_dtype(self) -> None:
        """FP4 dtype on nodes should be detected."""
        graph = build_graph_from_ops([
            {"name": "quant", "category": "QUANTIZE", "dtype": "fp4"},
        ])

        engine = SWEmulationEngine(auto_emulate=True)
        plans = engine.detect_nvidia_features(graph)

        fp4_plan = next(p for p in plans if p.feature == "fp4")
        assert fp4_plan.detected

    def test_detect_fp8_by_op_name(self) -> None:
        """FP8 operation names should be detected."""
        graph = build_graph_from_ops([
            {"name": "tt.fp8_cast", "category": "CAST", "dtype": "fp8_e4m3"},
            {"name": "matmul", "category": "MATMUL", "dtype": "fp32"},
        ])

        engine = SWEmulationEngine(auto_emulate=True)
        plans = engine.detect_nvidia_features(graph)

        fp8_plan = next(p for p in plans if p.feature == "fp8")
        assert fp8_plan.detected

    def test_detect_fp8_by_dtype(self) -> None:
        """FP8 dtype should be detected."""
        graph = build_graph_from_ops([
            {"name": "te_gemm", "category": "MATMUL", "dtype": "fp8_e5m2"},
        ])

        engine = SWEmulationEngine(auto_emulate=True)
        plans = engine.detect_nvidia_features(graph)

        fp8_plan = next(p for p in plans if p.feature == "fp8")
        assert fp8_plan.detected

    def test_detect_transformer_engine(self) -> None:
        """Transformer Engine operations should be detected."""
        graph = build_graph_from_ops([
            {"name": "te.fp8_gemm", "category": "MATMUL",
             "dtype": "fp8_e4m3", "attributes": {"delayed_scaling": True}},
            {"name": "te.amax_update", "category": "ELEMENTWISE",
             "dtype": "fp32"},
        ])

        engine = SWEmulationEngine(auto_emulate=True)
        plans = engine.detect_nvidia_features(graph)

        te_plan = next(p for p in plans if p.feature == "transformer_engine")
        assert te_plan.detected

    def test_detect_transformer_engine_pattern(self) -> None:
        """FP8 matmuls + AMAX tracking should be detected as TE."""
        graph = build_graph_from_ops([
            {"name": "gemm_0", "category": "MATMUL", "dtype": "fp8_e4m3"},
            {"name": "gemm_1", "category": "MATMUL", "dtype": "fp8_e4m3"},
            {"name": "amax_tracker", "category": "ELEMENTWISE",
             "dtype": "fp32", "attributes": {"amax": 1.0}},
        ])

        engine = SWEmulationEngine(auto_emulate=True)
        plans = engine.detect_nvidia_features(graph)

        te_plan = next(p for p in plans if p.feature == "transformer_engine")
        assert te_plan.detected

    def test_detect_emulation_auto_on_non_nvidia(self) -> None:
        """Emulation should be auto-enabled when no Nvidia GPU."""
        graph = build_graph_from_ops([
            {"name": "tt.fp8_cast", "category": "CAST", "dtype": "fp8_e4m3"},
        ])

        engine = SWEmulationEngine(auto_emulate=True)
        plans = engine.detect_nvidia_features(graph)

        fp8_plan = next(p for p in plans if p.feature == "fp8")
        # This machine likely has no Nvidia GPU, so emulated should be True
        # (or if it does have one, the test still passes -- we check the flag is set)
        assert fp8_plan.emulated == (not engine._has_nvidia and engine.auto_emulate)

    def test_estimate_impact_no_features(self) -> None:
        """No detected features should give impact 1.0."""
        plans = [
            EmulationPlan(feature="fp4", detected=False, emulated=False),
            EmulationPlan(feature="fp8", detected=False, emulated=False),
            EmulationPlan(feature="transformer_engine", detected=False, emulated=False),
        ]
        engine = SWEmulationEngine(auto_emulate=True)
        impact = engine.estimate_impact(plans)
        assert impact == 1.0

    def test_estimate_impact_multiplicative(self) -> None:
        """Multiple emulations should compound multiplicatively."""
        plans = [
            EmulationPlan(feature="fp4", detected=True, emulated=True, performance_impact=4.0),
            EmulationPlan(feature="fp8", detected=True, emulated=True, performance_impact=2.0),
        ]
        engine = SWEmulationEngine(auto_emulate=True)
        impact = engine.estimate_impact(plans)
        assert impact == 8.0

    def test_estimate_impact_only_active(self) -> None:
        """Only emulated plans should contribute to impact."""
        plans = [
            EmulationPlan(feature="fp4", detected=True, emulated=True, performance_impact=4.0),
            EmulationPlan(feature="fp8", detected=True, emulated=False, performance_impact=2.0),
        ]
        engine = SWEmulationEngine(auto_emulate=True)
        impact = engine.estimate_impact(plans)
        assert impact == 4.0  # Only fp4 contributes


# ===================================================================
# Apply emulation
# ===================================================================


class TestApplyEmulation:
    """Graph rewriting tests."""

    def test_no_emulation_needed(self) -> None:
        """Graph with no detected features should pass through unchanged."""
        graph = build_graph_from_ops([
            {"name": "add", "category": "ELEMENTWISE", "dtype": "fp32"},
        ])
        plans = [EmulationPlan(feature="fp4", detected=False)]
        plans.append(EmulationPlan(feature="fp8", detected=False))
        plans.append(EmulationPlan(feature="transformer_engine", detected=False))

        engine = SWEmulationEngine(auto_emulate=True)
        result = engine.apply_emulation(plans, graph)

        assert "add" in result.nodes
        assert result.metadata.get("emulation_active") is None

    def test_emulate_fp4_node(self) -> None:
        """FP4 node should be rewritten with emulated op."""
        graph = build_graph_from_ops([
            {"name": "fp4_cast", "category": "CAST", "dtype": "fp4"},
            {"name": "matmul", "category": "MATMUL", "dtype": "fp16"},
        ])
        plans = [
            EmulationPlan(feature="fp4", detected=True, emulated=True),
            EmulationPlan(feature="fp8", detected=False, emulated=False),
            EmulationPlan(feature="transformer_engine", detected=False, emulated=False),
        ]

        engine = SWEmulationEngine(auto_emulate=True)
        result = engine.apply_emulation(plans, graph)

        # The FP4 node should be replaced
        fp4_emulated = [n for n in result.nodes.values() if "fp4_emulated" in n.name]
        assert len(fp4_emulated) >= 1
        assert fp4_emulated[0].attributes.get("emulation_type") == "fp4"

        # The non-FP4 node should pass through
        assert any(n.name == "matmul" for n in result.nodes.values())

    def test_emulate_fp8_node(self) -> None:
        """FP8 node should be rewritten with emulated op."""
        graph = build_graph_from_ops([
            {"name": "tt.fp8_cast", "category": "CAST", "dtype": "fp8_e4m3"},
        ])
        plans = [
            EmulationPlan(feature="fp4", detected=False, emulated=False),
            EmulationPlan(feature="fp8", detected=True, emulated=True),
            EmulationPlan(feature="transformer_engine", detected=False, emulated=False),
        ]

        engine = SWEmulationEngine(auto_emulate=True)
        result = engine.apply_emulation(plans, graph)

        fp8_emulated = [n for n in result.nodes.values() if "fp8_emulated" in n.name]
        assert len(fp8_emulated) >= 1

    def test_emulate_te_node(self) -> None:
        """Transformer Engine node should be rewritten with emulated op."""
        graph = build_graph_from_ops([
            {"name": "te.fp8_gemm", "category": "MATMUL", "dtype": "fp8_e4m3",
             "attributes": {"delayed_scaling": True}},
        ])
        plans = [
            EmulationPlan(feature="fp4", detected=False, emulated=False),
            EmulationPlan(feature="fp8", detected=False, emulated=False),
            EmulationPlan(feature="transformer_engine", detected=True, emulated=True),
        ]

        engine = SWEmulationEngine(auto_emulate=True)
        result = engine.apply_emulation(plans, graph)

        te_emulated = [n for n in result.nodes.values() if "te_emulated" in n.name]
        assert len(te_emulated) >= 1
        assert te_emulated[0].attributes.get("emulation_type") == "transformer_engine"

    def test_emulation_metadata(self) -> None:
        """Emulated graph should have metadata marker."""
        graph = build_graph_from_ops([
            {"name": "fp4_cast", "category": "CAST", "dtype": "fp4"},
        ])
        plans = [
            EmulationPlan(feature="fp4", detected=True, emulated=True),
        ]

        engine = SWEmulationEngine(auto_emulate=True)
        result = engine.apply_emulation(plans, graph)

        assert result.metadata.get("emulation_active") is True

    def test_multiple_emulations(self) -> None:
        """Multiple features should all be emulated."""
        graph = build_graph_from_ops([
            {"name": "fp4_cast", "category": "CAST", "dtype": "fp4"},
            {"name": "tt.fp8_cast", "category": "CAST", "dtype": "fp8_e4m3"},
            {"name": "te.fp8_gemm", "category": "MATMUL", "dtype": "fp8_e4m3",
             "attributes": {"delayed_scaling": True}},
        ])
        plans = [
            EmulationPlan(feature="fp4", detected=True, emulated=True),
            EmulationPlan(feature="fp8", detected=True, emulated=True),
            EmulationPlan(feature="transformer_engine", detected=True, emulated=True),
        ]

        engine = SWEmulationEngine(auto_emulate=True)
        result = engine.apply_emulation(plans, graph)

        emulated_names = [
            n.name for n in result.nodes.values() if "_emulated" in n.name
        ]
        assert len(emulated_names) >= 3


# ===================================================================
# EmulationPlan
# ===================================================================


class TestEmulationPlan:
    """EmulationPlan dataclass tests."""

    def test_defaults(self) -> None:
        """Default EmulationPlan should have no detection/emulation."""
        plan = EmulationPlan(feature="fp4")
        assert plan.feature == "fp4"
        assert not plan.detected
        assert not plan.emulated
        assert plan.performance_impact == 1.0

    def test_to_dict(self) -> None:
        """to_dict should return a clean dict."""
        plan = EmulationPlan(
            feature="fp8",
            detected=True,
            emulated=True,
            performance_impact=2.5,
            details="FP8 emulation active",
        )
        d = plan.to_dict()
        assert d["feature"] == "fp8"
        assert d["detected"] is True
        assert d["emulated"] is True
        assert d["performance_impact"] == 2.5


# ===================================================================
# Engine summary
# ===================================================================


class TestEngineSummary:
    """SWEmulationEngine summary output tests."""

    def test_get_summary_no_features(self) -> None:
        """Summary with no features."""
        plans = [
            EmulationPlan(feature="fp4", detected=False),
            EmulationPlan(feature="fp8", detected=False),
            EmulationPlan(feature="transformer_engine", detected=False),
        ]
        engine = SWEmulationEngine(auto_emulate=True)
        summary = engine.get_summary(plans)

        assert "has_nvidia_gpu" in summary
        assert "auto_emulate" in summary
        assert "emulation_active" in summary
        assert summary["emulation_active"] is False
        assert len(summary["plans"]) == 3
        assert summary["total_impact"] == 1.0

    def test_get_summary_active_emulation(self) -> None:
        """Summary with active emulation."""
        plans = [
            EmulationPlan(
                feature="fp8", detected=True, emulated=True, performance_impact=2.0
            ),
        ]
        engine = SWEmulationEngine(auto_emulate=True)
        summary = engine.get_summary(plans)

        assert summary["emulation_active"] is True
        assert summary["total_impact"] == 2.0


# ===================================================================
# build_graph_from_ops
# ===================================================================


class TestBuildGraph:
    """Convenience graph builder tests."""

    def test_simple_ops(self) -> None:
        """Building from simple op dicts should work."""
        graph = build_graph_from_ops([
            {"name": "a", "category": "MATMUL", "dtype": "fp32"},
            {"name": "b", "category": "ELEMENTWISE", "dtype": "fp16"},
        ])
        assert len(graph.nodes) == 2
        assert graph.nodes["a"].category == OpCategory.MATMUL
        assert graph.nodes["b"].dtype == "fp16"

    def test_with_attributes(self) -> None:
        """Attributes should be preserved."""
        graph = build_graph_from_ops([
            {
                "name": "test",
                "category": "MATMUL",
                "attributes": {"key": "value"},
                "inputs": ["x", "y"],
                "outputs": ["z"],
            },
        ])
        assert graph.nodes["test"].attributes["key"] == "value"
        assert graph.nodes["test"].inputs == ["x", "y"]

    def test_unknown_category(self) -> None:
        """Unknown category should become UNKNOWN."""
        graph = build_graph_from_ops([
            {"name": "weird", "category": "NONEXISTENT"},
        ])
        assert graph.nodes["weird"].category == OpCategory.UNKNOWN

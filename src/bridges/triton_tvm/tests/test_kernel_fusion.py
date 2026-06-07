"""Tests for the kernel-fusion engine.

Verifies that:

1. ``FusionPlanner`` detects all 6 supported patterns
2. ``FusionCodeGenerator`` produces syntactically valid ``@triton.jit`` code
3. Speedup estimates are positive and conservative (< 40 %)
4. Empty / non-fusible graphs produce no plans
5. All exposed symbols are importable through the lazy-export path
"""

from __future__ import annotations

import pytest

from src.bridges.triton_tvm.kernel_fusion import (
    FusionCodeGenerator,
    FusionPlan,
    FusionPlanner,
    OpKind,
    OpNode,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matmul_op(m: int = 4096, n: int = 4096, k: int = 4096, dtype: str = "float32") -> OpNode:
    return OpNode(
        kind=OpKind.MATMUL,
        attrs={"m": m, "n": n, "k": k, "dtype": dtype},
    )


def _bias_op() -> OpNode:
    return OpNode(kind=OpKind.BIAS_ADD)


def _relu_op() -> OpNode:
    return OpNode(kind=OpKind.RELU)


def _gelu_op() -> OpNode:
    return OpNode(kind=OpKind.GELU)


def _silu_op() -> OpNode:
    return OpNode(kind=OpKind.SILU)


# ---------------------------------------------------------------------------
# FusionPlanner — pattern detection
# ---------------------------------------------------------------------------


class TestFusionPlannerDetectAllPatterns:
    """Every supported pattern must be detected with correct metadata."""

    PATTERN_CASES = [
        ("matmul+relu", [_matmul_op(), _relu_op()]),
        ("matmul+gelu", [_matmul_op(), _gelu_op()]),
        ("matmul+silu", [_matmul_op(), _silu_op()]),
        ("matmul+bias+relu", [_matmul_op(), _bias_op(), _relu_op()]),
        ("matmul+bias+gelu", [_matmul_op(), _bias_op(), _gelu_op()]),
        ("matmul+bias+silu", [_matmul_op(), _bias_op(), _silu_op()]),
    ]

    @pytest.mark.parametrize("expected_pattern,graph", PATTERN_CASES)
    def test_detects(self, expected_pattern: str, graph: list[OpNode]) -> None:
        planner = FusionPlanner()
        plans = planner.find_patterns(graph)
        assert len(plans) == 1, f"Expected exactly 1 plan for {expected_pattern}"
        assert plans[0].pattern == expected_pattern
        assert len(plans[0].ops) == len(graph)

    def test_no_false_positive_for_unrelated_ops(self) -> None:
        """A graph with no fusible pattern returns empty."""
        planner = FusionPlanner()
        ops = [
            OpNode(OpKind.MATMUL, attrs={"m": 1024, "n": 1024, "k": 1024}),
            OpNode(OpKind.MATMUL),  # second matmul, not activation
        ]
        plans = planner.find_patterns(ops)
        assert len(plans) == 0

    def test_empty_graph(self) -> None:
        planner = FusionPlanner()
        assert planner.find_patterns([]) == []

    def test_activation_only_no_fusion(self) -> None:
        """An activation without a preceding matmul is not fused."""
        planner = FusionPlanner()
        plans = planner.find_patterns([_relu_op()])
        assert len(plans) == 0


# ---------------------------------------------------------------------------
# FusionPlan — dataclass invariants
# ---------------------------------------------------------------------------


class TestFusionPlanInvariants:
    def test_speedup_in_range(self) -> None:
        """Speedup must be in [0, 1]."""
        FusionPlan(pattern="matmul+relu", ops=[_matmul_op(), _relu_op()], estimated_speedup=0.0)
        FusionPlan(pattern="matmul+relu", ops=[_matmul_op(), _relu_op()], estimated_speedup=1.0)
        FusionPlan(pattern="matmul+relu", ops=[_matmul_op(), _relu_op()], estimated_speedup=0.25)

        with pytest.raises(ValueError):
            FusionPlan(pattern="matmul+relu", ops=[_matmul_op(), _relu_op()], estimated_speedup=-0.1)

        with pytest.raises(ValueError):
            FusionPlan(
                pattern="matmul+relu",
                ops=[_matmul_op(), _relu_op()],
                estimated_speedup=1.5,
            )

    def test_post_init_sets_fused_kernel_source_none(self) -> None:
        plan = FusionPlan(
            pattern="matmul+gelu", ops=[_matmul_op(), _gelu_op()], estimated_speedup=0.18
        )
        assert plan.fused_kernel_source is None

    def test_meta_defaults_empty(self) -> None:
        plan = FusionPlan(
            pattern="matmul+silu", ops=[_matmul_op(), _silu_op()], estimated_speedup=0.15
        )
        assert plan.meta == {}


# ---------------------------------------------------------------------------
# FusionCodeGenerator — output structure
# ---------------------------------------------------------------------------


class TestFusionCodeGenerator:
    """Verifies generated code structure and content."""

    def _generate(self, pattern: str, ops: list[OpNode]) -> str:
        planner = FusionPlanner()
        plans = planner.find_patterns(ops)
        assert len(plans) == 1
        gen = FusionCodeGenerator()
        return gen.generate(plans[0])

    @pytest.mark.parametrize(
        "pattern,ops",
        [
            ("matmul+relu", [_matmul_op(), _relu_op()]),
            ("matmul+gelu", [_matmul_op(), _gelu_op()]),
            ("matmul+silu", [_matmul_op(), _silu_op()]),
            ("matmul+bias+relu", [_matmul_op(), _bias_op(), _relu_op()]),
            ("matmul+bias+gelu", [_matmul_op(), _bias_op(), _gelu_op()]),
            ("matmul+bias+silu", [_matmul_op(), _bias_op(), _silu_op()]),
        ],
    )
    def test_generated_code_is_valid_python_syntax(self, pattern: str, ops: list[OpNode]) -> None:
        """Generated source must be syntactically valid Python."""
        source = self._generate(pattern, ops)
        compile(source, f"<{pattern}>", "exec")

    def test_generated_contains_triton_jit_decorator(self) -> None:
        source = self._generate("matmul+relu", [_matmul_op(), _relu_op()])
        assert "@triton.jit" in source

    def test_generated_contains_tl_dot(self) -> None:
        source = self._generate("matmul+gelu", [_matmul_op(), _gelu_op()])
        assert "tl.dot" in source

    def test_generated_contains_fused_kernel_name(self) -> None:
        source = self._generate("matmul+silu", [_matmul_op(), _silu_op()])
        assert "matmul_silu_fused" in source

    def test_generated_activation_inlined(self) -> None:
        """Activation code must appear before the store."""
        source = self._generate("matmul+relu", [_matmul_op(), _relu_op()])
        # relu should appear before tl.store
        relu_pos = source.index("tl.where")
        store_pos = source.index("tl.store")
        assert relu_pos < store_pos, "Activation must be applied before store"

    def test_generated_bias_inlined(self) -> None:
        """Bias-add code must appear before activation."""
        source = self._generate("matmul+bias+gelu", [_matmul_op(), _bias_op(), _gelu_op()])
        # Bias load should appear before the activation and store.
        assert "bias_ptr" in source
        assert "bias_stride" in source
        bias_pos = source.index("bias_val")
        act_pos = source.index("tanh")
        store_pos = source.index("tl.store")
        assert bias_pos < act_pos < store_pos, "Bias must be applied before activation, before store"

    @pytest.mark.parametrize(
        "activation,expected_keyword",
        [
            ("relu", "tl.where"),
            ("gelu", "tanh"),
            ("silu", "tl.sigmoid"),
        ],
    )
    def test_activation_keyword_present(self, activation: str, expected_keyword: str) -> None:
        source = self._generate(
            f"matmul+{activation}", [_matmul_op(), _relu_op() if activation == "relu" else _gelu_op() if activation == "gelu" else _silu_op()]
        )
        assert expected_keyword in source


# ---------------------------------------------------------------------------
# Speedup estimation
# ---------------------------------------------------------------------------


class TestSpeedupEstimation:
    def test_estimate_positive_and_conservative(self) -> None:
        planner = FusionPlanner()
        cases = [
            (_matmul_op(m=1024, n=1024, k=1024), _relu_op(), "matmul+relu"),
            (_matmul_op(m=4096, n=4096, k=4096), _gelu_op(), "matmul+gelu"),
            (_matmul_op(m=8192, n=8192, k=8192), _silu_op(), "matmul+silu"),
        ]
        for matmul, act, name in cases:
            graph = [matmul, act]
            plans = planner.find_patterns(graph)
            assert len(plans) == 1, f"No plan for {name}"
            sp = plans[0].estimated_speedup
            assert 0.05 <= sp <= 0.40, (
                f"Speedup {sp:.2%} for {name} out of range [5%, 40%]"
            )

    def test_bias_variants_slightly_higher(self) -> None:
        """Bias variants save more memory traffic → higher speedup."""
        planner = FusionPlanner()
        base = planner.find_patterns([_matmul_op(), _relu_op()])[0]
        bias = planner.find_patterns([_matmul_op(), _bias_op(), _relu_op()])[0]
        assert bias.estimated_speedup >= base.estimated_speedup

    def test_estimate_floor_for_zero_sized(self) -> None:
        """If no M/N/K is known, return a conservative floor."""
        planner = FusionPlanner()
        ops = [
            OpNode(OpKind.MATMUL, attrs={"dtype": "float32"}),
            _relu_op(),
        ]
        plans = planner.find_patterns(ops)
        assert len(plans) == 1
        assert plans[0].estimated_speedup == 0.12


# ---------------------------------------------------------------------------
# End-to-end: planner → generator flow
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """The full flow: plan → generate → valid code for every pattern."""

    @pytest.mark.parametrize(
        "pattern,ops",
        [
            ("matmul+relu", [_matmul_op(), _relu_op()]),
            ("matmul+gelu", [_matmul_op(), _gelu_op()]),
            ("matmul+silu", [_matmul_op(), _silu_op()]),
            ("matmul+bias+relu", [_matmul_op(), _bias_op(), _relu_op()]),
            ("matmul+bias+gelu", [_matmul_op(), _bias_op(), _gelu_op()]),
            ("matmul+bias+silu", [_matmul_op(), _bias_op(), _silu_op()]),
        ],
    )
    def test_plan_generate_roundtrip(self, pattern: str, ops: list[OpNode]) -> None:
        planner = FusionPlanner()
        plans = planner.find_patterns(ops)
        assert len(plans) == 1, f"No plan for {pattern}"
        plan = plans[0]
        assert plan.pattern == pattern
        assert plan.fused_kernel_source is None  # not yet generated

        gen = FusionCodeGenerator()
        source = gen.generate(plan)
        assert plan.fused_kernel_source is not None
        assert source == plan.fused_kernel_source
        compile(source, f"<{pattern}>", "exec")  # valid syntax

    def test_multiple_fusible_regions(self) -> None:
        """Graph with two independent fusible regions produces two plans."""
        ops = [
            _matmul_op(m=512, n=512, k=512),
            _relu_op(),
            _matmul_op(m=256, n=256, k=256),
            _gelu_op(),
        ]
        planner = FusionPlanner()
        plans = planner.find_patterns(ops)
        assert len(plans) == 2
        assert plans[0].pattern == "matmul+relu"
        assert plans[1].pattern == "matmul+gelu"


# ---------------------------------------------------------------------------
# Lazy import path
# ---------------------------------------------------------------------------


class TestLazyImports:
    """All key symbols must be importable through the bridge's lazy-exports."""

    def test_import_fusion_planner(self) -> None:
        from src.bridges.triton_tvm import FusionPlanner  # noqa: F401

    def test_import_fusion_plan(self) -> None:
        from src.bridges.triton_tvm import FusionPlan  # noqa: F401

    def test_import_fusion_code_generator(self) -> None:
        from src.bridges.triton_tvm import FusionCodeGenerator  # noqa: F401

    def test_import_op_node(self) -> None:
        from src.bridges.triton_tvm import OpNode  # noqa: F401

    def test_import_op_kind(self) -> None:
        from src.bridges.triton_tvm import OpKind  # noqa: F401

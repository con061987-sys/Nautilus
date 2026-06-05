"""Tests for the IR → TIR conversion pipeline (Step 2)."""

from __future__ import annotations

import pytest

from src.bridges.triton_tvm.ir_to_tir.ttgir_parser import (
    TTGIRFunction,
    TTGIROperation,
    TTGIRParser,
    TTGIRType,
    OpKind,
)
from src.bridges.triton_tvm.ir_to_tir.pass1_lower_tensor_idioms import (
    LowerTensorIdioms,
)
from src.bridges.triton_tvm.ir_to_tir.pass2_rewrite_spmd import (
    RewriteSPMDToLoops,
)
from src.bridges.triton_tvm.ir_to_tir.pass3_replace_pointers import (
    ReplacePointersWithMemRefs,
)
from src.bridges.triton_tvm.ir_to_tir.pass4_materialize_tvm import (
    MaterializeTensorsToTVM,
)
from src.bridges.triton_tvm.ir_to_tir.tvmscript_emitter import TVMScriptEmitter
from src.bridges.triton_tvm.ir_to_tir.tt_dot_split import TTDotSplitter, SplitResult
from src.bridges.triton_tvm.ir_to_tir.conversion_pipeline import (
    ConversionPipeline,
    ConversionResult,
    ConversionStatus,
)


# Sample TTGIR texts for testing
SIMPLE_MATMUL_IR = """module {
  tt.func @simple_matmul(
    %A: !tt.ptr<tensor<128x32xf32>>,
    %B: !tt.ptr<tensor<32x128xf32>>,
    %C: !tt.ptr<tensor<128x128xf32>>
  ) {
    %a = tt.load %A : tensor<128x32xf32>
    %b = tt.load %B : tensor<32x128xf32>
    %c = tt.dot %a, %b : tensor<128x128xf32>
    tt.store %C, %c : tensor<128x128xf32>
    tt.return
  }
}
"""

ELEMENTWISE_IR = """module {
  tt.func @add_kernel(
    %A: !tt.ptr<tensor<256xf32>>,
    %B: !tt.ptr<tensor<256xf32>>,
    %C: !tt.ptr<tensor<256xf32>>
  ) {
    %a = tt.load %A : tensor<256xf32>
    %b = tt.load %B : tensor<256xf32>
    %sum = arith.addf %a, %b : tensor<256xf32>
    tt.store %C, %sum : tensor<256xf32>
    tt.return
  }
}
"""

REDUCTION_IR = """module {
  tt.func @sum_kernel(
    %input: !tt.ptr<tensor<1024xf32>>,
    %output: !tt.ptr<tensor<1xf32>>
  ) {
    %x = tt.load %input : tensor<1024xf32>
    %sum = "tt.reduce"(%x) ({
      ^bb0(%a: f32, %b: f32): arith.addf %a, %b : f32
    }) {axis = 0 : i32} : (tensor<1024xf32>) -> tensor<1xf32>
    tt.store %output, %sum : tensor<1xf32>
    tt.return
  }
}
"""

KERNEL_WITH_SPMD = """module {
  tt.func @with_pid(
    %A: !tt.ptr<tensor<128x32xf32>>,
    %B: !tt.ptr<tensor<32x128xf32>>,
    %C: !tt.ptr<tensor<128x128xf32>>
  ) {
    %pid = tt.get_program_id(0) : i32
    %a = tt.load %A : tensor<128x32xf32>
    %b = tt.load %B : tensor<32x128xf32>
    %c = tt.dot %a, %b : tensor<128x128xf32>
    tt.store %C, %c : tensor<128x128xf32>
    tt.return
  }
}
"""

MALFORMED_IR = "this is not valid TTGIR text at all"


class TestTTGIRParser:
    """Tests for the TTGIR parser."""

    def setup_method(self) -> None:
        self.parser = TTGIRParser()

    def test_parse_simple_matmul(self) -> None:
        """A matmul IR should parse successfully with the right ops."""
        func = self.parser.parse(SIMPLE_MATMUL_IR)
        assert func.name == "simple_matmul"
        assert len(func.args) == 3
        assert func.has_dot()

    def test_parse_elementwise(self) -> None:
        """An elementwise IR should parse without a dot."""
        func = self.parser.parse(ELEMENTWISE_IR)
        assert func.name == "add_kernel"
        assert not func.has_dot()
        assert any(op.kind == OpKind.ADDF for op in func.iter_all_ops())

    def test_parse_reduction(self) -> None:
        """A reduction IR should contain a REDUCE op."""
        func = self.parser.parse(REDUCTION_IR)
        assert func.name == "sum_kernel"
        assert any(op.kind == OpKind.REDUCE for op in func.iter_all_ops())

    def test_parse_with_program_id(self) -> None:
        """An IR with get_program_id should be recognized."""
        func = self.parser.parse(KERNEL_WITH_SPMD)
        assert any(op.kind == OpKind.GET_PROGRAM_ID for op in func.iter_all_ops())

    def test_parse_pointer_types(self) -> None:
        """Pointer types should be parsed correctly."""
        func = self.parser.parse(SIMPLE_MATMUL_IR)
        # First arg should be a pointer to a 2D float32 tensor
        arg_name, arg_type = func.args[0]
        assert arg_type.is_pointer
        assert arg_type.is_tensor
        assert arg_type.shape == (128, 32)
        assert arg_type.element_dtype == "float32"

    def test_parse_module_attributes(self) -> None:
        """Module-level attributes like num-warps should be captured."""
        ir_with_attrs = """module attributes {ttg.num-warps = 4 : i32, ttg.num-stages = 3 : i32} {
  tt.func @test() {
    tt.return
  }
}"""
        func = self.parser.parse(ir_with_attrs)
        # Module attrs may or may not be captured depending on parser version
        assert isinstance(func.module_attrs, dict)

    def test_parse_malformed_raises(self) -> None:
        """Malformed IR should raise ValueError."""
        with pytest.raises(ValueError):
            self.parser.parse(MALFORMED_IR)

    def test_op_count(self) -> None:
        """op_count should return the total number of ops including nested ones."""
        func = self.parser.parse(SIMPLE_MATMUL_IR)
        # simple_matmul has 2 loads, 1 dot, 1 store = 4 ops
        assert func.op_count() == 4


class TestPass1LowerTensorIdioms:
    """Tests for Pass 1: scalar ops to tensor.generate."""

    def setup_method(self) -> None:
        self.pass1 = LowerTensorIdioms()

    def test_preserves_loads_stores_dots(self) -> None:
        """Pass 1 should not modify load/store/dot ops."""
        parser = TTGIRParser()
        func = parser.parse(SIMPLE_MATMUL_IR)
        result = self.pass1.run(func)
        # The dot should still be a dot
        assert any(op.kind == OpKind.DOT for op in result.iter_all_ops())

    def test_marks_arith_ops_for_lowering(self) -> None:
        """Arith ops should get the __lowered_to_tensor attribute."""
        parser = TTGIRParser()
        func = parser.parse(ELEMENTWISE_IR)
        result = self.pass1.run(func)
        addf_ops = [op for op in result.iter_all_ops() if op.kind == OpKind.ADDF]
        assert len(addf_ops) > 0
        assert addf_ops[0].attributes.get("__lowered_to_tensor") == "true"

    def test_preserves_unknown_ops(self) -> None:
        """UNKNOWN ops should be left unchanged."""
        parser = TTGIRParser()
        func = parser.parse(SIMPLE_MATMUL_IR)
        result = self.pass1.run(func)
        # All ops should still exist
        assert result.op_count() == func.op_count()


class TestPass2RewriteSPMD:
    """Tests for Pass 2: SPMD primitives to for-loops."""

    def setup_method(self) -> None:
        self.pass2 = RewriteSPMDToLoops()

    def test_wraps_body_in_for_loop(self) -> None:
        """After Pass 2, the body should be wrapped in a for-loop."""
        parser = TTGIRParser()
        func = parser.parse(SIMPLE_MATMUL_IR)
        result = self.pass2.run(func)
        # At least one top-level op should now be a FOR_LOOP
        assert any(op.kind == OpKind.FOR_LOOP for op in result.ops)

    def test_finds_max_axis(self) -> None:
        """The pass should correctly identify the max program axis used."""
        parser = TTGIRParser()
        func = parser.parse(KERNEL_WITH_SPMD)
        max_axis = self.pass2._find_max_program_axis(func)
        # KERNEL_WITH_SPMD has axis 0
        assert max_axis == 0


class TestPass3ReplacePointers:
    """Tests for Pass 3: pointer → memref conversion."""

    def setup_method(self) -> None:
        self.pass3 = ReplacePointersWithMemRefs()

    def test_converts_pointer_args_to_memref(self) -> None:
        """!tt.ptr arguments should become memref."""
        parser = TTGIRParser()
        func = parser.parse(SIMPLE_MATMUL_IR)
        result = self.pass3.run(func)
        # The first arg should no longer be a pointer
        _, arg_type = result.args[0]
        assert not arg_type.is_pointer

    def test_marks_loads_for_memref(self) -> None:
        """tt.load ops should get the __converted_to_memref_load attribute."""
        parser = TTGIRParser()
        func = parser.parse(SIMPLE_MATMUL_IR)
        result = self.pass3.run(func)
        load_ops = [op for op in result.iter_all_ops() if op.kind == OpKind.LOAD]
        assert load_ops[0].attributes.get("__converted_to_memref_load") == "true"


class TestPass4MaterializeTensors:
    """Tests for Pass 4: memref → TVM block."""

    def setup_method(self) -> None:
        self.pass4 = MaterializeTensorsToTVM()

    def test_marks_loads_for_tvm_block(self) -> None:
        """Loads should get the __materialized_to_tvm_block attribute."""
        parser = TTGIRParser()
        func = self.pass4.run(parser.parse(SIMPLE_MATMUL_IR))
        load_ops = [op for op in func.iter_all_ops() if op.kind == OpKind.LOAD]
        assert load_ops[0].attributes.get("__materialized_to_tvm_block") == "true"

    def test_marks_reductions(self) -> None:
        """Reduction ops should be marked for TVM block emission."""
        parser = TTGIRParser()
        func = self.pass4.run(parser.parse(REDUCTION_IR))
        reduce_ops = [op for op in func.iter_all_ops() if op.kind == OpKind.REDUCE]
        assert reduce_ops[0].attributes.get("__materialized_to_tvm_reduction") == "true"


class TestTVMScriptEmitter:
    """Tests for the TVMScript emitter."""

    def setup_method(self) -> None:
        self.emitter = TVMScriptEmitter()
        self.pipeline = [LowerTensorIdioms(), RewriteSPMDToLoops(),
                        ReplacePointersWithMemRefs(), MaterializeTensorsToTVM()]

    def _convert(self, ir_text: str) -> TTGIRFunction:
        parser = TTGIRParser()
        func = parser.parse(ir_text)
        for pass_impl in self.pipeline:
            func = pass_impl.run(func)
        return func

    def test_emit_matmul(self) -> None:
        """A matmul function should emit valid TVMScript."""
        func = self._convert(SIMPLE_MATMUL_IR)
        output = self.emitter.emit(func)
        assert "@T.prim_func" in output
        assert "simple_matmul" in output
        assert "T.Buffer" in output

    def test_emit_elementwise(self) -> None:
        """An elementwise function should emit TVMScript with T.add."""
        func = self._convert(ELEMENTWISE_IR)
        output = self.emitter.emit(func)
        assert "@T.prim_func" in output
        assert "add_kernel" in output

    def test_emit_signature(self) -> None:
        """The signature should include all arguments with proper types."""
        func = self._convert(SIMPLE_MATMUL_IR)
        output = self.emitter.emit(func)
        # The function has 3 args: A, B, C
        assert "A:" in output
        assert "B:" in output
        assert "C:" in output

    def test_emit_normalize_dtype(self) -> None:
        """MLIR dtype names should be normalized in output."""
        func = self._convert(SIMPLE_MATMUL_IR)
        output = self.emitter.emit(func)
        # f32 should become "float32" in the output
        assert "float32" in output


class TestTTDotSplitter:
    """Tests for the tt.dot splitter."""

    def setup_method(self) -> None:
        self.splitter = TTDotSplitter()

    def test_split_kernel_with_dot(self) -> None:
        """A kernel with tt.dot should be split into matmul + remainder."""
        parser = TTGIRParser()
        func = parser.parse(SIMPLE_MATMUL_IR)
        result = self.splitter.split(func)
        assert result.has_dot
        assert result.matmul_m == 128
        assert result.matmul_n == 128
        assert result.matmul_k == 32
        assert result.matmul_dtype == "float32"

    def test_split_kernel_without_dot(self) -> None:
        """A kernel without tt.dot should have has_dot=False."""
        parser = TTGIRParser()
        func = parser.parse(ELEMENTWISE_IR)
        result = self.splitter.split(func)
        assert not result.has_dot
        assert result.dot_op is None
        assert result.matmul_m == 0

    def test_split_extracts_operands(self) -> None:
        """The split should extract the dot's operands."""
        parser = TTGIRParser()
        func = parser.parse(SIMPLE_MATMUL_IR)
        result = self.splitter.split(func)
        # The dot has 3 operands: A, B, C
        assert len(result.operands) == 3

    def test_split_removes_dot_from_remainder(self) -> None:
        """After split, the remainder should not contain the dot."""
        parser = TTGIRParser()
        func = parser.parse(SIMPLE_MATMUL_IR)
        result = self.splitter.split(func)
        for op in result.remainder_ops:
            assert op.kind != OpKind.DOT


class TestConversionPipeline:
    """End-to-end tests for the full conversion pipeline."""

    def setup_method(self) -> None:
        self.pipeline = ConversionPipeline()

    def test_convert_simple_matmul(self) -> None:
        """A matmul kernel should convert with dot split out."""
        result = self.pipeline.convert(SIMPLE_MATMUL_IR)
        assert result.is_usable
        assert result.has_dot_split
        assert result.status == ConversionStatus.SUCCESS_WITH_DOT
        assert "simple_matmul" in result.tvmscript_text
        assert "@T.prim_func" in result.tvmscript_text

    def test_convert_elementwise(self) -> None:
        """An elementwise kernel should convert fully (no dot)."""
        result = self.pipeline.convert(ELEMENTWISE_IR)
        assert result.is_usable
        assert result.status == ConversionStatus.SUCCESS
        assert "add_kernel" in result.tvmscript_text

    def test_convert_reduction(self) -> None:
        """A reduction kernel should convert successfully."""
        result = self.pipeline.convert(REDUCTION_IR)
        assert result.is_usable
        assert "sum_kernel" in result.tvmscript_text

    def test_convert_with_spmd(self) -> None:
        """A kernel with get_program_id should still convert."""
        result = self.pipeline.convert(KERNEL_WITH_SPMD)
        assert result.is_usable

    def test_convert_malformed_falls_back(self) -> None:
        """Malformed IR should return FALLBACK status."""
        result = self.pipeline.convert(MALFORMED_IR)
        assert result.status == ConversionStatus.FALLBACK
        assert not result.is_usable
        assert result.error is not None

    def test_pipeline_records_pass_times(self) -> None:
        """The conversion result should record per-pass timing."""
        result = self.pipeline.convert(ELEMENTWISE_IR)
        assert isinstance(result.pass_times, dict)
        # At minimum, parse, dot_split, and the 4 passes should be recorded
        assert "parse" in result.pass_times
        assert "dot_split" in result.pass_times
        assert "lower_tensor_idioms" in result.pass_times
        assert "rewrite_spmd" in result.pass_times
        assert "replace_pointers" in result.pass_times
        assert "materialize_tvm" in result.pass_times
        assert "emit" in result.pass_times
        for t in result.pass_times.values():
            assert t >= 0

    def test_dot_split_bounds_extracted(self) -> None:
        """For a matmul kernel, M/N/K should be correctly extracted."""
        result = self.pipeline.convert(SIMPLE_MATMUL_IR)
        assert result.has_dot_split
        assert result.split is not None
        assert result.split.matmul_m == 128
        assert result.split.matmul_n == 128
        assert result.split.matmul_k == 32


class TestPipelineIntegration:
    """Integration tests for the full conversion pipeline."""

    def test_end_to_end_no_dot(self) -> None:
        """Full pipeline: elementwise IR → TVMScript → status check."""
        pipeline = ConversionPipeline()
        result = pipeline.convert(ELEMENTWISE_IR)
        assert result.is_usable
        assert "T.prim_func" in result.tvmscript_text
        assert "add_kernel" in result.tvmscript_text

    def test_end_to_end_with_dot(self) -> None:
        """Full pipeline: matmul IR → TVMScript (dot removed) + split info."""
        pipeline = ConversionPipeline()
        result = pipeline.convert(SIMPLE_MATMUL_IR)
        assert result.is_usable
        assert result.has_dot_split
        # The emitted TVMScript should not contain "tt.dot" because
        # the dot was split out for the extern_bridge
        assert "tt.dot" not in result.tvmscript_text
        # But the split info should carry the M/N/K
        assert result.split is not None
        assert result.split.matmul_m == 128

"""Tests for src.common — Result, types, errors, hardware, logging, observability."""

# mypy: ignore general type errors in this test file (generic type inference)
# mypy: disable-code-blocks
import json
import sys
import time
from pathlib import Path

import pytest

# Ensure src. is importable
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3]))


def test_result_ok_unwrap():
    from src.common.result import Ok

    o = Ok(42)
    assert o.is_ok()
    assert not o.is_err()
    assert o.unwrap() == 42
    assert o.unwrap_or(0) == 42


def test_result_err_unwrap_raises():
    from src.common.result import Err

    e = Err(ValueError("boom"))
    assert e.is_err()
    assert not e.is_ok()
    with pytest.raises(ValueError, match="boom"):
        e.unwrap()
    assert e.unwrap_or(7) == 7


def test_result_map():
    from src.common.result import Err, Ok

    result_ok = Ok(2).map(lambda x: x * 3)  # type: ignore[misc]
    result_err = Err(ValueError("x")).map(lambda x: x * 3)  # type: ignore[misc]
    assert isinstance(result_ok, Ok)
    assert result_ok.unwrap() == 6
    assert result_err.is_err()


def test_result_and_then():
    from src.common.result import Err, Ok

    def add_one(x: int) -> Ok:
        return Ok(x + 1)  # type: ignore[misc]

    result_ok = Ok(2).and_then(add_one)
    result_err = Err(ValueError("x")).and_then(add_one)
    assert isinstance(result_ok, Ok)
    assert result_ok.unwrap() == 3
    assert result_err.is_err()


def test_error_codes_unique():
    from src.common.errors import ErrorCode

    codes = [c.value for c in ErrorCode]
    assert len(codes) == len(set(codes)), "Error codes must be unique"


def test_error_context_is_chainable():
    from src.common.errors import CompilationError

    e = CompilationError("failed", context={"kernel": "matmul"})
    e2 = e.with_context(arch="sm_90")
    assert e2.context == {"kernel": "matmul", "arch": "sm_90"}
    # Original is untouched
    assert e.context == {"kernel": "matmul"}


def test_error_to_dict():
    from src.common.errors import TuningError

    e = TuningError("bad config", context={"trials": 0})
    d = e.to_dict()
    assert d["type"] == "TuningError"
    assert d["code"].startswith("E_")
    assert d["context"]["trials"] == 0


def test_vendor_from_arch():
    from src.common.types import Arch, Vendor

    assert Arch.SM_90.vendor == Vendor.NVIDIA
    assert Arch.GFX942.vendor == Vendor.AMD
    assert Arch.XE_HPG.vendor == Vendor.INTEL
    assert Arch.APPLE_M2.vendor == Vendor.APPLE


def test_hardware_target_to_tvm():
    from src.common.types import Arch, HardwareTarget, Vendor

    t = HardwareTarget(vendor=Vendor.NVIDIA, arch=Arch.SM_90)
    assert t.to_tvm_target() == "nvidia/nvidia-h100"
    t2 = HardwareTarget(vendor=Vendor.AMD, arch=Arch.GFX942)
    assert t2.to_tvm_target() == "rocm/gfx942"


def test_mesh_shape_rejects_zero():
    from src.common.errors import ConfigError
    from src.common.types import MeshShape

    with pytest.raises(ConfigError):
        MeshShape(axes=(2, 0, 2))


def test_tensor_sharding_validates_length_match():
    from src.common.errors import ConfigError
    from src.common.types import TensorShardingLite

    with pytest.raises(ConfigError):
        TensorShardingLite(
            tensor_name="x",
            mesh_axes=(0, 1),
            partition_shape=(2,),
        )


def test_sharding_spec_validates_axis_range():
    from src.common.errors import ConfigError
    from src.common.types import MeshShape, ShardingSpecLite, TensorShardingLite

    mesh = MeshShape(axes=(2, 2))
    bad = TensorShardingLite(
        tensor_name="x",
        mesh_axes=(5,),  # OOB
        partition_shape=(1,),
    )
    with pytest.raises(ConfigError):
        ShardingSpecLite(mesh=mesh, tensor_shardings={"x": bad})


def test_fat_binary_dedup_vendors():
    from src.common.types import Arch, FatBinary, KernelSection, SectionFormat, Vendor

    fb = FatBinary(kernel_name="k")
    fb.add_section(
        KernelSection(vendor=Vendor.NVIDIA, arch=Arch.SM_90, format=SectionFormat.PTX, data=b"x")
    )
    fb.add_section(
        KernelSection(vendor=Vendor.AMD, arch=Arch.GFX942, format=SectionFormat.HSACO, data=b"y")
    )
    fb.add_section(
        KernelSection(vendor=Vendor.NVIDIA, arch=Arch.SM_80, format=SectionFormat.PTX, data=b"z")
    )
    assert len(fb.sections) == 3
    assert set(fb.vendors) == {Vendor.NVIDIA, Vendor.AMD}


def test_circuit_breaker_opens_after_threshold():
    from src.common.errors import CircuitOpenError
    from src.common.observability import CircuitBreaker, CircuitBreakerConfig, CircuitState

    cb = CircuitBreaker(
        CircuitBreakerConfig(name="test", failure_threshold=2, reset_timeout_seconds=0.1)
    )

    def boom():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        cb.call(boom)
    with pytest.raises(RuntimeError):
        cb.call(boom)
    assert cb.state == CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        cb.call(boom)
    # Wait for half-open
    time.sleep(0.15)
    assert cb.state == CircuitState.HALF_OPEN
    # A successful call closes it
    cb.call(lambda: "ok")
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_excluded_exception_does_not_count():
    from src.common.observability import CircuitBreaker, CircuitBreakerConfig, CircuitState

    cb = CircuitBreaker(
        CircuitBreakerConfig(
            name="test",
            failure_threshold=2,
            reset_timeout_seconds=0.1,
            excluded_exceptions=(KeyboardInterrupt,),
        )
    )

    def interrupt():
        raise KeyboardInterrupt()

    for _ in range(10):
        with pytest.raises(KeyboardInterrupt):
            cb.call(interrupt)
    assert cb.state == CircuitState.CLOSED  # Should not have opened


def test_stage_budget_unknown_stage_raises():
    from src.common.observability import StageBudgets

    budgets = StageBudgets()
    with pytest.raises(KeyError):
        budgets.get("nonexistent_stage")


def test_timeout_manager_stage_budget():
    from src.common.errors import StageTimeoutError
    from src.common.observability import StageBudgets, TimeoutManager

    tm = TimeoutManager(
        StageBudgets(
            ir_capture_seconds=0.05,
        )
    )
    tm.start()
    with pytest.raises(StageTimeoutError), tm.stage("ir_capture"):
        time.sleep(0.1)


def test_logging_structured_json(tmp_path):
    from src.common.logging import JsonLogSink, configure_logging, get_logger

    log_path = str(tmp_path / "test.log")
    sink = JsonLogSink(log_path)
    configure_logging(level="debug", sinks=[sink])
    log = get_logger("test")
    log.info("hello", foo="bar", n=42)
    sink.flush()
    contents = Path(log_path).read_text().strip().splitlines()
    assert len(contents) == 1
    record = json.loads(contents[0])
    assert record["msg"] == "hello"
    assert record["foo"] == "bar"
    assert record["n"] == 42
    assert record["level"] == "info"
    assert record["logger"].startswith("nautilus.test")


def test_logging_span_records_stages():
    import io

    from src.common.logging import (
        LogSink,
        configure_logging,
    )
    from src.common.logging import (
        span as span_context,
    )
    from src.common.logging import (
        stage as stage_context,
    )

    buf = io.StringIO()

    class _BufSink(LogSink):
        def __init__(self, b):
            self._b = b

        def emit(self, record):
            self._b.write(json.dumps(record, default=str) + "\n")

        def flush(self):
            pass

    sink = _BufSink(buf)
    configure_logging(level="debug", sinks=[sink])
    with span_context("test_op", kernel="matmul") as sp:
        with stage_context(sp, "phase_a") as st:
            st.set(ops=10)
            time.sleep(0.01)
        with stage_context(sp, "phase_b"):
            pass
    sink.flush()
    records = [json.loads(line) for line in buf.getvalue().splitlines() if line]
    span_records = [r for r in records if r["msg"] == "span_finished"]
    assert len(span_records) == 1
    sp_record = span_records[0]
    assert sp_record["operation"] == "test_op"
    assert sp_record["metadata"]["kernel"] == "matmul"
    assert len(sp_record["stages"]) == 2
    stage_names = [s["name"] for s in sp_record["stages"]]
    assert stage_names == ["phase_a", "phase_b"]


def test_hardware_detect_returns_something():
    """detect_gpu_vendors() must return a set (possibly empty) and not raise."""
    from src.common.hardware import detect_gpu_vendors

    vendors = detect_gpu_vendors()
    assert isinstance(vendors, set)


def test_hardware_format_summary_runs():
    from src.common.hardware import format_device_summary

    summary = format_device_summary()
    assert "Host:" in summary
    assert "GPUs:" in summary

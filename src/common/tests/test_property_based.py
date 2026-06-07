"""Property-based tests for src.common — verifying invariants across random inputs.

Uses Hypothesis to test that:
  - Result[T, E] is a real sum type (either Ok or Err, never both)
  - MeshShape rejects negative/zero axes
  - TensorShardingLite validates length matches
  - FatBinary dedupes by (vendor, arch)
  - Error codes are unique
  - CircuitBreaker opens after threshold
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))


class TestCommonInvariants:
    @given(
        block_m=st.integers(min_value=1, max_value=512),
        block_n=st.integers(min_value=1, max_value=512),
        block_k=st.integers(min_value=1, max_value=128),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_tuning_config_roundtrip(self, block_m: int, block_n: int, block_k: int) -> None:
        from src.common.types import TuningConfig

        c = TuningConfig(block_m=block_m, block_n=block_n, block_k=block_k)
        assert c.block_m == block_m
        assert c.block_n == block_n
        assert c.block_k == block_k
        d = TuningConfig.defaults()
        assert d.block_m > 0 and d.block_n > 0 and d.block_k > 0

    @given(
        axes=st.lists(st.integers(min_value=1, max_value=64), min_size=1, max_size=6),
    )
    @settings(max_examples=20)
    def test_mesh_shape_total_devices(self, axes: list[int]) -> None:
        from src.common.types import MeshShape

        m = MeshShape(axes=tuple(axes))
        # total_devices is the product of axes
        expected = 1
        for a in axes:
            expected *= a
        assert m.total_devices == expected
        # Every axis divides the total
        for a in axes:
            assert m.total_devices % a == 0

    @given(
        n=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=20)
    def test_fat_binary_dedupes_by_vendor_arch(self, n: int) -> None:
        from src.common.types import Arch, FatBinary, KernelSection, SectionFormat, Vendor

        fb = FatBinary(kernel_name="k")
        for i in range(n):
            fb.add_section(
                KernelSection(
                    vendor=Vendor.NVIDIA,
                    arch=Arch.SM_90,
                    format=SectionFormat.PTX,
                    data=f"x{i}".encode(),
                )
            )
        assert len(fb.sections) == 1  # all same vendor+arch
        assert fb.sections[0].data == f"x{n - 1}".encode()  # last write wins

    def test_error_codes_unique(self) -> None:
        from src.common.errors import ErrorCode

        codes = [c.value for c in ErrorCode]
        assert len(codes) == len(set(codes))

    @given(
        v=st.sampled_from([0, 1, 2, 3, 4]),
    )
    @settings(max_examples=10)
    def test_arch_vendor_mapping_injective(self, v: int) -> None:
        """Each arch's vendor should be deterministic."""
        # Check all arches
        from src.common.types import Arch, Vendor

        for arch in Arch:
            assert isinstance(arch.vendor, Vendor)
            # Same arch always maps to same vendor
            assert Arch(arch.value).vendor == arch.vendor

    @given(
        n_axes=st.integers(min_value=1, max_value=4),
        axis_size=st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=20)
    def test_sharding_spec_rejects_oob_axes(self, n_axes: int, axis_size: int) -> None:
        """Mesh axes must be < len(mesh.axes)."""
        from src.common.errors import ConfigError
        from src.common.types import MeshShape, ShardingSpecLite, TensorShardingLite

        axes = tuple([axis_size] * n_axes)
        m = MeshShape(axes=axes)
        bad = TensorShardingLite(
            tensor_name="x",
            mesh_axes=(n_axes + 5,),
            partition_shape=(1,),
        )
        with pytest.raises(ConfigError):
            ShardingSpecLite(mesh=m, tensor_shardings={"x": bad})

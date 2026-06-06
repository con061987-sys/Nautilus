"""Tests for the pointer flow analyzer.

Covers the Wave 2.7 alias-tracking work documented in
docs/cuda_ingestion_architecture.md:
  - restrict-qualified parameters are disjoint
  - unrestricted parameters form a single alias set
  - local pointers inherit alias info from their initializer
  - shared memory allocations are disjoint from all pointers
"""

from __future__ import annotations

from typing import Any

import pytest

from src.bridges.cuda_ingest.parser import CudaParser
from src.bridges.cuda_ingest.pointer_analysis import (
    AliasInfo,
    PointerAnalyzer,
    PointerLayout,
)

SAMPLE_RESTRICT_KERNEL = '''
__global__ void restrict_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
'''


SAMPLE_ALIASED_KERNEL = '''
__global__ void aliased_kernel(
    const float* A,
    const float* B,
    float* C,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
'''


SAMPLE_LOCAL_ASSIGN_KERNEL = '''
__global__ void local_assign_kernel(
    const float* __restrict__ A,
    float* C,
    int n
) {
    const float* p = A;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = p[i] + 1.0f;
    }
}
'''


SAMPLE_SHMEM_KERNEL = '''
__global__ void smem_kernel(
    const float* A,
    float* C,
    int n
) {
    __shared__ float tile[16][16];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        tile[0][0] = A[i];
        C[i] = tile[0][0];
    }
}
'''


def make_kernel(source: str) -> Any:
    parser = CudaParser()
    kernels = parser.parse_source(source)
    return kernels[0] if kernels else None


class TestPointerAnalyzer:
    """Tests for the PointerAnalyzer class."""

    def test_analyzer_init(self) -> None:
        analyzer = PointerAnalyzer()
        assert analyzer is not None

    def test_restrict_pointers_are_disjoint(self) -> None:
        """Two __restrict__-qualified parameters must not alias each other."""
        kernel = make_kernel(SAMPLE_RESTRICT_KERNEL)
        analyzer = PointerAnalyzer()
        alias_map = analyzer.analyze_aliases(kernel)

        assert "A" in alias_map
        assert "B" in alias_map
        assert "C" in alias_map
        assert alias_map["A"].is_restrict
        assert alias_map["B"].is_restrict
        assert alias_map["C"].is_restrict

        # No pair of restrict pointers shares an alias entry.
        for n1, info1 in alias_map.items():
            for n2 in info1.may_alias:
                info2 = alias_map[n2]
                assert n1 not in info2.may_alias or n1 == n2, (
                    f"restrict pointers {n1} and {n2} should not alias"
                )

    def test_unrestricted_parameters_form_alias_set(self) -> None:
        """Unrestricted parameters must all alias each other."""
        kernel = make_kernel(SAMPLE_ALIASED_KERNEL)
        analyzer = PointerAnalyzer()
        alias_map = analyzer.analyze_aliases(kernel)

        assert not alias_map["A"].is_restrict
        assert not alias_map["B"].is_restrict
        # A and B should alias each other (and C).
        assert "B" in alias_map["A"].may_alias
        assert "C" in alias_map["A"].may_alias
        assert "A" in alias_map["B"].may_alias

    def test_local_pointer_inherits_alias_info(self) -> None:
        """`float* p = A;` makes p inherit A's alias info."""
        kernel = make_kernel(SAMPLE_LOCAL_ASSIGN_KERNEL)
        analyzer = PointerAnalyzer()
        alias_map = analyzer.analyze_aliases(kernel)

        # p should be in the alias map (created by the assign propagation)
        assert "p" in alias_map, "local pointer p should be tracked"
        assert "A" in alias_map["p"].may_alias
        # Inheriting from a restrict pointer makes p restrict.
        assert alias_map["p"].is_restrict

    def test_shmem_disjoint_from_pointers(self) -> None:
        """__shared__ allocations must never appear in pointer alias sets."""
        kernel = make_kernel(SAMPLE_SHMEM_KERNEL)
        analyzer = PointerAnalyzer()
        alias_map = analyzer.analyze_aliases(kernel)

        for name, info in alias_map.items():
            assert "tile" not in info.may_alias, (
                f"shared memory 'tile' must not alias with pointer {name}"
            )

    def test_aliases_nothing_property(self) -> None:
        """`aliases_nothing` is True only for restrict pointers that may not
        alias with any other pointer (the C99 restrict contract)."""
        kernel = make_kernel(SAMPLE_RESTRICT_KERNEL)
        analyzer = PointerAnalyzer()
        alias_map = analyzer.analyze_aliases(kernel)

        # All three restrict pointers are disjoint from every other restrict
        # pointer, so they `aliases_nothing`.
        for name, info in alias_map.items():
            assert info.aliases_nothing, (
                f"restrict pointer {name} should aliases_nothing"
            )

    def test_layouts_recorded_for_parameters(self) -> None:
        """analyze_kernel should produce a PointerLayout for each pointer param."""
        kernel = make_kernel(SAMPLE_RESTRICT_KERNEL)
        analyzer = PointerAnalyzer()
        layouts = analyzer.analyze_kernel(kernel)
        assert "A" in layouts
        assert "B" in layouts
        assert "C" in layouts
        for layout in layouts.values():
            assert isinstance(layout, PointerLayout)
            assert layout.element_type in ("float",)

    def test_restrict_flag_propagates_to_layout(self) -> None:
        """PointerLayout.is_restrict must reflect the __restrict__ qualifier."""
        kernel = make_kernel(SAMPLE_RESTRICT_KERNEL)
        analyzer = PointerAnalyzer()
        layouts = analyzer.analyze_kernel(kernel)
        assert layouts["A"].is_restrict
        assert layouts["B"].is_restrict
        assert layouts["C"].is_restrict

    def test_non_restrict_layout(self) -> None:
        """A non-restrict parameter must produce is_restrict=False layout."""
        kernel = make_kernel(SAMPLE_ALIASED_KERNEL)
        analyzer = PointerAnalyzer()
        layouts = analyzer.analyze_kernel(kernel)
        assert not layouts["A"].is_restrict
        assert not layouts["B"].is_restrict

    def test_alias_info_dataclass_fields(self) -> None:
        """AliasInfo must expose the documented fields."""
        info = AliasInfo(
            name="p",
            is_restrict=True,
            is_parameter=True,
            provenance="kernel parameter",
        )
        assert info.name == "p"
        assert info.is_restrict is True
        assert info.is_parameter is True
        assert info.provenance == "kernel parameter"
        assert info.aliases_nothing is True

    def test_empty_kernel_no_aliases(self) -> None:
        """A kernel with no parameters has an empty alias map."""
        source = '''
__global__ void empty() {
    int i = threadIdx.x;
}
'''
        kernel = make_kernel(source)
        analyzer = PointerAnalyzer()
        alias_map = analyzer.analyze_aliases(kernel)
        assert alias_map == {}

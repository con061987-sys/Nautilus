"""Tests for the shared memory analyzer."""

from __future__ import annotations

import pytest

from src.bridges.cuda_ingest.shared_memory import (
    SharedMemoryAnalyzer,
    SharedMemPlan,
    SharedMemAllocation,
)
from src.bridges.cuda_ingest.parser import (
    CudaParser,
    CudaKernel,
    CudaStatement,
)


SAMPLE_KERNEL_WITH_SHMEM = '''
__global__ void matmul_with_shmem(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float tile_a[16][16];
    __shared__ float tile_b[16][16];
    __shared__ float accum[256];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        for (int k = 0; k < K; k++) {
            tile_a[threadIdx.y][threadIdx.x] = A[row * K + k];
            tile_b[threadIdx.y][threadIdx.x] = B[k * N + col];
            __syncthreads();
        }
    }
}
'''


def make_kernel_with_shmem() -> CudaKernel:
    """Parse a sample kernel with shared memory declarations."""
    parser = CudaParser()
    kernels = parser.parse_source(SAMPLE_KERNEL_WITH_SHMEM)
    return kernels[0]


class TestSharedMemoryAnalyzer:
    """Tests for the SharedMemoryAnalyzer class."""

    def test_analyzer_init(self) -> None:
        """SharedMemoryAnalyzer should initialise without error."""
        analyzer = SharedMemoryAnalyzer()
        assert analyzer is not None

    def test_analyze_static_shared(self) -> None:
        """A kernel with static __shared__ arrays should be analyzed."""
        kernel = make_kernel_with_shmem()
        analyzer = SharedMemoryAnalyzer()
        plan = analyzer.analyze(kernel)
        assert plan.num_allocations >= 3  # tile_a, tile_b, accum
        for alloc in plan.allocations:
            assert alloc.is_static
            assert alloc.array_size > 0
            assert alloc.element_type == "float"

    def test_total_bytes_calculation(self) -> None:
        """total_bytes should estimate the memory usage."""
        kernel = make_kernel_with_shmem()
        analyzer = SharedMemoryAnalyzer()
        plan = analyzer.analyze(kernel)
        # tile_a is 16*16 floats = 1024 bytes
        # tile_b is 16*16 floats = 1024 bytes
        # accum is 256 floats = 1024 bytes
        # Total: 3072 bytes
        assert plan.total_static_bytes == 3072

    def test_bank_conflict_detection(self) -> None:
        """Arrays that are multiples of 32 should trigger bank conflict warnings."""
        analyzer = SharedMemoryAnalyzer()
        # Create allocations that will trigger warnings
        plan = SharedMemPlan()
        plan.allocations = [
            SharedMemAllocation(name="data", element_type="float", array_size=64, is_static=True, raw_declaration=""),
            SharedMemAllocation(name="other", element_type="float", array_size=128, is_static=True, raw_declaration=""),
        ]
        warnings = analyzer._analyze_bank_conflicts(plan.allocations)
        assert len(warnings) > 0

    def test_generate_triton_allocation_static(self) -> None:
        """Static allocation should produce tl.zeros."""
        analyzer = SharedMemoryAnalyzer()
        alloc = SharedMemAllocation(
            name="tile", element_type="float", array_size=256,
            is_static=True, raw_declaration="",
        )
        code = analyzer.generate_triton_allocation(alloc)
        assert "tile_smem" in code
        assert "tl.zeros" in code
        assert "256" in code
        assert "tl.float32" in code

    def test_generate_triton_allocation_dynamic(self) -> None:
        """Dynamic allocation should use BLOCK_SIZE."""
        analyzer = SharedMemoryAnalyzer()
        alloc = SharedMemAllocation(
            name="data", element_type="float", array_size=0,
            is_static=False, raw_declaration="",
        )
        code = analyzer.generate_triton_allocation(alloc)
        assert "BLOCK_SIZE" in code
        assert "tl.zeros" in code

    def test_generate_all_allocations(self) -> None:
        """generate_all_allocations should produce code for all allocations."""
        analyzer = SharedMemoryAnalyzer()
        plan = SharedMemPlan()
        plan.allocations = [
            SharedMemAllocation("a", "float", 64, True, ""),
            SharedMemAllocation("b", "int", 32, True, ""),
        ]
        code = analyzer.generate_all_allocations(plan)
        assert "a_smem" in code
        assert "b_smem" in code

    def test_no_shared_memory(self) -> None:
        """A kernel without __shared__ should produce an empty plan."""
        source = '''
__global__ void no_shmem(float* x) {
    int i = threadIdx.x;
    x[i] = 1.0f;
}
'''
        parser = CudaParser()
        kernels = parser.parse_source(source)
        analyzer = SharedMemoryAnalyzer()
        plan = analyzer.analyze(kernels[0])
        assert plan.num_allocations == 0
        assert plan.total_static_bytes == 0

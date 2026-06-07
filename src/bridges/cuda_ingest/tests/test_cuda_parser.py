"""Tests for the CUDA C++ parser."""

from __future__ import annotations

from src.bridges.cuda_ingest.parser import (
    CudaParser,
    CudaStatementType,
)

SAMPLE_MATMUL_CUDA = """
#include <cuda_runtime.h>

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    __shared__ float tile[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }

    __syncthreads();
    atomicAdd(&counter, 1);
}

__device__ float helper(float x) {
    return x * 2.0f;
}
"""


SAMPLE_SIMPLE_KERNEL = """
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
"""


class TestCudaParser:
    """Tests for the CUDA C++ parser."""

    def test_parser_init(self) -> None:
        """CudaParser should initialise without error."""
        parser = CudaParser()
        assert parser is not None

    def test_parse_simple_kernel(self) -> None:
        """A simple kernel should parse to one CudaKernel."""
        parser = CudaParser()
        kernels = parser.parse_source(SAMPLE_SIMPLE_KERNEL)
        assert len(kernels) == 1
        kernel = kernels[0]
        assert kernel.name == "vector_add"
        assert kernel.is_global
        assert kernel.num_params == 4

    def test_parse_matmul_kernel(self) -> None:
        """A matmul kernel should parse with correct params and shared mem."""
        parser = CudaParser()
        kernels = parser.parse_source(SAMPLE_MATMUL_CUDA)
        # Should have 2 kernels: matmul_kernel (global) and helper (device)
        assert len(kernels) == 2
        matmul = kernels[0]
        assert matmul.name == "matmul_kernel"
        assert matmul.is_global
        # 6 params: const float* A, const float* B, float* C, int M, int N, int K
        assert matmul.num_params == 6
        assert matmul.num_params >= 5

    def test_parse_shared_memory(self) -> None:
        """__shared__ declarations should be extracted."""
        parser = CudaParser()
        kernels = parser.parse_source(SAMPLE_MATMUL_CUDA)
        matmul = kernels[0]
        assert len(matmul.shared_mem_declarations) >= 1
        shared = matmul.shared_mem_declarations[0]
        assert shared["name"] == "tile"

    def test_parse_sync_threads(self) -> None:
        """__syncthreads() should be identified as a statement."""
        parser = CudaParser()
        kernels = parser.parse_source(SAMPLE_MATMUL_CUDA)
        matmul = kernels[0]
        sync_stmts = [s for s in matmul.body if s.stmt_type == CudaStatementType.SYNC_THREADS]
        assert len(sync_stmts) >= 1

    def test_parse_atomic_op(self) -> None:
        """atomicAdd should be identified as an atomic op."""
        parser = CudaParser()
        kernels = parser.parse_source(SAMPLE_MATMUL_CUDA)
        matmul = kernels[0]
        atomic_stmts = [s for s in matmul.body if s.stmt_type == CudaStatementType.ATOMIC_OP]
        assert len(atomic_stmts) >= 1
        assert "atomicAdd" in atomic_stmts[0].raw_text

    def test_parse_device_function(self) -> None:
        """__device__ functions should be identified as device, not global."""
        parser = CudaParser()
        kernels = parser.parse_source(SAMPLE_MATMUL_CUDA)
        # Find the helper function
        device_funcs = [k for k in kernels if k.is_device]
        assert len(device_funcs) == 1
        assert device_funcs[0].name == "helper"

    def test_parse_empty_source(self) -> None:
        """Empty source should return empty kernel list."""
        parser = CudaParser()
        kernels = parser.parse_source("")
        assert kernels == []

    def test_parse_no_kernels(self) -> None:
        """Source without __global__ should return empty kernel list."""
        parser = CudaParser()
        source = "int main() { return 0; }"
        kernels = parser.parse_source(source)
        assert kernels == []

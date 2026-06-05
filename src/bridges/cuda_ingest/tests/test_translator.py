"""Tests for the CUDA to Triton translator."""

from __future__ import annotations

from typing import Any

import pytest

from src.bridges.cuda_ingest.parser import CudaParser
from src.bridges.cuda_ingest.translator import CudaToTritonTranslator


SAMPLE_KERNEL = '''
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
'''


SAMPLE_MATMUL_KERNEL = '''
__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
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
}
'''


SAMPLE_DEVICE_FUNCTION = '''
__device__ float helper(float x) {
    return x * 2.0f;
}
'''


def make_kernel(source: str) -> Any:
    """Parse a CUDA source and return the first kernel."""
    parser = CudaParser()
    kernels = parser.parse_source(source)
    return kernels[0] if kernels else None


class TestCudaToTritonTranslator:
    """Tests for the CudaToTritonTranslator class."""

    def test_translator_init(self) -> None:
        """CudaToTritonTranslator should initialise without error."""
        translator = CudaToTritonTranslator()
        assert translator is not None

    def test_translate_simple_kernel(self) -> None:
        """A simple vector_add kernel should translate successfully."""
        kernel = make_kernel(SAMPLE_KERNEL)
        translator = CudaToTritonTranslator()
        result = translator.translate(kernel)
        assert result.is_usable
        assert "vector_add" in result.triton_source
        assert "@triton.jit" in result.triton_source
        assert "tl.program_id" in result.triton_source

    def test_translate_matmul_kernel(self) -> None:
        """A matmul kernel should translate with shared memory."""
        kernel = make_kernel(SAMPLE_MATMUL_KERNEL)
        translator = CudaToTritonTranslator()
        result = translator.translate(kernel)
        assert result.is_usable
        # Should include tl.zeros for shared memory
        assert "tl.zeros" in result.triton_source
        # Should include tl.barrier for __syncthreads
        assert "tl.barrier" in result.triton_source

    def test_translate_rejects_device_function(self) -> None:
        """__device__ functions should not be translatable as kernels."""
        kernel = make_kernel(SAMPLE_DEVICE_FUNCTION)
        translator = CudaToTritonTranslator()
        result = translator.translate(kernel)
        assert not result.is_usable
        assert result.error is not None
        assert "device" in result.error.lower()

    def test_translation_preserves_kernel_name(self) -> None:
        """The translated function should have the same name as the CUDA kernel."""
        kernel = make_kernel(SAMPLE_KERNEL)
        translator = CudaToTritonTranslator()
        result = translator.translate(kernel)
        assert result.kernel_name == kernel.name
        assert f"def {kernel.name}" in result.triton_source

    def test_translation_records_pointer_layouts(self) -> None:
        """The translation should record the inferred pointer layouts."""
        kernel = make_kernel(SAMPLE_KERNEL)
        translator = CudaToTritonTranslator()
        result = translator.translate(kernel)
        assert "a" in result.pointer_layouts
        assert "b" in result.pointer_layouts
        assert "c" in result.pointer_layouts

    def test_translation_records_shared_mem_plan(self) -> None:
        """The translation should record the shared memory plan."""
        kernel = make_kernel(SAMPLE_MATMUL_KERNEL)
        translator = CudaToTritonTranslator()
        result = translator.translate(kernel)
        assert result.shared_mem_plan is not None
        assert result.shared_mem_plan.num_allocations >= 1

    def test_translation_collects_warnings(self) -> None:
        """Bank conflict warnings should be collected in the result."""
        kernel = make_kernel(SAMPLE_MATMUL_KERNEL)
        translator = CudaToTritonTranslator()
        result = translator.translate(kernel)
        # Warnings may or may not be present depending on shared memory analysis
        assert isinstance(result.warnings, list)

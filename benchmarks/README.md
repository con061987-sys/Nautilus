# Nautilus Benchmarks

This directory contains the benchmark suite for Nautilus. The
suite measures auto-tuning speedups vs. default Triton configs
across multiple kernels and vendors.

## Quick start

```bash
# Run a single kernel
python benchmarks/run_benchmarks.py --kernel matmul

# Run all kernels and write results to JSON
python benchmarks/run_benchmarks.py --output benchmarks/results.json

# Compare against a reference
python benchmarks/run_benchmarks.py --kernel matmul --reference pytorch
```

## Kernel suite

| Kernel | Description | Triton |
|--------|-------------|--------|
| `matmul.py` | SGEMM: C = A @ B | yes |
| `attention.py` | Scaled dot-product attention | yes |
| `layer_norm.py` | Layer normalization | yes |
| `softmax.py` | Softmax | yes |
| `gelu.py` | GELU activation | yes |
| `conv2d.py` | 2D convolution (implicit GEMM) | yes |
| `embedding.py` | Embedding lookup | yes |
| `reduce.py` | Sum / max reduction | yes |
| `scan.py` | Prefix scan (cumulative sum) | yes |
| `fused_attention.py` | Fused QKV + attention | yes |

## Metrics

For each (kernel, vendor) pair we measure:

- `baseline_tflops` — default Triton config, no tuning
- `tuned_tflops` — best config from TVM MetaSchedule (64 trials)
- `speedup` — `tuned_tflops / baseline_tflops`
- `tuning_time_s` — wall-clock time spent in MetaSchedule
- `compilation_time_s` — AOT compilation time per vendor

## Result schema

```json
{
  "kernel": "matmul",
  "shape": {"M": 1024, "N": 1024, "K": 1024},
  "results": {
    "nvidia/sm_90": {
      "baseline_tflops": 312.5,
      "tuned_tflops": 425.0,
      "speedup": 1.36,
      "tuning_time_s": 12.3,
      "compilation_time_s": 1.8
    },
    "amd/gfx942": {
      "baseline_tflops": 280.0,
      "tuned_tflops": 395.0,
      "speedup": 1.41,
      "tuning_time_s": 14.1,
      "compilation_time_s": 2.3
    }
  }
}
```

## Adding a new kernel

1. Create `benchmarks/kernels/<name>.py`
2. Define a `run_<name>(shape, target) -> Callable` function
3. The kernel must be a `@triton.jit` function
4. Add a row to the table in this README

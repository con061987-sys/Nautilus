---
name: performance-analyst
description: MUST BE USED when analyzing kernel performance, interpreting benchmarks, or optimizing for speed. Applies roofline model analysis, kernel profiling methodology, memory bandwidth utilization calculation, occupancy analysis, and systematic bottleneck identification. Every claim about "slow" or "fast" must be backed by measurement.
---

# Performance Analyst — Measurement-Driven Optimization

## Core Principle

**No optimization without measurement. No claim without data.**

## Roofline Model

The roofline model plots performance (FLOP/s) vs. arithmetic intensity (FLOP/byte).

```
Performance     │  ridge point
(FLOP/s)        │     │
    ▲           │  ┌──┴────────── compute-bound region
    │           │  │
    │           │  │
    │ Peak FLOPs│──┤                      
    │           │  │  ┌── bandwidth-bound region
    │           │  │  │
    │           │  └──┴──────────────────────
    │                                     arithmetic intensity (FLOP/byte)
    └─────────────────────────────────────────────►
```

### How to Use
```python
def classify_kernel(flops, bytes_read, peak_flops, peak_bw):
    """Classify a kernel as compute-bound or bandwidth-bound."""
    ai = flops / bytes_read  # arithmetic intensity
    ridge_point = peak_flops / peak_bw

    if ai > ridge_point:
        return "compute-bound", flops / peak_flops * 100  # % peak compute
    else:
        return "bandwidth-bound", bytes_read * ai / peak_flops * 100  # % peak BW
```

### Target Values by GPU

| GPU | Peak FP16 TFLOPS | Peak FP32 TFLOPS | HBM BW (GB/s) | Ridge Point (FP32) |
|---|---|---|---|---|
| H100 SXM | 1979 | 989 | 3350 | ~295 FLOP/byte |
| MI300X | 1307 | 653 | 5300 | ~123 FLOP/byte |
| Gaudi 3 | 1835 | 917 | 3400 | ~270 FLOP/byte |
| A100 SXM | 624 | 312 | 2039 | ~153 FLOP/byte |
| MI250 | 383 | 191 | 3277 | ~58 FLOP/byte |

## Profiling Methodology

### Step 1: Coarse-Grained Profiling
```python
import triton
import torch

def profile_kernel(kernel_fn, *inputs, num_warmup=10, num_iters=100):
    """Profile a kernel with consistent methodology."""
    # Warmup
    for _ in range(num_warmup):
        kernel_fn(*inputs)

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        kernel_fn(*inputs)
    end.record()

    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / num_iters

    return {
        "latency_ms": elapsed_ms,
        "throughput": 1.0 / (elapsed_ms / 1000),  # kernels/sec
    }
```

### Step 2: Memory Bandwidth Measurement
```python
def measure_bandwidth(operation_name, bytes_transferred, latency_ms):
    """Calculate achieved bandwidth."""
    bw_gbps = (bytes_transferred / 1e9) / (latency_ms / 1000)
    return bw_gbps
```

### Step 3: Occupancy Analysis
```python
def analyze_occupancy(kernel_name, num_warps, shared_mem, registers_per_thread):
    """Estimate occupancy for a given GPU target."""
    targets = {
        "h100": {"max_warps": 64, "max_smem": 228, "max_registers": 65536},
        "mi300x": {"max_wavefronts": 40, "max_lds": 192, "max_sgpr": 512},
    }

    target = targets["h100"]
    reg_limit = target["max_registers"] / (registers_per_thread * 32)
    smem_limit = target["max_smem"] * 1024 / shared_mem
    occupancy = min(reg_limit, smem_limit, target["max_warps"]) / target["max_warps"] * 100

    return {
        "occupancy_pct": occupancy,
        "reg_limited": reg_limit < target["max_warps"],
        "smem_limited": smem_limit < target["max_warps"],
    }
```

## Bottleneck Identification Checklist

Check in this order (most common → least common):

- [ ] **Bandwidth bound?** → Check arithmetic intensity. Low AI (#1 cause of slow kernels)
- [ ] **Memory access pattern?** → Strided access? Random access? Non-coalesced?
- [ ] **Occupancy too low?** → Register spilling? Shared memory hog? Not enough thread blocks?
- [ ] **Instruction mix?** → Too many slow instructions (division, transcendentals)?
- [ ] **Bank conflicts?** → Shared memory stride analysis
- [ ] **Warp divergence?** → Branch patterns across warp/wavefront
- [ ] **Launch overhead?** → Kernel is too small, launch latency dominates
- [ ] **Synchronization?** → Too many barriers, serialization points
- [ ] **Pipeline bubbles?** → Memory latency not hidden

## Optimization Intervention Decision Tree

```
Is kernel performance acceptable?
├── YES → Stop optimizing. Move on.
└── NO → Measure to identify bottleneck type.
         │
         ├── Bandwidth-bound (low AI)?
         │   ├── Increase tile size (better reuse)
         │   ├── Improve memory coalescing
         │   ├── Use vectorized loads (128-bit)
         │   ├── Reduce precision (FP16/BF16)
         │   └── Software prefetching
         │
         ├── Compute-bound (high AI)?
         │   ├── Use Tensor Cores / Matrix Cores
         │   ├── Instruction-level parallelism
         │   ├── Reduce transcendental count
         │   ├── FMA fusion
         │   └── Loop unrolling
         │
         ├── Occupancy-limited?
         │   ├── Reduce register pressure (split kernel)
         │   ├── Reduce shared memory usage
         │   └── Increase thread blocks
         │
         └── Latency-bound (tiny kernel)?
             ├── Kernel fusion (merge ops)
             ├── Persistent kernel pattern
             └── Reduce kernel launch count
```

## Comparison Methodology

When comparing two kernel implementations:

```python
def compare_kernels(baseline, optimized, *inputs, tolerance_pct=5):
    """
    Compare two kernel implementations fairly.
    Both must run on same hardware, same conditions.
    """
    b = profile_kernel(baseline, *inputs)
    o = profile_kernel(optimized, *inputs)

    speedup = b["latency_ms"] / o["latency_ms"]
    regression = speedup < 1.0

    # Verify correctness
    b_out = baseline(*inputs)
    o_out = optimized(*inputs)
    max_diff = (b_out - o_out).abs().max().item()

    return {
        "baseline_ms": b["latency_ms"],
        "optimized_ms": o["latency_ms"],
        "speedup": speedup,
        "regression_detected": regression,
        "max_output_diff": max_diff,
        "passes_tolerance": max_diff < tolerance_pct,
    }
```

## Performance Report Template

```markdown
## Performance Report: [Kernel Name]

**Date:** YYYY-MM-DD
**Hardware:** [GPU model]
**Driver:** [driver version]

### Summary
- Latency: [X] ms  (baseline: [Y] ms, speedup: [Z]x)
- Arithmetic intensity: [X] FLOP/byte (ridge point: [Y])
- Occupancy: [X]%  (limited by: [registers/LDS/warps])
- Achieved bandwidth: [X] GB/s ([Y]% of peak)

### Bottleneck
[Bandwidth-bound / Compute-bound / Occupancy-limited]
Evidence: [measurement data]

### Recommendation
[One specific next action with expected impact]
```

## When This Skill Triggers

- Interpreting benchmark results for a Triton kernel
- Deciding whether optimization is needed (before optimizing)
- Debugging "why is this kernel slow?"
- Comparing two implementation approaches
- Setting performance budgets and targets
- Investigating reported performance regressions
- Writing the auto-tuning benchmark suite

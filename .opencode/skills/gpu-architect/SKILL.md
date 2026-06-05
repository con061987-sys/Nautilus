---
name: gpu-architect
description: MUST BE USED when designing GPU kernels, analyzing kernel performance, or working with hardware-specific optimizations. Deep knowledge of AMD CDNA3, Nvidia Hopper/Blackwell, Intel Xe GPU architectures — memory hierarchy, cache topology, occupancy calculation, warp scheduling, shared memory patterns, and instruction-level optimization.
---

# GPU Architect — Hardware Architecture Deep Knowledge

## Nvidia GPU Architecture (Hopper H100 / Blackwell)

### Memory Hierarchy
| Level | Size | Latency | Bandwidth |
|---|---|---|---|
| Register File (per SM) | 256KB | 1 cycle | ~20 TB/s |
| L1/Shared Memory (per SM) | 256KB (Hopper) | ~30 cycles | ~12 TB/s |
| L2 Cache | 60MB | ~200 cycles | ~3 TB/s |
| HBM3 | 80GB | ~800 cycles | 3.35 TB/s |

### Occupancy Calculation
```
Max warps per SM: 64 (Hopper)
Max threads per SM: 2048
Max registers per SM: 65536

Occupancy = (active_warps / max_warps) × 100%

Register-limited occupancy = min(65536 / (registers_per_thread × 32), 64)
Shared_memory-limited occupancy = min(228KB / shared_mem_per_block, 64)
```

### Key Optimization Targets (Nvidia)
- **Maximize arithmetic intensity:** Compute ÷ bandwidth. Aim for >10:1 for compute-bound (vs. bandwidth-bound)
- **Coalesced memory access:** Adjacent threads access adjacent memory addresses
- **Shared memory bank conflicts:** 32 banks, stride-1 access is conflict-free, stride-2 has 2-way conflicts
- **Tensor Cores (H100):** `tcgen05` instructions, 16×8×16 MMA, requires specific tile shapes (16×16×16, 32×8×16)
- **Hopper new features:** DPX (dynamic programming), Transformer Engine (FP8), 4th-gen Tensor Cores

## AMD GPU Architecture (CDNA3 / MI300X)

### Memory Hierarchy
| Level | Size | Latency | Bandwidth |
|---|---|---|---|
| SGPR per CU | 128KB | 1 cycle | ~18 TB/s |
| LDS (per CU) | 128KB + 64KB (raw) | ~25 cycles | ~10 TB/s |
| L2 Cache (per GCD) | 4-8MB | ~150 cycles | ~2.7 TB/s |
| HBM3 (per GCD) | 96-192GB | ~700 cycles | 5.3 TB/s total (3.2 TB/s per stack) |

### Key Differences from Nvidia
- **Wavefront size: 64 threads** vs. Nvidia's 32-thread warp. Changes everything: memory coalescing width, thread block sizes, shared memory patterns
- **SIMD vs SIMT:** AMD uses explicit SIMD (each CU has 4 SIMDs, each SIMD executes a wavefront). Occupancy planning is about wavefronts per SIMD, not warps per SM
- **LDS is split:** 128KB + 64KB RAW (Read-After-Write) partition. RAW partition is for atomic operations
- **Matrix Cores (MI300X):** Similar to Tensor Cores, 16×16×16, 32×32×32, require `__builtin_amdgcn_mfma_*` intrinsics
- **Cache hierarchy:** 2-level rather than 3-level typical. L1 doesn't exist in CDNA2+, compute units use LDS directly

### AMD-Specific Optimization Rules
```
Wavefront occupancy = active_wavefronts / 40 (max per CU)
SGPR-limited: 128KB / (sgpr_per_wavefront × 64)
LDS-limited: 192KB / lds_per_workgroup

Memory coalescing: wavefront of 64 accesses consecutive 256B chunk
Bank conflicts in LDS: 32 banks, wavefront is 64 threads → 2× access to same bank on consecutive pairs
```

## Intel GPU Architecture (Xe / Xe2 / Xe3)

### Memory Hierarchy
| Level | Size | Latency | Bandwidth |
|---|---|---|---|
| GRF (per EU) | 128KB | 1 cycle | ~15 TB/s |
| SLM (per Subslice) | 64-128KB | ~35 cycles | ~8 TB/s |
| L3 Cache | 12-24MB | ~180 cycles | ~2 TB/s |
| HBM2e (max config) | 48GB | ~700 cycles | ~1.2 TB/s |

### Key Differences
- **Execution Unit (EU):** Smallest compute unit. Each EU has 7 hyperthreads (aka threads), each thread supports 8-wide SIMD (SIMD8). A subslice has 8 EUs
- **SIMD flexibility:** Intel supports SIMD8, SIMD16, SIMD32. Smaller SIMD = better occupancy but worse memory efficiency. Intel's compiler often chooses for you
- **SLM (Shared Local Memory):** Intel's equivalent of shared memory/LDS. Located at subslice level, shared across 8 EUs
- **Thread (EU thread):** Not a GPU thread in Nvidia/AMD sense. Each EU thread holds 8 GRF registers. Intel maps work-items to EU threads in a SIMD fashion

### Key Optimization Rules (Intel)
- **SIMD lane width:** For compute-bound kernels, SIMD32 is best. For divergent code, SIMD8 is better
- **SLM bank conflicts:** 16 banks, 64 bytes per bank. Consecutive 64B chunks map to different banks
- **Intel uses `__intel` SPIR-V extensions** for function-attributed optimizations
- **Bfloat16/FP16** are natively supported in Xe2+

## Universal GPU Optimization Principles

### Roofline Model
```
Peak performance (FLOP/s) = min(peak compute, arithmetic_intensity × peak bandwidth)

Compute bound if: arithmetic_intensity > peak_compute / peak_bandwidth (ridge point)
Bandwidth bound if: arithmetic_intensity < ridge point
```

### Memory Access Patterns (Ranked by Efficiency)
1. **Coalesced, aligned, sequential** — 100% bandwidth utilization
2. **Coalesced, misaligned** — 90-100% (modern GPUs handle alignment)
3. **Strided access (stride=2)** — ~50% bandwidth
4. **Random access** — <10% bandwidth
5. **Atomic operations** — serialized, avoid in perf path

### Shared Memory Patterns
- **Tiling:** Load tile from global → compute on tile → write tile back
- **Bank conflict formula:** `(banks / gcd(stride, banks))` way conflicts
- **Padding trick:** Add `+ 1` to LDS allocation to shift banks on strided access

### Occupancy Trade-offs
| Low Occupancy | High Occupancy |
|---|---|
| More registers per thread | More thread-level parallelism |
| Better single-thread perf | Better latency hiding |
| Works for compute-bound | Better for bandwidth-bound |

## When This Skill Triggers

- Designing new Triton kernels
- Analyzing kernel benchmark results
- Debugging poor GPU utilization
- Choosing block/tile sizes for a target architecture
- Writing hardware-specific optimization code
- Adding support for a new GPU target

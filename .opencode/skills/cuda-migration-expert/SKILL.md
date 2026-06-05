---
name: cuda-migration-expert
description: MUST BE USED when translating CUDA code to Triton, mapping Nvidia intrinsics to AMD/Intel equivalents, or working with the CUDA ingestion bridge (Phase 4). Exhaustive knowledge of CUDA ↔ AMD GCN ↔ Intel SPIR-V intrinsic mappings, warp-level primitive translation, memory model semantics, and synchronization pattern porting.
---

# CUDA Migration Expert — Cross-Platform Intrinsic Mapping

## Intrinsic Mapping Tables

### Memory Operations
| CUDA (Nvidia PTX) | AMD (GCN/ROCm) | Intel (SPIR-V) | Triton |
|---|---|---|---|
| `ld.global.f32` | `global_load_dword` | `OpLoad` (CrossWorkgroup) | `tl.load` |
| `st.global.f32` | `global_store_dword` | `OpStore` (CrossWorkgroup) | `tl.store` |
| `ld.shared.f32` | `ds_read_b32` | `OpLoad` (Workgroup) | `tl.load` (shared) |
| `st.shared.f32` | `ds_write_b32` | `OpStore` (Workgroup) | `tl.store` (shared) |
| `ldg` (read-only) | `global_load_dword` (readonly) | `OpLoad` (CrossWorkgroup, NonWritable) | `tl.load` (eviction_policy="evict_last") |
| `atom.global.add` | `buffer_atomic_add` | `OpAtomicIAdd` | `tl.atomic_add` |
| `atom.global.cas` | `buffer_atomic_cmpswap` | `OpAtomicCompareExchange` | `tl.atomic_cas` |

### Synchronization
| CUDA | AMD | Intel | Semantics |
|---|---|---|---|
| `__syncthreads()` | `__threadfence_block()` (LDS) + `s_barrier` | `OpControlBarrier` (Workgroup) | Full workgroup barrier |
| `__threadfence()` | `__threadfence_system()` (for sys) | `OpMemoryBarrier` (Device) | Device-wide visibility |
| `__threadfence_block()` | `s_waitcnt lgkmcnt(0)` | `OpMemoryBarrier` (Workgroup) | Block-level ordering |
| `__syncwarp()` | VOTE + `s_barrier` (within wave) | `OpGroupNonUniformBarrier` (Subgroup) | Warp/subgroup sync |

### Math Intrinsics
| CUDA | AMD | Intel |
|---|---|---|
| `__fmaf_rn` | `v_fma_f32` | `OpFOrd` + FMul+FAdd |
| `__expf` | `v_exp_f32` (approx) | `OpExp` (or SPIR-V extended) |
| `__logf` | `v_log_f32` (approx) | `OpLog` |
| `__sinf` / `__cosf` | `v_sin_f32` / `v_cos_f32` | `OpSin` / `OpCos` |
| `__powf` | `v_pow_f32` | `OpPow` |
| `__saturatef` | `v_med3_f32` (clamp 0-1) | N/A (use clamp) |
| `__float2half_rn` | `v_cvt_f32_f16` chain | `OpFConvert` (to 16-bit) |
| `__half2float` | `v_cvt_f16_f32` | `OpFConvert` (to 32-bit) |

### Warp/Subgroup Level Primitives
| CUDA (Warp) | AMD (Wavefront) | Intel (Subgroup) | Notes |
|---|---|---|---|
| `__shfl_sync(val, src)` | `__builtin_amdgcn_mov_dpp` | `OpGroupNonUniformShuffle` | Thread-IDX 1 thread mapping |
| `__shfl_up_sync(val, delta)` | `__builtin_amdgcn_mov_dpp` (row_up) | `OpGroupNonUniformShuffleUp` | Shift within subgroup |
| `__shfl_down_sync(val, delta)` | `__builtin_amdgcn_mov_dpp` (row_down) | `OpGroupNonUniformShuffleDown` | Shift within subgroup |
| `__shfl_xor_sync(val, mask)` | `__builtin_amdgcn_mov_dpp` (lane broadcast) | `OpGroupNonUniformShuffleXor` | Butterfly |
| `__any_sync(pred)` | `__any(pred)` (wave-wide) | `OpGroupNonUniformAny` | ANY reduction |
| `__all_sync(pred)` | `__all(pred)` (wave-wide) | `OpGroupNonUniformAll` | ALL reduction |
| `__ballot_sync(pred)` | `__builtin_amdgcn_ballot_w32` (or w64) | `OpGroupNonUniformBallot` | Bitmask of active lanes |

### Key Translation Notes

**Warp vs. Wavefront size difference (32 vs. 64):**
```python
# CUDA code using __shfl_sync assumes 32 threads
# AMD transliteration must account for wavefront of 64

# Solution: Abstract with TL_STATIC_CONSTANT
TL_WARP_SIZE = 32  # Triton standardizes on 32
# AMD backend maps: 2 consecutive wavefronts of 32
```

**Memory model differences:**
```cpp
// CUDA: Weakly-ordered memory model, acquire-release since CUDA 11+
__threadfence_block();  // Sequential consistency within block

// AMD: Stronger ordering, but LDS is separate from global
// s_waitcnt ensures completion, not ordering
// For acquire semantics: s_waitcnt vmcnt(0) + s_waitcnt lgkmcnt(0)

// Intel: OpenCL 2.0 memory ordering
// memory_order_acq_rel for release/acquire semantics
```

## Common Porting Patterns

### Pattern 1: Parallel Reduction
```python
# CUDA:
#   extern __shared__ float sdata[];
#   sdata[tid] = input[tid];
#   __syncthreads();
#   for (int s = 1; s < blockDim.x; s *= 2) {
#       if (tid % (2*s) == 0) sdata[tid] += sdata[tid + s];
#       __syncthreads();
#   }

# Triton:
#   tl.store(shared, input_ptrs, mask)
#   tl.accumulate_along_dim(shared, dim=0)
```

### Pattern 2: Matrix Transpose
```python
# CUDA shared memory transpose:
#   sdata[tx][ty] = input[tx][ty];  // coalesced write
#   __syncthreads();
#   output[ty][tx] = sdata[tx][ty];  // coalesced read

# Triton:
#   tl.store(shared, input, mask)   # coalesced
#   tl.debug_barrier()
#   output = tl.load(shared, mask)   # transposed via block pointers
```

### Pattern 3: GEMM
```python
# CUDA tensor core GEMM:
#   wmma::fragment<wmma::matrix_a, 16, 16, 16, half> a_frag;
#   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
#   wmma::load_matrix_sync(a_frag, A, lda);
#   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

# Triton:
#   @triton.jit
#   def matmul_kernel(A, B, C, M, N, K, BLOCK: tl.constexpr):
#       a = tl.load(A_ptrs, mask=m_mask[:, None] & k_mask[None, :])
#       b = tl.load(B_ptrs, mask=k_mask[:, None] & n_mask[None, :])
#       c = tl.dot(a, b)  # Auto-maps to tensor/matrix cores
```

## When This Skill Triggers

- Translating CUDA code to Triton for the ingestion bridge (Phase 4)
- Debugging cross-platform correctness issues
- Understanding why a kernel works on Nvidia but not AMD/Intel
- Writing the intrinsic mapping tables for the CUDA parser
- Analyzing memory model differences between platforms
- Porting warp-level synchronization patterns

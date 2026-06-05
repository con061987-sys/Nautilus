---
name: compiler-architect
description: MUST BE USED when designing compiler passes, IR transformations, dialect conversions, or any MLIR/LLVM-level work. Deep expertise in LLVM pass pipeline architecture, MLIR dialect conversion legality, TVM TIR schedule primitives, XLA HLO optimization passes, and IR design principles. The difference between a working bridge and an optimizing bridge.
---

# Compiler Architect — MLIR/LLVM/TVM/XLA Pass Expertise

## MLIR Architecture

### Dialect Hierarchy
```
StableHLO (XLA) ──► Tensor ──► Linalg ──► Vector ──► LLVM
     │                                                   
     ▼                                                    
TTGIR (Triton) ──► TritonGPU ──► Vector ──► LLVM
     │
     ▼
TIR (TVM) ──────► Builtin ──► LLVM
```

### Dialect Conversion Legality
The #1 failure mode in MLIR work is illegal dialect conversion.

```
A conversion from Dialect A to Dialect B is LEGAL if:
1. Every op in A has a conversion pattern registered for B
2. All result types in A are convertible to types in B
3. All operand types in A are convertible to types in B
4. The conversion preserves SSA dominance (no def-after-use violations)

Common illegal conversion patterns:
- Scalar → Vector without vectorization pattern
- Integer → Float without type conversion pattern
- Unranked → Ranked tensor without shape inference
```

### Standard Conversion Pass Pipeline
```python
# For normalizing any high-level dialect to LLVM:
# 1. Convert to builtin + vector dialect
#    --convert-scf-to-cf     (structured control flow → branches)
#    --convert-arith-to-llvm (arithmetic ops → LLVM intrinsics)
#    --convert-vector-to-llvm(vector ops → LLVM vector ops)
#    --convert-func-to-llvm  (function ops → LLVM funcs)
#
# 2. Target-specific lowering
#    For Nvidia: --convert-vector-to-gpu (→ PTX)
#    For AMD:    --convert-vector-to-amdgpu (→ HSACO)
```

## TVM TIR Schedule Primitives

Understanding TVM's schedule language is essential for the Phase 1 bridge:

```python
# Key schedule primitives for auto-tuning
def tune_schedule(tir_func):
    sch = tvm.tir.Schedule(tir_func)

    # Loop transformations
    sch.split(loop, factors=[8, 16])        # Split loop into outer + inner
    sch.reorder(loop_a, loop_b)              # Reorder loops
    sch.fuse(loop_a, loop_b)                 # Fuse loops
    sch.vectorize(inner_loop)                # Vectorize inner loop
    sch.bind(block_loop, "blockIdx.x")       # Bind to GPU block
    sch.bind(thread_loop, "threadIdx.x")     # Bind to GPU thread

    # Memory transformations
    sch.cache_read(block, "shared", [tile])  # Cache to shared memory
    sch.cache_write(block, "shared")
    sch.set_scope(block, "global")           # Set memory scope

    # Pipeline
    sch.pipeline(producer, consumer, depth=4) # Software pipeline

    return sch
```

MetaSchedule automatically searches over combinations of these primitives.

## XLA HLO Optimization Passes

Understanding XLA's pipeline helps the Phase 3 bridge:

```
Input: StableHLO Module
    │
    ▼
[HLO Pass Pipeline]
    ├── DCE (Dead Code Elimination)
    ├── CSE (Common Subexpression Elimination)
    ├── Algebraic Simplifier
    ├── Dot Dimension Merger (contract dot dims)
    ├── BatchDot Normalization
    ├── Layout Assignment
    ├── HW-Specific Fusion (fuse ops for target)
    │
    ▼
[GSPMD Partitioner]
    │── PartitionAssignment (sharding propagation)
    │── CollectiveInsertion (all-reduce, all-gather)
    │── SPMDPartitioning (split computation)
    │
    ▼
[Backend Lowering]
    Nvidia: HLO → LLVM → PTX (via Triton or XLA/GPU)
    AMD:    HLO → LLVM → HSACO (via XLA/AMD or AOTriton)
```

## LLVM Pass Pipeline

```python
# Standard LLVM optimization pipeline for GPU kernels
# --O2 equivalent:
opt -S -O2 kernel.ll -o kernel_opt.ll
# Pipeline: instcombine → simplifycfg → gvn → licm → sroa → loop-vectorize → ...

# Custom passes we may need to write:
# 1. Triton-specific intrinsic lowering
# 2. Target-specific memory fence insertion
# 3. Workgroup size optimization
```

## IR Design Principles

When designing a new IR or bridge between dialects:

1. **One op = one semantics.** Don't overload operations with side effects
2. **SSA form is non-negotiable.** Every value is defined exactly once
3. **Type-carrying IR is better than attribute-carrying IR.** Types are checked by the verifier, attributes are not
4. **Round-trip fidelity.** If IR_A → IR_B → IR_A, you should get semantically identical IR back
5. **Verifier before passes.** A good verifier catches bad input before it causes silent miscompilation

## When This Skill Triggers

- Designing the IR normalization pipeline between Triton and TVM
- Writing MLIR conversion patterns
- Adding a new compiler pass
- Debugging illegal dialect conversion errors
- Optimizing the compilation flow for latency or throughput
- Understanding why a kernel doesn't compile for a specific target

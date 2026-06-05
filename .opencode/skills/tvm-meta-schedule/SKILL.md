---
name: tvm-meta-schedule
description: Deep knowledge of Apache TVM MetaSchedule API, evolutionary search for kernel optimization, TIR dialect, cost models, and tuning integration. Use when working with auto-tuning, TVM integration, or the triton_tvm bridge.
---

# TVM MetaSchedule Skill

## Overview

Apache TVM's MetaSchedule uses reinforcement learning to automatically discover optimal kernel configurations (block sizes, tile shapes, pipeline depths) for any target hardware architecture.

## Core API

### Basic MetaSchedule Workflow

```python
import tvm
from tvm import meta_schedule as ms

# 1. Load or create a TIR module (our normalized IR)
mod = tvm.IRModule.from_expr(tir_function)

# 2. Define the target hardware
target = tvm.target.Target("hip")  # AMD ROCm
# or: "cuda" for Nvidia, "llvm -mtriple=spirv64-unknown-unknown" for Intel

# 3. Create database for storing tuning results
db = ms.database.MemoryDatabase()

# 4. Run evolutionary search
with ms.ApplyHistoryBest(db):
    tuned = ms.tune_tir(
        mod=mod,
        target=target,
        database=db,
        # Parameters for the search
        num_trials=1000,          # How many configs to try
        max_trials_per_task=100,  # Per-task budget
        work_dir="./tune_logs",
        # Task extraction method
        task_extraction="from_tir",  # Extract compute tasks from TIR
        # Builder & Runner
        builder=ms.builder.LocalBuilder(),
        runner=ms.runner.LocalRunner(
            evaluator_config=ms.runner.EvaluatorConfig(
                number=1,  # Number of repeats for timing
                repeat=10,  # Number of measurement repeats
                min_repeat_ms=100,  # Minimum measurement time
            )
        ),
    )

# 5. Extract best config
best_config = db.get_best_record(mod)
```

### Extracting Tuning Parameters

```python
# The database stores:
# - Block tile sizes (e.g., [128, 256] for M and N dimensions)
# - Thread block configuration
# - Pipeline depth (num_stages)
# - Unrolling factors
# - Vectorization widths

best_config = db.query_tuning_record(mod, target)
if best_config:
    params = best_config.as_json()
    # Example output:
    # {
    #   "tile_m": 128,
    #   "tile_n": 256,
    #   "tile_k": 32,
    #   "num_stages": 4,
    #   "num_warps": 8,
    #   "vectorize_width": 4,
    # }
```

## Integration with Our Bridge

The bridge normalizes Triton's TTGIR to TVM's TIR dialect, then runs MetaSchedule:

```python
# ir_normalizer.py: TTGIR → MLIR Vector → TVM TIR
def normalize_for_tvm(ttgir_module) -> tvm.IRModule:
    # Step 1: Lower TTGIR to MLIR Vector Dialect
    # Step 2: Convert MLIR Vector dialect to TVM TIR
    # Step 3: Return tvm.IRModule ready for tuning
    pass
```

## Critical Knowledge

1. **MetaSchedule is NOT a neural network.** It's an evolutionary search combined with a learned cost model. It generates candidate configurations, measures them on the target hardware (or simulated), and uses the results to refine its cost model.
2. **The cost model transfers across similar kernels.** Once MetaSchedule tunes a matmul kernel for MI300X, tuning a slightly different matmul takes 10-100x fewer trials.
3. **TIR (Tensor IR)** is TVM's low-level IR. It represents compute with explicit loop nests, memory accesses, and thread hierarchies.
4. **Database persistence matters.** MetaSchedule stores results in a JSON database. We should keep this across sessions to amortize tuning costs.
5. **Supported targets:** CUDA (Nvidia), HIP/ROCm (AMD), OpenCL, Vulkan, Metal, LLVM (CPU), and Hexagon.

## Common Tuning Scenarios

| Kernel Type | Key Parameters | Search Space Size |
|---|---|---|
| GEMM (matmul) | tile_m, tile_n, tile_k, num_stages | ~10,000 configs |
| Attention | block_m, block_n, num_warps, num_stages | ~5,000 configs |
| LayerNorm | block_size, num_warps, vectorize_width | ~500 configs |
| Reduction | tile_size, num_warps, unroll_factor | ~1,000 configs |

## When This Skill Triggers

- Working on `src/bridges/triton_tvm/` bridge code
- Setting up or tuning MetaSchedule configuration
- Debugging TVM integration or IR conversion
- Analyzing tuning results or database records
- Running benchmarks via `src.bridges.triton_tvm.benchmark`

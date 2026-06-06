# Nautilus User Guide

Nautilus is a cross-vendor AI compilation framework. This guide
walks you through installation, first compile, and common workflows.

## What Nautilus does

Nautilus takes standard PyTorch model code and:

1. **Auto-tunes** GPU kernels using TVM MetaSchedule (replaces
   the manual "find the best block size" work that CUDA engineers
   do today)
2. **AOT-compiles** the kernel for every supported GPU vendor
   (Nvidia, AMD, Intel, Apple) and bundles the result into a
   single **fat binary** that dispatches at runtime
3. **Auto-shards** the model across a heterogeneous cluster
   (mixed AMD/Intel/Nvidia) using Google XLA's GSPMD algorithm
4. **Ingests** legacy CUDA C++ code and translates it to portable
   Triton kernels

## Installation

### Option 1: One-command install (recommended)

```bash
git clone https://github.com/nvindia-cud/NVINDIA_CUD
cd nautilus
./scripts/setup-cuda.sh    # or setup-rocm.sh for AMD dev
```

This creates a venv, installs all deps, and verifies the env.

### Option 2: Manual

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[nvidia,sharding,dev]   # pick your vendor
```

### Verify the install

```bash
nautilus verify
```

Output:
```
=== Nautilus Environment Verification (target: all) ===

  [REQ] [✓] gcc  (/usr/bin/gcc)
  [REQ] [✓] clang  (/usr/bin/clang)
  [opt] [✗] nvcc   — Install CUDA toolkit: ...
  [opt] [✗] torch  — pip install torch
  [opt] [✗] triton — pip install triton
  ...

  Result:  WARNING
  Summary: OK with warnings: 4 optional tool(s) missing
```

## Your first compile

Create a Triton kernel in `matmul.py`:

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16))
```

### Step 1: Tune the kernel

```bash
nautilus tune matmul.py --target nvidia/sm_90 --trials 64
```

Output:
```
Nautilus best config for 'matmul_kernel' on nvidia/sm_90:
  BLOCK_M = 128
  BLOCK_N = 128
  BLOCK_K = 32
  num_warps  = 8
  num_stages = 3
```

### Step 2: Build a fat binary

```bash
nautilus build matmul.py \
  --target nvidia/sm_90 \
  --target amd/gfx942 \
  -o matmul.fat.o
```

The output is a single ELF object file that contains:
- The Nvidia PTX (for H100/A100)
- The AMD HSACO (for MI300X)
- The Intel SPIR-V (if you added `--target intel/xe_hpg`)
- A C runtime stub that probes `/dev/nvidia*`, `/dev/kfd`,
  `/dev/dri/renderD*` at startup and dispatches to the right one

### Step 3: Use the fat binary

```c
#include "nautilus_dispatch.h"

int main() {
    // Allocate matrices
    // ...
    // Call the fat binary
    nautilus_dispatch(args);
    // ...
}
```

Or via Python ctypes:

```python
import ctypes
lib = ctypes.CDLL("./matmul.fat.o")
lib.nautilus_dispatch.argtypes = [ctypes.c_void_p]
lib.nautilus_dispatch.restype = ctypes.c_int
lib.nautilus_dispatch(my_args)
```

## Auto-sharding a model

```python
import torch
import torch.nn as nn
from src.bridges.pytorch_xla import AutoShardingBridge
from src.common.types import MeshShape

class TwoLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 1024)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = TwoLayerMLP()
mesh = MeshShape(axes=(2, 2))  # 2x2 device grid
bridge = AutoShardingBridge()
result = bridge.shard(
    model=model,
    example_inputs=(torch.randn(1, 1024),),
    device_mesh=mesh,
)
print(f"sharded across {mesh.total_devices} devices")
print(f"comm volume: {result.gspmd_result.sharding_spec.estimated_comm_volume_bytes} bytes")
```

Or via CLI:
```bash
nautilus shard my_model.py --mesh 2,2 --output-dir ./shards
```

## CUDA ingest

```bash
nautilus ingest my_legacy_kernel.cu \
  --output my_legacy_kernel.py \
  --target nvidia/sm_90
```

This parses the .cu file, translates intrinsics to Triton, then
runs the standard compile pipeline.

## Next steps

- [Architecture](ARCHITECTURE.md) — how Nautilus is wired
- [Contributing](CONTRIBUTING.md) — how to add a backend
- [Troubleshooting](TROUBLESHOOTING.md) — common issues

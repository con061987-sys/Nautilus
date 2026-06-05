---
name: xla-sharding
description: Deep knowledge of Google XLA/OpenXLA compiler, StableHLO dialect, GSPMD auto-sharding, PyTorch FX graph capture, and distributed tensor operations. Use when implementing auto-sharding, PyTorch integration, or the pytorch_xla bridge.
---

# XLA / Auto-Sharding Skill

## Overview

Google's XLA (Accelerated Linear Algebra) compiler provides state-of-the-art automatic sharding via its GSPMD (Generalized SPMD) partitioner. Combined with OpenXLA's StableHLO intermediate representation, it can automatically compute optimal distributed execution plans for any model graph.

## Pipeline Overview

```
PyTorch Model
    │
    ▼
torch.compile() ──► TorchFX Graph (functional, traceable IR)
    │
    ▼
FX → StableHLO Converter ──► mlir.Module in StableHLO dialect
    │
    ▼
OpenXLA Compiler ──► GSPMD Partitioner
    │                       │
    │                  ├── Input: Mesh topology (devices, connectivity)
    │                  ├── Input: Cost model (bandwidth, latency)
    │                  └── Output: Sharding specification per op
    │
    ▼
Sharding Spec ──► PyTorch DTensor ──► Distributed Execution
```

## TorchFX Graph Capture

```python
import torch
import torch._dynamo as dynamo
import torch.fx as fx

# Method 1: Using torch.compile() with fullgraph
@torch.compile(backend="nvindia", fullgraph=True)
def forward(x, w):
    return torch.matmul(x, w)

# Method 2: Explicit FX trace for more control
def capture_graph(model, example_inputs):
    # Use Dynamo to capture the graph
    graph = dynamo.export(model, *example_inputs)
    return graph  # fx.GraphModule

    # Alternative: raw FX symbolic trace
    tracer = fx.Tracer()
    graph_module = fx.symbolic_trace(model)
    return graph_module
```

## FX → StableHLO Conversion

```python
# Using torch_xla for FX → StableHLO conversion
import torch_xla
from torch_xla.stablehlo import exported_program_to_stablehlo

def convert_to_stablehlo(fx_graph: fx.GraphModule, example_inputs) -> str:
    """
    Convert a TorchFX graph to StableHLO MLIR text.
    """
    # Export via torch.export
    exported = torch.export.export(fx_graph, example_inputs)

    # Convert to StableHLO module
    stablehlo_module = exported_program_to_stablehlo(exported)

    # Return MLIR text
    return str(stablehlo_module)
```

## GSPMD Auto-Sharding

```python
# Using OpenXLA's PJRT API to run GSPMD
# This is the standard way to invoke XLA's sharding pass

from openxla.python import pjrt
import openxla_python_mlir as mlir

def run_gspmd(stablehlo_module: str, device_mesh: dict):
    """
    device_mesh: {
        "devices": [0, 1, 2, 3, 4, 5, 6, 7],
        "mesh_shape": [2, 4],
        "device_ids": [[0, 1, 2, 3], [4, 5, 6, 7]],
    }
    """
    # Parse the StableHLO MLIR
    mlir_module = mlir.parse_module(stablehlo_module)

    # Run GSPMD partitioning
    # GSPMD:
    # 1. Analyzes the computation graph
    # 2. Propagates sharding constraints
    # 3. Inserts collectives (all-reduce, all-gather, etc.)
    # 4. Partitions each op across the device mesh
    partitioned = pjrt.run_gspmd(
        mlir_module,
        num_partitions=len(device_mesh["devices"]),
        mesh_shape=device_mesh["mesh_shape"],
        device_ids=device_mesh["device_ids"],
        # Optional: Manual sharding hints
        entry_computation_layout=None,
    )

    return partitioned  # Partitioned HLO with sharding annotations
```

## Sharding Spec → PyTorch DTensor

```python
from torch.distributed._tensor import DTensor, DeviceMesh, Shard, Replicate

def apply_sharding_spec(model, sharding_spec, device_mesh):
    """
    Convert XLA's sharding decisions into PyTorch DTensor dimensions.
    """
    mesh = DeviceMesh("cuda", list(range(device_mesh["num_devices"])))

    for name, param in model.named_parameters():
        if name in sharding_spec:
            spec = sharding_spec[name]
            # spec is a list of placements:
            # Shard(0) = split along dim 0
            # Shard(1) = split along dim 1
            # Replicate() = full copy on each device
            placements = [
                Shard(d) if isinstance(s, int) else Replicate()
                for s in spec
            ]
            param.data = DTensor.from_local(
                param.data, mesh, placements,
                run_check=False
            )
```

## Critical Knowledge

1. **StableHLO is the interchange format.** It's a portable MLIR dialect that can be consumed by XLA, TensorFlow, and increasingly by TVM. Always normalize through StableHLO, never use JAX-specific HLO.
2. **GSPMD is automatic but not magic.** It needs accurate cost models (bandwidth between devices, compute capacity) to make optimal decisions. Supply real measurements from your cluster.
3. **GSPMD vs manual sharding.** GSPMD can match or beat hand-tuned sharding for standard transformer architectures. For custom models with unusual communication patterns, manual hints help.
4. **Communication primitives GSPMD inserts:** all-reduce, all-gather, reduce-scatter, all-to-all. These map to NCCL (Nvidia), RCCL (AMD), or oneCCL (Intel) depending on hardware.
5. **PJRT (Plugin-driven JIT Runtime)** is the standard XLA runtime API. All major hardware vendors provide PJRT plugins.

## Common Sharding Strategies

| Model Type | GSPMD Strategy | Communication Pattern |
|---|---|---|
| Transformer (decode) | Tensor parallelism along hidden dim | All-reduce per layer |
| Transformer (training) | Pipeline + data parallelism | P2P + all-reduce |
| MoE (Mixture of Experts) | Expert sharding | All-to-all |
| Convolutional | Spatial sharding | All-gather |

## When This Skill Triggers

- Working on `src/bridges/pytorch_xla/` bridge code
- Capturing PyTorch graphs for sharding
- Debugging GSPMD partitioning decisions
- Configuring OpenXLA or PJRT runtime
- Analyzing communication patterns in distributed execution

"""PyTorch ↔ OpenXLA Auto-Sharding Bridge (Phase 3 of Nautilus).

This package implements the auto-sharding pipeline that takes a
standard PyTorch model and distributes it across a heterogeneous
cluster of AMD, Intel, and Nvidia GPUs.

Architecture:
    [PyTorch Model]
        │
        ▼
    [torch.compile() / torch.export()] ──► [TorchFX Graph]
        │
        ▼
    [FX → StableHLO Converter] ──► [StableHLO Module in MLIR]
        │
        ▼
    [GSPMD Partitioner] ──► [Sharded StableHLO + ShardingSpec]
        │                       │
        │                       ▼
        │               [DTensor Conversion]
        │
        ▼
    [Fat Binary Builder] ──► Per-shard fat binaries
        │
        ▼
    [Cluster Executor] ──► Distributed execution with comm backend

The auto-sharding is production-grade:
  - Circuit breaker per dependency (TVM, XLA, PyTorch, fat binary builder)
  - Per-stage timeouts
  - Persistent sharding cache
  - Heterogeneous cluster support (mixed AMD/Intel/Nvidia)
  - Communication backend abstraction (NCCL/RCCL/oneCCL/UALink)
  - Hardware validation per shard
  - Graceful degradation chain

Modules:
  graph_capture.py        - torch.compile / torch.export wrapper
  stablehlo_export.py     - FX → StableHLO MLIR conversion
  gspmd_runner.py         - GSPMD auto-sharding invocation
  dtensor_apply.py        - Sharding spec → PyTorch DTensor
  device_mesh.py          - Mixed device cluster management
  comm_backend.py         - NCCL/RCCL/oneCCL/UALink communication
  pipeline_orchestrator.py - Main pipeline coordinator
  hardware_orchestrator.py - Per-shard kernel dispatch
"""

from .graph_capture import GraphCapture, CapturedGraph, GraphMetadata
from .stablehlo_export import StableHLOExporter, StableHLOModule
from .gspmd_runner import GSPMDRunner, GSPMDResult, ShardingSpec
from .dtensor_apply import DTensorApplier, DTensorPlan
from .device_mesh import DeviceMesh, MeshDevice, MeshTopology
from .comm_backend import CommBackend, CommGroup, CollectiveOp
from .pipeline_orchestrator import AutoShardingBridge, ShardingConfig, ShardingResult
from .hardware_orchestrator import ShardExecutor, ShardExecutionResult

__all__ = [
    "GraphCapture",
    "CapturedGraph",
    "GraphMetadata",
    "StableHLOExporter",
    "StableHLOModule",
    "GSPMDRunner",
    "GSPMDResult",
    "ShardingSpec",
    "DTensorApplier",
    "DTensorPlan",
    "DeviceMesh",
    "MeshDevice",
    "MeshTopology",
    "CommBackend",
    "CommGroup",
    "CollectiveOp",
    "AutoShardingBridge",
    "ShardingConfig",
    "ShardingResult",
    "ShardExecutor",
    "ShardExecutionResult",
]

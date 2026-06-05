# Technical Architecture Specification: NVINDIA_CUD

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                          │
│  Python API  │  CLI (click)  │  PyTorch Integration  │  Jupyter     │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      COMPILER ORCHESTRATION LAYER                    │
│                                                                      │
│  ┌──────────────┐   ┌────────────────┐   ┌──────────────────────┐   │
│  │ Auto-Tuner   │   │ AOT Packager   │   │ Auto-Sharding Engine │   │
│  │ (Phase 1)    │──▶│ (Phase 2)      │──▶│ (Phase 3)            │   │
│  └───────┬──────┘   └───────┬────────┘   └──────────┬───────────┘   │
│          │                  │                        │               │
└──────────┼──────────────────┼────────────────────────┼───────────────┘
           │                  │                        │
           ▼                  ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     INTEGRATION BRIDGE LAYER                         │
│                                                                      │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────────┐       │
│  │ Triton ↔ TVM   │  │ AOTriton +   │  │ PyTorch FX ↔ XLA   │       │
│  │ Bridge         │  │ LLVM Linker  │  │ Bridge             │       │
│  └───────┬────────┘  └──────┬───────┘  └─────────┬──────────┘       │
└──────────┼───────────────────┼────────────────────┼──────────────────┘
           │                   │                    │
           ▼                   ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DEPENDENCY ABSTRACTION LAYER                       │
│                                                                      │
│  C-API Wrappers  │  Pinned Git Submodules  │  MLIR Normalizer       │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │ Triton   │  │ TVM      │  │ XLA      │  │ LLVM / MLIR      │    │
│  │ C-API    │  │ C-API    │  │ C-API    │  │ Toolchain        │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. Phase 1: Auto-Tuning Bridge (Triton ↔ TVM MetaSchedule)

### 2.1 Data Flow

```
[Triton Kernel (Python)] 
        │
        ▼
[Triton JIT Compiler] ──► [TTGIR (Triton GPU IR)]
        │
        ▼
[IR Interceptor] ──► Extracts: matmul_size, tile_bounds, data_types, num_warps
        │
        ▼
[TVM MetaSchedule Bridge] ──► Normalizes IR to TVM TIR format
        │
        ▼
[Evolutionary Search] ──► TVM runs RL-based search for optimal config
        │
        ▼
[Config Feedback] ──► block_size_M, block_size_N, block_size_K, num_stages, num_warps
        │
        ▼
[Triton Recompile] ──► Compiles kernel with tuned parameters
```

### 2.2 Key Interfaces

```python
# triton_tvm_bridge.py
class AutoTuningBridge:
    def intercept_ir(self, kernel_fn) -> TritonIR: ...
    def normalize_to_tvm(self, ir: TritonIR) -> TVMIR: ...
    def run_metaschedule(self, ir: TVMIR, target: HardwareTarget) -> TuningConfig: ...
    def apply_config(self, config: TuningConfig, kernel_fn) -> CompiledKernel: ...
```

### 2.3 MLIR Normalization Pipeline

```
TTGIR (Triton) ──► De-Sugar Pass ──► Strip vendor-specific layout descriptors
    │
    ▼
Standard MLIR Vector Dialect ──► Pure mathematical graph
    │
    ▼
TVM Relay/TIR ──► MetaSchedule-compatible representation
```

## 3. Phase 2: AOT Fat Binary Packaging

### 3.1 Build Pipeline

```
[Triton Kernel]
    │
    ├──► [AMD AOTriton] ──► kernel.hsaco
    │
    ├──► [Intel oneAPI] ──► kernel.spv
    │
    ├──► [Nvidia PTX]   ──► kernel.ptx (via Triton JIT + save)
    │
    └──► [LLVM Linker (lld)] ──► fat_binary.o
                                     │
                                     ▼
                            [C Runtime Stub]
                            Checks vendor CPUID at startup
                            Selects correct binary block
```

### 3.2 Fat Binary Format

```
┌─────────────────────────────────────┐
│  ELF Header                         │
├─────────────────────────────────────┤
│  C Runtime Stub (code)              │
│  - Detect CPU vendor (CPUID)        │
│  - Detect GPU via /dev/kfd, /dev/dri│
│  - Jump to matching backend         │
├─────────────────────────────────────┤
│  Section: .nv_kernel (PTX)          │
├─────────────────────────────────────┤
│  Section: .amd_kernel (HSACO)       │
├─────────────────────────────────────┤
│  Section: .intel_kernel (SPIR-V)    │
├─────────────────────────────────────┤
│  Section: .apple_kernel (Metal AIR) │
└─────────────────────────────────────┘
```

### 3.3 Key Interfaces

```python
# aot_packager.py
class FatBinaryBuilder:
    def compile_for_amd(self, kernel_ir: str) -> bytes: ...
    def compile_for_intel(self, kernel_ir: str) -> bytes: ...
    def compile_for_nvidia(self, kernel_ir: str) -> bytes: ...
    def link_fat_binary(self, backends: dict[str, bytes]) -> ELFBinary: ...
    def emit_c_stub(self, backends: list[str]) -> CSource: ...
```

## 4. Phase 3: Auto-Sharding Bridge (PyTorch ↔ OpenXLA)

### 4.1 Data Flow

```
[PyTorch Model]
    │
    ▼
[torch.compile()] ──► [TorchFX Graph]
    │
    ▼
[FX → StableHLO Converter] ──► Translates graph to StableHLO (MLIR)
    │
    ▼
[OpenXLA Compiler] ──► [GSPMD Partitioner]
    │                       │
    │                  Calculates optimal sharding
    │                  based on: device topology, mesh shape, cost model
    │                       │
    ▼                       ▼
[Sharding Instructions] ──► [PyTorch DTensor Conversion]
    │
    ▼
[Phase 2 Fat Binaries] ──► Execute sharded computation on cluster
```

### 4.2 Key Interfaces

```python
# pytorch_xla_bridge.py
class AutoShardingBridge:
    def capture_graph(self, model: nn.Module, example_inputs: tuple) -> FXGraph: ...
    def convert_to_stablehlo(self, graph: FXGraph) -> StableHLOProgram: ...
    def run_gspmd(self, program: StableHLOProgram, mesh: DeviceMesh) -> ShardingSpec: ...
    def apply_sharding(self, spec: ShardingSpec, model: nn.Module) -> ShardedModel: ...
```

## 5. Dependency Abstraction Layer

### 5.1 C-API Wrapper Pattern

Every external dependency is wrapped behind a stable C-API:

```c
// triton_c_api.h — NEVER changes signature
typedef struct triton_kernel_s triton_kernel_t;
triton_kernel_t* triton_compile(const char* ir_source, const char* target);
int triton_set_tuning_param(triton_kernel_t* kernel, const char* param, int value);
int triton_get_binary(triton_kernel_t* kernel, void** out_data, size_t* out_size);
void triton_free_kernel(triton_kernel_t* kernel);
```

When upstream Triton changes internal APIs, only the implementation of these 5 functions changes. All bridge code stays untouched.

### 5.2 Git Submodule Pinning

```bash
# third_party/ — all pinned to specific commits
third_party/triton    @ v3.2.0        # OpenAI Triton
third_party/tvm       @ v0.18.0       # Apache TVM
third_party/xla       @ commit abc123 # OpenXLA (pinned, not branch)
third_party/llvm-project @ v19.1.0   # LLVM + MLIR
```

### 5.3 Drift Detection CI

```yaml
# .github/workflows/drift-detection.yml
on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM UTC
jobs:
  drift:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python scripts/check_upstream_drift.py
      - run: python scripts/run_validation_tests.py
      - if: failure()
        run: python scripts/open_github_issue.py —title "Drift detected: ${{ matrix.dep }}"
```

## 6. Runtime Layer

### 6.1 Deterministic Memory Reclaimer

```python
class MemoryReclaimer:
    """
    Prevents OOM during dynamic tuning phases.
    Forces GPU driver to flush cached allocators between tuning iterations
    without interrupting the model execution context.
    """
    def reclaim(self, device: str) -> int: ...  # Returns bytes reclaimed
    def set_watermark(self, threshold_mb: int): ...
    def auto_reclaim(self, interval_seconds: float): ...
```

### 6.2 Fault Tolerance

```python
class AsyncCheckpointer:
    """
    Saves micro-checkpoints of model weights to system RAM every N seconds.
    On node failure, rebuilds computation graph for surviving nodes
    and resumes within 3 seconds.
    """
    def start(self, interval_seconds: float = 5.0): ...
    def on_node_failure(self, dead_node_id: str): ...
    def rebuild_topology(self, alive_nodes: list[str]): ...
```

### 6.3 IEEE-754 Math Validator

```python
class MathValidator:
    """
    Injects bit-exact rounding validation into compiled kernels.
    Sacrifices marginal speed for guaranteed deterministic output.
    """
    def enable_bit_exact_mode(self): ...
    def validate_kernel(self, kernel_path: str) -> ValidationReport: ...
    def insert_rounding_correction(self, ir: MLIRModule) -> MLIRModule: ...
```

## 7. Project Structure (Detailed)

```
src/
├── bridges/
│   ├── triton_tvm/
│   │   ├── __init__.py
│   │   ├── bridge.py           # AutoTuningBridge orchestrator
│   │   ├── ir_interceptor.py   # Captures Triton IR before JIT
│   │   ├── ir_normalizer.py    # TTGIR → MLIR Vector Dialect
│   │   ├── tvm_adapter.py      # MLIR → TVM MetaSchedule API
│   │   ├── config_applier.py   # Tuned params → Triton recompile
│   │   └── benchmarks/
│   │       ├── matmul.py
│   │       ├── attention.py
│   │       └── layer_norm.py
│   ├── aot_packager/
│   │   ├── __init__.py
│   │   ├── builder.py          # FatBinaryBuilder
│   │   ├── amd_backend.py      # AOTriton wrapper
│   │   ├── intel_backend.py    # oneAPI wrapper
│   │   ├── nvidia_backend.py   # PTX capture
│   │   ├── linker.py           # LLVM lld wrapper
│   │   ├── runtime_stub.c      # C vendor-detection stub
│   │   └── tests/
│   │       └── test_fat_binary.py
│   ├── pytorch_xla/
│   │   ├── __init__.py
│   │   ├── bridge.py           # AutoShardingBridge
│   │   ├── graph_capture.py    # torch.compile → FX
│   │   ├── stablehlo_export.py # FX → StableHLO
│   │   ├── gspmd_runner.py     # OpenXLA GSPMD invocation
│   │   ├── dtensor_apply.py    # Sharding spec → DTensor
│   │   └── tests/
│   │       └── test_sharding.py
│   └── cuda_ingest/
│       ├── __init__.py
│       ├── parser.py           # CUDA C++ parser
│       ├── translator.py       # CUDA → Triton IR
│       └── tests/
│           └── test_translation.py
├── runtime/
│   ├── memory_reclaimer.py
│   ├── async_checkpointer.py
│   ├── math_validator.py
│   └── fault_detector.py
├── common/
│   ├── types.py                # Shared type definitions
│   ├── hardware.py             # Hardware detection & abstraction
│   ├── logging.py              # Structured logging
│   └── errors.py               # Error types and handling
├── c_api/
│   ├── triton_c_api.h
│   ├── tvm_c_api.h
│   ├── xla_c_api.h
│   └── wrappers/               # C-API implementation for each dep
├── cli/
│   ├── __init__.py
│   ├── main.py                 # click CLI entry point
│   └── commands/
│       ├── build.py
│       ├── tune.py
│       ├── shard.py
│       └── serve.py
└── tests/
    ├── conftest.py
    ├── test_auto_tuning.py
    ├── test_fat_binary.py
    ├── test_sharding.py
    └── integration/
        ├── test_full_pipeline.py
        └── test_cluster.py
```

## 8. Hardware Target Matrix

| Target | Binary Format | Compiler | Test Infrastructure |
|---|---|---|---|
| Nvidia H100/A100 | PTX + cubin | Triton JIT | Available locally |
| AMD MI300X | HSACO | AOTriton | AMD Developer Cloud (free) |
| Intel Gaudi 2/3 | SPIR-V | oneAPI | Intel Tiber AI Cloud (free) |
| Apple M-series | Metal AIR | (future) | Local Mac |
| AMD MI250 | HSACO | AOTriton | AMD Developer Cloud (free) |

# NVINDIA_CUD — Universal Compiler Framework

## Project Vision

Build a neutral, open-source compiler framework that breaks Nvidia's CUDA monopoly by wiring together existing open-source infrastructure. The framework lets AI models run on **any hardware** (AMD, Intel, Apple Silicon) at near-native performance without manual tuning.

**Core principle:** This is a *wiring* project, not an *invention* project. The execution engines already exist — we connect them.

## Architecture Overview

```
[Legacy CUDA Code] ──► [CUDA Ingestor] ──┐
                                          ├──► [Triton Core Engine] ──► [Auto-Tuning AI] ──► Max Performance on ANY Chip
[New Python Code]  ──► [Auto-Sharding]  ──┘
```

## Tech Stack

| Component | Role | Source |
|---|---|---|
| **OpenAI Triton** | Kernel language, compiler frontend | openai/triton |
| **TVM MetaSchedule** | RL-based auto-tuning backend | apache/tvm |
| **AMD AOTriton** | AOT compilation for AMD GPUs | ROCm/AOTriton |
| **Intel oneAPI** | AOT compilation for Intel GPUs | intel/llvm |
| **Google XLA / OpenXLA** | Auto-sharding via GSPMD/StableHLO | openxla/xla |
| **LLVM/MLIR** | IR normalization, fat binary packaging | llvm/llvm-project |
| **PyTorch 2.x** | Model capture via torch.compile/FX | pytorch/pytorch |

## Four-Phase Execution Plan

### Phase 1: Auto-Tuning Bridge (Month 1)
Wire Triton kernels → TVM MetaSchedule → optimal block configs.
- Intercept Triton IR before kernel compilation
- Extract mathematical bounds (matrix sizes, block shapes)
- Feed to TVM MetaSchedule via Python bridge
- Plug optimized configs back into Triton compiler

### Phase 2: Universal AOT Package (Month 2)
Bundle binaries for AMD, Intel, Nvidia into single "Fat Binary".
- Call AMD AOTriton compiler for .hsaco binary
- Call Intel oneAPI compiler for .spv binary
- Use LLVM linker (lld) to bundle into single object
- C-stub at entry checks hardware vendor at runtime

### Phase 3: Auto-Sharding Bridge (Month 3)
Wire PyTorch models → OpenXLA/GSPMD → optimal distributed cuts.
- Capture PyTorch model as TorchFX graph via torch.compile()
- Pass to OpenXLA via Python/C++ bindings
- GSPMD partitioner calculates optimal slicing
- Route execution commands to Phase 2 Fat Binaries

### Phase 4: CUDA Ingestion & Hardening (Months 4+)
- Build loss-less CUDA-to-Triton frontend compiler
- Deterministic memory reclaimer (prevent OOM)
- Asynchronous fault tolerance for mixed clusters
- IEEE-754 bit-exact math validation

## Conventions

### Code Style
- Python: PEP 8, type hints on all public functions, Google-style docstrings
- C++: LLVM coding style, RAII, explicit error handling via Expected<T>
- All bridge code must have integration tests
- All public APIs need docstrings with usage examples

### Repository Structure
```
NVINDIA_CUD/
├── opencode.json          # OpenCode project config
├── AGENTS.md              # This file — AI agent instructions
├── docs/
│   ├── PRD.md             # Product Requirements Document
│   ├── TECH_SPEC.md       # Technical Architecture Spec
│   ├── A.md through G.md  # Original strategy docs
│   └── PATTERNS.md        # Code patterns & conventions
├── .opencode/
│   ├── skills/            # Project-specific AI skills
│   ├── agents/            # Specialized sub-agents
│   ├── commands/          # Custom command shortcuts
│   └── plugins/           # Custom OpenCode plugins
├── src/
│   ├── bridges/           # Cross-system integration code
│   │   ├── triton_tvm/    # Phase 1: Triton ↔ TVM
│   │   ├── aot_packager/  # Phase 2: Fat binary build
│   │   ├── pytorch_xla/   # Phase 3: PyTorch ↔ XLA
│   │   └── cuda_ingest/   # Phase 4: CUDA ingestion
│   ├── runtime/           # Runtime support (memory, fault tolerance)
│   ├── common/            # Shared utilities
│   └── tests/             # Integration & regression tests
└── third_party/           # Git submodules (pinned commits)
    ├── triton/
    ├── tvm/
    └── xla/
```

### Bridge Pattern
Every bridge follows this contract:
1. **Intercept** — capture IR/graph from source system
2. **Normalize** — convert to neutral representation (MLIR Vector Dialect)
3. **Translate** — pass to target system via C-API or stable interface
4. **Verify** — validate output correctness

### Version Drift Strategy (from docs/E)
- C-API abstraction layers around each dependency
- Git submodules pinned to specific commits
- Automated drift-detection CI (daily builds against latest nightlies)

## Build & Test Commands

```
# Build all bridges
python -m venv .venv && source .venv/bin/activate && pip install -e .

# Run bridge integration tests
python -m pytest src/tests/

# Validate MLIR dialect
python -m scripts/validate_mlir.py

# Build fat binary for current platform
python -m src.bridges.aot_packager.build --target all

# Run auto-tuning benchmark
python -m src.bridges.triton_tvm.benchmark --kernel matmul
```

## MCP Tool Usage

When you need external information:
- **`use context7`** — look up API docs for Triton, TVM, XLA, LLVM
- **`use gh_grep`** — search GitHub for real-world usage patterns
- **`use github`** — repo operations (issues, PRs, code search)
- **`use sequential-thinking`** — complex multi-step reasoning for architecture decisions

## Prohibited Patterns

- No `# type: ignore` or `Any` to bypass type checking
- No hardcoded hardware-specific values (block sizes, thread counts)
- No direct imports from unstable internal APIs — always use C-API wrappers
- No single monolithic bridge files — split by direction (triton→tvm, tvm→triton)
- No "works on my machine" patterns — all platform-specific logic must be abstracted

## Critical Knowledge

1. **MLIR dialect mismatch is the #1 failure mode.** Triton uses TTGIR, TVM uses Relay/TIR, XLA uses StableHLO. Always normalize through LLVM Vector/Math Dialect.
2. **Version drift breaks everything.** Pin submodules. Never depend on `main` branch of any dependency.
3. **AMD + Intel dev clouds are free.** Use Intel Tiber AI Cloud + AMD Developer Cloud for testing. Never assume Nvidia availability.
4. **This is a wiring project.** If you're writing complex algorithms from scratch, you've lost the plot. The engines exist — connect them.

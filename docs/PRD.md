# Product Requirements Document: NVINDIA_CUD

## 1. Executive Summary

**Product:** NVINDIA_CUD — Universal Compiler Framework
**Tagline:** Write once, run at max performance on ANY chip.
**Duration:** 12 months to production (3-month PoC)

NVINDIA_CUD is an open-source compiler framework that extends OpenAI Triton into a multi-vendor, auto-tuning, auto-sharding universal compiler. It breaks Nvidia's CUDA monopoly by allowing AI models to run at near-native performance on AMD, Intel, and Apple Silicon hardware without manual tuning or code changes.

## 2. Problem Statement

Nvidia dominates AI hardware with ~88% market share, largely due to CUDA's software moat — 15+ years of optimized libraries, tooling, and developer habits that lock users into Nvidia hardware. While open alternatives exist (AMD ROCm, Intel oneAPI, Apple Metal), they suffer from:

- **Manual tuning burden** — developers must hand-optimize for each target
- **No unified toolchain** — separate compilers, separate workflows
- **Hardware lock-in** — code written for one vendor doesn't work on another
- **No auto-sharding** — multi-GPU training across mixed hardware is infeasible

## 3. Target Audience

- AI/ML engineers deploying models on non-Nvidia hardware
- Cloud providers wanting hardware diversity (avoid Nvidia markup)
- Edge/on-device AI needing portable pre-compiled binaries
- Researchers experimenting with novel hardware architectures

## 4. Functional Requirements

### F-1: Auto-Tuning Engine
**Priority:** P0 (Phase 1, Month 1)

- MUST intercept Triton kernel IR before compilation
- MUST extract mathematical bounds (tensor shapes, data types)
- MUST feed to TVM MetaSchedule for optimal block/tile configuration
- MUST return tuned parameters to Triton compiler
- MUST support AMD MI300X, Intel Gaudi, and Nvidia H100 targets
- SHOULD complete tuning in < 5 seconds per kernel
- MUST NOT require user intervention in tuning process

### F-2: Multi-Platform AOT Compilation
**Priority:** P0 (Phase 2, Month 2)

- MUST compile Triton kernels ahead-of-time for AMD (HSACO) and Intel (SPIR-V)
- MUST bundle all platform binaries into single "Fat Binary" executable
- MUST detect target hardware at runtime and select correct binary
- MUST support Nvidia PTX as compilation target
- SHOULD support Apple Metal backend
- MUST produce binaries that run without runtime compilation

### F-3: Auto-Sharding for Distributed Training
**Priority:** P1 (Phase 3, Month 3)

- MUST capture PyTorch model graphs via torch.compile() / FX
- MUST pass graph to OpenXLA for GSPMD-based auto-sharding
- MUST automatically slice model across available devices
- MUST support heterogeneous clusters (mixed AMD/Intel/Nvidia)
- MUST handle communication protocol differences (PCIe, Ethernet, UALink)
- SHOULD match NVLink performance within 15% for homogeneous clusters

### F-4: Legacy CUDA Ingestion
**Priority:** P2 (Phase 4, Months 4+)

- MUST parse standard CUDA C++ (.cu) files
- MUST identify Nvidia-specific intrinsics and pointer semantics
- MUST translate to Triton IR
- MUST output optimized binary for any supported target
- SHOULD achieve >90% translation accuracy for common kernel patterns

### F-5: Deterministic Math Guarantee
**Priority:** P1 (Phase 4)

- MUST validate IEEE-754 compliance across all targets
- MUST detect rounding differences between hardware vendors
- MUST provide bit-exact mode sacrificing speed for deterministic output
- SHOULD be user-selectable (speed vs. determinism toggle)

### F-6: Fault Tolerance
**Priority:** P2 (Phase 4)

- MUST implement asynchronous state checkpointing during training
- MUST detect node failure in < 1 second
- MUST resume training within 3 seconds of node failure
- MUST support mixed-vendor cluster recovery

## 5. Non-Functional Requirements

### Performance
- Auto-tuned kernels must achieve ≥90% of hand-optimized CUDA performance
- Fat binary startup overhead must be < 10ms
- Auto-sharding must achieve ≥85% of NVLink bandwidth efficiency on standard Ethernet

### Compatibility
- Support Linux (primary), macOS (development), Windows (via WSL)
- Support AMD ROCm 6.x, Intel oneAPI 2025+, CUDA 12.x
- Python 3.10+ (primary API), C++17 (runtime layer)

### Reliability
- 99.9% uptime for runtime components (non-training)
- Zero silent correctness errors in bit-exact mode
- Graceful degradation when target hardware lacks features

## 6. Architecture Constraints

- **Wiring, not inventing.** All major algorithmic components must leverage existing open-source infrastructure (Triton, TVM, XLA, LLVM).
- **Neutral third-party.** Must not favor any hardware vendor. All targets get equal optimization effort.
- **C-API isolation.** Every external dependency must be wrapped behind a stable C-API to survive version drift.
- **Pinned submodules.** All upstream dependencies are pinned to specific git commits, not version ranges.

## 7. Success Metrics

| Metric | Target | Measurement |
|---|---|---|
| Kernels supported for auto-tuning | 50+ common patterns | Integration test count |
| Hardware targets supported | 5+ (Nvidia, AMD x2, Intel x2, Apple) | Build matrix |
| Auto-tuning speedup vs. default Triton | ≥30% on non-Nvidia targets | Benchmark suite |
| Fat binary overhead | < 1% performance penalty | Microbenchmarks |
| CI breaking-change detection | < 24 hours from upstream change | Drift CI pipeline |
| Developer onboarding time | < 1 hour to first compiled kernel | Docs + tutorial |

## 8. Release Plan

| Phase | Duration | Deliverable |
|---|---|---|
| Phase 1: Auto-Tuning | Month 1 | Triton ↔ TVM bridge, CLI tool, 10 benchmark kernels |
| Phase 2: AOT Packaging | Month 2 | Fat binary builder, runtime loader, AMD + Intel targets |
| Phase 3: Auto-Sharding | Month 3 | PyTorch ↔ XLA bridge, multi-node demo on dev cloud |
| Phase 4: Hardening | Months 4-6 | CUDA ingestion, memory reclaimer, fault tolerance |
| Beta Release | Month 6 | Public alpha with all 4 phases functional |
| v1.0 Release | Month 12 | Production-ready with CI, docs, package distribution |

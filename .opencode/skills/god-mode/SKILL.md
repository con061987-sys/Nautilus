---
name: god-mode
description: MUST BE USED for system-wide review, cross-cutting concern detection, or before any major merge/deploy. Activates holistic system awareness across ALL dimensions — correctness, performance, safety, maintainability, security, compatibility, and observability. Catches issues that domain-specific skills miss because they span boundaries.
---

# God Mode — Holistic System Awareness

## The All-Dimensions Checklist

Before any significant deliverable (new bridge, major refactor, release), run through ALL dimensions:

### 1. Correctness
- [ ] Type safety — any `Any`, `# type: ignore`, `as any`, `@ts-ignore`?
- [ ] Edge cases — empty tensors, zero-dim, NaN, inf, negative dimensions
- [ ] Error paths — what happens when each dependency fails? (Triton crash, TVM timeout, XLA OOM)
- [ ] Idempotency — running the same operation twice gives the same result
- [ ] Determinism — same input → same output (or documented why not)
- [ ] Round-trip — if IR goes Triton→TVM→Triton, is it lossless?

### 2. Performance
- [ ] Kernel launch overhead — are we re-compiling unnecessarily?
- [ ] Memory bandwidth — are we bandwidth-bound or compute-bound?
- [ ] Copy overhead — how many host↔device transfers per operation?
- [ ] Overlap potential — can compute and communication overlap?
- [ ] Caching — are tuning results cached across runs?

### 3. Safety & Robustness
- [ ] Error handling — every fallible operation has a fallback
- [ ] Timeouts — every external call (Triton, TVM, XLA) has a timeout
- [ ] Memory leaks — are GPU allocations freed on error paths?
- [ ] Crash isolation — if one bridge fails, does the whole system crash?
- [ ] Input validation — are malformed inputs caught early?
- [ ] Resource limits — what happens with a 100GB model on a 16GB GPU?

### 4. Maintainability
- [ ] Documentation — is there a docstring for every public API?
- [ ] Test coverage — is there a test for the happy path AND the failure path?
- [ ] Duplication — could this be unified with an existing pattern?
- [ ] Coupling — does this change require touching other bridges?
- [ ] Version drift surface — how many direct calls to unstable upstream APIs?

### 5. Compatibility
- [ ] Cross-platform — does this work on Linux? macOS? (Windows WSL?)
- [ ] Cross-vendor — does this assume Nvidia-specific behavior?
- [ ] Python version — does it work on 3.10, 3.11, 3.12?
- [ ] Dependency versions — what happens with Triton 3.1 vs 3.2?
- [ ] Hardware fallback — graceful degradation when target HW is unavailable

### 6. Security
- [ ] Unsafe deserialization — `pickle`, `eval`, `yaml.load` without safe loader?
- [ ] Shell injection — are command-line args properly escaped?
- [ ] Temporary files — written to safe directories, cleaned up on exit?
- [ ] Secrets — any API keys or tokens in code, logs, or error messages?

### 7. Observability
- [ ] Logging — are errors logged with enough context to debug?
- [ ] Metrics — can we measure performance changes over time?
- [ ] Debug mode — can we dump IR at each pipeline stage?
- [ ] Profiling — can we identify where time is spent?

## Cross-Cutting Pattern Detection

Beyond the checklist, scan for these systemic anti-patterns:

**Leaky Abstractions:**
Does a bridge expose details of one system to another? (e.g., Triton TTGIR internals leaking into TVM bridge)

**Hidden Coupling:**
Are two seemingly independent modules implicitly coupled through shared global state, file paths, or timing assumptions?

**Premature Optimization:**
Is there complex code that exists for performance reasons without benchmarks proving it matters?

**Technical Debt Acceleration:**
Does this change make future changes harder? (increased complexity, more coupling, less testability)

## Review Cadence

| Trigger | Depth |
|---|---|
| Single file edit | Quick scan (correctness + safety) |
| New bridge/module | Full checklist (all 7 dimensions) |
| Phase completion | Full checklist + cross-cutting scan |
| Pre-release | Full checklist + external review (Oracle) |

## When This Skill Triggers

- Before merging any significant PR
- Completing a Phase milestone
- Adding a new hardware target
- Modifying the IR pipeline (highest risk area)
- Before any release or deployment
- When asked to "review the whole system" or "check everything"

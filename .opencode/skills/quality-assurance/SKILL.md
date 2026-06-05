---
name: quality-assurance
description: MUST BE USED before calling any task complete, before merging, or when asked to review code. Systematic multi-layer review process covering correctness, safety, style, documentation, and edge cases. Enforces rigorous verification before anything ships.
---

# Quality Assurance — Systematic Review & Verification

## The QA Process

Every deliverable goes through 4 review layers. **Do not skip layers.**

### Layer 1: Automated Verification
Run these before any human/agent review:
- [ ] **LSP diagnostics:** `lsp_diagnostics` on all changed files — zero errors, zero warnings
- [ ] **Build:** project builds with exit code 0
- [ ] **Tests:** `python -m pytest src/tests/` — all pass (or pre-existing failures documented)
- [ ] **Lint:** `ruff` or `flake8` on Python, `clang-tidy` on C++
- [ ] **Type check:** `mypy` (Python) or `tsc --noEmit` (TypeScript)
- [ ] **Format:** code matches project style (black for Python, clang-format for C++)

If ANY automated check fails, STOP. Do not proceed to layer 2 until fixed.

### Layer 2: Structural Review
- [ ] **Single responsibility:** Does each function/class do one thing?
- [ ] **Naming:** Do names communicate intent? (not `tmp`, `data`, `helper`)
- [ ] **Duplication:** Is there code that duplicates existing project patterns?
- [ ] **Complexity:** Can any function be split? (rule of thumb: >50 lines needs justification)
- [ ] **Dependencies:** Are new imports justified? Could existing utilities be reused?
- [ ] **API surface:** Is every public API documented? (docstring with Args/Returns/Raises)
- [ ] **Error handling:** Are error paths handled, not just happy paths?

### Layer 3: Behavioral Review
- [ ] **Correctness:** Does the output match expected results for known inputs?
- [ ] **Edge cases tested (minimum):**
  - Empty input (zero tensors, empty strings, 0-length arrays)
  - Boundary values (max int, NaN, infinity)
  - Null/None/missing values (Optional params omitted)
  - Unexpected types (if function expects int, what happens with float?)
  - Concurrent access (if shared state, is it thread-safe?)
- [ ] **Graceful degradation:** If a dependency fails (Triton crash, TVM timeout, XLA OOM), does error handling hide the complexity?
- [ ] **Idempotency:** Running the same operation twice with the same state produces the same result.

### Layer 4: Domain-Specific Review

Select the review checklist relevant to this change:

**Bridge code checklist:**
- [ ] IR round-trip: source → normalize → translate produces correct output
- [ ] Error codes: every C-API call's return code is checked
- [ ] Version compat: each bridge target's dependency API hasn't drifted
- [ ] Logging: IR dumps are available in debug mode

**Kernel code checklist:**
- [ ] Occupancy analysis done (registers, shared memory budgeted)
- [ ] Memory coalescing verified (adjacent threads → adjacent addresses)
- [ ] Bank conflicts checked (shared memory stride analysis)
- [ ] Tensor Core/Matrix Core tiles aligned (16×16, 32×8, etc.)
- [ ] Boundary checking: out-of-bounds access impossible by construction

**Runtime code checklist:**
- [ ] Memory leaks: every allocation has a paired deallocation
- [ ] Timeouts: every external call has a timeout
- [ ] Signal safety: no unsafe operations in signal handlers
- [ ] Resource cleanup: temp files, GPU memory freed on all exit paths

## Review Depth by Risk

| Risk Level | Layers Required | Who Reviews |
|---|---|---|
| Trivial (typo, comment, formatting) | L1 only | Self |
| Small change (< 20 lines, no new API) | L1 + L2 | Self |
| Normal feature (new function/module) | L1 + L2 + L3 | Self + peer agent |
| High-risk (bridge, C-API, runtime) | L1 + L2 + L3 + L4 | Self + Oracle |
| Security-sensitive | Full + security audit | Self + Oracle + user |

## Bug Tracking

Every bug found during QA must be documented:
```
Location: [file:line]
Severity: [crash/incorrect/perf/docs]
Root cause: [one sentence]
Fix: [one sentence]
Regression test: [test name]
```

## Exit Criteria

A task is complete ONLY when ALL of:
1. All L1 checks pass
2. All applicable L2-L4 checks pass or are explicitly waived
3. The deliverable actually runs and produces correct output
4. The user confirms acceptance (if behavioral change)

## When This Skill Triggers

- Before marking any task complete
- Before merging any PR
- Before releasing any version
- When asked to "review this code"
- Any change to bridge code (Triton↔TVM, AOT, XLA)
- Any change to C-API or runtime code

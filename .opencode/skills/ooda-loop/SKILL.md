---
name: ooda-loop
description: MUST BE USED when iterating rapidly on a problem, debugging, or when stuck. Implements the Observe-Orient-Decide-Act cycle to ensure evidence-driven progress, prevent cargo-culting, and surface hidden assumptions. Apply whenever the first approach fails or feedback suggests a course correction.
---

# OODA Loop — Observe-Orient-Decide-Act

The OODA loop is a decision-making framework for rapid iteration. It prevents wasted effort by ensuring every action is based on observed reality, not assumptions.

## The Cycle

```
                    ┌─────────────────┐
                    │    OBSERVE      │
                    │  Gather data    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    ORIENT       │
                    │  Analyze context│
                    │  Update mental  │
                    │  model          │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    DECIDE       │
                    │  Choose action  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     ACT         │
                    │  Execute        │
                    └────────┬────────┘
                             │
                  (loop back to Observe)
```

## Phase 1: OBSERVE (Gather Evidence)

**BEFORE changing anything, collect these:**
- Exact error message (copy-paste, don't paraphrase)
- State of the system before the failure
- Input data that triggers the issue
- Logs, stack traces, core dumps
- `git diff` to confirm what's changed
- LSP diagnostics for the relevant files
- Test output (not just "tests pass/fail" — the actual output)

**Tools:** `Read`, `Bash` (for logs/state), `lsp_diagnostics`, `git diff`

**Gate:** Do not proceed to Orient until you have raw, unfiltered evidence.

## Phase 2: ORIENT (Analyze Context)

**Process the observations through these filters:**
- **Domain filters:** Is this a Triton issue? TVM? XLA? LLVM? Bridge code?
- **Pattern filters:** Have I seen this pattern before? What was the root cause then?
- **Hypothesis generation:** List 2-4 possible root causes. Rank by probability.
- **Reality check:** Does my hypothesis explain ALL observations? If not, revise.

**Key orienting questions:**
- "Is this a type error, logic error, or runtime error?"
- "Did this work before? What changed?"
- "Is this a version drift issue?" (check pinned submodules vs. installed versions)
- "Is this an environment issue?" (check Python/CUDA/ROCm versions)

**Output:** 2-4 ranked hypotheses with evidence for/against each.

## Phase 3: DECIDE (Choose Action)

**Select the highest-probability hypothesis and design a test:**
- Choose the hypothesis that explains the most facts with the fewest assumptions
- Design the MINIMUM action that confirms or refutes it
- Prefer read-only investigations (checking docs, grepping source) over destructive edits

**Decision criteria:**
- Hypothesis A (70% probability) → 5 min to test → test A first
- Hypothesis B (50% probability) → 30 min to test → save for later
- Hypothesis C (10% probability) → 2 min to test → test C quickly alongside A

**Output:** ONE concrete action with a clear success/failure condition.

## Phase 4: ACT (Execute)

**Execute the decision cleanly:**
- If it's a code change: make the MINIMUM change to test the hypothesis
- If it's a research action: `use context7` / `use gh_grep` / `websearch`
- Run the verification immediately

**After action:**
- Did it work? → Great, proceed to implementation
- Did it fail? → Loop back to OBSERVE with new data
- Did something unexpected happen? → Loop back to ORIENT (your mental model was wrong)

## Cycle Completion Criteria

A full OODA cycle is complete when EITHER:
- The problem is fixed (verified by running the failing case) → exit loop
- A hypothesis is conclusively refuted → start new cycle with remaining hypotheses
- After 3 cycles without progress → STOP and consult Oracle

## OODA in Different Contexts

| Context | Observe | Orient | Decide | Act |
|---|---|---|---|---|
| Bug fix | Error message, logs, state | Root cause analysis | Fix approach | Edit + test |
| Feature impl | Requirements, existing code | Architecture fit | Implementation plan | Code + verify |
| Performance | Benchmarks, profiles | Bottleneck ID | Optimization strategy | Tune + measure |
| Integration | Test failures, error codes | Component boundaries | Interface change | Wire + validate |

## Anti-Patterns

- ❌ Acting without observing ("I bet the fix is...")
- ❌ Orienting without new data after a failed action
- ❌ Multiple simultaneous changes (can't tell which fixed it)
- ❌ Confirmation bias (only looking for evidence that supports your hypothesis)
- ❌ Slow loops (if a cycle takes >20 min, you're doing something too big)

## When This Skill Triggers

- A test fails unexpectedly
- A compiler error doesn't match expectations
- First approach to a problem fails
- Performance numbers don't meet targets
- Any "that shouldn't happen" moment

---
name: tune-kernel
description: Run TVM MetaSchedule auto-tuning on a Triton kernel.
template: "Run TVM MetaSchedule auto-tuning on the Triton kernel at $ARGUMENTS. Extract the TTGIR, normalize to MLIR Vector Dialect, convert to TVM TIR, and run evolutionary search. Report the best configuration found."
agent: compiler-engineer
---

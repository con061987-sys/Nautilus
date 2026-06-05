---
name: compile-kernel
description: Compile a Triton kernel through the full pipeline with auto-tuning, AOT packaging, and deployment.
template: "Run the full compilation pipeline for the Triton kernel at $ARGUMENTS. Include auto-tuning via TVM MetaSchedule, AOT compilation for all targets, and fat binary packaging."
agent: compiler-engineer
---

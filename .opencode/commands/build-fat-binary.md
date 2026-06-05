---
name: build-fat-binary
description: Build the fat binary for current platform or all targets.
template: "Build the fat binary for the Triton kernel at $ARGUMENTS. Compile for all available backends (AMD, Intel, Nvidia) and link using LLVM lld. Output the fat binary .o file and the runtime C stub."
agent: compiler-engineer
---

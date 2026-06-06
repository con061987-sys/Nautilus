# third_party/ — Pinned upstream dependencies.

This directory contains pinned Git submodules for all major upstream
dependencies. Each submodule is pinned to a specific release tag,
providing exact source-trees for C-API builds and reproducible builds.

To initialize all submodules after cloning:
```bash
git submodule update --init --recursive
```

## Recommended submodules (pinned commits)

| Repo | Pin | Notes |
|------|-----|-------|
| triton | `v3.0.0` (55a4ab0) | https://github.com/triton-lang/triton |
| tvm | `v0.18.0` (22a9d38) | https://github.com/apache/tvm |
| xla (openxla) | `e115cfc` | https://github.com/openxla/xla |
| llvm-project | `llvmorg-19.1.0` (a4bf6cd) | https://github.com/llvm/llvm-project |
| aotriton | `0.10` (future) | https://github.com/ROCm/aotriton |

## How to add a submodule

```bash
# Add a submodule pinned to a specific commit
git submodule add https://github.com/triton-lang/triton.git third_party/triton
cd third_party/triton
git checkout v3.0.0
cd ../..
git add third_party/triton
git commit -m "third_party: pin triton to v3.0.0"
```

## Updating a pin

```bash
# Update to a new commit
cd third_party/triton
git fetch
git checkout v3.1.0
cd ../..
git add third_party/triton
git commit -m "third_party: bump triton to v3.1.0"
# Then update pyproject.toml to match.
```

## Why pinned submodules?

When upstream libraries (Triton, TVM, XLA) change their APIs,
any code that calls them directly breaks. The C-API wrapper
(`src/c_api/`) is supposed to insulate us from this, but the
Python bindings still need to track upstream. Pinning the exact
commit means we control when the API changes affect us, not
arbitrary upstream release schedules.

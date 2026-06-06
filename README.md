# Nautilus — Cross-Vendor AI Compilation Framework

Nautilus is an open-source framework that compiles AI models for
**any** GPU vendor (Nvidia, AMD, Intel, Apple) at near-native
performance, without manual tuning, and shards them automatically
across mixed-vendor clusters.

## What it does

- **Auto-tunes** GPU kernels using TVM MetaSchedule (replaces
  hand-tuning)
- **Bundles** per-vendor binaries into a single fat binary that
  dispatches at runtime
- **Auto-shards** models across mixed AMD/Intel/Nvidia clusters
  using XLA GSPMD
- **Ingests** legacy CUDA C++ and translates to portable Triton

## Quick start

```bash
# Install
git clone https://github.com/nvindia-cud/nautilus
cd nautilus
./scripts/setup-cuda.sh       # or setup-rocm.sh

# Verify
nautilus verify

# Compile a Triton kernel into a fat binary
nautilus tune my_kernel.py --target nvidia/sm_90
nautilus build my_kernel.py --target nvidia/sm_90 --target amd/gfx942 -o my.fat.o

# Shard a PyTorch model
nautilus shard my_model.py --mesh 2,2 --output-dir ./shards
```

## Documentation

- [User Guide](docs/USER_GUIDE.md) — installation, first compile
- [Architecture](docs/ARCHITECTURE.md) — how Nautilus is wired
- [Contributing](docs/CONTRIBUTING.md) — how to add code
- [PRD](docs/PRD.md) — product requirements
- [Tech Spec](docs/TECH_SPEC.md) — technical architecture

## Status

Nautilus is in active development. The current version implements
the full 4-phase architecture with all bridges wired end-to-end,
but compilation requires the corresponding hardware SDKs (CUDA
toolkit, ROCm, oneAPI). See [Troubleshooting](docs/TROUBLESHOOTING.md)
if you hit issues.

## License

Apache 2.0 — see [LICENSE](LICENSE).

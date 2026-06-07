# Nautilus Benchmarks

The benchmark suite measures compile time, kernel execution time,
peak memory, and per-vendor binary size across the full Nautilus
pipeline. It is wired into the CLI as ``nautilus bench``.

## Quick start

```bash
# 1. Run every discovered benchmark. Writes a result set to
#    benchmarks/results/bench_<id>.json.
nautilus bench run

# 2. Compare the two most recent runs. Exits non-zero on regression
#    when combined with --exit-on-regression.
nautilus bench compare --latest 2 --exit-on-regression

# 3. Show the trend for a metric over the last 10 runs.
nautilus bench history --benchmark kernels/matmul --metric exec_time_s --last 10

# 4. Smoke check that discovery sees your new benchmark.
nautilus bench list
```

## Suite structure

```
benchmarks/
├── __init__.py             # package + schema re-exports
├── results.py              # ResultSet, BenchmarkResult, compare, history
├── runner.py               # BenchmarkRunner, BenchmarkProtocol, RunContext
├── kernels/
│   ├── matmul_bench.py     # SGEMM (Triton + numpy fallback)
│   ├── attention_bench.py  # flash-style attention (Triton + numpy)
│   └── layer_norm_bench.py # row-wise LayerNorm (Triton + numpy)
├── models/
│   ├── __init__.py
│   ├── resnet50_bench.py   # ResNet-50 (torch.compile + eager)
│   └── llama_tiny_bench.py # LLaMA-tiny (torch.generate + torch.compile)
└── run_benchmarks.py       # legacy argparse runner (kept for back-compat)
```

Discovery convention: a module ending in ``_bench.py`` that exports
a ``BENCHMARK`` instance (with ``name()``, ``targets()``, and ``run()``
methods) is picked up automatically by ``nautilus bench run``.

## What each benchmark measures

Every benchmark records four headline metrics per (benchmark, vendor)
pair, plus benchmark-specific extras:

| Metric          | Unit | What it tells you                                  |
|-----------------|------|----------------------------------------------------|
| `compile_time_s`| s    | Wall-clock time to produce the per-vendor binary   |
| `exec_time_s`   | s    | Median kernel or forward-pass time (over trials)   |
| `memory_mb`     | MiB  | Peak RSS during execution                          |
| `binary_size_b` | B    | Size of the per-vendor binary blob                 |

A benchmark with an unavailable dependency (no CUDA, no transformers)
is recorded as ``status="skipped"`` rather than crashing the suite.

## Regression thresholds

Defaults (overridable via CLI flags):

| Metric          | Threshold | Min absolute delta |
|-----------------|-----------|--------------------|
| `exec_time_s`   |  5%       | 1 ms               |
| `compile_time_s`| 10%       | 50 ms              |
| `binary_size_b` | 20%       | 1 KiB              |
| `memory_mb`     | 15%       | 1 MiB              |

A metric is flagged only if **both** the fractional threshold and
the absolute minimum are exceeded. This avoids noise on tiny
baselines (a 0.5 ms regression on a 1 ms baseline is not a 50% loss).

## CLI reference

### `nautilus bench run`

```bash
nautilus bench run \
  [--benchmark NAME ...] \
  [--target VENDOR/ARCH ...] \
  [--output-dir DIR] \
  [--trials N] [--warmup N] [--timeout SECS] \
  [--fail-fast/--no-fail-fast] \
  [--json]
```

### `nautilus bench compare`

```bash
nautilus bench compare \
  [--baseline FILE] [--candidate FILE] \
  [--latest N] \
  [--results-dir DIR] \
  [--direction all|regressions|improvements] \
  [--exec-threshold F] [--compile-threshold F] \
  [--binary-size-threshold F] [--memory-threshold F] \
  [--format table|json|markdown] \
  [--exit-on-regression]
```

### `nautilus bench history`

```bash
nautilus bench history \
  [--benchmark NAME] [--vendor VENDOR/ARCH] \
  [--metric exec_time_s|compile_time_s|binary_size_b|memory_mb] \
  [--results-dir DIR] [--last N] [--all] \
  [--format table|json]
```

### `nautilus bench list`

Lists every discovered benchmark + its declared targets.

## Result schema

```json
{
  "run_id": "20250607T094215Z",
  "started_at": "2025-06-07T09:42:15+00:00",
  "finished_at": "2025-06-07T09:43:01+00:00",
  "git_sha": "abc1234",
  "host": "runner-1",
  "nautilus_version": "0.1.0",
  "schema_version": 1,
  "results": [
    {
      "benchmark": "kernels/matmul",
      "vendor": "nvidia/sm_90",
      "status": "ok",
      "compile_time_s": 1.842,
      "exec_time_s": 0.0042,
      "memory_mb": 412.0,
      "binary_size_b": 184320,
      "params": {"M": 1024, "N": 1024, "K": 1024, "dtype": "float16"},
      "extras": {"tflops": 512.3, "grid": [8, 8]}
    }
  ]
}
```

The on-disk schema is versioned via ``schema_version``. Loaders
refuse to read mismatched versions; bump the constant in
``benchmarks/results.py`` and write a migration if you change it.

## Adding a new benchmark

1. Create ``benchmarks/kernels/<name>_bench.py`` (or
   ``benchmarks/models/<name>_bench.py`` for model-level work).
2. Implement a class with:
   - ``name() -> str`` — e.g. ``"kernels/my_kernel"``
   - ``targets() -> list[str]`` — e.g. ``["nvidia/sm_90", "amd/gfx942", "cpu"]``
   - ``run(target, ctx) -> dict`` — must return a dict with
     ``status`` plus any of ``compile_time_s``, ``exec_time_s``,
     ``memory_mb``, ``binary_size_b``, ``params``, ``extras``.
3. Expose ``BENCHMARK = MyBenchmark()`` at module level.
4. Verify discovery: ``nautilus bench list`` should print your
   benchmark with its declared targets.

## Nightly benchmarks

``.github/workflows/nightly-benchmarks.yml`` runs the suite daily at
06:00 UTC across CPU + AMD Developer Cloud + Intel Tiber AI Cloud.
The CPU job is the gate; on a regression it opens a GitHub issue
with a markdown diff and uploads all result sets as artifacts.

To trigger a manual run: GitHub → Actions → Nightly Benchmarks →
Run workflow.

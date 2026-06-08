"""`nautilus inspect` — Deep system verification commands.

Subcommands
-----------
* ``fat-binary <file>``  inspect a fat binary file (legacy, unchanged)
* ``topology``            device discovery and interconnect
* ``toolchain``           build tool availability (gcc, lld, llvm-spirv, ...)
* ``backends``            test each vendor backend availability
* ``compliance``          IEEE-754 math validation settings
* ``pipeline``            full pipeline dry-run with timing

All subcommands accept ``--format {text,json,yaml}`` (default: text)
and ``topology`` accepts ``--detailed`` for verbose output. Each
subcommand is independent: missing tools, missing hardware, or
missing optional dependencies never crash the command; they are
reported as fields in the output so callers can detect degradation
programmatically.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import click

from src.common.logging import get_logger

log = get_logger("nautilus.cli.inspect")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


# Try to import yaml at module load time. PyYAML is a runtime dep, but
# if it ever goes missing we still want the other formats to work —
# yaml() then raises a clean Click error.
try:
    import yaml  # type: ignore[import-untyped]

    _HAVE_YAML = True
except ImportError:  # pragma: no cover - PyYAML is in pyproject
    _HAVE_YAML = False


def _render(data: Any, fmt: str, detailed: bool = False) -> str:
    """Render ``data`` in the requested format.

    The ``detailed`` flag only affects the ``text`` renderer — for
    json/yaml, callers can pass any shape they want and it is
    serialised verbatim.
    """
    if fmt == "json":
        return json.dumps(data, indent=2, default=str, sort_keys=False)
    if fmt == "yaml":
        if not _HAVE_YAML:
            raise click.ClickException(
                "PyYAML is required for --format yaml. Install with: pip install pyyaml",
            )
        return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    # text
    return _render_text(data, detailed=detailed)


def _render_text(data: Any, detailed: bool = False, indent: int = 0) -> str:
    """Render arbitrary nested data as a human-readable text table.

    Falls back to ``repr`` for values that don't fit the known shape
    (dict, list, scalar). Designed for the "quick eyeball" case —
    structured consumers should always pick ``--format json``.
    """
    pad = "  " * indent
    if isinstance(data, dict):
        if not data:
            return f"{pad}(empty)"
        # Find the widest key for column alignment.
        width = max(len(str(k)) for k in data)
        lines: list[str] = []
        for k, v in data.items():
            key = f"{pad}{k:<{width}}"
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{key} :")
                lines.append(_render_text(v, detailed=detailed, indent=indent + 1))
            elif v is None or v == "":
                lines.append(f"{key} : <missing>")
            elif isinstance(v, bool):
                marker = "YES" if v else "no"
                lines.append(f"{key} : {marker}")
            else:
                lines.append(f"{key} : {v}")
        return "\n".join(lines)
    if isinstance(data, list):
        if not data:
            return f"{pad}(empty list)"
        lines = []
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}[{i}]")
                lines.append(_render_text(item, detailed=detailed, indent=indent + 1))
            else:
                lines.append(f"{pad}- {item}")
        return "\n".join(lines)
    return f"{pad}{data!r}"


def _echo(data: Any, fmt: str, detailed: bool = False) -> None:
    """Render + echo to stdout in one step."""
    click.echo(_render(data, fmt, detailed=detailed))


# ---------------------------------------------------------------------------
# Tool detection helpers
# ---------------------------------------------------------------------------


def _which(*names: str) -> str | None:
    """Return the first existing path among ``names``, else None."""
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def _tool_record(*names: str, version_arg: str = "--version") -> dict[str, Any]:
    """Probe a tool: returns ``{path, available, version}``.

    Never raises — missing tool returns ``available=False`` with the
    candidates it tried, so callers can surface that in the report.
    """
    path = _which(*names)
    record: dict[str, Any] = {
        "candidates": list(names),
        "path": path,
        "available": path is not None,
    }
    if path is not None:
        record["version"] = _safe_version(path, version_arg)
    else:
        record["version"] = None
    return record


def _safe_version(path: str, version_arg: str) -> str | None:
    """Best-effort ``<tool> --version`` capture.

    Returns the first non-empty line trimmed, or None on any failure.
    Bounded by a short timeout so a hung tool can't hang inspection.
    """
    import subprocess

    try:
        out = subprocess.run(
            [path, version_arg],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout or out.stderr or "").strip()
    if not text:
        return None
    return text.splitlines()[0][:200]


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------


@click.group(
    "inspect",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def cli() -> None:
    """Deep system verification commands.

    Every subcommand is independent and degrades gracefully when
    tools, hardware, or optional dependencies are missing — those
    are reported as fields in the output, never raised as crashes.
    """


# Common options applied to most subcommands. Reused via ``@click.option``
# + decorator stacking on each command below.
_FMT_OPTION = click.option(
    "--format",
    "fmt",
    default="text",
    type=click.Choice(["text", "json", "yaml"], case_sensitive=False),
    show_default=True,
    help="Output format.",
)


# ---------------------------------------------------------------------------
# fat-binary (legacy — unchanged behaviour, still outputs JSON)
# ---------------------------------------------------------------------------


@cli.command("fat-binary")
@click.argument("fat_binary", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def fat_binary_cmd(fat_binary: Path) -> None:
    """Inspect a fat binary file and print its metadata as JSON."""
    from src.bridges.aot_packager.fat_binary import FatBinary

    fb = FatBinary.from_bytes(fat_binary.read_bytes())
    click.echo(
        json.dumps(
            {
                "kernel_name": fb.kernel_name,
                "sections": [
                    {
                        "vendor": s.vendor,
                        "format": s.format.value,
                        "size": len(s.data),
                    }
                    for s in fb.sections
                ],
                "total_size": fb.total_size,
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------


@cli.command("topology")
@_FMT_OPTION
@click.option(
    "--detailed",
    is_flag=True,
    default=False,
    help="Include per-PCIe BDF, NUMA, and lspci raw data in the output.",
)
def topology_cmd(fmt: str, detailed: bool) -> None:
    """Device discovery and interconnect.

    Discovers every GPU/accelerator on the system via
    ``discover_topology()`` and reports the host, per-device details,
    bandwidth matrix, and (with ``--detailed``) raw probe records.
    """
    from src.common.hardware import discover_topology

    topo = discover_topology()
    payload = topo.to_dict()
    if not detailed:
        # Trim ``raw`` fields from each device — those are diagnostic-only.
        for dev in payload.get("devices", []):
            dev.pop("raw", None)
    _echo(payload, fmt, detailed=detailed)


# ---------------------------------------------------------------------------
# toolchain
# ---------------------------------------------------------------------------


# Canonical tool set. Keeping it in one place so the JSON / YAML keys
# stay stable across formats and so adding a new tool is a one-line
# change.
_TOOL_PROBES: list[tuple[str, tuple[str, ...]]] = [
    # Compilers
    ("gcc", ("gcc",)),
    ("g++", ("g++",)),
    ("clang", ("clang",)),
    ("clang++", ("clang++",)),
    ("rustc", ("rustc",)),
    ("cmake", ("cmake",)),
    ("ninja", ("ninja",)),
    # Linker
    ("lld", ("lld", "ld.lld")),
    # SPIR-V / Vulkan toolchain
    ("llvm-spirv", ("llvm-spirv",)),
    ("spirv-val", ("spirv-val",)),
    ("spirv-link", ("spirv-link",)),
    # Vendor toolchains
    ("nvcc", ("nvcc",)),
    ("nvidia-smi", ("nvidia-smi",)),
    ("amdclang", ("amdclang", "amdclang++")),
    ("rocm-smi", ("rocm-smi",)),
    ("aotriton", ("aotriton",)),
    ("icpx", ("icpx",)),
    # Misc
    ("python", ("python", sys.executable)),
    ("git", ("git",)),
    ("lspci", ("lspci",)),
]


@cli.command("toolchain")
@_FMT_OPTION
def toolchain_cmd(fmt: str) -> None:
    """List build-tool availability (path + version)."""
    tools: dict[str, dict[str, Any]] = {}
    for name, candidates in _TOOL_PROBES:
        tools[name] = _tool_record(*candidates)

    available = [n for n, r in tools.items() if r["available"]]
    missing = [n for n, r in tools.items() if not r["available"]]

    summary: dict[str, Any] = {
        "available_count": len(available),
        "missing_count": len(missing),
        "available": sorted(available),
        "missing": sorted(missing),
        "tools": tools,
    }
    _echo(summary, fmt)


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


def _probe_backend(vendor: str, module_attr: str) -> dict[str, Any]:
    """Try to import + instantiate one vendor backend module.

    Returns a structured record so the CLI output is consistent
    whether the backend is wired or not.
    """
    rec: dict[str, Any] = {
        "vendor": vendor,
        "importable": False,
        "instantiable": False,
        "class": None,
        "error": None,
    }
    try:
        mod = __import__(
            f"src.bridges.aot_packager.{module_attr}",
            fromlist=["*"],
        )
        rec["importable"] = True
    except Exception as exc:  # pragma: no cover - exercised in CI via missing deps
        rec["error"] = f"import: {type(exc).__name__}: {exc}"
        return rec

    # Heuristic: pick the first class defined in the module that
    # isn't imported from elsewhere. The vendor backends follow the
    # ``<Vendor>Backend`` naming convention.
    cls_name = f"{vendor.capitalize()}Backend"
    cls = getattr(mod, cls_name, None)
    if cls is None:
        # Fall back: any class whose name ends in 'Backend'.
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name, None)
            if isinstance(obj, type) and attr_name.endswith("Backend"):
                cls = obj
                cls_name = attr_name
                break
    if cls is None:
        rec["error"] = f"no Backend class found in {module_attr}.py"
        return rec
    rec["class"] = cls_name
    try:
        instance = cls()
        rec["instantiable"] = True
        rec["class_repr"] = type(instance).__name__
    except Exception as exc:
        rec["error"] = f"instantiate: {type(exc).__name__}: {exc}"
    return rec


@cli.command("backends")
@_FMT_OPTION
def backends_cmd(fmt: str) -> None:
    """Test each vendor backend availability.

    For each of Nvidia / AMD / Intel / Apple, attempts to import the
    backend module and instantiate the backend class. A backend that
    is wired but missing the vendor toolchain is still reported as
    ``instantiable=true`` — runtime compilation is what will fail
    later, not the wiring.
    """
    vendor_modules = {
        "nvidia": "nvidia_backend",
        "amd": "amd_backend",
        "intel": "intel_backend",
        "apple": "metal_backend",
    }
    records: dict[str, dict[str, Any]] = {}
    for vendor, mod in vendor_modules.items():
        records[vendor] = _probe_backend(vendor, mod)

    # Also probe the orchestrator builder — it ties all backends together.
    builder_rec: dict[str, Any] = {
        "importable": False,
        "instantiable": False,
        "class": None,
        "error": None,
    }
    try:
        from src.bridges.aot_packager.builder import FatBinaryBuilder

        builder_rec["importable"] = True
        builder_rec["class"] = "FatBinaryBuilder"
        try:
            FatBinaryBuilder(cache_dir="/tmp/nautilus-inspect-builder")
            builder_rec["instantiable"] = True
        except Exception as exc:
            builder_rec["error"] = f"instantiate: {type(exc).__name__}: {exc}"
    except Exception as exc:
        builder_rec["error"] = f"import: {type(exc).__name__}: {exc}"
    records["builder"] = builder_rec

    available = [v for v, r in records.items() if r.get("instantiable")]
    summary: dict[str, Any] = {
        "available_count": len(available),
        "available": sorted(available),
        "backends": records,
    }
    _echo(summary, fmt)


# ---------------------------------------------------------------------------
# compliance
# ---------------------------------------------------------------------------


@cli.command("compliance")
@_FMT_OPTION
@click.option(
    "--emulate/--no-emulate",
    "emulate",
    default=True,
    show_default=True,
    help="Enable software emulation of Nvidia features on non-Nvidia hardware.",
)
def compliance_cmd(fmt: str, emulate: bool) -> None:
    """Validate IEEE-754 math and Nvidia feature emulation status.

    Reports the current ``MathValidator`` configuration: bit-exact
    mode, default strictness level, and the per-op overrides
    registered so far. Also shows the software emulation status for
    Nvidia-specific features (FP4, FP8, Transformer Engine) when
    running on non-Nvidia hardware.

    This is a *configuration* check — actually validating that a
    kernel produces IEEE-754-correct results requires running the
    kernel on real hardware and is out of scope for ``inspect``.
    """
    from src.runtime.math_validator import MathValidator, StrictnessLevel

    try:
        validator = MathValidator()
        stats = validator.get_stats()
        op_specs = {
            name: {
                "strictness": spec.strictness.name,
                "tolerance": spec.tolerance,
                "notes": spec.notes,
            }
            for name, spec in validator._op_specs.items()  # diagnostic-only read
        }
        summary: dict[str, Any] = {
            "available": True,
            "bit_exact_mode": stats["bit_exact_mode"],
            "default_strictness": stats["default_strictness"],
            "op_overrides": stats["op_overrides"],
            "strictness_levels": [s.name for s in StrictnessLevel],
            "op_specs": op_specs,
            "notes": (
                "Bit-exact mode forces IEEE-754 strict-fp; default "
                "strictness is ULP_4 (4 ULPs tolerance). Override "
                "per-op via MathValidator.set_op_strictness()."
            ),
        }
    except Exception as exc:
        summary = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        from src.bridges.triton_tvm.sw_emulation import SWEmulationEngine, ModelGraph

        engine = SWEmulationEngine(auto_emulate=emulate)
        plans = engine.detect_nvidia_features(ModelGraph())
        summary["sw_emulation"] = engine.get_summary(plans)
    except Exception as exc:
        summary["sw_emulation"] = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    _echo(summary, fmt)


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def _clean_stage_summary(outcome: Any) -> dict[str, Any]:
    """Tidy the per-stage summary dict for the inspector output.

    The pipeline populates ``summary["would_run"]`` with the bound
    method's ``__name__`` which, for lambda-wired handlers, comes
    out as the literal string ``"<lambda>"`` — useless to consumers.
    Rewrite that to a clean ``stage_<name>`` so the output is
    stable, and drop the field entirely on non-dry-run stages.
    """
    raw = outcome.summary or {}
    cleaned: dict[str, Any] = dict(raw)
    if "would_run" in cleaned:
        value = cleaned["would_run"]
        if value == "<lambda>":
            cleaned["would_run"] = f"stage_{outcome.stage.value}"
        elif not isinstance(value, str):
            cleaned["would_run"] = str(value)
    return cleaned


_DEFAULT_PIPELINE_INPUT = Path("benchmarks/kernels/matmul.py")


@cli.command("pipeline")
@_FMT_OPTION
@click.option(
    "--input",
    "input_file",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
    default=_DEFAULT_PIPELINE_INPUT,
    show_default=True,
    help="Input Triton kernel / PyTorch model to dry-run.",
)
@click.option(
    "--target",
    "-t",
    "targets",
    multiple=True,
    help="Target hardware as 'vendor/arch' (e.g. nvidia/sm_90). Default: one per vendor.",
)
def pipeline_cmd(fmt: str, input_file: Path, targets: tuple[str, ...]) -> None:
    """Run the full pipeline dry-run with timing.

    Executes every stage (``capture`` → ``shard`` → ``extract`` →
    ``tune`` → ``build`` → ``dispatch``) but does NOT do the
    expensive work — each stage returns a plan with measured
    duration ``0.0`` while the rest of the pipeline orchestrator
    still records which stages *would* run and in what order.

    If the input file is missing, the report contains an explicit
    ``input_error`` field and stage outcomes show ``skipped=true``
    rather than crashing.
    """
    from src.cli.commands.pipeline import (
        Pipeline,
        PipelineContext,
        PipelineStage,
    )
    from src.cli.commands.pipeline import _parse_targets as parse_targets

    hardware_targets = parse_targets(tuple(targets))

    summary: dict[str, Any] = {
        "input": str(input_file),
        "input_exists": input_file.exists(),
        "dry_run": True,
        "targets": [t.to_tvm_target() for t in hardware_targets],
    }

    if not input_file.exists():
        summary["input_error"] = f"input file not found: {input_file}"
        summary["stages"] = []
        summary["total_duration_ms"] = 0.0
        _echo(summary, fmt)
        return

    # Run in-memory: no state.json / manifest written because the
    # Pipeline class guards both with ``if not self.dry_run``. Pass
    # ``output_dir=None`` so no temp dir is created.
    ctx = PipelineContext(
        input_path=input_file,
        target_strings=[t.to_tvm_target() for t in hardware_targets],
        output_dir=None,
        dry_run=True,
    )
    pipeline = Pipeline(
        ctx=ctx,
        targets=hardware_targets,
        dry_run=True,
        parallel=True,
    )

    t0 = time.perf_counter()
    try:
        outcomes = pipeline.run()
    except Exception as exc:  # pragma: no cover - orchestration path
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["stages"] = []
        summary["total_duration_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        _echo(summary, fmt)
        return
    total_ms = round((time.perf_counter() - t0) * 1000, 3)

    summary["stages"] = [
        {
            "stage": o.stage.value,
            "success": o.success,
            "skipped": o.skipped,
            "dry_run": o.dry_run,
            "duration_ms": round(o.duration_ms, 3),
            "summary": _clean_stage_summary(o),
            "error": o.error,
        }
        for o in outcomes
    ]
    summary["total_duration_ms"] = total_ms
    summary["all_success"] = all(o.success or o.skipped for o in outcomes)
    summary["stage_count"] = len(outcomes)
    summary["stages_attempted"] = [o.stage.value for o in outcomes]
    # Surface the canonical stage list so consumers can diff
    # ``stages_attempted`` against ``stages_canonical`` to detect
    # short-circuited runs.
    summary["stages_canonical"] = [s.value for s in PipelineStage]
    _echo(summary, fmt)


__all__ = ["cli"]

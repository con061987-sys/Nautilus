"""Pytest configuration for integration tests.

Adds fixtures for skipping GPU-dependent tests when no GPU is available,
and for the optional dependency marker.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_COLLECTION_IMPORT_ERRORS: list[tuple[str, str]] = []


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip tests based on missing prerequisites.

    - @pytest.mark.gpu: skip if no nvidia-smi / rocm-smi / lspci
    - @pytest.mark.cuda: skip if no nvcc
    - @pytest.mark.rocm: skip if no /opt/rocm
    - @pytest.mark.intel: skip if no spirv-val
    - @pytest.mark.requires_deps: skip if torch/tvm/triton not importable
    """
    has_cuda = bool(shutil.which("nvidia-smi") or shutil.which("nvcc"))
    has_rocm = Path("/opt/rocm").exists() or bool(shutil.which("rocm-smi"))
    has_intel = bool(shutil.which("spirv-val")) or bool(shutil.which("ocloc"))
    has_gpu = has_cuda or has_rocm or has_intel

    has_torch = False
    has_tvm = False
    has_triton = False
    has_torch_xla = False
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        pass
    try:
        import tvm  # noqa: F401
        has_tvm = True
    except ImportError:
        pass
    try:
        import triton  # noqa: F401
        has_triton = True
    except ImportError:
        pass
    try:
        import torch_xla  # noqa: F401
        has_torch_xla = True
    except ImportError:
        pass

    for item in items:
        markers = {m.name for m in item.iter_markers()}
        if "gpu" in markers and not has_gpu:
            item.add_marker(pytest.mark.skip(reason="No GPU detected (nvidia-smi/rocm-smi/lspci missing)"))
        if "cuda" in markers and not has_cuda:
            item.add_marker(pytest.mark.skip(reason="CUDA toolkit not installed"))
        if "rocm" in markers and not has_rocm:
            item.add_marker(pytest.mark.skip(reason="ROCm not installed at /opt/rocm"))
        if "intel" in markers and not has_intel:
            item.add_marker(pytest.mark.skip(reason="Intel oneAPI / SPIRV-Tools not installed"))
        requires_deps = item.get_closest_marker("requires_deps")
        if requires_deps:
            for dep in requires_deps.args:
                try:
                    __import__(dep)
                except ImportError:
                    item.add_marker(pytest.mark.skip(reason=f"requires {dep}"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Path to the Nautilus repo root."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def any_gpu_available() -> bool:
    return bool(shutil.which("nvidia-smi") or Path("/opt/rocm").exists() or shutil.which("spirv-val"))


@pytest.fixture
def clean_cache(tmp_path: Path) -> Path:
    """Provide an isolated cache directory so tests don't pollute user cache."""
    return tmp_path


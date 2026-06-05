"""Build script for the Triton ↔ TVM bridge.

This script handles:
  1. Building the C++ shared library (lib/CMakeLists.txt)
  2. Installing the Python package with all entry points
  3. Verifying the build is consistent
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def build_cpp_plugin() -> None:
    """Build the C++ shared library plugin.

    Skipped silently if CMake or the build tools are not available —
    the Python-only mode still works via TRITON_PLUGIN_DIRS pointing
    to the Python backend package.
    """
    lib_dir = Path(__file__).parent / "src" / "bridges" / "triton_tvm" / "lib"
    if not lib_dir.exists():
        return

    # Only build if user explicitly requests it
    if "--no-cpp" in sys.argv:
        sys.argv.remove("--no-cpp")
        return

    if not (lib_dir / "CMakeLists.txt").exists():
        return

    triton_src = os.environ.get("TRITON_SRC_DIR")
    if not triton_src:
        print(
            "Note: TRITON_SRC_DIR not set — skipping C++ plugin build.\n"
            "The Python backend works without the C++ plugin via TRITON_PLUGIN_DIRS.\n"
            "To build the C++ plugin for native IR capture:\n"
            "  export TRITON_SRC_DIR=/path/to/triton/source\n"
            "  export MLIR_DIR=/path/to/llvm/build/lib/cmake/mlir\n"
            "  pip install -e . --no-build-isolation\n",
            file=sys.stderr,
        )
        return

    build_dir = lib_dir / "build"
    build_dir.mkdir(exist_ok=True)

    mlir_dir = os.environ.get("MLIR_DIR")
    if not mlir_dir:
        print(
            "Warning: MLIR_DIR not set — cannot build C++ plugin.\n"
            "Set MLIR_DIR=/path/to/llvm/build/lib/cmake/mlir to enable.",
            file=sys.stderr,
        )
        return

    print(f"Building C++ plugin: triton_src={triton_src} mlir_dir={mlir_dir}")

    # Configure
    subprocess.run([
        "cmake", "-GNinja",
        f"-DTRITON_SRC_DIR={triton_src}",
        f"-DMLIR_DIR={mlir_dir}",
        "-S", str(lib_dir),
        "-B", str(build_dir),
    ], check=True)

    # Build
    subprocess.run([
        "cmake", "--build", str(build_dir),
    ], check=True)

    print("C++ plugin built successfully.")


# Hook into setuptools
if __name__ == "__main__":
    # When run directly (not via pip), just build the C++ plugin
    build_cpp_plugin()
else:
    # When imported by setuptools, hook into the build process
    try:
        from setuptools import setup as _setup
        from setuptools.command.build_py import build_py as _build_py
        from setuptools.command.develop import develop as _develop

        class CustomBuildPy(_build_py):
            def run(self) -> None:
                build_cpp_plugin()
                super().run()

        class CustomDevelop(_develop):
            def run(self) -> None:
                build_cpp_plugin()
                super().run()

        # Re-export with custom commands
        # Note: this is a simplified version; a real build would use
        # scikit-build-core or py-build for proper C++ integration
    except ImportError:
        pass

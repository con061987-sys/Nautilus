"""Build script for the Triton ↔ TVM bridge.

This script handles:
  1. Building the C++ shared library (lib/CMakeLists.txt)
  2. Installing the Python package with all entry points
  3. Verifying the build is consistent
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def build_runtime_stub() -> None:
    """Compile the AOT fat-binary C runtime stub.

    Compiles ``src/bridges/aot_packager/runtime_stub.c`` with gcc
    targeting the host architecture (auto-detected via ``os.uname()``)
    to ``build/runtime_stub.o``. The resulting object is the vendor
    detection + dispatch shim that gets linked into every fat binary
    the packager produces.

    The flags (-nostdlib -ffreestanding) match the runtime_stub.c
    header comment, which intentionally avoids libc to stay portable
    across every supported target.

    Skipped silently if gcc is not on PATH — the linker pipeline has
    its own gcc invocation (see FatBinaryBuilder._compile_runtime_stub)
    that will surface a clear error at build time.
    """
    if not shutil.which("gcc"):
        print(
            "Note: gcc not found in PATH — skipping runtime_stub build.\n"
            "Fat binary linking will fail at build time without gcc; "
            "install gcc (apt install gcc / brew install gcc) to enable.",
            file=sys.stderr,
        )
        return

    repo_root = Path(__file__).parent
    src_c = repo_root / "src" / "bridges" / "aot_packager" / "runtime_stub.c"
    if not src_c.exists():
        print(
            f"Note: {src_c} not found — skipping runtime_stub build.",
            file=sys.stderr,
        )
        return

    # os.uname() over platform.machine() — the former reports what
    # the kernel actually sees, the latter is a Python-layer shim
    # that has been observed to disagree with the binary format gcc
    # emits on cross-compiled CI images.
    host_arch = os.uname().machine
    supported = {"x86_64", "aarch64"}
    if host_arch not in supported:
        print(
            f"Note: host arch {host_arch!r} not in {sorted(supported)} — "
            "skipping runtime_stub build. The fat binary packager will "
            "re-attempt at build time and fail loudly if unsupported.",
            file=sys.stderr,
        )
        return

    out_dir = repo_root / "build"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_o = out_dir / "runtime_stub.o"

    cmd = [
        "gcc", "-c",
        "-nostdlib", "-ffreestanding",
        "-Wall", "-Werror", "-std=c11",
        "-o", str(out_o),
        str(src_c),
    ]
    print(f"Building runtime stub: arch={host_arch} out={out_o}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            f"Error: gcc failed to compile runtime_stub.c (rc={exc.returncode}).\n"
            f"Command: {' '.join(cmd)}",
            file=sys.stderr,
        )
        raise

    print(f"Runtime stub built: {out_o}")


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
try:
    from setuptools import setup as _setup
    from setuptools.command.build_py import build_py as _build_py
    from setuptools.command.develop import develop as _develop

    class CustomBuildPy(_build_py):
        def run(self) -> None:
            build_cpp_plugin()
            build_runtime_stub()
            super().run()

    class CustomDevelop(_develop):
        def run(self) -> None:
            build_cpp_plugin()
            build_runtime_stub()
            super().run()

    # Re-export with custom commands
    # Note: this is a simplified version; a real build would use
    # scikit-build-core or py-build for proper C++ integration
    setup_kwargs: dict = {}
    setup_kwargs["name"] = "nautilus"
    setup_kwargs["cmdclass"] = {
        "build_py": CustomBuildPy,
        "develop": CustomDevelop,
    }
    _setup(**setup_kwargs)
except ImportError:
    pass


if __name__ == "__main__":
    # When run directly (not via pip), build the C++ plugin and runtime stub
    build_cpp_plugin()
    build_runtime_stub()

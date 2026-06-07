"""Bridge between TVM TIR and Triton's tt.dot for matmul kernels.

This module solves the hardest problem in the bridge: TVM TIR has no
native representation for tensor-core matrix multiply. The solution:

  1. Identify matmul sections in the captured Triton IR
  2. Compile those sections separately with Triton's normal compiler
     (so we keep Tensor Core performance)
  3. In TIR, generate a call to the Triton-compiled matmul via tvm.extern
  4. The TIR handles the rest of the kernel (elementwise, reduction)
  5. MetaSchedule tunes the non-matmul parts (memory access, tiling)

This is the only production-quality path because:
  - Full IR conversion (TTGIR -> TIR for matmul) is not yet solved in
    any open-source project (triton-tvm has no tt.dot support)
  - Pure Triton autotune misses MetaSchedule's RL search
  - Pure TVM MetaSchedule has no tensor-core equivalent

The hybrid approach gives us the best of both worlds.
"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.logging import get_logger

try:
    from tvm.script import tir as _tir

    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False

from src.common.errors import CompilationError, DependencyMissingError

from .ir_capture import IRBounds

logger = get_logger(__name__)


@dataclass(frozen=True)
class CompiledMatmul:
    """A Triton-compiled matmul ready to be called from TVM via tvm.extern."""

    name: str
    source_hash: str
    cubin_path: Path  # or hsaco, spv depending on target
    metadata: dict[str, Any]
    m: int
    n: int
    k: int
    dtype: str

    def cache_key(self) -> str:
        parts = [self.name, self.source_hash, str(self.m), str(self.n), str(self.k), self.dtype]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()


@dataclass
class ExternMatmulBuilder:
    """Compiles matmul ops with Triton, exposes them as TVM extern calls.

    This builder:
      1. Takes the matmul bounds (M, N, K, dtype) extracted from TTGIR
      2. Generates a Triton kernel for the matmul
      3. AOT-compiles it to the target binary (cubin/hsaco/spv)
      4. Returns metadata TVM can call via tvm.extern
    """

    def __init__(
        self,
        cache_dir: str = "/tmp/nvindia_cud_extern_cache",
        default_num_warps: int = 8,
        default_num_stages: int = 3,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_num_warps = default_num_warps
        self.default_num_stages = default_num_stages
        self._cache: dict[str, CompiledMatmul] = {}

    def build_matmul(
        self,
        name: str,
        bounds: IRBounds,
        target: str = "cuda",
        source_hash: str = "unknown",
    ) -> CompiledMatmul:
        """Build a Triton matmul for the given bounds and target.

        Args:
            name: Logical name (e.g. "matmul_qk").
            bounds: M, N, K, dtype from the captured Triton IR.
            target: Target backend ("cuda", "rocm", "metal", "intel").
            source_hash: For cache key derivation.

        Returns:
            CompiledMatmul ready to be referenced from a TVM TIR.
        """
        if bounds.m is None or bounds.n is None or bounds.k is None:
            raise ValueError(f"Cannot build matmul without M/N/K bounds: {bounds}")

        cache_key = f"{name}:{bounds.m}:{bounds.n}:{bounds.k}:{bounds.data_dtype}:{target}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Generate Triton source
        triton_source = self._generate_triton_matmul(
            name=name,
            m=bounds.m,
            n=bounds.n,
            k=bounds.k,
            dtype=bounds.data_dtype,
            num_warps=self.default_num_warps,
            num_stages=self.default_num_stages,
        )

        # Compile to target binary
        binary_path = self._compile_triton_to_binary(
            triton_source=triton_source,
            target=target,
            name=name,
        )

        result = CompiledMatmul(
            name=name,
            source_hash=source_hash,
            cubin_path=binary_path,
            metadata={
                "num_warps": self.default_num_warps,
                "num_stages": self.default_num_stages,
            },
            m=bounds.m,
            n=bounds.n,
            k=bounds.k,
            dtype=bounds.data_dtype,
        )
        self._cache[cache_key] = result
        logger.info(
            "Built matmul: name=%s m=%d n=%d k=%d target=%s → %s",
            name,
            bounds.m,
            bounds.n,
            bounds.k,
            target,
            binary_path,
        )
        return result

    def generate_tir_extern_call(
        self,
        matmul: CompiledMatmul,
        buffer_args: list[Any],
    ) -> Any:
        """Generate a TIR PrimFunc that calls the compiled matmul via tvm.extern.

        This is the integration point: a TIR function that wraps the
        Triton-compiled matmul as an external call. The rest of the
        kernel (surrounding elementwise/reduction) can call this.
        """
        if not TVM_AVAILABLE:
            raise RuntimeError("TVM required for tvm.extern generation")

        # Read the binary contents
        with open(matmul.cubin_path, "rb") as f:
            f.read()

        @_tir.prim_func
        def matmul_extern(
            a_handle: _tir.handle,
            b_handle: _tir.handle,
            c_handle: _tir.handle,
        ) -> None:
            a = _tir.match_buffer(a_handle, (matmul.m, matmul.k), matmul.dtype)
            b = _tir.match_buffer(b_handle, (matmul.k, matmul.n), matmul.dtype)
            c = _tir.match_buffer(c_handle, (matmul.m, matmul.n), matmul.dtype)

            # Call the Triton-compiled matmul via tvm.extern
            _tir.evaluate(
                _tir.tvm_call_cpacked(
                    "triton_matmul_run",
                    _tir.tvm_stack_make_array(
                        a.data,
                        b.data,
                        c.data,
                        _tir.tvm_stack_make_shape(
                            matmul.m,
                            matmul.k,
                            matmul.k,
                            matmul.n,
                            matmul.m,
                            matmul.n,
                        ),
                        _tir.float32(0.0),  # alpha
                        _tir.float32(0.0),  # beta
                    ),
                )
            )

        return matmul_extern

    # ------------------------------------------------------------------
    # Internal compilation pipeline
    # ------------------------------------------------------------------

    def _generate_triton_matmul(
        self,
        name: str,
        m: int,
        n: int,
        k: int,
        dtype: str,
        num_warps: int,
        num_stages: int,
    ) -> str:
        """Generate Triton source for a generic matmul."""
        return f'''"""Auto-generated matmul for {name}: [{m}, {n}, {k}], dtype={dtype}."""
import triton
import triton.language as tl


@triton.jit
def {name}_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    A_block = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B_block = B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kk in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A_block, mask=rm[:, None] < M, other=0.0)
        b = tl.load(B_block, mask=rn[None, :] < N, other=0.0)
        acc += tl.dot(a, b)
        A_block += BLOCK_SIZE_K * stride_ak
        B_block += BLOCK_SIZE_K * stride_bk

    C_block = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_block, acc.to(tl.{_triton_dtype(dtype)}), mask=mask)


def {name}(A, B):
    """Run the matmul: C = A @ B."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
    {name}_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        num_warps={num_warps},
        num_stages={num_stages},
    )
    return C
'''

    def _compile_triton_to_binary(
        self,
        triton_source: str,
        target: str,
        name: str,
    ) -> Path:
        """Compile a Triton source file to the target binary.

        Returns the path to the binary (.cubin, .hsaco, .spv, etc.)
        """
        target_ext = {
            "cuda": "cubin",
            "rocm": "hsaco",
            "metal": "metallib",
            "intel": "spv",
        }.get(target, "bin")

        # Hash-based output path
        source_hash = hashlib.sha256(triton_source.encode()).hexdigest()[:16]
        output_path = self.cache_dir / f"{name}_{source_hash}.{target_ext}"

        if output_path.exists():
            logger.debug("Reusing cached binary: %s", output_path)
            return output_path

        # Write source to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            dir=str(self.cache_dir),
        ) as f:
            f.write(triton_source)
            source_file = Path(f.name)

        try:
            # Use Triton's AOT compile to produce the target binary.
            # This requires Triton with the relevant backend installed
            # (AOTriton for AMD, Triton CPU/cuda hooks, etc.). If the
            # backend is missing or the compile fails we surface a
            # typed DependencyMissingError — never silently fall back
            # to a non-existent touch-file (the previous behaviour
            # produced confusing downstream errors when downstream
            # code tried to load the empty placeholder).
            self._run_triton_aot(
                source_file=source_file,
                output_path=output_path,
                target=target,
            )
            return output_path
        except DependencyMissingError:
            raise
        except Exception as exc:
            raise DependencyMissingError(
                message=(
                    f"Triton AOT compile failed for target={target!r} "
                    f"(name={name!r}): {exc}. Install the matching "
                    "Triton backend / AOTriton and retry."
                ),
                cause=exc,
                context={
                    "target": target,
                    "name": name,
                    "source_hash": source_hash,
                },
            ) from exc
        finally:
            source_file.unlink(missing_ok=True)

    def _run_triton_aot(
        self,
        source_file: Path,
        output_path: Path,
        target: str,
    ) -> None:
        """Invoke Triton's AOT compiler.

        This calls the triton_aot CLI (when available) or uses the
        Python triton.compile API directly to produce the target
        binary. Falls back to dump-ttgir-style output.
        """
        try:
            # Try the triton_aot CLI (Triton 3.5+)
            cmd = [
                "python",
                "-m",
                "triton.tools.aot",
                "--source",
                str(source_file),
                "--target",
                target,
                "--output",
                str(output_path),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                return
            logger.debug("triton_aot returned %d: %s", result.returncode, result.stderr)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback: use triton.compile via Python
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("kernel_mod", source_file)
            if spec is None or spec.loader is None:
                raise CompilationError(
                    f"Could not load kernel module from {source_file}",
                    context={"source_file": str(source_file)},
                )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Find the kernel function
            kernel_fn = None
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if hasattr(attr, "run") and callable(attr):
                    kernel_fn = attr
                    break

            if kernel_fn is None:
                raise RuntimeError("No Triton kernel found in source")

            # Create a fake tensor for compilation
            import torch

            a = torch.zeros(
                128, 128, device="cuda" if target == "cuda" else "cpu", dtype=torch.float32
            )
            b = torch.zeros(
                128, 128, device="cuda" if target == "cuda" else "cpu", dtype=torch.float32
            )

            # Trigger JIT compile
            grid = (1,)
            kernel_fn[grid](a, b, a, 128, 128, 128, 128, 1, 128, 1, 128, 1)

            # Do NOT rglob-copy from ~/.triton/cache/ — filenames are
            # hash-derived and shared with other kernels in the same
            # process, so a glob would silently pick a stale/unrelated
            # binary. Fail loudly instead.
            raise RuntimeError(
                "Triton JIT cached the binary but its filename is not "
                "predictable; AOT compile paths are required for the "
                "extern bridge."
            )
        except Exception as exc:
            raise DependencyMissingError(
                message=(
                    f"All AOT compile paths failed: {exc}. "
                    "Install the matching Triton backend / AOTriton "
                    "and retry."
                ),
                cause=exc,
                context={"source_file": str(source_file), "target": target},
            ) from exc


def _triton_dtype(dtype: str) -> str:
    """Map canonical dtype name to Triton language constant."""
    mapping = {
        "float32": "float32",
        "float16": "float16",
        "bfloat16": "bfloat16",
        "float64": "float64",
        "int32": "int32",
        "int64": "int64",
    }
    return mapping.get(dtype, "float32")

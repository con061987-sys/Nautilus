# Code Patterns & Conventions

## Python Code Style

### Imports
```python
# Standard library
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

# Third-party
import torch
import triton

# Project imports
from src.common.types import DeviceMesh, KernelConfig
from src.common.hardware import detect_hardware
```

### Type Hints
All public functions MUST have type hints:
```python
def compile_kernel(
    kernel_source: str,
    target: HardwareTarget,
    num_warps: int = 4,
    num_stages: int = 3,
) -> CompiledKernel:
    """Compile a Triton kernel for the specified target.

    Args:
        kernel_source: Triton Python kernel source code.
        target: Target hardware specification.
        num_warps: Number of warps per thread block.
        num_stages: Number of pipeline stages.

    Returns:
        CompiledKernel object containing binary and metadata.
    """
    ...
```

### Google-Style Docstrings
Always include Args, Returns, Raises sections for public APIs.

## Bridge Pattern

Every bridge follows this 4-step contract:

```python
class SomeBridge:
    def intercept(self, input_data: Any) -> IntermediateIR:
        """Extract IR/graph from the source system."""
        ...

    def normalize(self, ir: IntermediateIR) -> NeutralIR:
        """Convert to neutral representation (MLIR Vector Dialect)."""
        ...

    def translate(self, normalized: NeutralIR, target: Target) -> Result:
        """Pass to target system via C-API or stable interface."""
        ...

    def verify(
        self, input_data: Any, result: Result, reference: Any
    ) -> ValidationReport:
        """Validate output correctness against reference."""
        ...
```

## Error Handling

- Use `Result[T, Error]` pattern (not exceptions) for expected failure modes
- Use exceptions only for programmer errors (wrong args, impossible states)
- All C-API wrappers must translate C error codes into project error types

```python
from src.common.errors import Result, CompileError, TuningError

def tune_kernel(kernel_ir: str, target: str) -> Result[TuningConfig, TuningError]:
    ...
```

## C++ Code Style (LLVM)

- Follow LLVM coding style (120 cols, 2-space indent, `_` separated names)
- Use `llvm::Expected<T>` for error handling
- RAII for all resource management
- No raw pointers in public APIs

## Testing

- Every bridge needs: unit tests + integration test + regression test
- Tests must not depend on specific hardware unless in `tests/hardware/`
- Use pytest fixtures for shared setup
- Name pattern: `test_<function_name>_<scenario>.py`

```bash
# Run all tests
python -m pytest src/tests/

# Run specific bridge tests
python -m pytest src/tests/test_auto_tuning.py -v

# Run hardware-dependent tests
python -m pytest src/tests/hardware/ --hw-test
```

## File Organization

- One file per major class/function
- Bridge files split by direction (triton→tvm, tvm→triton)
- C-API headers in `src/c_api/`
- Tests mirror source structure under `src/tests/`

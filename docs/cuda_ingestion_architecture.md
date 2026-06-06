# CUDA Ingestion Bridge Architecture

## Overview

The CUDA ingestion bridge (Phase 4) translates standard CUDA C++ (`.cu`) kernels into Triton Python source code. The translated Triton code is then fed into the Phase 1/2 pipeline (auto-tuning → fat binary) to run on any supported hardware.

## Architecture Decision: tree-sitter over Regex

The original parser used regex patterns on raw C++ text. This approach failed on:
- Template parameters (`template<typename T> __global__ void kernel(T* x)`)
- Nested namespaces (`namespace A { namespace B { __global__ void ... }}`)
- Complex for-loops with multiple init/update statements (`for (int i = 0, j = n-1; i < n && j > 0; i++, j--)`)
- Function pointers and lambda expressions
- Nested angle brackets (`vector<vector<int>>`)

**Solution:** Use `tree-sitter-cpp` (via the `tree-sitter` Python bindings) to parse the full C++ grammar. This gives us:
- Correct handling of all C++ constructs without custom regex
- AST-level node types for precise statement classification
- Access to field expressions (`threadIdx.x`) as structured `field_expression` nodes, distinguishable from regular struct member access
- Natural handling of template and namespace nesting

## Architecture

```
Source (.cu)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  parser.py: TreeSitterCudaParser                             │
│                                                              │
│  1. tree-sitter-cpp parses source into AST                   │
│  2. Recursive walk finds function_definition nodes           │
│  3. CUDA qualifiers (__global__/__device__) detected textually│
│     (tree-sitter-cpp doesn't understand CUDA keywords)       │
│  4. Parameters extracted from parameter_list children        │
│  5. Body statements classified by node type + metadata       │
│  6. CUDA constructs (threadIdx, atomicAdd) tagged in metadata│
│                                                              │
│  Output: list[CudaKernel]                                    │
│    ├── name, qualifier, return_type                          │
│    ├── parameters: [{"name": str, "type": str}]              │
│    ├── body: [CudaStatement]                                 │
│    │    ├── stmt_type: CudaStatementType enum                │
│    │    ├── raw_text: original source text                   │
│    │    ├── line_number: source line                         │
│    │    └── metadata: dict with AST-level info               │
│    │         ├── has_sync, has_atomic, has_cuda_field        │
│    │         ├── sync_calls: [{"name": "__syncthreads"}]     │
│    │         ├── atomic_ops: [{"name": "atomicAdd", args}]   │
│    │         └── cuda_fields: [{"object": "threadIdx",       │
│    │                              "field": "x"}]             │
│    └── shared_mem_declarations: [...]                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  translator.py: CudaToTritonTranslator                       │
│                                                              │
│  1. For each CudaStatement, dispatch by stmt_type enum       │
│  2. If metadata.has_cuda_field:                              │
│       → structured replacement using cuda_fields list        │
│  3. If metadata.has_sync:                                    │
│       → generate tl.debug_barrier() with comment             │
│  4. If metadata.has_atomic:                                  │
│       → map function name via _ATOMIC_TO_TRITON table        │
│       → insert type casts on numeric arguments               │
│       → raise IngestionUnsupportedIntrinsicError for unknown │
│  5. Otherwise: pass through with text-level math renaming   │
│                                                              │
│  Output: TranslationResult                                   │
│    ├── triton_source: Python Triton kernel                   │
│    ├── shared_mem_plan: SharedMemPlan                        │
│    ├── pointer_layouts: dict[str, PointerLayout]             │
│    └── warnings: list of bank conflict warnings              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
  [intrinsic_mapper.py — usage change]
    Before: mapper.transform_text(stmt.raw_text) — blind text replacement
    After:  mapper.get_mapping("atomicAdd").triton_name — lookup table reference

    The IntrinsicMapper is now used as a LOOKUP TABLE, not a transform engine.
    The translator builds Triton code from AST metadata and the mapping table,
    not by running regex substitution on raw text.
```

## Key Design Decisions

### 1. Statement metadata bridges parser and translator

The `CudaStatement.metadata` dict is the contract between parser and translator:

```python
metadata = {
    "ast_type": "expression_statement",
    "has_sync": False,
    "has_atomic": True,
    "has_cuda_field": True,
    "cuda_fields": [
        {"object": "threadIdx", "field": "x", "text": "threadIdx.x"}
    ],
    "atomic_ops": [
        {"name": "atomicAdd", "args": ["&counter", "1"]}
    ],
    "sync_calls": [],
}
```

This allows:
- **Precise dispatch**: Translator knows exactly which CUDA constructs are present
- **Context-aware replacement**: Uses the structured field expression info, not regex
- **Unsupported detection**: If a CUDA construct isn't in the mapping tables, it raises an error

### 2. CUDA qualifier detection is textual, but everything else is AST

Tree-sitter-cpp does not natively understand `__global__`, `__device__`, or `__shared__` as CUDA qualifiers. These are detected by checking the leading text of the node. However:

- **Function structure** (parameters, body, for/if/while) is from the AST
- **Field expressions** like `threadIdx.x` are from the AST (`field_expression` node)
- **Function calls** like `atomicAdd()` are from the AST (`call_expression` node)

This hybrid approach gives us 95%+ of the benefit of full CUDA AST parsing without needing a custom grammar.

### 3. No silent no-ops

The translator NEVER silently drops unsupported constructs. Instead:
- Unsupported atomics → `IngestionUnsupportedIntrinsicError`
- Unsupported CUDA fields → `IngestionUnsupportedIntrinsicError`
- Unsupported statement types → log + return as TODO comment

### 4. IntrinsicMapper kept as lightweight mapping table

The `intrinsic_mapper.py` file is unchanged — it remains the single source of truth for what each CUDA intrinsic maps to in Triton. The translator uses `get_mapping()` for API-style access and `_ATOMIC_TO_TRITON`/`_CUDA_FIELD_TO_TRITON` dictionaries for internal dispatch.

## Limitations (known, tracked)

| Limitation | Impact | Future Fix |
|---|---|---|
| `__shared__` multi-dimensional arrays (e.g. `tile[16][16]`) extract only first dim | Shared memory allocation size may be undercounted | Update `SharedMemoryAnalyzer._parse_declaration` to multiply all dims |
| `blockIdx.x * blockDim.x + threadIdx.x` maps to `tl.program_id(0) * tl.num_programs(0) + tl.program_id(0)` | Incorrect semantics for global thread index | Requires pattern-matching the `blockIdx*blockDim + threadIdx` idiom and rewriting to `pid * BLOCK_SIZE + offsets` |
| Pointer declarations (`int* ptr`) are kept as comments | Manual review needed for pointer-based kernels | Full pointer flow analysis |
| Compound assignment (`+=`, `-=`) in AST is treated as ASSIGNMENT | Works for simple cases but doesn't decompose into `load → modify → store` | Add compound-assignment decomposition |
| C++11+ features (move semantics, auto, decltype) not tested | May produce weird output | Iterative grammar coverage expansion |

## Upgrade Impact on Existing Tests

The existing test files import the same public API:

| Test file | Changes needed |
|---|---|
| `test_cuda_parser.py` | **Minimal.** `CudaParser` still works. `CudaStatementType` enum values changed (was `auto()`-based, still `auto()`-based). Compatibility depends on comparing by enum member, not numeric `.value`. |
| `test_translator.py` | **Minimal.** `CudaToTritonTranslator` interface unchanged. Output Triton source uses `tl.debug_barrier()` instead of `tl.barrier()`. |
| `test_intrinsic_mapper.py` | **None.** File unchanged. |
| `test_shared_memory.py` | **None.** SharedMemoryAnalyzer API unchanged. |
| `test_ptr_analysis.py` | **None.** PointerAnalyzer API unchanged. |

**Critical fix for tests:** The original `test_translator.py` checks for `tl.barrier()` in the output. The new translator generates `tl.debug_barrier()` (per the spec requirement). Any test asserting `tl.barrier` needs updating to `tl.debug_barrier`.

## Dependency Installation

```bash
# Development install
pip install -e '.[cuda]'

# Or directly
pip install tree-sitter>=0.20 tree-sitter-cpp>=0.20

# Verify
python -c "import tree_sitter_cpp; from tree_sitter import Language; Language(tree_sitter_cpp.language())"
```

Minimum versions tested: `tree-sitter==0.25.2`, `tree-sitter-cpp==0.23.4`.

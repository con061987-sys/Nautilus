---
name: fat-binary-packaging
description: Knowledge of multi-vendor AOT binary compilation, fat binary format using LLVM linker, C runtime stubs for hardware detection, and cross-platform binary bundling. Use when working with AOT compilation, binary packaging, or the aot_packager bridge.
---

# Fat Binary Packaging Skill

## Overview

The fat binary system compiles Triton kernels ahead-of-time for multiple hardware targets and bundles them into a single ELF binary with a runtime C stub that selects the correct backend at startup.

## AOT Compilation Backends

### AMD AOTriton
```python
# AMD's AOTriton: Triton → HSACO (AMD GPU binary)
from aotriton import compile as aot_compile

hsaco = aot_compile(
    triton_kernel_source,    # Triton Python kernel as string
    kernel_name="matmul",
    signature="*fp32, *fp32, *fp32, i32, i32, i32",
    device="hip",            # Target: AMD HIP
    arch="gfx942",           # MI300X architecture
    options={
        "num_warps": 8,
        "num_stages": 4,
    }
)
# Returns bytes containing .hsaco binary
```

### Intel oneAPI
```python
# Intel oneAPI: Triton → SPIR-V (Intel GPU binary)
import subprocess

# oneAPI uses the llvm-spirv toolchain
# Step 1: Triton → LLVM IR
# Step 2: LLVM IR → SPIR-V via llvm-spirv
result = subprocess.run([
    "llvm-spirv", "-o", "kernel.spv",
    "kernel.ll",
    "--spirv-target=spv64",
    "--spirv-math=fast",
], capture_output=True)

# Or use the SYCL compiler for OpenCL-style compilation
result = subprocess.run([
    "icpx", "-fsycl", "-fintelfpga",
    "-o", "kernel.spv",
    "kernel.cpp",
])
```

### Nvidia PTX
```python
# Nvidia: Triton JIT → PTX (intermediate representation)
import triton
from triton.compiler import compile, ASTSource

compiled = compile(
    ASTSource(kernel_fn),
    target="ptx",  # Intercept at PTX level
    options={
        "num_warps": 8,
        "num_stages": 4,
    }
)
ptx_bytes = compiled.asm["ptx"]
```

## Fat Binary Linking (LLVM lld)

```python
# Combine all binaries into a single ELF object using LLVM linker (lld)
import subprocess

def link_fat_binary(backends, output_path="fat_binary.o"):
    """
    backend: dict of {name: binary_bytes}
    Creates an ELF relocatable object with dedicated sections per backend.
    """
    # Write individual section files
    for name, data in backends.items():
        write_section(f".{name}_kernel.elf", data)

    # Link using lld
    subprocess.run([
        "ld.lld", "-r",  # -r = relocatable (partial link)
        "-o", output_path,
        "runtime_stub.o",  # C stub for vendor detection
        *[f".{name}_kernel.elf" for name in backends],
        # Define section symbols for the C stub to use
        "--defsym", f"_start_{name}=.",
    ])
```

## C Runtime Stub

```c
// runtime_stub.c — compiled with: gcc -c -nostdlib runtime_stub.c
// This stub runs before main() to select the correct kernel binary.

// External symbols defined by linker for each backend section
extern char _binary_nv_kernel_start, _binary_nv_kernel_end;
extern char _binary_amd_kernel_start, _binary_amd_kernel_end;
extern char _binary_intel_kernel_start, _binary_intel_kernel_end;

// Simple vendor detection via CPUID
static int detect_vendor(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned int eax, ebx, ecx, edx;
    __asm__ volatile("cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(0));
    // EBX = "Genu" for Intel, ECX = "ntel" for Intel
    // EBX = "Auth" for AMD, ECX = "enti" for AMD
    if (ebx == 0x756e6547 && ecx == 0x6c65746e) return 0; // Intel
    if (ebx == 0x68747541 && ecx == 0x444d4163) return 1; // AMD
    return 2; // Unknown
#elif defined(__arm__) || defined(__aarch64__)
    return 3; // Apple Silicon
#endif
}

// Called by the compiled kernel dispatch layer
void* get_kernel_binary(const char* kernel_name) {
    int vendor = detect_vendor();
    switch (vendor) {
        case 0: return &_binary_nv_kernel_start;    // Nvidia PTX
        case 1: return &_binary_amd_kernel_start;   // AMD HSACO
        case 2:
            // Check for Intel GPU via /dev/dri
            if (access("/dev/dri/renderD128", F_OK) == 0)
                return &_binary_intel_kernel_start; // Intel SPIR-V
            return &_binary_nv_kernel_start;        // Fallback
        case 3: return &_binary_amd_kernel_start;   // Apple → AMD through MoltenVK
    }
    return NULL;
}
```

## Section Naming Convention

Each backend binary gets its own ELF section:

| Section | Backend | Format |
|---|---|---|
| `.nv_kernel` | Nvidia CUDA | PTX text |
| `.amd_kernel` | AMD ROCm | HSACO binary |
| `.intel_kernel` | Intel oneAPI | SPIR-V binary |
| `.apple_kernel` | Apple Metal | AIR binary |

## Critical Knowledge

1. **ELF is the container, not the code.** We use ELF as a packaging format. The actual kernel binaries (HSACO, SPIR-V) are stored as opaque data sections.
2. **PTX is forward-compatible.** Nvidia PTX runs on future GPU generations. HSACO and SPIR-V are more target-specific.
3. **`ld.lld -r` (relocatable link)** combines multiple object files into one without resolving all symbols. This is the standard way to bundle object files.
4. **Avoid GNU ld.** Use LLVM's `ld.lld` — it has consistent behavior across platforms and supports all our targets.
5. **The stub is ~50 lines.** Keep it minimal. Every byte in the stub is overhead on every kernel launch.

## When This Skill Triggers

- Working on `src/bridges/aot_packager/` bridge code
- Adding or updating compiler backends (AMD, Intel, Nvidia)
- Modifying the C runtime stub
- Debugging binary linking or section layout
- Testing fat binary on different hardware targets

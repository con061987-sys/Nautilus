#ifndef NAUTILUS_TRITON_C_API_H
#define NAUTILUS_TRITON_C_API_H

/*
 * triton_c_api.h — Stable C ABI for the OpenAI Triton compiler.
 *
 * This header defines the ONLY signatures that Nautilus code calls
 * into Triton's C++ internals. The implementation (in
 * src/c_api/wrappers/triton_wrapper.cpp) is updated when upstream
 * Triton breaks its API; everything else in Nautilus is insulated.
 *
 * Lifetime rules:
 *   - All `const char*` parameters are borrowed; must outlive the call.
 *   - `nautilus_kernel_t*` returned by compile() is opaque; the caller
 *     must release it with nautilus_release().
 *   - nautilus_compile() is thread-safe IF Triton's underlying
 *     GIL/compile lock is held; the wrapper serializes via a mutex.
 *
 * Error model:
 *   - Functions returning int return 0 on success, non-zero on failure.
 *   - On failure, nautilus_last_error_message() returns a human-readable
 *     description. The string is valid until the next call to any
 *     function in this header on the same thread.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque kernel handle. */
typedef struct nautilus_kernel_s nautilus_kernel_t;

/* Vendor for which a kernel is being compiled. */
typedef enum {
    NAUTILUS_VENDOR_NVIDIA  = 0,
    NAUTILUS_VENDOR_AMD     = 1,
    NAUTILUS_VENDOR_INTEL   = 2,
    NAUTILUS_VENDOR_APPLE   = 3,
    NAUTILUS_VENDOR_UNKNOWN = -1,  /* no GPU detected / runtime fallback */
} nautilus_vendor_t;

/* Compute capability / arch id. */
typedef enum {
    NAUTILUS_ARCH_SM_70   = 70,
    NAUTILUS_ARCH_SM_75   = 75,
    NAUTILUS_ARCH_SM_80   = 80,
    NAUTILUS_ARCH_SM_86   = 86,
    NAUTILUS_ARCH_SM_89   = 89,
    NAUTILUS_ARCH_SM_90   = 90,
    NAUTILUS_ARCH_SM_100  = 100,
    NAUTILUS_ARCH_SM_120  = 120,
    NAUTILUS_ARCH_GFX900  = 900,
    NAUTILUS_ARCH_GFX906  = 906,
    NAUTILUS_ARCH_GFX908  = 908,
    NAUTILUS_ARCH_GFX90A  = 910,
    NAUTILUS_ARCH_GFX942  = 942,
    NAUTILUS_ARCH_GFX950  = 950,
    NAUTILUS_ARCH_XE_LP   = 1200,
    NAUTILUS_ARCH_XE_HPG  = 1201,
    NAUTILUS_ARCH_XE_HPC  = 1202,
    NAUTILUS_ARCH_XE2     = 1203,
    NAUTILUS_ARCH_GAUDI2  = 2002,
    NAUTILUS_ARCH_GAUDI3  = 2003,
} nautilus_arch_t;

/*
 * Compile a Triton kernel (Python source as a string) for the given
 * target. The kernel is exposed as a function pointer at runtime.
 *
 * @param source         Triton @triton.jit function source code
 * @param kernel_name    Name of the @triton.jit function to compile
 * @param vendor         Target vendor
 * @param arch           Target arch
 * @param num_warps      Warps per block (vendor-specific mapping)
 * @param num_stages     Pipeline stages
 * @param block_m        Tile size M (matmul-style; 0 if N/A)
 * @param block_n        Tile size N
 * @param block_k        Tile size K
 * @param out            Receives an opaque handle; must be freed
 * @return               0 on success, non-zero on failure
 */
int nautilus_compile(
    const char* source,
    const char* kernel_name,
    nautilus_vendor_t vendor,
    nautilus_arch_t arch,
    int num_warps,
    int num_stages,
    int block_m,
    int block_n,
    int block_k,
    nautilus_kernel_t** out
);

/*
 * Retrieve the binary for a compiled kernel. Format is vendor-specific
 * (PTX text, CUBIN binary, HSACO binary, SPIR-V binary, etc.).
 *
 * @param kernel         Handle from nautilus_compile()
 * @param out_data       Receives a pointer to the binary data
 * @param out_size       Receives the size in bytes
 * @param out_format     Receives a string identifying the format
 *                       ("ptx", "cubin", "hsaco", "spv", "metallib")
 * @return               0 on success
 */
int nautilus_get_binary(
    nautilus_kernel_t* kernel,
    const uint8_t** out_data,
    size_t* out_size,
    const char** out_format
);

/*
 * Release all resources held by a kernel handle. After this call,
 * the handle is invalid and must not be reused.
 */
void nautilus_release(nautilus_kernel_t* kernel);

/*
 * Set a tuning parameter on a compiled kernel. Used by MetaSchedule
 * to push optimized block sizes into a pre-compiled kernel without
 * recompiling. Not all backends support this; some return
 * NAUTILUS_ERR_UNSUPPORTED.
 *
 * @param kernel   Handle
 * @param name     Parameter name (e.g. "num_warps", "num_stages",
 *                 "BLOCK_M", "BLOCK_N", "BLOCK_K")
 * @param value    Integer value
 * @return         0 on success, NAUTILUS_ERR_UNSUPPORTED if backend
 *                 doesn't support runtime tuning
 */
int nautilus_set_tuning_param(
    nautilus_kernel_t* kernel,
    const char* name,
    int64_t value
);

/* Error codes. */
#define NAUTILUS_OK                     0
#define NAUTILUS_ERR_GENERIC           -1
#define NAUTILUS_ERR_INVALID_ARG       -2
#define NAUTILUS_ERR_BACKEND_MISSING   -3
#define NAUTILUS_ERR_BACKEND_VERSION   -4
#define NAUTILUS_ERR_COMPILE_FAILED    -5
#define NAUTILUS_ERR_COMPILE_TIMEOUT   -6
#define NAUTILUS_ERR_UNSUPPORTED       -7
#define NAUTILUS_ERR_OOM               -8
#define NAUTILUS_ERR_INTERNAL          -99

/*
 * Return a human-readable error message for the last failed call
 * on the current thread. The string is valid until the next call
 * to any function in this header on the same thread.
 */
const char* nautilus_last_error_message(void);

/*
 * Return the version of the underlying Triton library. Format is
 * MAJOR.MINOR.PATCH (e.g. "3.0.0").
 */
const char* nautilus_triton_version(void);

#ifdef __cplusplus
}
#endif

#endif /* NAUTILUS_TRITON_C_API_H */

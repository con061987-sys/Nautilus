// src/c_api/triton_wrapper.cpp — C wrapper around the OpenAI Triton
// compiler runtime.
//
// This translation unit implements the `nautilus_*` symbols declared
// in triton_c_api.h.
//
// Upstream Triton (triton-lang/triton) does NOT expose a stable public
// C ABI. The Python bindings call C++ internals directly via
// pybind11. The "stable" surface we work with is:
//
//   1. `triton.runtime.driver.active.get_current_target(...)` — Python
//      side; the C++ equivalent is the `triton::driver::Driver` API
//      which is built into `libtriton.so` as a set of free functions.
//
//   2. The precompiled kernel binary that we ultimately need: PTX
//      text (Nvidia), CUBIN bytes (Nvidia), HSACO (AMD via
//      AOTriton, when AOTRITON_AVAILABLE), SPIR-V (Intel via
//      oneAPI, when INTEL_AOT_AVAILABLE).
//
// To insulate ourselves from Triton's fast-moving internals, this
// wrapper uses the following well-defined entry points when present
// (looked up via dlsym; if not present, returns
// NAUTILUS_ERR_BACKEND_MISSING with a clear hint):
//
//   * triton_compile_kernel_v3  — Triton >= 3.0 plugin entry point.
//     Signature: extern "C" int triton_compile_kernel_v3(
//         const char* source,
//         const char* kernel_name,
//         const char* target_triple,    // e.g. "cuda:sm_90"
//         int num_warps, int num_stages,
//         int block_m, int block_n, int block_k,
//         uint8_t** out_binary,
//         size_t*  out_size,
//         char*    out_format, size_t format_buf_size,
//         char*    err_buf, size_t err_buf_size);
//     Returns 0 on success, non-zero otherwise. out_format and
//     err_buf are caller-allocated C strings.
//
//   * triton_get_compiler_version — returns the upstream version.
//
//   * triton_kernel_set_tuning_param_v3 — optional runtime-tuning
//     entry point. If absent, set_tuning_param returns
//     NAUTILUS_ERR_UNSUPPORTED.
//
// If the symbol lookup fails (older Triton, or Triton not built with
// plugin support, or libtriton.so simply missing on this machine),
// the wrapper still produces a loadable libnautilus_c_api.so: it
// just reports NAUTILUS_ERR_BACKEND_MISSING with a build hint. This
// is the graceful-degradation contract documented in __init__.py.
//
// We never dlclose() the upstream library (see internal.h).

#include "triton_c_api.h"
#include "internal.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

// ---------------------------------------------------------------------
// Opaque kernel handle (what nautilus_kernel_t points to).
// ---------------------------------------------------------------------

struct nautilus_kernel_s {
    std::vector<uint8_t> binary;       // owned copy of the kernel binary
    std::string          format;       // "ptx", "cubin", "hsaco", "spv", "metallib"
    bool                 released = false;
    // An optional pointer to an upstream-side handle. When Triton
    // exposes a v3 plugin entry point, that handle is owned by the
    // upstream library and must be released by the matching
    // triton_kernel_release_v3 symbol. Otherwise this stays null.
    void*                upstream_handle = nullptr;
};

// Tracks live kernel handles. Every entry point that takes a
// nautilus_kernel_t* must check `kernel_registry().contains(p)`
// before dereferencing — bogus pointers (e.g. 0xDEADBEEF) are
// no-ops instead of segfaults.
static nautilus_internal::HandleRegistry<nautilus_kernel_s>& kernel_registry() {
    static nautilus_internal::HandleRegistry<nautilus_kernel_s> r;
    return r;
}

// ---------------------------------------------------------------------
// Cached dlopen handle for libtriton.so
// ---------------------------------------------------------------------

namespace {
nautilus_internal::DlHandle& triton_dl() {
    static nautilus_internal::DlHandle h;
    return h;
}
}  // namespace

// Try to load libtriton from a list of candidate paths. The first
// successful load wins. We try both the SONAME-style name (which
// respects LD_LIBRARY_PATH / rpath) and a few absolute fallbacks
// relative to the env vars set by the build.
static bool try_load_triton(char* err_slot) {
    using nautilus_internal::set_error;

    // Allow override.
    if (const char* env = std::getenv("NAUTILUS_TRITON_LIB")) {
        if (triton_dl().load(env, err_slot) != nullptr) return true;
    }
    // SONAME lookup (lets the loader find the system-installed copy).
    static const char* kCandidates[] = {
        "libtriton.so",
        "libtriton.so.3",
        "libtriton.so.3.0.0",
        "libtriton_runtime.so",
    };
    for (const char* name : kCandidates) {
        if (triton_dl().load(name, err_slot) != nullptr) {
            return true;
        }
    }
    // No candidate worked. Restore the most-recent error so the caller
    // can surface it.
    set_error(err_slot,
        "could not find libtriton.so. Set NAUTILUS_TRITON_LIB=/path/to/libtriton.so, "
        "or install OpenAI Triton (>= 3.0) built with the v3 C-plugin ABI. "
        "The Python layer will fall back to subprocess invocation.");
    return false;
}

// A compile lock to serialize calls into Triton's compiler — the
// upstream is not generally thread-safe at the C level, and Python
// holds the GIL during compiles, so we mirror that here.
static std::mutex g_compile_mu;

// ---------------------------------------------------------------------
// Public C ABI
// ---------------------------------------------------------------------

extern "C" {

NAUTILUS_API int nautilus_compile(
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
) {
    using nautilus_internal::set_error;
    char* err = nautilus_internal::triton_error_slot();
    set_error(err, "");

    if (out == nullptr) {
        set_error(err, "nautilus_compile: out is null");
        return NAUTILUS_ERR_INVALID_ARG;
    }
    *out = nullptr;
    if (source == nullptr || kernel_name == nullptr) {
        set_error(err, "nautilus_compile: source/kernel_name is null");
        return NAUTILUS_ERR_INVALID_ARG;
    }

    if (!try_load_triton(err)) {
        return NAUTILUS_ERR_BACKEND_MISSING;
    }

    // Build the target triple upstream expects, e.g. "cuda:sm_90".
    const char* vendor_str = nautilus_internal::vendor_name(static_cast<int>(vendor));
    std::string target = std::string(vendor_str) + ":" + nautilus_internal::arch_target_string(
        static_cast<int>(vendor), static_cast<int>(arch));

    // Look up the v3 plugin entry point. We accept several aliases
    // because upstream has changed the symbol name across versions.
    using compile_fn_t = int (*)(
        const char*, const char*, const char*,
        int, int, int, int, int,
        uint8_t**, size_t*, char*, size_t,
        char*, size_t);

    compile_fn_t compile_fn = nullptr;
    {
        // Try the canonical name first.
        void* p = triton_dl().sym("triton_compile_kernel_v3", err);
        if (p != nullptr) {
            compile_fn = reinterpret_cast<compile_fn_t>(p);
        } else {
            // Older names: triton_compile_kernel (no version suffix).
            p = triton_dl().sym("triton_compile_kernel", err);
            if (p != nullptr) {
                compile_fn = reinterpret_cast<compile_fn_t>(p);
            }
        }
    }
    if (compile_fn == nullptr) {
        // No upstream entry point: surface a clear message. We do NOT
        // pretend to have compiled the kernel — that would let bugs
        // into the fat-binary pipeline.
        set_error(err,
            "libtriton.so loaded but does not export triton_compile_kernel_v3. "
            "Rebuild OpenAI Triton with the v3 C-plugin ABI (cmake -DTRITON_BUILD_C_API=ON) "
            "or use the Python bridge path.");
        return NAUTILUS_ERR_BACKEND_VERSION;
    }

    // Allocate a handle and run the compile under the lock.
    std::lock_guard<std::mutex> lock(g_compile_mu);

    char upstream_err[512] = {0};
    char format_buf[32]    = {0};
    uint8_t* bin_ptr       = nullptr;
    size_t   bin_size      = 0;

    int rc = compile_fn(
        source, kernel_name, target.c_str(),
        num_warps, num_stages, block_m, block_n, block_k,
        &bin_ptr, &bin_size,
        format_buf, sizeof(format_buf),
        upstream_err, sizeof(upstream_err));

    if (rc != 0 || bin_ptr == nullptr) {
        std::string msg = "triton_compile failed: ";
        msg += upstream_err;
        set_error(err, msg.c_str());
        // Some upstream variants allocate the buffer; others don't.
        // To stay safe, we don't try to free it — Python ctypes
        // memory ownership is documented to be retained by the
        // upstream library.
        return NAUTILUS_ERR_COMPILE_FAILED;
    }

    auto* handle = new nautilus_kernel_s();
    handle->binary.assign(bin_ptr, bin_ptr + bin_size);
    handle->format = format_buf;
    kernel_registry().add(handle);

    *out = handle;
    return NAUTILUS_OK;
}

NAUTILUS_API int nautilus_get_binary(
    nautilus_kernel_t* kernel,
    const uint8_t** out_data,
    size_t* out_size,
    const char** out_format
) {
    using nautilus_internal::set_error;
    char* err = nautilus_internal::triton_error_slot();
    set_error(err, "");

    if (kernel == nullptr) {
        set_error(err, "nautilus_get_binary: kernel is null");
        return NAUTILUS_ERR_INVALID_ARG;
    }
    if (!kernel_registry().contains(kernel)) {
        set_error(err, "nautilus_get_binary: handle was not allocated by this library");
        return NAUTILUS_ERR_INVALID_ARG;
    }
    if (kernel->released) {
        set_error(err, "nautilus_get_binary: kernel already released");
        return NAUTILUS_ERR_INVALID_ARG;
    }
    if (out_data == nullptr || out_size == nullptr || out_format == nullptr) {
        set_error(err, "nautilus_get_binary: out parameters are null");
        return NAUTILUS_ERR_INVALID_ARG;
    }
    *out_data   = kernel->binary.data();
    *out_size   = kernel->binary.size();
    *out_format = kernel->format.c_str();
    return NAUTILUS_OK;
}

NAUTILUS_API void nautilus_release(nautilus_kernel_t* kernel) {
    if (kernel == nullptr) return;
    if (!kernel_registry().contains(kernel)) return;
    if (kernel->released) return;
    kernel->released = true;

    // If Triton gave us an upstream-side handle, hand it back.
    if (kernel->upstream_handle != nullptr && triton_dl().loaded()) {
        using release_fn_t = void (*)(void*);
        char err_buf[256] = {0};
        void* sym = triton_dl().sym("triton_kernel_release_v3", err_buf);
        if (sym != nullptr) {
            reinterpret_cast<release_fn_t>(sym)(kernel->upstream_handle);
        }
    }
    kernel_registry().remove(kernel);
    delete kernel;
}

NAUTILUS_API int nautilus_set_tuning_param(
    nautilus_kernel_t* kernel,
    const char* name,
    int64_t value
) {
    using nautilus_internal::set_error;
    char* err = nautilus_internal::triton_error_slot();
    set_error(err, "");

    if (kernel == nullptr || name == nullptr) {
        set_error(err, "nautilus_set_tuning_param: null argument");
        return NAUTILUS_ERR_INVALID_ARG;
    }
    if (!kernel_registry().contains(kernel)) {
        set_error(err, "nautilus_set_tuning_param: handle was not allocated by this library");
        return NAUTILUS_ERR_INVALID_ARG;
    }
    if (kernel->released) {
        set_error(err, "nautilus_set_tuning_param: kernel already released");
        return NAUTILUS_ERR_INVALID_ARG;
    }
    if (!triton_dl().loaded()) {
        if (!try_load_triton(err)) {
            return NAUTILUS_ERR_BACKEND_MISSING;
        }
    }
    using set_fn_t = int (*)(void*, const char*, int64_t);
    void* sym = triton_dl().sym("triton_kernel_set_tuning_param_v3", err);
    if (sym == nullptr) {
        // Older Triton or no runtime-tuning ABI.
        set_error(err,
            "upstream Triton does not support runtime tuning (no "
            "triton_kernel_set_tuning_param_v3 export). Recompile with "
            "the tuned config instead via nautilus_compile().");
        return NAUTILUS_ERR_UNSUPPORTED;
    }
    auto fn = reinterpret_cast<set_fn_t>(sym);
    // Pass the upstream handle if we have one; else pass the address
    // of our own opaque struct. The upstream ABI is allowed to
    // ignore it.
    void* handle_ptr = kernel->upstream_handle != nullptr
        ? kernel->upstream_handle
        : static_cast<void*>(kernel);
    return fn(handle_ptr, name, value);
}

NAUTILUS_API const char* nautilus_last_error_message(void) {
    return nautilus_internal::triton_error_slot();
}

NAUTILUS_API const char* nautilus_triton_version(void) {
    // The build string is the *wrapper* version — caller correlates
    // with the .so they loaded. We return it directly without
    // touching upstream: even if Triton is not present, this string
    // is meaningful.
    static thread_local std::string s;
    s = std::string(NAUTILUS_BUILD_STRING) + " (wrapper)";
    if (triton_dl().loaded()) {
        using ver_fn_t = const char* (*)();
        char err_buf[256] = {0};
        void* sym = triton_dl().sym("triton_get_compiler_version", err_buf);
        if (sym != nullptr) {
            const char* upstream = reinterpret_cast<ver_fn_t>(sym)();
            if (upstream != nullptr && upstream[0] != '\0') {
                s += "+upstream=";
                s += upstream;
            }
        }
    }
    return s.c_str();
}

}  // extern "C"

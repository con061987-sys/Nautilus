// src/c_api/stubs.cpp — placeholder C++ implementation.
//
// The full wrapper that calls into Triton's C++ internals, TVM's
// PackedFunc API, and XLA's PJRT bindings is a future file. The
// Python ctypes bindings in __init__.py work without it because they
// fall back to subprocess-based invocations of upstream CLIs.
//
// This file exists so that `cmake --build` produces a real shared
// library that can be linked against; otherwise the C-API surface
// declared in the headers would be unfulfilled.

#include <cstring>
#include "triton_c_api.h"
#include "tvm_c_api.h"
#include "xla_c_api.h"

extern "C" {

static thread_local char g_last_error[1024] = {0};

static void set_error(const char* msg) {
    std::strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
    g_last_error[sizeof(g_last_error) - 1] = '\0';
}

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
) {
    if (!source || !kernel_name || !out) {
        set_error("invalid argument: null pointer");
        return NAUTILUS_ERR_INVALID_ARG;
    }
    set_error("nautilus_c_api stubs compiled without Triton/TVM/XLA backend; "
              "the Python layer handles calls via subprocess. "
              "Rebuild with -DNAUTILUS_USE_TRITON_LIBS=ON to use the C++ path.");
    *out = nullptr;
    return NAUTILUS_ERR_BACKEND_MISSING;
}

int nautilus_get_binary(
    nautilus_kernel_t* kernel,
    const uint8_t** out_data,
    size_t* out_size,
    const char** out_format
) {
    set_error("nautilus_get_binary: no kernel available (stub build)");
    return NAUTILUS_ERR_BACKEND_MISSING;
}

void nautilus_release(nautilus_kernel_t* kernel) {}

int nautilus_set_tuning_param(
    nautilus_kernel_t* kernel,
    const char* name,
    int64_t value
) {
    return NAUTILUS_ERR_UNSUPPORTED;
}

const char* nautilus_last_error_message(void) {
    return g_last_error;
}

const char* nautilus_triton_version(void) {
    return "0.0.0-stub";
}

int nautilus_tir_parse(
    const char* text, size_t text_len, const char* target,
    nautilus_tir_module_t** out
) {
    set_error("nautilus_tir_parse: stub build (Python handles via TVM Python API)");
    return NAUTILUS_TUNING_ERR_BACKEND;
}

void nautilus_tir_release(nautilus_tir_module_t* mod) {}

int nautilus_tune(
    nautilus_tir_module_t* mod,
    int max_trials, int num_trials_per_iter,
    nautilus_tuning_strategy_t strategy,
    int timeout_seconds,
    nautilus_tuning_record_t** out_record
) {
    set_error("nautilus_tune: stub build");
    return NAUTILUS_TUNING_ERR_BACKEND;
}

void nautilus_tuning_record_release(nautilus_tuning_record_t* rec) {}

int nautilus_record_get_int(
    nautilus_tuning_record_t* rec, const char* name, int64_t* out_value
) {
    return NAUTILUS_TUNING_ERR_GENERIC;
}

const char* nautilus_tvm_last_error_message(void) { return g_last_error; }
const char* nautilus_tvm_version(void) { return "0.0.0-stub"; }

int nautilus_stablehlo_from_fx(
    const char* fx_graph_json, size_t json_len, const char* function_name,
    nautilus_stablehlo_t** out
) {
    set_error("nautilus_stablehlo_from_fx: stub build");
    return NAUTILUS_XLA_ERR_EXPORT;
}

void nautilus_stablehlo_release(nautilus_stablehlo_t* mod) {}

int nautilus_stablehlo_get_mlir_text(
    nautilus_stablehlo_t* mod, const char** out_text, size_t* out_len
) {
    return NAUTILUS_XLA_ERR_GENERIC;
}

int nautilus_mesh_create(const int64_t* axes, size_t n_axes, nautilus_mesh_t** out) {
    return NAUTILUS_XLA_ERR_MESH;
}

void nautilus_mesh_release(nautilus_mesh_t* mesh) {}

int nautilus_gspmd_shard(
    nautilus_stablehlo_t* mod, nautilus_mesh_t* mesh,
    nautilus_shard_strategy_t strategy, int timeout_seconds,
    nautilus_sharding_spec_t** out_spec
) {
    return NAUTILUS_XLA_ERR_GSPMD;
}

void nautilus_sharding_spec_release(nautilus_sharding_spec_t* spec) {}

const char* nautilus_xla_last_error_message(void) { return g_last_error; }
const char* nautilus_xla_version(void) { return "0.0.0-stub"; }

}  // extern "C"

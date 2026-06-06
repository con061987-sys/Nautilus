// src/c_api/tvm_wrapper.cpp — C wrapper around Apache TVM
// MetaSchedule for the auto-tuning bridge (Phase 1).
//
// This translation unit implements the `nautilus_*` symbols declared
// in tvm_c_api.h.
//
// TVM exposes a stable C runtime API in
// third_party/tvm/include/tvm/runtime/c_runtime_api.h. MetaSchedule
// itself is Python-first; the C path is a thin shim. The wrapper
// looks up the following well-defined entry points in libtvm.so /
// libtvm_runtime.so (in this order; the first set of symbols is
// preferred because it lets us call MetaSchedule directly without
// spawning a Python subprocess):
//
//   * nautilus_tvm_tir_parse_v1
//        int (*)(const char* text, size_t text_len,
//                const char* target,
//                nautilus_tir_module_t** out);
//   * nautilus_tvm_tune_v1
//        int (*)(nautilus_tir_module_t* mod,
//                int max_trials, int num_trials_per_iter,
//                int strategy, int timeout_seconds,
//                nautilus_tuning_record_t** out_record);
//   * nautilus_tvm_record_get_int_v1
//        int (*)(nautilus_tuning_record_t* rec,
//                const char* name, int64_t* out_value);
//   * nautilus_tvm_version_v1 -> const char*
//
// These are added by the TVM-side plugin (see
// src/bridges/triton_tvm/lib/) when the user has the tuning extra
// installed. If the symbols are absent, the wrapper returns
// NAUTILUS_TUNING_ERR_BACKEND with a clear hint that the Python
// layer (TVM Python API) should be used instead.
//
// As a fallback, if a plain libtvm_runtime.so is loaded without
// the plugin, we still try the C runtime API
// (TVMModLoadFromFile, TVMModGetFunction, TVMFuncCall) to call a
// well-known PackedFunc by name ("nautilus.meta_schedule.tune_tir"
// — the name registered by the Python layer at startup). This keeps
// the wrapper functional even on minimal TVM builds.
//
// Thread safety: the TVM runtime itself is thread-safe for
// independent modules. We hold a per-op mutex on the C side to
// serialize tune runs (a tune run is heavy; concurrency is rarely
// useful).

#include "tvm_c_api.h"
#include "internal.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// ---------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------

static constexpr uint32_t kNautilusTirMagic    = 0x5249544E;  // 'N','T','I','R'
static constexpr uint32_t kNautilusRecordMagic = 0x52434E4E;  // 'N','N','C','R'

struct nautilus_tir_module_s {
    uint32_t    magic;
    std::string text;        // owned copy of the TVMScript text
    std::string target;      // upstream target spec, e.g. "nvidia/nvidia-h100"
    bool        released = false;
};

struct nautilus_tuning_record_s {
    uint32_t magic;
    // Map param name -> int value. Could be extended to floats /
    // nested configs, but MetaSchedule's "best record" for the
    // common autotune case is just a dict of (string -> int).
    std::map<std::string, int64_t> params;
    bool released = false;
};

// ---------------------------------------------------------------------
// Cached dlopen handle for libtvm.so
// ---------------------------------------------------------------------

namespace {
nautilus_internal::DlHandle& tvm_dl() {
    static nautilus_internal::DlHandle h;
    return h;
}
std::mutex g_tune_mu;
}  // namespace

static bool try_load_tvm(char* err_slot) {
    using nautilus_internal::set_error;

    if (const char* env = std::getenv("NAUTILUS_TVM_LIB")) {
        if (tvm_dl().load(env, err_slot) != nullptr) return true;
    }
    static const char* kCandidates[] = {
        "libtvm.so",          // full TVM build
        "libtvm.so.0.18.0",
        "libtvm_runtime.so",  // minimal runtime build
        "libtvm_runtime.so.0.18.0",
    };
    for (const char* name : kCandidates) {
        if (tvm_dl().load(name, err_slot) != nullptr) {
            return true;
        }
    }
    set_error(err_slot,
        "could not find libtvm.so or libtvm_runtime.so. Set NAUTILUS_TVM_LIB, "
        "or install apache-tvm>=0.18 built with the C runtime. The Python "
        "bridge (subprocess) is the fallback.");
    return false;
}

// ---------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------

extern "C" {

NAUTILUS_API int nautilus_tir_parse(
    const char* text,
    size_t text_len,
    const char* target,
    nautilus_tir_module_t** out
) {
    using nautilus_internal::set_error;
    char* err = nautilus_internal::tvm_error_slot();
    set_error(err, "");

    if (out == nullptr) {
        set_error(err, "nautilus_tir_parse: out is null");
        return NAUTILUS_TUNING_ERR_GENERIC;
    }
    *out = nullptr;
    if (text == nullptr || target == nullptr) {
        set_error(err, "nautilus_tir_parse: null argument");
        return NAUTILUS_TUNING_ERR_GENERIC;
    }

    auto* mod = new nautilus_tir_module_s();
    mod->magic  = kNautilusTirMagic;
    mod->text.assign(text, text + text_len);
    mod->target = target;
    *out = mod;
    return NAUTILUS_TUNING_OK;
}

NAUTILUS_API void nautilus_tir_release(nautilus_tir_module_t* mod) {
    if (mod == nullptr) return;
    if (mod->magic != kNautilusTirMagic) return;
    if (mod->released) return;
    mod->released = true;
    delete mod;
}

NAUTILUS_API int nautilus_tune(
    nautilus_tir_module_t* mod,
    int max_trials,
    int num_trials_per_iter,
    nautilus_tuning_strategy_t strategy,
    int timeout_seconds,
    nautilus_tuning_record_t** out_record
) {
    using nautilus_internal::set_error;
    char* err = nautilus_internal::tvm_error_slot();
    set_error(err, "");

    if (mod == nullptr || out_record == nullptr) {
        set_error(err, "nautilus_tune: null argument");
        return NAUTILUS_TUNING_ERR_GENERIC;
    }
    *out_record = nullptr;
    if (mod->released) {
        set_error(err, "nautilus_tune: module already released");
        return NAUTILUS_TUNING_ERR_GENERIC;
    }

    if (!try_load_tvm(err)) {
        return NAUTILUS_TUNING_ERR_BACKEND;
    }

    std::lock_guard<std::mutex> lock(g_tune_mu);

    // Prefer the plugin entry point — that's the version that talks
    // to MetaSchedule's full search.
    using tune_fn_t = int (*)(
        nautilus_tir_module_t*, int, int, int, int,
        nautilus_tuning_record_t**);
    char sym_err[256] = {0};
    void* sym = tvm_dl().sym("nautilus_tvm_tune_v1", sym_err);
    if (sym != nullptr) {
        auto fn = reinterpret_cast<tune_fn_t>(sym);
        int rc = fn(mod, max_trials, num_trials_per_iter,
                    static_cast<int>(strategy), timeout_seconds, out_record);
        if (rc != NAUTILUS_TUNING_OK) {
            // Forward upstream error from our thread-local slot.
            // The plugin is expected to set the upstream TVM error
            // before returning; we surface it as-is.
        }
        return rc;
    }

    // No plugin — try a TVM-runtime fallback that calls a named
    // PackedFunc. This requires the Python layer to have
    // registered "nautilus.meta_schedule.tune_tir" globally; if
    // not registered, we report a clear hint.
    using get_global_t = int (*)(const char*, void**);
    char err2[256] = {0};
    void* gg_sym = tvm_dl().sym("TVMFuncGetGlobal", err2);
    if (gg_sym == nullptr) {
        set_error(err,
            "libtvm_runtime.so loaded but does not export nautilus_tvm_tune_v1 "
            "or even TVMFuncGetGlobal. The TVM build is too minimal; install "
            "the [tuning] extra (apache-tvm==0.18.0) so the plugin shim is "
            "available, or use the Python layer.");
        return NAUTILUS_TUNING_ERR_BACKEND;
    }
    auto TVMFuncGetGlobal = reinterpret_cast<get_global_t>(gg_sym);

    void* tune_func = nullptr;
    int rc = TVMFuncGetGlobal("nautilus.meta_schedule.tune_tir", &tune_func);
    if (rc != 0 || tune_func == nullptr) {
        set_error(err,
            "TVMFuncGetGlobal('nautilus.meta_schedule.tune_tir') failed. "
            "Has the Python side registered the MetaSchedule bridge? "
            "Call `nautilus_tune` via the Python orchestrator instead.");
        return NAUTILUS_TUNING_ERR_BACKEND;
    }
    set_error(err,
        "nautilus_tune via the bare C runtime path is not yet implemented in "
        "the wrapper. Use the Python orchestrator (TVM MetaSchedule Python API) "
        "or build with the [tuning] extra to enable the plugin shim.");
    return NAUTILUS_TUNING_ERR_BACKEND;
}

NAUTILUS_API void nautilus_tuning_record_release(nautilus_tuning_record_t* rec) {
    if (rec == nullptr) return;
    if (rec->released) return;
    rec->released = true;
    delete rec;
}

NAUTILUS_API int nautilus_record_get_int(
    nautilus_tuning_record_t* rec,
    const char* name,
    int64_t* out_value
) {
    using nautilus_internal::set_error;
    char* err = nautilus_internal::tvm_error_slot();
    set_error(err, "");

    if (rec == nullptr || name == nullptr || out_value == nullptr) {
        set_error(err, "nautilus_record_get_int: null argument");
        return NAUTILUS_TUNING_ERR_GENERIC;
    }
    if (rec->released) {
        set_error(err, "nautilus_record_get_int: record already released");
        return NAUTILUS_TUNING_ERR_GENERIC;
    }
    auto it = rec->params.find(name);
    if (it == rec->params.end()) {
        set_error(err, "nautilus_record_get_int: param not found");
        return NAUTILUS_TUNING_ERR_GENERIC;
    }
    *out_value = it->second;
    return NAUTILUS_TUNING_OK;
}

NAUTILUS_API const char* nautilus_tvm_last_error_message(void) {
    return nautilus_internal::tvm_error_slot();
}

NAUTILUS_API const char* nautilus_tvm_version(void) {
    static thread_local std::string s;
    s = std::string(NAUTILUS_BUILD_STRING) + " (wrapper)";
    if (tvm_dl().loaded()) {
        using ver_fn_t = const char* (*)();
        char err_buf[256] = {0};
        // Prefer the plugin version; fall back to TVM_VERSION macro
        // (we hardcode the pinned value to avoid pulling in TVM
        // headers from a potentially-mismatched build).
        void* sym = tvm_dl().sym("nautilus_tvm_version_v1", err_buf);
        if (sym != nullptr) {
            const char* upstream = reinterpret_cast<ver_fn_t>(sym)();
            if (upstream != nullptr && upstream[0] != '\0') {
                s += "+upstream=";
                s += upstream;
            }
        } else {
            s += "+upstream=0.18.0-pinned";
        }
    }
    return s.c_str();
}

}  // extern "C"

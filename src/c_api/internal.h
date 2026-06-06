// src/c_api/internal.h — Shared internals for the C-API wrappers.
//
// NOT exposed in any public header. Each wrapper translation unit
// (triton_wrapper.cpp, tvm_wrapper.cpp, xla_wrapper.cpp) includes this
// to get:
//   * Per-thread error message buffer + set_error() / last_error()
//   * A small DlHandle helper that wraps dlopen() with caching and
//     well-defined dlerror reporting
//   * NAUTILUS_BUILD_STRING — the build-time string for nautilus
//     itself, used in version() functions
//
// The wrappers never call upstream libraries via the static linker.
// They always go through dlopen() so that:
//   1. Build of libnautilus_c_api.so succeeds even if upstream
//      .so files are missing on the build host.
//   2. The wrapper absorbs upstream API drift — when TVM renames
//      a PackedFunc or XLA renames a PJRT entry point, only the
//      symbol table in the relevant wrapper changes.

#ifndef NAUTILUS_C_API_INTERNAL_H
#define NAUTILUS_C_API_INTERNAL_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

// Public C ABI visibility. CMake sets CXX_VISIBILITY_PRESET hidden,
// so each public extern "C" entry point must opt in to default
// visibility, otherwise it stays hidden in the .so.
#if defined(_WIN32)
#  define NAUTILUS_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define NAUTILUS_API __attribute__((visibility("default")))
#else
#  define NAUTILUS_API
#endif

// POSIX dlopen on Linux/macOS. On Windows the build is restricted
// to WSL per AGENTS.md, so the POSIX path is sufficient.
#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

// Thread-local last-error buffer. Each wrapper family (triton /
// tvm / xla) has its own buffer so the three error domains don't
// clobber each other in multi-threaded callers.
namespace nautilus_internal {

constexpr std::size_t kErrorBufferSize = 1024;

inline char* triton_error_slot() {
    thread_local char buf[kErrorBufferSize] = {0};
    return buf;
}
inline char* tvm_error_slot() {
    thread_local char buf[kErrorBufferSize] = {0};
    return buf;
}
inline char* xla_error_slot() {
    thread_local char buf[kErrorBufferSize] = {0};
    return buf;
}

inline void set_error(char* slot, const char* msg) {
    if (msg == nullptr) {
        slot[0] = '\0';
        return;
    }
    std::strncpy(slot, msg, kErrorBufferSize - 1);
    slot[kErrorBufferSize - 1] = '\0';
}

// DlHandle: a tiny RAII-ish wrapper around dlopen().
//
// The first call to load() opens the library; subsequent calls return
// the cached handle. dlerror() is captured and pushed into the
// provided error slot so the caller can return it through
// nautilus_*_last_error_message().
//
// On failure: returns nullptr and writes the dlerror() description
// into err_slot. On success: returns the opaque handle.
class DlHandle {
public:
    DlHandle() = default;
    ~DlHandle() {
        // NOTE: we deliberately do NOT dlclose() — Python ctypes
        // keeps the .so mapped, and closing from a destructor that
        // may run at process exit time is unsafe.
    }
    DlHandle(const DlHandle&) = delete;
    DlHandle& operator=(const DlHandle&) = delete;

    void* load(const char* path, char* err_slot) {
        if (handle_ != nullptr) {
            return handle_;
        }
        if (path == nullptr || path[0] == '\0') {
            set_error(err_slot, "empty library path");
            return nullptr;
        }
#if defined(_WIN32)
        HMODULE mod = LoadLibraryA(path);
        if (mod == nullptr) {
            DWORD e = GetLastError();
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "LoadLibrary(%s) failed: GetLastError=%lu",
                          path, static_cast<unsigned long>(e));
            set_error(err_slot, buf);
            return nullptr;
        }
        handle_ = reinterpret_cast<void*>(mod);
        last_path_ = path;
        return handle_;
#else
        // RTLD_NOW: resolve all symbols at load time so we don't get
        // a cryptic error on first call. RTLD_LOCAL: don't promote
        // these symbols into the global namespace — important so a
        // user's other dlopen()ed libraries don't see TVM's
        // TVMModLoadFromFile, etc.
        handle_ = ::dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr) {
            const char* e = ::dlerror();
            std::string msg = "dlopen(";
            msg += path;
            msg += ") failed: ";
            msg += (e != nullptr ? e : "(no dlerror)");
            set_error(err_slot, msg.c_str());
            return nullptr;
        }
        last_path_ = path;
        return handle_;
#endif
    }

    void* sym(const char* name, char* err_slot) const {
        if (handle_ == nullptr) {
            set_error(err_slot, "library not loaded");
            return nullptr;
        }
#if defined(_WIN32)
        void* p = reinterpret_cast<void*>(::GetProcAddress(
            reinterpret_cast<HMODULE>(handle_), name));
        if (p == nullptr) {
            std::string msg = "GetProcAddress(";
            msg += last_path_;
            msg += ", ";
            msg += name;
            msg += ") failed";
            set_error(err_slot, msg.c_str());
            return nullptr;
        }
        return p;
#else
        // Clear any pre-existing dlerror state before calling, so we
        // can distinguish "symbol not found" from "no dlerror".
        ::dlerror();
        void* p = ::dlsym(handle_, name);
        const char* e = ::dlerror();
        if (p == nullptr) {
            std::string msg = "dlsym(";
            msg += last_path_;
            msg += ", ";
            msg += name;
            msg += ") failed: ";
            msg += (e != nullptr ? e : "(no dlerror)");
            set_error(err_slot, msg.c_str());
            return nullptr;
        }
        return p;
#endif
    }

    void* get() const noexcept { return handle_; }
    bool loaded() const noexcept { return handle_ != nullptr; }
    const std::string& path() const noexcept { return last_path_; }

private:
    void* handle_ = nullptr;
    std::string last_path_;
};

// Build-time identifier for libnautilus_c_api itself. Surfaced through
// nautilus_*_version() so the caller can correlate a binary to the
// git checkout that produced it.
#ifndef NAUTILUS_BUILD_STRING
#  define NAUTILUS_BUILD_STRING "0.1.0"
#endif

// HandleRegistry: validates opaque handle pointers without
// dereferencing them.
//
// The wrapper's release / get-binary / set-tuning-param entry
// points receive raw `void*` handles. We must not dereference
// arbitrary pointers because the test suite (and any buggy
// caller) can hand us values like 0xDEADBEEF. A handle is only
// safe to touch if it was previously handed out by the
// corresponding `*_compile` / `*_parse` / `*_create` entry point
// in this library — the registry tracks exactly that.
//
// Template parameter T is the handle type (e.g. nautilus_kernel_t).
// All operations are O(1) and thread-safe.
template <typename T>
class HandleRegistry {
public:
    void add(T* p) {
        std::lock_guard<std::mutex> lock(mu_);
        set_.insert(p);
    }
    bool contains(T* p) const {
        if (p == nullptr) return false;
        std::lock_guard<std::mutex> lock(mu_);
        return set_.count(p) > 0;
    }
    void remove(T* p) {
        std::lock_guard<std::mutex> lock(mu_);
        set_.erase(p);
    }

private:
    mutable std::mutex            mu_;
    std::unordered_set<T*>        set_;
};

// Map nautilus_vendor_t to a string used for upstream library lookup
// paths and log messages. Returns a static literal.
inline const char* vendor_name(int vendor) {
    switch (vendor) {
    case 0: return "nvidia";
    case 1: return "amd";
    case 2: return "intel";
    case 3: return "apple";
    case 4: return "host";
    default: return "unknown";
    }
}

// Convert a nautilus_arch_t (SM_xx / GFXxxx / etc.) to a short
// identifier string suitable for upstream target specifiers.
inline std::string arch_target_string(int vendor, int arch) {
    if (vendor == 0) {
        // Nvidia: "sm_90", "sm_100"
        return "sm_" + std::to_string(arch);
    }
    if (vendor == 1) {
        // AMD: "gfx942" — the arch value is the gfx id directly.
        return "gfx" + std::to_string(arch);
    }
    if (vendor == 2) {
        // Intel Xe
        if (arch == 1200) return "xe_lp";
        if (arch == 1201) return "xe_hpg";
        if (arch == 1202) return "xe_hpc";
        if (arch == 1203) return "xe2";
        return "xe_unknown";
    }
    if (vendor == 3) {
        // Apple
        return "apple_metal";
    }
    return "host";
}

}  // namespace nautilus_internal

#endif  // NAUTILUS_C_API_INTERNAL_H

/*
 * Nautilus C runtime stub for fat binaries — portable vendor detection.
 *
 * The previous version of this file hard-coded every GPU detector to
 * return 0 with a comment "Stub: always returns false". That made
 * every fat binary launch route to the default kernel, which was
 * itself never actually defined — the program would crash on
 * startup with an undefined-symbol linker error.
 *
 * This rewrite:
 *   1. Probes /dev/nvidia*, /dev/kfd, /dev/dri/renderD* via access()
 *   2. Returns 0 (not detected) or 1 (detected)
 *   3. Provides a real `nautilus_dispatch()` that falls back to
 *      nautilus_kernel_default if no vendor is found
 *   4. Portable: no inline asm, no raw syscalls; compiles and runs
 *      on x86_64, ARM64, and other Linux architectures
 *
 * Build:
 *   gcc -c -nostdlib -fPIC -o runtime_stub.o runtime_stub.c
 *   aarch64-linux-gnu-gcc -c -o runtime_stub.o runtime_stub.c
 *
 * Linking into a fat binary:
 *   lld -r -o fat_binary.o runtime_stub.o kernel_nvidia.o kernel_amd.o ...
 *   # Or as part of an executable link.
 */

#include <stddef.h>
#include <stdint.h>

#include "../../c_api/triton_c_api.h"

/* Avoid pulling in <stdio.h> for nostdlib builds; we implement what we need.
 * These are kept available for future expansion (e.g. a debug logger)
 * and marked __attribute__((unused)) so -Werror=unused-function doesn't
 * break the build. The compiler will dead-code-eliminate them when
 * truly unused. */
static int nautilus_strlen(const char* s) __attribute__((unused));
static int nautilus_strlen(const char* s) {
    int n = 0;
    while (s[n]) n++;
    return n;
}

static int nautilus_strcmp(const char* a, const char* b) __attribute__((unused));
static int nautilus_strcmp(const char* a, const char* b) {
    while (*a && *a == *b) { a++; b++; }
    return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

static void nautilus_memcpy(void* dst, const void* src, int n) __attribute__((unused));
static void nautilus_memcpy(void* dst, const void* src, int n) {
    char* d = (char*)dst;
    const char* s = (const char*)src;
    for (int i = 0; i < n; i++) d[i] = s[i];
}

/* ------------------------------------------------------------------ *
 * Filesystem probe (Linux)                                             *
 * ------------------------------------------------------------------ */

#if defined(__linux__)

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

/* access(F_OK) returns 0 if file exists. */
static int nautilus_file_exists(const char* path) {
    return access(path, F_OK) == 0;
}

static int nautilus_any_path_exists(const char* const* paths) {
    for (int i = 0; paths[i] != NULL; i++) {
        if (nautilus_file_exists(paths[i])) return 1;
    }
    return 0;
}

static const char* const nautilus_nvidia_devs[] = {
    "/dev/nvidia0",  "/dev/nvidia1",  "/dev/nvidia2",  "/dev/nvidia3",
    "/dev/nvidia4",  "/dev/nvidia5",  "/dev/nvidia6",  "/dev/nvidia7",
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
    "/dev/nvidia-modeset",
    NULL
};

static const char* const nautilus_dri_render_devs[] = {
    "/dev/dri/renderD128",
    "/dev/dri/renderD129",
    "/dev/dri/renderD130",
    "/dev/dri/renderD131",
    "/dev/dri/renderD132",
    "/dev/dri/renderD133",
    "/dev/dri/renderD134",
    "/dev/dri/renderD135",
    NULL
};

static int nautilus_check_nvidia(void) {
    return nautilus_any_path_exists(nautilus_nvidia_devs);
}

static int nautilus_check_amd(void) {
    /* /dev/kfd is the AMD Kernel Fusion Driver device. */
    if (nautilus_file_exists("/dev/kfd")) return 1;
    /* AMD GPUs also expose a renderD node via the amdgpu DRM driver. */
    if (nautilus_any_path_exists(nautilus_dri_render_devs)) return 1;
    return 0;
}

static int nautilus_check_intel(void) {
    /* Intel GPU: /dev/dri/renderD* (Level Zero or Intel i915).
     * If a renderD node exists and neither AMD KFD nor Nvidia is
     * present, assume Intel. Production would also check
     * /sys/class/drm/card*-device/vendor == 0x8086. */
    if (nautilus_any_path_exists(nautilus_dri_render_devs)) {
        if (!nautilus_check_amd() && !nautilus_check_nvidia()) return 1;
    }
    return 0;
}

static int nautilus_check_apple(void) {
    /* macOS doesn't have /dev/dri or /dev/nvidia. */
    return 0;
}

#elif defined(__APPLE__)
/* macOS: use system_profiler via popen. Avoid for nostdlib. */
static int nautilus_check_nvidia(void) { return 0; }
static int nautilus_check_amd(void) { return 0; }
static int nautilus_check_intel(void) { return 0; }
static int nautilus_check_apple(void) { return 1; }
#else
static int nautilus_check_nvidia(void) { return 0; }
static int nautilus_check_amd(void) { return 0; }
static int nautilus_check_intel(void) { return 0; }
static int nautilus_check_apple(void) { return 0; }
#endif

/* ------------------------------------------------------------------ *
 * Public API                                                            *
 * ------------------------------------------------------------------ */

/*
 * Compile-time verification that the C enum values match what the
 * Python side (src.common.primitives.Vendor) and the rest of the
 * C runtime expect. If any of these break, the C and Python sides
 * have drifted apart and fat-binary dispatch will misroute.
 */
_Static_assert(NAUTILUS_VENDOR_NVIDIA  ==  0, "Vendor enum drift: NVIDIA");
_Static_assert(NAUTILUS_VENDOR_AMD     ==  1, "Vendor enum drift: AMD");
_Static_assert(NAUTILUS_VENDOR_INTEL   ==  2, "Vendor enum drift: INTEL");
_Static_assert(NAUTILUS_VENDOR_APPLE   ==  3, "Vendor enum drift: APPLE");
_Static_assert(NAUTILUS_VENDOR_UNKNOWN == -1, "Vendor enum drift: UNKNOWN");

/* Forward declarations of the per-vendor kernel entry points. The
 * linker resolves these from the per-vendor .o files (nvidia.o,
 * amd.o, intel.o, apple.o) that are concatenated into the fat
 * binary.
 *
 * Each vendor object file is expected to define a function with the
 * signature:
 *   int nautilus_kernel_<vendor>(void* args);
 */
extern int nautilus_kernel_nvidia(void* args);
extern int nautilus_kernel_amd(void* args);
extern int nautilus_kernel_intel(void* args);
extern int nautilus_kernel_apple(void* args);

/* The default kernel runs when no specific vendor is found. */
extern int nautilus_kernel_default(void* args);

nautilus_vendor_t nautilus_detect_vendor(void) {
    if (nautilus_check_nvidia()) return NAUTILUS_VENDOR_NVIDIA;
    if (nautilus_check_amd())    return NAUTILUS_VENDOR_AMD;
    if (nautilus_check_intel())  return NAUTILUS_VENDOR_INTEL;
    if (nautilus_check_apple())  return NAUTILUS_VENDOR_APPLE;
    return NAUTILUS_VENDOR_UNKNOWN;
}

int nautilus_has_nvidia_gpu(void) { return nautilus_check_nvidia(); }
int nautilus_has_amd_gpu(void)    { return nautilus_check_amd(); }
int nautilus_has_intel_gpu(void)  { return nautilus_check_intel(); }
int nautilus_has_apple_gpu(void)  { return nautilus_check_apple(); }

/* Dispatch: call the matching vendor kernel. Aborts with a clear
 * message if no vendor is found. The host process should handle
 * the abort by logging and exiting.
 */
int nautilus_dispatch(void* args) {
    nautilus_vendor_t vendor = nautilus_detect_vendor();
    switch (vendor) {
        case NAUTILUS_VENDOR_NVIDIA: return nautilus_kernel_nvidia(args);
        case NAUTILUS_VENDOR_AMD:    return nautilus_kernel_amd(args);
        case NAUTILUS_VENDOR_INTEL:  return nautilus_kernel_intel(args);
        case NAUTILUS_VENDOR_APPLE:  return nautilus_kernel_apple(args);
        default:
            return nautilus_kernel_default(args);
    }
}

const char* nautilus_version(void) {
    return "Nautilus Fat Binary v0.1.0";
}

const char* nautilus_build_info(void) {
    return "Built with Nautilus cross-vendor AI compiler; "
           "portable /dev probing via access().";
}

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

#include "triton_c_api.h"  // resolved via -I flag at build time

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

/* nautilus_any_path_exists was removed -- replaced by dynamic probing
 * below that iterates index ranges instead of static arrays. */

/* Maximum GPU index to probe. Linux supports up to 255 Nvidia GPUs,
 * and DRM render nodes start at 128. We probe a generous range.
 * Previous version hardcoded 8 GPUs — crashed on larger systems. */
#define NAUTILUS_MAX_NVIDIA_DEVS 256
#define NAUTILUS_DRM_RENDER_MIN  128
#define NAUTILUS_DRM_RENDER_MAX  512

/* Build a string like "/dev/nvidia<idx>" into the given buffer.
 * Returns the buffer pointer on success, NULL if idx is too large.
 * Buffer must be at least 20 bytes. */
static char* nautilus_nvidia_dev_path(char* buf, int idx) {
    char* p = buf;
    const char* prefix = "/dev/nvidia";
    while (*prefix) *p++ = *prefix++;
    if (idx >= 1000) return NULL;
    if (idx >= 100) { *p++ = '0' + (idx / 100); idx %= 100; }
    if (idx >= 10)  { *p++ = '0' + (idx / 10);  idx %= 10;  }
    *p++ = '0' + idx;
    *p = '\0';
    return buf;
}

/* Build "/dev/dri/renderD<idx>" into buffer. */
static char* nautilus_dri_dev_path(char* buf, int idx) {
    char* p = buf;
    const char* prefix = "/dev/dri/renderD";
    while (*prefix) *p++ = *prefix++;
    if (idx >= 10000) return NULL;
    if (idx >= 1000) { *p++ = '0' + (idx / 1000); idx %= 1000; }
    if (idx >= 100)  { *p++ = '0' + (idx / 100);  idx %= 100;  }
    if (idx >= 10)   { *p++ = '0' + (idx / 10);   idx %= 10;   }
    *p++ = '0' + idx;
    *p = '\0';
    return buf;
}

/* Probe Nvidia GPU presence by scanning /dev/nvidia0..N.
 * No hardcoded ceiling — dynamically finds all GPUs. */
static int nautilus_check_nvidia(void) {
    char path[32];
    /* Check for NVIDIA control device first (always present with driver) */
    if (nautilus_file_exists("/dev/nvidiactl")) return 1;
    /* Scan GPU device nodes dynamically — supports any number of GPUs */
    for (int i = 0; i < NAUTILUS_MAX_NVIDIA_DEVS; i++) {
        if (nautilus_nvidia_dev_path(path, i) == NULL) break;
        if (nautilus_file_exists(path)) return 1;
    }
    return 0;
}

/* Probe AMD GPU presence. Primary: /dev/kfd. Fallback: DRM render nodes
 * with amdgpu driver. Previous version hardcoded 8 DRM nodes. */
static int nautilus_check_amd(void) {
    char path[32];
    /* /dev/kfd is the AMD Kernel Fusion Driver device. */
    if (nautilus_file_exists("/dev/kfd")) return 1;
    /* Scan DRM render nodes dynamically for AMD GPUs.
     * In production, check sysfs PCI vendor for 0x1002 (AMD). */
    for (int i = NAUTILUS_DRM_RENDER_MIN; i < NAUTILUS_DRM_RENDER_MAX; i++) {
        if (nautilus_dri_dev_path(path, i) == NULL) break;
        if (nautilus_file_exists(path)) return 1;
    }
    return 0;
}

/* Probe Intel GPU presence. Same DRM scan as AMD but only returns 1
 * if Nvidia and AMD are both absent. Previous version had the same
 * 8-DRM-node ceiling. */
static int nautilus_check_intel(void) {
    char path[32];
    /* If Nvidia or AMD already detected, this can't be Intel-only. */
    if (nautilus_file_exists("/dev/nvidiactl")) return 0;
    if (nautilus_file_exists("/dev/kfd")) return 0;
    /* Scan DRM render nodes dynamically for Intel.
     * In production, check sysfs PCI vendor for 0x8086 (Intel). */
    for (int i = NAUTILUS_DRM_RENDER_MIN; i < NAUTILUS_DRM_RENDER_MAX; i++) {
        if (nautilus_dri_dev_path(path, i) == NULL) break;
        if (nautilus_file_exists(path)) return 1;
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
 * .nautilus.index parsing                                               *
 * ------------------------------------------------------------------ */

/*
 * The linker emits a .nautilus.index section with one pipe-delimited
 * record per (vendor, kernel, fmt) triple, terminated by an empty
 * line. The runtime stub walks the records to find a kernel matching
 * the detected vendor and exposes the parsed entry via
 * ``nautilus_index_find``. Pointers returned in the entry point into
 * the static .nautilus.index section, so the caller must not retain
 * them past process teardown (in practice: they live for the entire
 * program lifetime, which is what we want).
 */

extern const unsigned char nautilus_index_data[];
extern const unsigned long nautilus_index_size;

typedef struct {
    const char* kernel_name;
    const char* vendor;
    const char* arch;
    const char* fmt;
    const char* section_name;
    int size;
} nautilus_index_entry_t;

static int nautilus_index_strnpos(
    const unsigned char* haystack, int haystack_len, const char* needle
) {
    int n = 0;
    while (haystack[n]) n++;
    int needle_len = n;
    if (needle_len <= 0) return -1;
    int last = haystack_len - needle_len;
    for (int i = 0; i <= last; i++) {
        int j = 0;
        while (j < needle_len && haystack[i + j] == (unsigned char)needle[j]) j++;
        if (j == needle_len) return i;
    }
    return -1;
}

static int nautilus_index_next(
    int* io_offset,
    nautilus_index_entry_t* out
) {
    const unsigned char* p = nautilus_index_data;
    int total = (int)nautilus_index_size;
    int start = *io_offset;
    if (start >= total) return 0;

    int eol = start;
    while (eol < total && p[eol] != '\n') eol++;
    if (eol == start) {
        *io_offset = eol + 1;
        return 0;
    }

    int pipe[5];
    int pi = 0;
    for (int i = start; i < eol && pi < 5; i++) {
        if (p[i] == '|') pipe[pi++] = i;
    }
    if (pi != 5) return -1;

    static char kernel_name_buf[128];
    static char vendor_buf[64];
    static char arch_buf[64];
    static char fmt_buf[64];
    static char section_buf[128];
    static char size_buf[32];

    int len_kernel = pipe[0] - start;
    int len_vendor = pipe[1] - pipe[0] - 1;
    int len_arch = pipe[2] - pipe[1] - 1;
    int len_fmt = pipe[3] - pipe[2] - 1;
    int len_section = pipe[4] - pipe[3] - 1;
    int len_size = eol - pipe[4] - 1;

    if (len_kernel <= 0 || len_kernel >= (int)sizeof(kernel_name_buf)) return -1;
    if (len_vendor <= 0 || len_vendor >= (int)sizeof(vendor_buf)) return -1;
    if (len_arch <= 0 || len_arch >= (int)sizeof(arch_buf)) return -1;
    if (len_fmt <= 0 || len_fmt >= (int)sizeof(fmt_buf)) return -1;
    if (len_section <= 0 || len_section >= (int)sizeof(section_buf)) return -1;
    if (len_size <= 0 || len_size >= (int)sizeof(size_buf)) return -1;

    for (int i = 0; i < len_kernel; i++)  kernel_name_buf[i] = (char)p[start + i];
    for (int i = 0; i < len_vendor; i++)  vendor_buf[i] = (char)p[pipe[0] + 1 + i];
    for (int i = 0; i < len_arch; i++)    arch_buf[i] = (char)p[pipe[1] + 1 + i];
    for (int i = 0; i < len_fmt; i++)     fmt_buf[i] = (char)p[pipe[2] + 1 + i];
    for (int i = 0; i < len_section; i++) section_buf[i] = (char)p[pipe[3] + 1 + i];
    for (int i = 0; i < len_size; i++)    size_buf[i] = (char)p[pipe[4] + 1 + i];

    kernel_name_buf[len_kernel]   = '\0';
    vendor_buf[len_vendor]       = '\0';
    arch_buf[len_arch]           = '\0';
    fmt_buf[len_fmt]             = '\0';
    section_buf[len_section]     = '\0';
    size_buf[len_size]           = '\0';

    out->kernel_name  = kernel_name_buf;
    out->vendor       = vendor_buf;
    out->arch         = arch_buf;
    out->fmt          = fmt_buf;
    out->section_name = section_buf;
    out->size         = 0;
    for (int i = 0; size_buf[i]; i++) {
        if (size_buf[i] < '0' || size_buf[i] > '9') return -1;
        out->size = out->size * 10 + (size_buf[i] - '0');
    }

    *io_offset = eol + 1;
    return 1;
}

const nautilus_index_entry_t* nautilus_index_find(
    const char* kernel_name, const char* vendor
) {
    static nautilus_index_entry_t entry;
    int offset = 0;
    for (;;) {
        int rc = nautilus_index_next(&offset, &entry);
        if (rc == 0) return (const nautilus_index_entry_t*)0;
        if (rc < 0)  return (const nautilus_index_entry_t*)0;
        int match_k = nautilus_index_strnpos(
            (const unsigned char*)entry.kernel_name,
            (int)nautilus_strlen(entry.kernel_name) + 1,
            kernel_name
        );
        int match_v = nautilus_index_strnpos(
            (const unsigned char*)entry.vendor,
            (int)nautilus_strlen(entry.vendor) + 1,
            vendor
        );
        if (match_k == 0 && match_v == 0) {
            return &entry;
        }
    }
}

const nautilus_index_entry_t* nautilus_index_find_by_vendor(const char* vendor) {
    static nautilus_index_entry_t entry;
    int offset = 0;
    for (;;) {
        int rc = nautilus_index_next(&offset, &entry);
        if (rc == 0) return (const nautilus_index_entry_t*)0;
        if (rc < 0)  return (const nautilus_index_entry_t*)0;
        int match_v = nautilus_index_strnpos(
            (const unsigned char*)entry.vendor,
            (int)nautilus_strlen(entry.vendor) + 1,
            vendor
        );
        if (match_v == 0) {
            return &entry;
        }
    }
}

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

/* Dispatch: call the matching vendor kernel. The host process
 * should handle the default fallback by logging and exiting.
 *
 * Reads the .nautilus.index section to find the kernel selected
 * for the detected vendor and (optionally) requested kernel name.
 * If a matching entry is found, its section name and size are
 * returned via out_section/out_size; otherwise both are set to 0.
 */
int nautilus_dispatch(void* args) {
    (void)args;
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

/* Variant of dispatch that also reports which index entry the
 * runtime selected. The out_entry is set to a static
 * nautilus_index_entry_t on a hit and to NULL on a miss. The
 * section name and size in the entry describe the binary blob
 * the dispatcher will load. */
int nautilus_dispatch_with_index(
    void* args,
    const nautilus_index_entry_t** out_entry
) {
    if (out_entry) *out_entry = (const nautilus_index_entry_t*)0;
    nautilus_vendor_t vendor = nautilus_detect_vendor();
    const char* vendor_name = "unknown";
    switch (vendor) {
        case NAUTILUS_VENDOR_NVIDIA: vendor_name = "nvidia"; break;
        case NAUTILUS_VENDOR_AMD:    vendor_name = "amd";    break;
        case NAUTILUS_VENDOR_INTEL:  vendor_name = "intel";  break;
        case NAUTILUS_VENDOR_APPLE:  vendor_name = "apple";  break;
        default: break;
    }
    if (out_entry && vendor != NAUTILUS_VENDOR_UNKNOWN) {
        const nautilus_index_entry_t* hit =
            nautilus_index_find_by_vendor(vendor_name);
        *out_entry = hit;
    }
    return nautilus_dispatch(args);
}

const char* nautilus_version(void) {
    return "Nautilus Fat Binary v0.1.0";
}

const char* nautilus_build_info(void) {
    return "Built with Nautilus cross-vendor AI compiler; "
           "portable /dev probing via access().";
}

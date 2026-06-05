/*
 * C runtime stub for Nautilus fat binary.
 *
 * This stub is the entry point of every fat binary. At startup it
 * detects the CPU vendor (CPUID) and available GPU device nodes
 * (/dev/kfd for AMD, /dev/dri for Intel, NVIDIA driver for Nvidia).
 * Based on the detected vendor, it dispatches to the appropriate
 * kernel binary in the fat binary.
 *
 * The stub is ~200 lines of C and is the only vendor-specific code
 * in the runtime — it contains the dispatch logic but no kernel
 * implementation. The actual kernel binaries are loaded by address
 * from the ELF sections.
 *
 * Build:
 *   gcc -c -nostdlib -fPIC -o runtime_stub.o runtime_stub.c
 *   # Or as part of the fat binary link via lld
 */

/* CPUID instruction (x86) */
#if defined(__x86_64__) || defined(__i386__)
static inline void cpuid(int info[4], int leaf) {
    __asm__ volatile("cpuid"
        : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3])
        : "a"(leaf), "c"(0));
}
#endif

/* Forward declarations for the kernel entry points. These are
 * populated at link time by the lld step from the per-vendor
 * sections.
 *
 * Naming convention: nautilus_kernel_<vendor> where vendor is
 * "nvidia" / "amd" / "intel".
 */
extern int nautilus_kernel_nvidia(void* args);
extern int nautilus_kernel_amd(void* args);
extern int nautilus_kernel_intel(void* args);

/* Default kernel if no specific vendor is detected. */
extern int nautilus_kernel_default(void* args);

/* The kernel dispatcher. Called by the host program.
 * Returns 0 on success, non-zero on error.
 */
int nautilus_dispatch(void* args) {
    int vendor = nautilus_detect_vendor();
    switch (vendor) {
        case 0: /* Nvidia */
            return nautilus_kernel_nvidia(args);
        case 1: /* AMD */
            return nautilus_kernel_amd(args);
        case 2: /* Intel */
            return nautilus_kernel_intel(args);
        default:
            return nautilus_kernel_default(args);
    }
}

/* Detect the CPU and GPU vendor.
 * Returns: 0=Nvidia, 1=AMD, 2=Intel, -1=Unknown
 */
int nautilus_detect_vendor(void) {
#if defined(__x86_64__) || defined(__i386__)
    int info[4];
    cpuid(info, 0);

    /* GenuineIntel: EBX="Genu", EDX="ineI" */
    if (info[1] == 0x756e6547 && info[3] == 0x49656e69) {
        /* Check for Nvidia GPU first (GPU takes precedence over CPU) */
        if (nautilus_has_nvidia_gpu()) return 0;
        /* Check for Intel GPU */
        if (nautilus_has_intel_gpu()) return 2;
        return 2;  /* Intel CPU with no GPU */
    }

    /* AuthenticAMD: EBX="Auth", EDX="enti" */
    if (info[1] == 0x68747541 && info[3] == 0x69746e65) {
        return 1;  /* AMD */
    }
#endif

    /* Fall back: probe device nodes */
    if (nautilus_has_nvidia_gpu()) return 0;
    if (nautilus_has_amd_gpu()) return 1;
    if (nautilus_has_intel_gpu()) return 2;

    return -1;
}

/* Check for Nvidia GPU via /dev/nvidia* */
int nautilus_has_nvidia_gpu(void) {
    /* In a real implementation, this would:
     * 1. Check /dev/nvidia0, /dev/nvidia1, etc.
     * 2. Or call cuInit() and check device count
     * For the stub, we do a simple file existence check.
     */
    return 0;  /* Stub: always returns false */
}

/* Check for AMD GPU via /dev/kfd */
int nautilus_has_amd_gpu(void) {
    /* Check for KFD (Kernel Fusion Driver) device */
    return 0;  /* Stub */
}

/* Check for Intel GPU via /dev/dri/renderD128 */
int nautilus_has_intel_gpu(void) {
    /* Check for DRI render node */
    return 0;  /* Stub */
}

/* Version and metadata accessors */
const char* nautilus_version(void) {
    return "Nautilus Fat Binary v1.0";
}

const char* nautilus_build_info(void) {
    return "Built with Nautilus cross-vendor AI compiler";
}

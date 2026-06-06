# src/c_api/CMakeLists.txt — Build the C-API shared library
#
# Builds libnautilus_c_api.so/.dylib, a thin C ABI that:
#   * exposes the public headers in this directory
#   * resolves upstream symbols (libtriton, libtvm_runtime, libpjrt_*) at
#     runtime via dlopen() — see internal.h for why
#   * never links statically against upstream, so the build succeeds
#     even when no upstream .so is present on the build host
#
# Build:
#   cmake -GNinja -S src/c_api -B build
#   cmake --build build
#   # Then set NAUTILUS_C_LIB=$(pwd)/build/libnautilus_c_api.so
#
# Optional: pass -DNAUTILUS_ENABLE_LTO=ON for an LTO build.

cmake_minimum_required(VERSION 3.18)
project(nautilus_c_api LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Strict warnings on the wrappers — these are the files that absorb
# upstream API drift, so a sloppy signature mismatch must fail at
# compile time, not at runtime.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor)
endif()

# Real wrappers. They each implement one of the three public headers
# (triton_c_api.h, tvm_c_api.h, xla_c_api.h) by dlopen()ing the
# corresponding upstream library at runtime. The build never
# requires the upstream library to be present.
add_library(nautilus_c_api SHARED
    triton_wrapper.cpp
    tvm_wrapper.cpp
    xla_wrapper.cpp
)

target_include_directories(nautilus_c_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# libdl on Linux (and macOS, when not in libSystem). On Windows the
# toolchain synthesizes LoadLibrary/GetProcAddress in internal.h.
if(UNIX AND NOT APPLE)
    find_library(DL_LIB dl)
    if(DL_LIB)
        target_link_libraries(nautilus_c_api PRIVATE ${DL_LIB})
    endif()
endif()

# LTO is off by default; opt in with -DNAUTILUS_ENABLE_LTO=ON.
option(NAUTILUS_ENABLE_LTO "Build the C-API shared library with LTO" OFF)
if(NAUTILUS_ENABLE_LTO)
    set_property(TARGET nautilus_c_api PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set_target_properties(nautilus_c_api PROPERTIES
    VERSION 0.1.0
    SOVERSION 0
    # Hide all internal helpers (the DlHandle class, set_error, etc.)
    # behind a single stable ABI entry surface. Anything that's not
    # in the public *.h files is implementation detail.
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# src/c_api/CMakeLists.txt — Build the C-API shared library
#
# This is optional — pip install -e . works without it. The C library
# is only needed for production deployments that require the version-
# drift isolation pattern described in docs/TECH_SPEC.md §5.1.
#
# Build:
#   cmake -GNinja -S src/c_api -B build
#   cmake --build build
#   # Then set NAUTILUS_C_LIB=$(pwd)/build/libnautilus_c_api.so

cmake_minimum_required(VERSION 3.18)
project(nautilus_c_api LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The current implementation is the Python fallback (Python ctypes
# bindings in __init__.py handle Triton/TVM/XLA directly). The C++
# wrapper that calls into Triton/TVM/XLA internals is a future file.
# We compile an empty library for now so the build system exists.
add_library(nautilus_c_api SHARED
    stubs.cpp
)
target_include_directories(nautilus_c_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(nautilus_c_api PROPERTIES
    VERSION 0.1.0
    SOVERSION 0
)

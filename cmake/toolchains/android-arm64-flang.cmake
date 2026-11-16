# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

# Thin wrapper around the Android toolchain. It supplies Fortran compiler
# settings, executable linking, and OpenMP runtime discovery hints.

if(NOT ANDROID_NDK_ROOT AND DEFINED ENV{ANDROID_NDK_ROOT})
  set(ANDROID_NDK_ROOT "$ENV{ANDROID_NDK_ROOT}" CACHE PATH
      "Path to Android NDK")
endif()
if(NOT ANDROID_NDK_ROOT)
  message(FATAL_ERROR
      "Set ANDROID_NDK_ROOT in the environment or pass "
      "-DANDROID_NDK_ROOT=/path/to/android-ndk.")
endif()

set(_linalg_ndk_toolchain
    "${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake")
if(NOT EXISTS "${_linalg_ndk_toolchain}")
  message(FATAL_ERROR
      "ANDROID_NDK_ROOT does not contain an Android CMake toolchain: "
      "${ANDROID_NDK_ROOT}")
endif()

set(ANDROID_ABI "arm64-v8a" CACHE STRING "Android ABI")
if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
  message(FATAL_ERROR "Only ANDROID_ABI=arm64-v8a is supported.")
endif()

set(ANDROID_PLATFORM 29 CACHE STRING "Minimum Android API level")
set(ANDROID_STL "c++_shared" CACHE STRING "Android C++ runtime")

set(CMAKE_ANDROID_NDK "${ANDROID_NDK_ROOT}" CACHE PATH "Android NDK" FORCE)
include("${_linalg_ndk_toolchain}")

# The NDK does not provide Flang. Compile with the external Fortran compiler,
# but link executables with the NDK C++ driver and an Android Flang runtime.
# Leave the runtime optional for library-only builds, which do not need it.
set(LINALG_ANDROID_FLANG_RUNTIME_LIBRARIES "" CACHE STRING
    "Semicolon-separated paths to Android AArch64 Flang runtime libraries")
set(CMAKE_Fortran_STANDARD_LIBRARIES "")
foreach(_runtime IN LISTS LINALG_ANDROID_FLANG_RUNTIME_LIBRARIES)
  if(NOT IS_ABSOLUTE "${_runtime}" OR NOT EXISTS "${_runtime}" OR
     IS_DIRECTORY "${_runtime}")
    message(FATAL_ERROR
        "Android Flang runtime library must be an existing absolute file path: ${_runtime}")
  endif()
  string(APPEND CMAKE_Fortran_STANDARD_LIBRARIES " \"${_runtime}\"")
endforeach()
string(APPEND CMAKE_Fortran_STANDARD_LIBRARIES
    " ${CMAKE_CXX_STANDARD_LIBRARIES_INIT}")
# Do not forward <FLAGS>: those are Fortran compilation flags, not C++ flags.
# <LINK_FLAGS> retains the NDK executable linker flags and user link options.
set(CMAKE_Fortran_LINK_EXECUTABLE
    "\"${CMAKE_CXX_COMPILER}\" --target=${CMAKE_CXX_COMPILER_TARGET} --sysroot=\"${CMAKE_SYSROOT}\" <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

# Static-library compiler probes cannot discover implicit link libraries.
# Help FindOpenMP locate the NDK runtime without requiring it for serial builds.
set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "OpenMP C++ libraries")
if(NOT OpenMP_omp_LIBRARY)
  file(GLOB _linalg_android_libomp_candidates
      "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/*/lib/clang/*/lib/linux/aarch64/libomp.so")
  if(_linalg_android_libomp_candidates)
    list(GET _linalg_android_libomp_candidates 0 _linalg_android_libomp)
    set(OpenMP_omp_LIBRARY "${_linalg_android_libomp}" CACHE FILEPATH
        "Path to the Android AArch64 OpenMP runtime")
  endif()
endif()

# The NDK toolchain replaces this list, so append our wrapper inputs only
# after including it. CMake then forwards them to compiler try_compile projects.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
    ANDROID_NDK_ROOT
    CMAKE_Fortran_COMPILER
    LINALG_ANDROID_FLANG_RUNTIME_LIBRARIES
    OpenMP_CXX_LIB_NAMES
    OpenMP_omp_LIBRARY)

set(CMAKE_Fortran_COMPILER_TARGET ${CMAKE_CXX_COMPILER_TARGET}
    CACHE STRING "Flang Android target triple" FORCE)
# Compiler checks must not require linking an executable with a Fortran runtime.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

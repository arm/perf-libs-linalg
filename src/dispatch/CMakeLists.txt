# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

include(ProcessorCount)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/msvc_runtime.cmake")

set(_dispatch_config "${LINALG_TARGET_OS}")
set(_dispatch_target_os "${LINALG_TARGET_OS}")

set(_dispatch_config_json_path
    "${CMAKE_CURRENT_SOURCE_DIR}/json/configs/${_dispatch_config}.json")
if(NOT EXISTS "${_dispatch_config_json_path}")
  message(FATAL_ERROR
      "Unknown dispatch configuration '${_dispatch_config}'. "
      "Expected dispatch config file to exist: ${_dispatch_config_json_path}"
  )
endif()

# Keep generated dispatch sources and objects in variant-specific build dirs.
set(_dispatch_variant "${LINALG_INTEGER_MODEL}-${LINALG_THREADING_MODEL}")
if(CMAKE_BUILD_TYPE)
  string(TOLOWER "${CMAKE_BUILD_TYPE}" _dispatch_build_type)
  string(APPEND _dispatch_variant "-${_dispatch_build_type}")
endif()
dispatch_get_msvc_runtime_compile_options(
    _dispatch_msvc_runtime_options
    _dispatch_msvc_runtime_variant
)
if(_dispatch_msvc_runtime_variant)
  string(APPEND _dispatch_variant "-${_dispatch_msvc_runtime_variant}")
endif()
set(_dispatch_generate_dir "${CMAKE_CURRENT_BINARY_DIR}/${_dispatch_variant}/generated")
set(_dispatch_build_dir "${CMAKE_CURRENT_BINARY_DIR}/${_dispatch_variant}/obj")

# Translate CMake build options into dispatch Makefile switches.
set(_dispatch_use_openmp 0)
if(LINALG_THREADING_MODEL STREQUAL "openmp")
  set(_dispatch_use_openmp 1)
endif()

set(_dispatch_int64 0)
if(LINALG_INTEGER_MODEL STREQUAL "ilp64")
  set(_dispatch_int64 1)
endif()

set(_dispatch_debug 0)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(_dispatch_debug 1)
endif()

option(LINALG_DISPATCH_QUIET_BUILD
    "Reduce dispatch Makefile terminal output and capture per-object logs"
    ON)
set(_dispatch_make_dev 0)
if(LINALG_DISPATCH_QUIET_BUILD)
  set(_dispatch_make_dev 1)
endif()

# Match Makefile parallelism to the host when CMake can detect it.
ProcessorCount(HOST_JOBS)
if(NOT HOST_JOBS OR HOST_JOBS EQUAL 0)
  set(HOST_JOBS 1)
endif()

# Base include roots needed by generated dispatch sources.
set(_dispatch_include_dirs
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/core"
    "${CMAKE_SOURCE_DIR}/src/common"
    "${CMAKE_SOURCE_DIR}/src/runtime"
)

# Collect include dirs from kernels target.
get_target_property(_kernel_include_dirs linalg_kernels INTERFACE_INCLUDE_DIRECTORIES)
if(_kernel_include_dirs)
  list(APPEND _dispatch_include_dirs ${_kernel_include_dirs})
endif()

# The dispatch Makefile takes raw compiler flags, not CMake target usage.
set(_dispatch_make_cxx_flags ${_dispatch_include_dirs})
list(TRANSFORM _dispatch_make_cxx_flags PREPEND "-I")
if(_dispatch_msvc_runtime_options)
  list(PREPEND _dispatch_make_cxx_flags ${_dispatch_msvc_runtime_options})
endif()
if(APPLE AND CMAKE_OSX_SYSROOT)
  list(PREPEND _dispatch_make_cxx_flags
      "-isysroot" "${CMAKE_OSX_SYSROOT}")
endif()
if(APPLE AND CMAKE_OSX_DEPLOYMENT_TARGET)
  list(PREPEND _dispatch_make_cxx_flags
      "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()
if(CMAKE_CXX_FLAGS)
  list(PREPEND _dispatch_make_cxx_flags "${CMAKE_CXX_FLAGS}")
endif()
string(JOIN " " _dispatch_make_cxx_flags ${_dispatch_make_cxx_flags})

find_program(MAKE_EXECUTABLE NAMES gmake make REQUIRED)
find_program(FIND_EXECUTABLE NAMES gfind find REQUIRED)
find_program(SH_EXECUTABLE NAMES sh REQUIRED)

# Makefile variables passed as command-line overrides.
set(_dispatch_make_vars
    "CXX=${CMAKE_CXX_COMPILER}"
    "AR=${CMAKE_AR}"
    "RANLIB=${CMAKE_RANLIB}"
    "GENERATE_DIR=${_dispatch_generate_dir}"
    "BUILD_DIR=${_dispatch_build_dir}"
    "CONFIG_JSON_PATH=${_dispatch_config_json_path}"
    "TARGET_OS=${_dispatch_target_os}"
    "EXTRA_CXXFLAGS=${_dispatch_make_cxx_flags}"
    "FIND=${FIND_EXECUTABLE}"
    "PYTHON=${Python3_EXECUTABLE}"
    "use_openmp=${_dispatch_use_openmp}"
    "int64=${_dispatch_int64}"
    "debug=${_dispatch_debug}"
    "dev=${_dispatch_make_dev}"
)

set(_dispatch_make_args
    "--no-print-directory"
    -C "${CMAKE_CURRENT_SOURCE_DIR}"
    ${_dispatch_make_vars}
)

set(_dispatch_make_base_command
    "${MAKE_EXECUTABLE}"
    ${_dispatch_make_args}
)

set(_dispatch_build_make_executable "${MAKE_EXECUTABLE}")
set(_dispatch_make_fallback_jobs "${HOST_JOBS}")
# TODO: Use JOB_SERVER_AWARE once the minimum CMake version is 3.28.
if(CMAKE_GENERATOR MATCHES "^(Unix|MSYS|MinGW) Makefiles$")
  set(_dispatch_build_make_executable "$(MAKE)")
endif()

set(_dispatch_make_build_command
    "${SH_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_make.sh"
    "${_dispatch_make_fallback_jobs}"
    "${_dispatch_build_make_executable}"
    ${_dispatch_make_args}
    build-objects
)

# Ask the Makefile for the objects it will build so CMake can archive them.
execute_process(
    COMMAND ${_dispatch_make_base_command} print-objects
    OUTPUT_VARIABLE _dispatch_object_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
string(REGEX REPLACE "[\r\n]+" ";" _dispatch_objects "${_dispatch_object_output}")

# Target to invoke the dispatch sub-build.
add_custom_target(dispatch_build
    COMMAND ${_dispatch_make_build_command}
    BYPRODUCTS ${_dispatch_objects}
    COMMENT "Building dispatch objects"
    USES_TERMINAL
    VERBATIM
)

# Define imported dispatch objects and associated interface library.
# Add a dependence on dispatch_build so that we ask Make about
# dependency changes on every re-build.
add_library(linalg_dispatch_objects OBJECT IMPORTED GLOBAL)
set_target_properties(linalg_dispatch_objects PROPERTIES IMPORTED_OBJECTS "${_dispatch_objects}")
add_dependencies(linalg_dispatch_objects dispatch_build)

add_library(linalg_dispatch INTERFACE)
add_dependencies(linalg_dispatch dispatch_build)
target_sources(linalg_dispatch INTERFACE $<TARGET_OBJECTS:linalg_dispatch_objects>)
target_include_directories(linalg_dispatch INTERFACE ${_dispatch_include_dirs})

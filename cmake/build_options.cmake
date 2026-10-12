# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

set(LINALG_THREADING_MODEL "serial" CACHE STRING "Threading backend to build: serial or openmp")
set_property(CACHE LINALG_THREADING_MODEL PROPERTY STRINGS serial openmp)
if(NOT LINALG_THREADING_MODEL MATCHES "^(serial|openmp)$")
  message(FATAL_ERROR "LINALG_THREADING_MODEL must be serial or openmp")
endif()

set(LINALG_INTEGER_MODEL "lp64" CACHE STRING "Public integer ABI to build: lp64 or ilp64")
set_property(CACHE LINALG_INTEGER_MODEL PROPERTY STRINGS lp64 ilp64)
if(NOT LINALG_INTEGER_MODEL MATCHES "^(lp64|ilp64)$")
  message(FATAL_ERROR "LINALG_INTEGER_MODEL must be lp64 or ilp64")
endif()

set(LINALG_SANITIZER "none" CACHE STRING "Sanitizer instrumentation: none or address")
set_property(CACHE LINALG_SANITIZER PROPERTY STRINGS none address)
if(NOT LINALG_SANITIZER MATCHES "^(none|address)$")
  message(FATAL_ERROR "LINALG_SANITIZER must be none or address")
endif()

option(LINALG_ENABLE_TUNING_OVERRIDES "Enable runtime tuning parameter overrides" OFF)

include(compile_options)

add_library(build_options INTERFACE)
target_compile_features(build_options INTERFACE cxx_std_23)
target_compile_options(build_options INTERFACE
  "$<$<COMPILE_LANGUAGE:C>:${LINALG_C_COMPILE_OPTIONS}>"
  "$<$<COMPILE_LANGUAGE:CXX>:${LINALG_CXX_COMPILE_OPTIONS}>"
  "$<$<COMPILE_LANGUAGE:Fortran>:${LINALG_Fortran_COMPILE_OPTIONS}>"
)

if(LINALG_INTEGER_MODEL STREQUAL "ilp64")
  target_compile_definitions(build_options INTERFACE INTEGER64)
endif()

if(LINALG_THREADING_MODEL STREQUAL "openmp")
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(build_options INTERFACE OpenMP::OpenMP_CXX)
endif()

option(LINALG_BUILD_BENCHMARKS "Build benchmarking tools for library development" OFF)

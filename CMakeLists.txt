# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

cmake_minimum_required(VERSION 3.21)
project(perf-libs-linalg VERSION 26.07 LANGUAGES C CXX Fortran ASM)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(platform)
include(build_options)
include(CTest)
include(GNUInstallDirs)

add_subdirectory(netlib)
add_subdirectory(src/runtime)
add_subdirectory(src/core)
add_subdirectory(src/kernels)
add_subdirectory(src/dispatch)

add_library(pl_linalg)
set_target_properties(pl_linalg PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(pl_linalg PRIVATE
    linalg_core
    linalg_kernels
    linalg_dispatch
    linalg_netlib
)

install(TARGETS pl_linalg)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" TYPE INCLUDE)

if(BUILD_TESTING)
  add_subdirectory(netlib/tests)
endif()

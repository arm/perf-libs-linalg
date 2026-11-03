# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

cmake_minimum_required(VERSION 3.21)
project(perf-libs-linalg VERSION 26.07 LANGUAGES C CXX Fortran ASM)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(platform)
include(build_options)
include(CTest)
include(GNUInstallDirs)

add_library(pl_linalg)
add_subdirectory(netlib)
add_subdirectory(src)

add_subdirectory(tools)

install(TARGETS pl_linalg)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" TYPE INCLUDE)
install(FILES LICENSE THIRD_PARTY_NOTICES.md
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/${PROJECT_NAME}")
install(DIRECTORY LICENSES
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/${PROJECT_NAME}")

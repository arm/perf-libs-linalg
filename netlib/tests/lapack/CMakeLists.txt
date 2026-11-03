# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

set(LAPACK_SOURCE_DIR "${reference_lapack_SOURCE_DIR}")
set(LAPACK_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(LAPACK_LIBRARIES pl_linalg)
set(BLAS_LIBRARIES "")
set(TMGLIB lapack_test_matgen)
set(BUILD_SHARED_LIBS OFF)
foreach(precision SINGLE DOUBLE COMPLEX COMPLEX16)
  set(BUILD_${precision} ON)
endforeach()
list(APPEND CMAKE_MODULE_PATH "${LAPACK_SOURCE_DIR}/CMAKE")
include(CheckLAPACKCompilerFlags)
CheckLAPACKCompilerFlags()
link_libraries(netlib_fortran_options)

add_subdirectory("${LAPACK_SOURCE_DIR}/TESTING/MATGEN" MATGEN)
add_subdirectory("${LAPACK_SOURCE_DIR}/TESTING" TESTING)

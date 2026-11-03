# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

set(cblas_source_dir "${reference_lapack_SOURCE_DIR}/CBLAS")
set(cblas_include_dir "${CMAKE_CURRENT_BINARY_DIR}/cblas/include")
set(cblas_private_include_dir "${CMAKE_CURRENT_BINARY_DIR}/cblas")

configure_file("${cblas_source_dir}/include/cblas.h"
  "${cblas_include_dir}/cblas.h" COPYONLY)
configure_file("${cblas_source_dir}/include/cblas_mangling_with_flags.h.in"
  "${cblas_private_include_dir}/cblas_mangling.h" COPYONLY)

target_include_directories(pl_linalg PUBLIC
  "$<BUILD_INTERFACE:${cblas_include_dir}>")

file(GLOB cblas_sources
  "${cblas_source_dir}/src/*.c" "${cblas_source_dir}/src/*.f")
# Use the existing Fortran error handler.
list(REMOVE_ITEM cblas_sources "${cblas_source_dir}/src/xerbla.c")
add_library(netlib_cblas_obj OBJECT ${cblas_sources})
target_link_libraries(netlib_cblas_obj PRIVATE
  build_options netlib_fortran_options)
target_include_directories(netlib_cblas_obj PRIVATE
  "${cblas_include_dir}" "${cblas_private_include_dir}" "${cblas_source_dir}/include")
target_compile_definitions(netlib_cblas_obj PRIVATE "$<$<COMPILE_LANGUAGE:C>:F77_INT=CBLAS_INT>")

# Allow cblas_xerbla overrides.
set_property(SOURCE "${cblas_source_dir}/src/cblas_xerbla.c" APPEND
  PROPERTY COMPILE_DEFINITIONS HAS_ATTRIBUTE_WEAK_SUPPORT)

target_link_libraries(linalg_netlib INTERFACE netlib_cblas_obj)
target_sources(linalg_netlib INTERFACE $<TARGET_OBJECTS:netlib_cblas_obj>)

install(FILES "${cblas_include_dir}/cblas.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

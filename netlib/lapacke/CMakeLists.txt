# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

set(lapacke_source_dir "${reference_lapack_SOURCE_DIR}/LAPACKE")
set(lapacke_include_dir "${CMAKE_CURRENT_BINARY_DIR}/include")

file(READ "${lapacke_source_dir}/include/lapacke_config.h"
  LAPACKE_CONFIG_HEADER)
file(READ "${lapacke_source_dir}/include/lapacke_mangling_with_flags.h.in"
  LAPACKE_MANGLING_HEADER)
configure_file("${lapacke_source_dir}/include/lapack.h.in"
  "${lapacke_include_dir}/lapack.h" @ONLY)
configure_file("${lapacke_source_dir}/include/lapacke.h"
  "${lapacke_include_dir}/lapacke.h" COPYONLY)

include("${lapacke_source_dir}/src/CMakeLists.txt")
foreach(precision SINGLE DOUBLE COMPLEX COMPLEX16)
  list(APPEND SOURCES ${SOURCES_${precision}})
endforeach()
list(TRANSFORM SOURCES PREPEND "${lapacke_source_dir}/src/")
include("${lapacke_source_dir}/utils/CMakeLists.txt")
list(TRANSFORM UTILS PREPEND "${lapacke_source_dir}/utils/")

add_library(netlib_lapacke_obj OBJECT ${SOURCES} ${UTILS})
target_link_libraries(netlib_lapacke_obj PRIVATE build_options)
target_include_directories(netlib_lapacke_obj PRIVATE
  "${lapacke_include_dir}" "${lapacke_source_dir}/include")

target_link_libraries(linalg_netlib INTERFACE netlib_lapacke_obj)
target_sources(linalg_netlib INTERFACE $<TARGET_OBJECTS:netlib_lapacke_obj>)
target_include_directories(pl_linalg PUBLIC
  "$<BUILD_INTERFACE:${lapacke_include_dir}>")

install(FILES "${lapacke_include_dir}/lapack.h" "${lapacke_include_dir}/lapacke.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

# Installation is handled by the enclosing pl_linalg target.
function(lapack_install_library library)
endfunction()

list(APPEND CMAKE_MODULE_PATH "${reference_lapack_SOURCE_DIR}/CMAKE")
include(CheckLAPACKCompilerFlags)
CheckLAPACKCompilerFlags()
set(LAPACKLIB netlib_lapack)
foreach(precision SINGLE DOUBLE COMPLEX COMPLEX16)
  set(BUILD_${precision} ON)
endforeach()
set(SECOND_SRC "${reference_lapack_SOURCE_DIR}/INSTALL/second_INT_CPU_TIME.f")
set(DSECOND_SRC "${reference_lapack_SOURCE_DIR}/INSTALL/dsecnd_INT_CPU_TIME.f")
add_subdirectory("${reference_lapack_SOURCE_DIR}/SRC"
  SRC EXCLUDE_FROM_ALL)

# These routines are already supplied by native code or existing Netlib sources.
get_target_property(sources netlib_lapack_obj SOURCES)
list(REMOVE_ITEM sources ../INSTALL/lsame.f xerbla.f)
foreach(precision s d c z)
  foreach(routine gebrd geqrf getrf labrd larf larfb larfg larft latrd potrf)
    list(REMOVE_ITEM sources ${precision}${routine}.f)
  endforeach()
endforeach()
foreach(precision s d)
  foreach(routine orglq orgqr ormlq ormqr sytrd)
    list(REMOVE_ITEM sources ${precision}${routine}.f)
  endforeach()
endforeach()
foreach(precision c z)
  foreach(routine hetrd rot spmv spr symv syr unglq ungqr unmlq unmqr)
    list(REMOVE_ITEM sources ${precision}${routine}.f)
  endforeach()
endforeach()
set_property(TARGET netlib_lapack_obj PROPERTY SOURCES "${sources}")

foreach(target mod_files netlib_lapack_obj)
  target_link_libraries(${target} build_options netlib_fortran_options)
  target_link_libraries(linalg_netlib INTERFACE ${target})
  target_sources(linalg_netlib INTERFACE $<TARGET_OBJECTS:${target}>)
endforeach()

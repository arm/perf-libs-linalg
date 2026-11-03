# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

set(CBLAS_TEST_SOURCE_DIR "${reference_lapack_SOURCE_DIR}/CBLAS/testing")

foreach(precision s d c z)
  foreach(level 1 2 3)
    set(name x${precision}cblat${level})
    set(sources c_${precision}blat${level}.f c_${precision}blas${level}.c)
    set(input_argument)
    if(level GREATER 1)
      list(APPEND sources c_${precision}${level}chke.c auxiliary.c)
      set(input_argument "-DINPUT=${CBLAS_TEST_SOURCE_DIR}/${precision}in${level}")
    endif()
    list(TRANSFORM sources PREPEND "${CBLAS_TEST_SOURCE_DIR}/")
    add_executable(${name} ${sources})
    set_target_properties(${name} PROPERTIES LINKER_LANGUAGE Fortran)
    target_include_directories(${name} PRIVATE
      "${CBLAS_TEST_SOURCE_DIR}/../include" "${cblas_private_include_dir}")
    target_link_libraries(${name} PRIVATE netlib_fortran_options pl_linalg)

    add_test(NAME netlib-cblas-${name} COMMAND "${CMAKE_COMMAND}"
      "-DTEST=$<TARGET_FILE:${name}>" ${input_argument}
      -P "${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake")
  endforeach()
endforeach()

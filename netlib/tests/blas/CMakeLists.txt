set(BLAS_TEST_SOURCE_DIR "${reference_lapack_SOURCE_DIR}/BLAS/TESTING")

function(add_blas_test name src)
  get_filename_component(basename "${src}" NAME_WE)
  set(test_input "${BLAS_TEST_SOURCE_DIR}/${basename}.in")
  set(input_argument)
  if(EXISTS "${test_input}")
    set(input_argument "-DINPUT=${test_input}")
  endif()

  add_executable(${name} "${BLAS_TEST_SOURCE_DIR}/${src}")
  set_target_properties(${name} PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(${name} PRIVATE netlib_fortran_options pl_linalg)
  add_test(NAME netlib-blas-${name} COMMAND "${CMAKE_COMMAND}"
    "-DTEST=$<TARGET_FILE:${name}>" ${input_argument}
    -P "${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake")
endfunction()

foreach(precision s d c z)
  foreach(level 1 2 3)
    add_blas_test(xblat${level}${precision} ${precision}blat${level}.f)
  endforeach()
endforeach()

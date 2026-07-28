# BLAS/Netlib test wiring.

set(BLAS_TEST_LIBRARY_TARGET pl_linalg)

# Precision selection for BLAS tests
option(BUILD_SINGLE "Build single precision real tests" ON)
option(BUILD_DOUBLE "Build double precision real tests" ON)
option(BUILD_COMPLEX "Build single precision complex tests" ON)
option(BUILD_COMPLEX16 "Build double precision complex tests" ON)

set(BLAS_TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/blas")
# Test runner script that executes BLAS tests and handles optional input files.
set(BLAS_TEST_RUNNER "${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake")

# Add a BLAS test executable and register it with CTest.
function(add_blas_test name src)
  set(test_src "${BLAS_TEST_SOURCE_DIR}/${src}")
  get_filename_component(basename "${src}" NAME_WE)
  set(test_input "${BLAS_TEST_SOURCE_DIR}/${basename}.in")

  if(NOT EXISTS "${test_src}")
    message(FATAL_ERROR "Missing BLAS test source file: ${test_src}")
  endif()

  # Create an executable for the BLAS test and link it against the library under test.
  add_executable(${name} "${test_src}")
  set_target_properties(${name} PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(${name} PRIVATE
    netlib_fortran_options
    ${BLAS_TEST_LIBRARY_TARGET}
  )

  # Register the test with CTest.
  # If an input file exists for the test, pass it as a variable to the test runner;
  # otherwise, just run the test without it.
  if(EXISTS "${test_input}")
    add_test(NAME netlib-blas-${name} COMMAND "${CMAKE_COMMAND}"
      -DTEST=$<TARGET_FILE:${name}>
      -DINPUT=${test_input}
      -P "${BLAS_TEST_RUNNER}")
  else()
    add_test(NAME netlib-blas-${name} COMMAND "${CMAKE_COMMAND}"
      -DTEST=$<TARGET_FILE:${name}>
      -P "${BLAS_TEST_RUNNER}")
  endif()
endfunction()

# Add tests for each precision as requested by the options.
if(BUILD_SINGLE)
  add_blas_test(xblat1s sblat1.f)
  add_blas_test(xblat2s sblat2.f)
  add_blas_test(xblat3s sblat3.f)
endif()

if(BUILD_DOUBLE)
  add_blas_test(xblat1d dblat1.f)
  add_blas_test(xblat2d dblat2.f)
  add_blas_test(xblat3d dblat3.f)
endif()

if(BUILD_COMPLEX)
  add_blas_test(xblat1c cblat1.f)
  add_blas_test(xblat2c cblat2.f)
  add_blas_test(xblat3c cblat3.f)
endif()

if(BUILD_COMPLEX16)
  add_blas_test(xblat1z zblat1.f)
  add_blas_test(xblat2z zblat2.f)
  add_blas_test(xblat3z zblat3.f)
endif()

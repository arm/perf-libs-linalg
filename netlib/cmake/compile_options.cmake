include(CheckCompilerFlag)
include(CheckFortranSourceCompiles)

# Select the first supported Fortran option.
function(linalg_select_fortran_flag output)
  foreach(flag IN LISTS ARGN)
    string(MAKE_C_IDENTIFIER "linalg_fortran_${flag}" supported)
    check_compiler_flag(Fortran "${flag}" ${supported})

    if(${supported})
      set(${output} "${flag}" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  message(FATAL_ERROR "No supported Fortran option for ${output}: ${ARGN}")
endfunction()

# Check that the selected flag makes default INTEGER 64 bits.
function(linalg_require_fortran_integer64 flag)
  set(CMAKE_REQUIRED_FLAGS "${flag}")

  check_fortran_source_compiles([[
    program check_default_integer
      integer, parameter :: expected_bits(64) = 0
      integer, parameter :: actual_bits(storage_size(0)) = expected_bits
      print *, actual_bits
    end program
  ]] linalg_fortran_integer_is_64 SRC_EXT F90)

  if(NOT linalg_fortran_integer_is_64)
    message(FATAL_ERROR "${flag} does not give 64-bit default INTEGER")
  endif()
endfunction()

linalg_select_fortran_flag(NETLIB_FORTRAN_COMPILE_OPTIONS
  -ffixed-line-length-132 -Mextend)

check_compiler_flag(Fortran "-Kieee" linalg_fortran_ieee)
if(linalg_fortran_ieee)
  list(APPEND NETLIB_FORTRAN_COMPILE_OPTIONS -Kieee)
endif()

if(LINALG_INTEGER_MODEL STREQUAL "ilp64")
  linalg_select_fortran_flag(linalg_fortran_integer_8
    -fdefault-integer-8 -i8)

  linalg_require_fortran_integer64("${linalg_fortran_integer_8}")
  list(APPEND NETLIB_FORTRAN_COMPILE_OPTIONS ${linalg_fortran_integer_8})
endif()

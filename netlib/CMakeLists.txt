set(NETLIB_FORTRAN_COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:Fortran>:-ffixed-line-length-132>
)

if(LINALG_INTEGER_MODEL STREQUAL "ilp64")
  include(CheckFortranCompilerFlag)
  check_fortran_compiler_flag("-fdefault-integer-8" _has_default_integer_8)
  if(_has_default_integer_8)
    list(APPEND NETLIB_FORTRAN_COMPILE_OPTIONS
        $<$<COMPILE_LANGUAGE:Fortran>:-fdefault-integer-8>
    )
  else()
    message(FATAL_ERROR
        "ILP64 netlib references need an 8-byte default INTEGER flag, "
        "but '-fdefault-integer-8' was not accepted by ${CMAKE_Fortran_COMPILER}."
    )
  endif()
endif()

add_library(netlib_fortran_options INTERFACE)
target_compile_options(netlib_fortran_options INTERFACE
    ${NETLIB_FORTRAN_COMPILE_OPTIONS}
)

file(GLOB_RECURSE REFERENCE_BLAS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/reference/blas/*.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/reference/blas/*.f90"
)

file(GLOB_RECURSE NETLIB_LAPACK_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/upstream/lapack/*.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/upstream/lapack/*.f90"
)

file(GLOB_RECURSE EXTENSION_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/extensions/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/extensions/*.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/extensions/*.f90"
)

file(GLOB_RECURSE NETLIB_UTILS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/upstream/utils/*.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/upstream/utils/*.f90"
)

add_library(reference_blas_obj OBJECT ${REFERENCE_BLAS_SOURCES} ${EXTENSION_SOURCES})
target_compile_options(reference_blas_obj PRIVATE
    $<$<COMPILE_LANGUAGE:Fortran>:-O2>
)
target_link_libraries(reference_blas_obj PRIVATE netlib_fortran_options)
target_link_libraries(reference_blas_obj PUBLIC build_options)
target_include_directories(reference_blas_obj PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
)

add_library(netlib_upstream_obj OBJECT ${NETLIB_LAPACK_SOURCES} ${NETLIB_UTILS_SOURCES})
target_link_libraries(netlib_upstream_obj PRIVATE netlib_fortran_options)
target_link_libraries(netlib_upstream_obj PUBLIC build_options)

add_library(linalg_netlib INTERFACE)
target_link_libraries(linalg_netlib INTERFACE
    reference_blas_obj
    netlib_upstream_obj
)
target_sources(linalg_netlib INTERFACE
    $<TARGET_OBJECTS:reference_blas_obj>
    $<TARGET_OBJECTS:netlib_upstream_obj>
)

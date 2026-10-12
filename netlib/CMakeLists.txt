include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/compile_options.cmake")

add_library(netlib_fortran_options INTERFACE)
target_compile_options(netlib_fortran_options INTERFACE
  "$<$<COMPILE_LANGUAGE:Fortran>:${NETLIB_FORTRAN_COMPILE_OPTIONS}>"
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

add_library(reference_blas_obj OBJECT ${REFERENCE_BLAS_SOURCES} ${EXTENSION_SOURCES})
target_compile_options(reference_blas_obj PRIVATE
    $<$<COMPILE_LANGUAGE:Fortran>:-O2>
)
target_link_libraries(reference_blas_obj PRIVATE netlib_fortran_options)
target_link_libraries(reference_blas_obj PUBLIC build_options)
target_include_directories(reference_blas_obj PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
)

add_library(netlib_upstream_obj OBJECT ${NETLIB_LAPACK_SOURCES})
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

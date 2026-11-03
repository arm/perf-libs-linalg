include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/compile_options.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/dependencies/lapack.cmake")

add_library(netlib_fortran_options INTERFACE)
target_compile_options(netlib_fortran_options INTERFACE
  "$<$<COMPILE_LANGUAGE:Fortran>:${NETLIB_FORTRAN_COMPILE_OPTIONS}>"
)

set(REFERENCE_BLAS_SOURCES)
foreach(routine
    sgemm dgemm cgemm zgemm
    ssymm dsymm csymm zsymm
    chemm zhemm
    ssyrk dsyrk csyrk zsyrk
    cherk zherk
    ssyr2k dsyr2k csyr2k zsyr2k
    cher2k zher2k
    strmm dtrmm ctrmm ztrmm
    strsm dtrsm ctrsm ztrsm
    strsv dtrsv ctrsv ztrsv)
  list(APPEND REFERENCE_BLAS_SOURCES
    "${reference_lapack_SOURCE_DIR}/BLAS/SRC/${routine}.f")
endforeach()

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

# Native factorization routines require these LARFT implementations.
# TODO: Evaluate using the default recursive LARFT implementation.
set(larft_source_dir "${reference_lapack_SOURCE_DIR}/SRC/VARIANTS/larft/LL-LVL2")
add_library(netlib_upstream_obj OBJECT
    "${larft_source_dir}/slarft.f"
    "${larft_source_dir}/dlarft.f"
    "${larft_source_dir}/clarft.f"
    "${larft_source_dir}/zlarft.f"
)
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

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/cblas.cmake")

if(LINALG_LAPACK_FALLBACK)
  add_subdirectory(lapack)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

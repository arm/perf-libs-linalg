add_subdirectory(blas)
add_subdirectory(cblas)

if(LINALG_LAPACK_FALLBACK)
  add_subdirectory(lapack)
endif()

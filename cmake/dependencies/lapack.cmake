# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

include_guard(GLOBAL)
include(FetchContent)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(LINALG_LAPACK_URL
  "https://github.com/Reference-LAPACK/lapack/archive/6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca.tar.gz"
  CACHE STRING "Reference LAPACK archive URL or local archive path")
set(_lapack_patches
  "${PROJECT_SOURCE_DIR}/netlib/patches/cblas-headers.patch"
  "${PROJECT_SOURCE_DIR}/netlib/patches/cblas-empty-vectors.patch"
  "${PROJECT_SOURCE_DIR}/netlib/patches/cblas-test-exclusions.patch"
  "${PROJECT_SOURCE_DIR}/netlib/patches/blas-nrm2-infinity.patch")
set(_lapack_patch_hash "")
foreach(_patch IN LISTS _lapack_patches)
  file(SHA256 "${_patch}" _hash)
  string(SHA256 _lapack_patch_hash "${_lapack_patch_hash}${_hash}")
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_lapack_patches})
if(NOT FETCHCONTENT_SOURCE_DIR_REFERENCE_LAPACK)
  find_package(Git REQUIRED)
endif()

FetchContent_Declare(reference_lapack
  URL "${LINALG_LAPACK_URL}"
  URL_HASH SHA256=6ce7f85a7e6ce57dfe21dac5090b2ede959394e4a52caa2dea46f8226b642030
  TLS_VERIFY TRUE
  # Extract fresh sources when the patches change.
  SOURCE_DIR "${FETCHCONTENT_BASE_DIR}/reference_lapack-${_lapack_patch_hash}"
  # Fetch the sources without configuring the upstream project.
  SOURCE_SUBDIR source-only
  PATCH_COMMAND "${GIT_EXECUTABLE}" --work-tree=. apply ${_lapack_patches}
)
FetchContent_MakeAvailable(reference_lapack)

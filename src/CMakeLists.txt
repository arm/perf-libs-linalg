# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

add_subdirectory(runtime)
add_subdirectory(core)
add_subdirectory(kernels)
add_subdirectory(dispatch)

set_target_properties(pl_linalg PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(pl_linalg PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)
target_compile_definitions(pl_linalg PUBLIC ${LINALG_ABI_COMPILE_DEFINITIONS})
target_link_options(pl_linalg PUBLIC ${LINALG_LINK_OPTIONS})
target_link_libraries(pl_linalg
  PRIVATE
    linalg_core
    linalg_runtime
    linalg_kernels
    linalg_dispatch
    linalg_netlib
)

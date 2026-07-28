# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

# Build all the objects
add_library(small_gemm_kernels_obj OBJECT)
kernel_target_sources(small_gemm_kernels_obj GLOB "*.cpp")
target_include_directories(small_gemm_kernels_obj
    PRIVATE ${INCLUDE_DIR}
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

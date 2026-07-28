# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

compile_mako(ASM_SOURCES GLOB "*.S.mako")

add_library(framework_kernels_obj OBJECT)
kernel_target_sources(framework_kernels_obj FILES xerbla.cpp)
kernel_target_sources(framework_kernels_obj FILES ${ASM_SOURCES} ISA FP16)
kernel_target_sources(framework_kernels_obj FILES vector_length_bytes_neon.cpp ISA NEON)
kernel_target_sources(framework_kernels_obj FILES vector_length_bytes_sve.cpp ISA SVE)
kernel_target_sources(framework_kernels_obj FILES vector_length_bytes_sme.cpp ISA SME2)

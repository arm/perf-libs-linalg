# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

compile_mako(ASM_SOURCES GLOB "*.S.mako")

add_library(misc_kernels_obj OBJECT)
kernel_target_sources(misc_kernels_obj FILES ${ASM_SOURCES})
kernel_target_sources(misc_kernels_obj FILES l2_norm_neon_kernel.cpp ISA NEON)
kernel_target_sources(misc_kernels_obj FILES l2_norm_sve_kernel.cpp ISA SVE)

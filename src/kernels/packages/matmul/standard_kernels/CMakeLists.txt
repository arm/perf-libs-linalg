# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

compile_mako(NEON_ASM_SOURCES GLOB "neon/*.S.mako")
compile_mako(BF16_ASM_SOURCES GLOB "bf16/*.S.mako")
compile_mako(FP16_ASM_SOURCES GLOB "fp16/*.S.mako")
compile_mako(SVE_ASM_SOURCES GLOB "sve/*.S.mako")

add_library(standard_kernels_obj OBJECT)
kernel_target_sources(standard_kernels_obj FILES ${NEON_ASM_SOURCES} GLOB "neon/*.cpp" ISA NEON)
kernel_target_sources(standard_kernels_obj FILES ${BF16_ASM_SOURCES} GLOB "bf16/*.cpp" ISA BF16)
kernel_target_sources(standard_kernels_obj FILES ${SVE_ASM_SOURCES} ISA SVE)
kernel_target_sources(standard_kernels_obj GLOB "sme2/*.cpp" ISA SME2)
kernel_target_sources(standard_kernels_obj FILES ${FP16_ASM_SOURCES} ISA FP16)

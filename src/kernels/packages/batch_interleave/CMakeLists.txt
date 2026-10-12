# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

add_library(batch_interleave_kernels_obj OBJECT)
kernel_target_sources(batch_interleave_kernels_obj GLOB "neon/*.cpp" ISA NEON)
kernel_target_sources(batch_interleave_kernels_obj GLOB "sve/*.cpp" ISA SVE)

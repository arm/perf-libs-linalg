/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SPEC_ROUTINE_SPECS_HPP
#define PERFLIBS_LINALG_SPEC_ROUTINE_SPECS_HPP

#include "framework/convert.hpp"

/*
 * We include all of the below header files because
 * these will define the function pointer types for
 * each of the types of kernel ie axpby_kernel_t<T>
 */
//LEVEL 1
#include "framework/axpby_kernels.hpp"
#include "framework/waxpby_kernels.hpp"
#include "framework/copy_kernels.hpp"
#include "framework/dot_kernels.hpp"
#include "framework/find_index_kernels.hpp"
#include "framework/l1_norm_kernels.hpp"
#include "framework/l2_norm_kernels.hpp"
#include "framework/rot_kernels.hpp"
#include "framework/swap_kernels.hpp"

//LEVEL 2
#include "framework/trsv_kernels.hpp"

//LEVEL 3
#include "kernel_specs/matmul3_kernels.hpp"
#include "framework/gemv_kernels.hpp"
#include "framework/trsm_kernels.hpp"


//TODO: delete this file

#endif //PERFLIBS_LINALG_SPEC_ROUTINE_SPECS_HPP

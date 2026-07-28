/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef GEMM_SMALL_VANILLA_H
#define GEMM_SMALL_VANILLA_H

#include "perflibs_complex.hpp"
#include "perflibs_util.hpp"
#include "perflibs_blas_types.hpp"


namespace perflibs::gemm {

void cgemm_small_vanilla(
	const kernel_inttype max_threads,
	const perflibs_trans transa, const perflibs_trans transb,
	const kernel_inttype m, const kernel_inttype n, const kernel_inttype k,
	const complex_float alpha, const complex_float *const a, const kernel_inttype lda,
	const complex_float *const b, const kernel_inttype ldb, const complex_float beta,
	complex_float *const c, const kernel_inttype ldc);
}

#endif // ifndef GEMM_SMALL_VANILLA_H

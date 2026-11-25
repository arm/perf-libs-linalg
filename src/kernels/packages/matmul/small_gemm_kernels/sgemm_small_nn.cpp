/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "perflibs_assert.hpp"
#include "gemm_small_generic_aarch64.hpp"

extern "C" void sgemm_small_kernel_nn_12x6(kernel_inttype m, kernel_inttype n, kernel_inttype k, float alpha,
                                        const float* a, kernel_inttype lda, const float* b, kernel_inttype ldb,
                                        float beta, float* c, kernel_inttype ldc);

void perflibs::gemm::sgemm_unpacked_nn_12x6(
	linalg::perflibs_trans transa, linalg::perflibs_trans transb, kernel_inttype m, kernel_inttype n, kernel_inttype k,
	float alpha, const float* a, kernel_inttype lda,
	const float* b, kernel_inttype ldb, float beta, float* c, kernel_inttype ldc) {
	PERFLIBS_ASSERT(transa == linalg::PERFLIBS_NOTRANS && transb == linalg::PERFLIBS_NOTRANS);

	sgemm_small_kernel_nn_12x6(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

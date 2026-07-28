/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "cgemm_unrolled_common.hpp"

namespace perflibs { namespace gemm {

void cgemm_unrolled_NN(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const std::complex<float> alpha,
                       const std::complex<float> * restrict A, const size_t lda,
                       const std::complex<float> * restrict B, const size_t ldb, const std::complex<float> beta,
                       std::complex<float> * restrict C, const size_t ldc) {
	if (m % 8 == 0) {
		if (n % 2 == 0 && k % 2 == 0) {
			cgemm_unrolled_kernel<'N', 'N', 2, 8, 2>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		} else if (n % 2 == 0) {
			cgemm_unrolled_kernel<'N', 'N', 2, 8, 1>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		} else if (k % 2 == 0) {
			cgemm_unrolled_kernel<'N', 'N', 1, 8, 2>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		} else {
			cgemm_unrolled_kernel<'N', 'N', 1, 8, 1>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		}
	} else {
		run_unrolled_impl<'N', 'N'>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}
}

}}

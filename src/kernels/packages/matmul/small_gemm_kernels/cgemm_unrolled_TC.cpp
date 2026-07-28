/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "cgemm_unrolled_common.hpp"

namespace perflibs { namespace gemm {

void cgemm_unrolled_TC(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const std::complex<float> alpha,
                       const std::complex<float> * restrict A, const size_t lda,
                       const std::complex<float> * restrict B, const size_t ldb, const std::complex<float> beta,
					   std::complex<float> * restrict C, const size_t ldc) {
	run_unrolled_impl<'T', 'C'>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

}}

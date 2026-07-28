/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_CGEMM_UNROLLED_H
#define PERFLIBS_CGEMM_UNROLLED_H

#include "perflibs_util.hpp"

#include <complex>

namespace perflibs { namespace gemm {

void cgemm_unrolled_NN(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const std::complex<float> alpha,
							const std::complex<float> * restrict A, const size_t lda,
							const std::complex<float> * restrict B, const size_t ldb, const std::complex<float> beta,
							std::complex<float> * restrict C, const size_t ldc);
void cgemm_unrolled_TN(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const std::complex<float> alpha,
							const std::complex<float> * restrict A, const size_t lda,
							const std::complex<float> * restrict B, const size_t ldb, const std::complex<float> beta,
							std::complex<float> * restrict C, const size_t ldc);
void cgemm_unrolled_CN(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const std::complex<float> alpha,
							const std::complex<float> * restrict A, const size_t lda,
							const std::complex<float> * restrict B, const size_t ldb, const std::complex<float> beta,
							std::complex<float> * restrict C, const size_t ldc);
void cgemm_unrolled_NT(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const std::complex<float> alpha,
							const std::complex<float> * restrict A, const size_t lda,
							const std::complex<float> * restrict B, const size_t ldb, const std::complex<float> beta,
							std::complex<float> * restrict C, const size_t ldc);
void cgemm_unrolled_TT(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const std::complex<float> alpha,
							const std::complex<float> * restrict A, const size_t lda,
							const std::complex<float> * restrict B, const size_t ldb, const std::complex<float> beta,
							std::complex<float> * restrict C, const size_t ldc);
void cgemm_unrolled_CT(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const std::complex<float> alpha,
							const std::complex<float> * restrict A, const size_t lda,
							const std::complex<float> * restrict B, const size_t ldb, const std::complex<float> beta,
							std::complex<float> * restrict C, const size_t ldc);
void cgemm_unrolled_NC(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const std::complex<float> alpha,
							const std::complex<float> * restrict A, const size_t lda,
							const std::complex<float> * restrict B, const size_t ldb, const std::complex<float> beta,
							std::complex<float> * restrict C, const size_t ldc);
void cgemm_unrolled_TC(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const std::complex<float> alpha,
							const std::complex<float> * restrict A, const size_t lda,
							const std::complex<float> * restrict B, const size_t ldb, const std::complex<float> beta,
							std::complex<float> * restrict C, const size_t ldc);
void cgemm_unrolled_CC(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const std::complex<float> alpha,
							const std::complex<float> * restrict A, const size_t lda,
							const std::complex<float> * restrict B, const size_t ldb, const std::complex<float> beta,
							std::complex<float> * restrict C, const size_t ldc);

}}

#endif

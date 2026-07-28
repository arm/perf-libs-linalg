/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "perflibs_util.hpp"

namespace perflibs { namespace gemm {

void sgemm_unrolled_NN(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const float alpha,
							const float * restrict A, const kernel_inttype lda,
							const float * restrict B, const kernel_inttype ldb, const float beta,
							float * restrict C, const kernel_inttype ldc);

void sgemm_unrolled_TN(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const float alpha,
							const float * restrict A, const kernel_inttype lda,
							const float * restrict B, const kernel_inttype ldb, const float beta,
							float * restrict C, const kernel_inttype ldc);

void sgemm_unrolled_NT(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const float alpha,
							const float * restrict A, const kernel_inttype lda,
							const float * restrict B, const kernel_inttype ldb, const float beta,
							float * restrict C, const kernel_inttype ldc);

void sgemm_unrolled_TT(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const float alpha,
							const float * restrict A, const kernel_inttype lda,
							const float * restrict B, const kernel_inttype ldb, const float beta,
							float * restrict C, const kernel_inttype ldc);

}}

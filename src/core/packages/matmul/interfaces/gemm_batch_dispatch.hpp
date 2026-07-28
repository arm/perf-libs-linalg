/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_GEMM_BATCH_DISPATCH_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_GEMM_BATCH_DISPATCH_HPP

#include "packages/matmul/interfaces/gemm.hpp"

namespace perflibs::gemm {

template<typename T, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void gemm_batch_dispatch_arch(
	const char *transa, const char *transb,
	kernel_inttype m, kernel_inttype n, kernel_inttype k,
	T alpha,
	const T *A, kernel_inttype lda,
	const T *B, kernel_inttype ldb,
	T beta,
	      T *C, kernel_inttype ldc) {

	perflibs::linalg::gemm<false, kernel_inttype, T, T, T, ArchitectureSpec>(
		transa,
		transb,
		&m, &n, &k,
		&alpha,
		A, &lda,
		B, &ldb,
		&beta,
		C, &ldc);
}

} // namespace perflibs::gemm

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_GEMM_BATCH_DISPATCH_HPP

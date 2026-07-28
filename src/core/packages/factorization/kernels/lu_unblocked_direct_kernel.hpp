/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_LU_UNBLOCKED_DIRECT_KERNEL_HPP
#define PERFLIBS_LINALG_LU_UNBLOCKED_DIRECT_KERNEL_HPP

#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

/**
 *  A basic implementation of LU factorization
 *  partial pivoting with row interchange
 *
 */
template<typename ArchitectureSpec, typename T, typename IntType>
void lu_unblocked_direct_kernel(const kernel_inttype m, const kernel_inttype n, T *A,
                                const kernel_inttype lda, IntType *ipiv, IntType& info) {

	PERFLIBS_ASSERT(m >= 0, " The number of rows in the matrix A must be greater or equal to zero");
	PERFLIBS_ASSERT(n >= 0, " The number of columns in the matrix A must be greater or equal to zero");
	PERFLIBS_ASSERT(lda >= max(1, m),
	             " The leading dimension the matrix A must be greater than the number of rows");

	// quick return like LAPACK
	if (m == 0 || n == 0) {
		return;
	}

	// initialize default pivot values
	const auto min_mn = min(m, n);
	for (kernel_inttype i = 0; i < min_mn; i++) {
		ipiv[i] = i;
	}

	// factorise A using the classical pivot LU
	for (kernel_inttype k = 0; k < min_mn; k++) {
		kernel_inttype r = k + 1;

		// find the largest absolute element in column k from current diagonal
		auto max_idx = k;
		auto max_elem = sum_abs(A[k * lda + k]);

		for (kernel_inttype i = r; i < m; i++) {
			auto curr_elem = sum_abs(A[k * lda + i]);
			if (curr_elem > max_elem) {
				max_elem = curr_elem;
				max_idx = i;
			}
		}

		// update the pivots
		// note we add 1 to return Fortran indices
		ipiv[k] = max_idx;

		// is the pivot 0?
		if (A[k * lda + max_idx] != T(0)) {
			// swap rows k and max_idx
			for (kernel_inttype j = 0; j < n; j++) {
				std::swap(A[j * lda + k], A[j * lda + max_idx]);
			}

			// update kth column below the diagonal
			//   A(k+1,k) = A(k+1,k)/A(k,k)
			T recipAkk = T(1) / A[k * lda + k];
			for (kernel_inttype i = r; i < m; i++) {
				A[k * lda + i] = A[k * lda + i] * recipAkk;
			}
		}
		else {
			if (info == 0) {
				// if this is the first 0 pivot value we have encountered
				// set info=k+1  (we are returning a 1-based (Fortran) index)
				info = k + 1;
			}
		}

		if (k < n - 1) {
			// rank-1 update (GER) of remainder of the matrix
			//    A(k+1,k+1) = A(k+1,k+1) - A(k+1,k) A(k,k+1)
			auto y_idx = r * lda + k;
			for (kernel_inttype j = r; j < n; j++) {
				auto Akr = A[y_idx];
				for (kernel_inttype i = r; i < m; i++) {
					A[j * lda + i] = A[j * lda + i] - A[k * lda + i] * Akr;
				}
				y_idx += lda;
			}
		}
	}
}

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_LU_UNBLOCKED_DIRECT_KERNEL_HPP

/*
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_LU_UNBLOCKED_NO_PIVOT_KERNEL_HPP
#define PERFLIBS_LINALG_LU_UNBLOCKED_NO_PIVOT_KERNEL_HPP

#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

/**
 *  A basic implementation of LU factorization without pivoting
 *  (right-looking; stops at the first exactly zero pivot)
 */
template<typename ArchitectureSpec, typename T, typename IntType>
void lu_unblocked_no_pivot_kernel(const kernel_inttype m, const kernel_inttype n, T *A,
                                  const kernel_inttype lda, IntType& info) {

	PERFLIBS_ASSERT(m >= 0, " The number of rows in the matrix A must be greater or equal to zero");
	PERFLIBS_ASSERT(n >= 0, " The number of columns in the matrix A must be greater or equal to zero");
	PERFLIBS_ASSERT(lda >= max(1, m),
	             " The leading dimension the matrix A must be greater than the number of rows");

	// quick return like LAPACK
	if (m == 0 || n == 0) {
		return;
	}

	const auto min_mn = min(m, n);
	for (kernel_inttype k = 0; k < min_mn; k++) {
		kernel_inttype r = k + 1;

		// stop at the first zero pivot, returning a 1-based (Fortran) index
		if (A[k * lda + k] == T(0)) {
			if (info == 0) {
				info = k + 1;
			}
			return;
		}

		// update kth column below the diagonal
		//   A(k+1,k) = A(k+1,k)/A(k,k)
		T recipAkk = T(1) / A[k * lda + k];
		for (kernel_inttype i = r; i < m; i++) {
			A[k * lda + i] = A[k * lda + i] * recipAkk;
		}

		// rank-1 update (GER) of the columns to the right
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

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_LU_UNBLOCKED_NO_PIVOT_KERNEL_HPP

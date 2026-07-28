/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_CHOLESKY_NON_RECURSIVE_KERNEL_HPP
#define PERFLIBS_LINALG_CHOLESKY_NON_RECURSIVE_KERNEL_HPP

#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

/**
 *  A basic implementation of cholesky factorization
 * This is non recursive version without call to BLAS
 */
template<typename ArchitectureSpec, typename T, typename IntType>
IntType cholesky_non_recursive_kernel(const perflibs_uplo uplo, const kernel_inttype n, T *A, const IntType lda) {

	PERFLIBS_ASSERT(n >= 0, " The number of columns in the matrix A must be greater or equal to zero");
	PERFLIBS_ASSERT(lda >= max(1, n), " The leading dimension the matrix A must be greater than the number of rows");

	// quick return like LAPACK
	if ( n == 0 ) return 0;

	using real_type = perflibs::remove_complex_t<T>;

	if (uplo == PERFLIBS_UPPER) {
		for (auto j = 0_ki; j < n; j++) {
			T column_dot_product = T(0);
			for (auto i = 0_ki; i < j; i++) {
				column_dot_product += conj(A[j*lda + i])*A[j*lda + i];
			}

			real_type ajj = perflibs::real(A[j*lda + j]) - real(column_dot_product);
			if (ajj <= real_type(0) || is_real_nan(ajj) || is_imag_nan(ajj)) {
				A[j*lda + j] = ajj;
				return j + 1;
			}
			ajj = std::sqrt(ajj);
			A[j*lda + j] = ajj;

			const real_type inverse_diag = real_type(1)/ajj;
			if (j < n - 1) {
				// Transposed GEMV on conjugated x with final scaling
				for (auto i = j + 1; i < n; i++) {
					for (auto k = 0_ki; k < j; k++) {
						A[lda*i + j] = A[lda*i + j] - A[lda*i + k]*conj(A[lda*j + k]);
					}
					A[lda*i + j] = A[lda*i + j]*inverse_diag;
				}
			}
		}
	}
	else {
		for (auto j = 0_ki; j < n; j++) {
			T column_dot_product = T(0);
			for (auto i = 0_ki; i < j; i++) {
				column_dot_product += conj(A[i*lda + j])*A[i*lda + j];
			}

			real_type ajj = real(A[j*lda + j]) - real(column_dot_product);
			if (ajj <= real_type(0) || is_real_nan(ajj) || is_imag_nan(ajj)) {
				A[j*lda + j] = ajj;
				return j + 1;
			}
			ajj = std::sqrt(ajj);
			A[j*lda + j] = ajj;

			const real_type inverse_diag = real_type(1)/ajj;
			if (j < n - 1) {
				// Non-transposed GEMV on conjugated x with final scaling. Reverse order
				// of loops compared with 'U' case above
				for (auto k = 0_ki; k < j; k++) {
					for (auto i = j + 1; i < n; i++) {
						A[lda*j + i] = A[lda*j + i] - A[lda*k + i]*conj(A[lda*k + j]);
					}
				}
				for (auto i = j + 1; i < n; i++) {
					A[lda*j + i] = A[lda*j + i]*inverse_diag;
				}
			}
		}
	}
	return 0;
}

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_CHOLESKY_NON_RECURSIVE_KERNEL_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_QR_NON_RECURSIVE_KERNEL_HPP
#define PERFLIBS_LINALG_QR_NON_RECURSIVE_KERNEL_HPP

#include "framework/linalg_util.hpp"
#include "packages/misc/interfaces/nrm2.hpp"
#include "packages/matmul/interfaces/scal.hpp"
#include "packages/factorization/helpers/householder_numeric.hpp"

namespace perflibs::linalg::factorization {

template<typename ArchitectureSpec, typename T>
void qr_non_recursive_kernel(const kernel_inttype m, const kernel_inttype n, T *A, const kernel_inttype lda, T *tau, T *work) {

	PERFLIBS_ASSERT(m >= 0, "The number of rows in the matrix A must be greater than or equal to zero");
	PERFLIBS_ASSERT(n >= 0, "The number of columns in the matrix A must be greater than or equal to zero");
	PERFLIBS_ASSERT(lda >= max(1, m), "The leading dimension of matrix A must be greater than the number of rows");

	using real_type = remove_complex_t<T>;

	constexpr pl_linalg_int_t increment_one = 1;
	constexpr real_type real_one        = real_type(1.0);
	constexpr T complex_one             = T(1.0);

	const real_type safe_min       = safe_scaling_threshold<T>();
	const auto reciprocal_safe_min = real_one / safe_min;

	for (auto col = 0_ki; col < min(m, n); col++) {
		const auto rows_in_householder     = m - col;
		const auto rows_below_diagonal_ki  = rows_in_householder - 1;
		const auto rows_below_diagonal     = static_cast<pl_linalg_int_t>(rows_below_diagonal_ki);
		const auto diagonal_element_idx = col * lda + col;

		// Generate elementary reflector H(i) to annihilate A(i+1:m,i)
		if (rows_in_householder > 1) {
			T diagonal_element = A[diagonal_element_idx];

			const auto subdiagonal_start_idx = col * lda + min(col + 1, m - 1);
			auto subcolumn_norm = perflibs::linalg::nrm2<false, pl_linalg_int_t, T, real_type, ArchitectureSpec>(&rows_below_diagonal, &A[subdiagonal_start_idx], &increment_one);

			if (subcolumn_norm == 0.0 && imag(diagonal_element) == 0.0) {
				tau[col] = 0.0;
			}
			else {
				auto householder_beta = compute_householder_beta(diagonal_element, subcolumn_norm);

				auto scaling_iterations = 0_ki;
				if (std::abs(householder_beta) < safe_min) {
					do {
						scaling_iterations++;
						perflibs::linalg::scal<false, pl_linalg_int_t, real_type, T, ArchitectureSpec>(&rows_below_diagonal, &reciprocal_safe_min, &A[subdiagonal_start_idx], &increment_one);
						householder_beta *= reciprocal_safe_min;
						diagonal_element *= reciprocal_safe_min;
					} while (std::abs(householder_beta) < safe_min && scaling_iterations < 20);

					subcolumn_norm = perflibs::linalg::nrm2<false, pl_linalg_int_t, T, real_type, ArchitectureSpec>(&rows_below_diagonal, &A[subdiagonal_start_idx], &increment_one);
					householder_beta = compute_householder_beta(diagonal_element, subcolumn_norm);
				}

				tau[col] = (householder_beta - diagonal_element) / householder_beta;

				T scaling_factor = complex_one / (diagonal_element - householder_beta);

				perflibs::linalg::scal<false, pl_linalg_int_t, T, T, ArchitectureSpec>(&rows_below_diagonal, &scaling_factor, &A[subdiagonal_start_idx], &increment_one);

				for (auto j = 0_ki; j < scaling_iterations; j++) {
					householder_beta *= safe_min;
				}
				A[diagonal_element_idx] = householder_beta;
			}
		}
		else {
			tau[col] = 0.0;
		}

		if (col < n - 1) {
			const auto next_col_start_idx = (col + 1) * lda + col;
			const auto remaining_cols = n - col - 1;

			T original_diagonal = A[diagonal_element_idx];
			A[diagonal_element_idx] = complex_one;

			T conjugate_tau = conj(tau[col]);

			// Apply H(i) to A(i:m,i+1:n) from the left
			if (conjugate_tau != T(0.0)) {
				// Compute w := A' * v
				for (auto j = 0_ki; j < remaining_cols; j++) {
					const kernel_inttype current_col_idx = next_col_start_idx + j * lda;
					T partial_sum0 = 0.0;
					T partial_sum1 = 0.0;
					T partial_sum2 = 0.0;
					T partial_sum3 = 0.0;

					// Unroll loop 4 times
					const kernel_inttype unrolled_limit = rows_in_householder - (rows_in_householder % 4);
					auto i = 0_ki;

					for (; i < unrolled_limit; i += 4) {
						partial_sum0 += conj(A[current_col_idx + i]) * A[diagonal_element_idx + i];
						partial_sum1 +=
						    conj(A[current_col_idx + i + 1]) * A[diagonal_element_idx + i + 1];
						partial_sum2 +=
						    conj(A[current_col_idx + i + 2]) * A[diagonal_element_idx + i + 2];
						partial_sum3 +=
						    conj(A[current_col_idx + i + 3]) * A[diagonal_element_idx + i + 3];
					}
					// Handle remaining elements
					for (; i < rows_in_householder; i++) {
						partial_sum0 += conj(A[current_col_idx + i]) * A[diagonal_element_idx + i];
					}
					work[j] = partial_sum0 + partial_sum1 + partial_sum2 + partial_sum3;
				}

				// Update A := A - v * w'
				for (auto j = 0_ki; j < remaining_cols; j++) {
					if (work[j] != T(0.0)) {
						const kernel_inttype current_col_idx = next_col_start_idx + j * lda;

						T update_factor = conjugate_tau * conj(work[j]);

						// Unroll loop 4 times
						const kernel_inttype unrolled_limit = rows_in_householder - (rows_in_householder % 4);
						auto i = 0_ki;

						for (; i < unrolled_limit; i += 4) {
							A[current_col_idx + i] -= A[diagonal_element_idx + i] * update_factor;
							A[current_col_idx + i + 1] -= A[diagonal_element_idx + i + 1] * update_factor;
							A[current_col_idx + i + 2] -= A[diagonal_element_idx + i + 2] * update_factor;
							A[current_col_idx + i + 3] -= A[diagonal_element_idx + i + 3] * update_factor;
						}
						// Handle remaining elements
						for (; i < rows_in_householder; i++) {
							A[current_col_idx + i] -= A[diagonal_element_idx + i] * update_factor;
						}
					}
				}
			}

			A[diagonal_element_idx] = original_diagonal;
		}
	}
}
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_QR_NON_RECURSIVE_KERNEL_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_NAIVE_TRMM
#define PERFLIBS_LINALG_NAIVE_TRMM

#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"

namespace perflibs::linalg {

struct naive_trmm {
	template<typename MatrixTypeA, typename MatrixTypeB, typename Scalar, typename... Args>
	void operator()(char side, char uplo, char transA, char diag, const MatrixTypeA &A, MatrixTypeB &B,
	                compute_position pos, Scalar alpha, Args &&... args) {

		// Let's use some sensible names for matrix dimensions
		auto m = B.cntg();
		auto n = B.strd();

		// These must have been converted to lowercase outside
		auto lside = side == 'l';
		auto upper = uplo == 'u';
		auto notrans = transA == 'n';
		auto nounit = diag == 'n';

		using T1 = typename MatrixTypeB::value_type;
		T1 temp = 0.0;

		if (lside) {
			if (notrans) {
				// Form B = alpha*A*B
				if (upper) { // left, notrans, upper
					for (kernel_inttype j = 0_ki; j < n; j++) {
						for (kernel_inttype k = 0_ki; k < m; k++) {
							// temp = alpha*b(k,j)
							temp = alpha * B(k, j);
							for (kernel_inttype i = 0_ki; i < k; i++) {
								// b(i,j) = b(i,j) + temp*a(i,k)
								B(i, j, write) = B(i, j) + temp * A(i, k);
							}
							if (nounit) {
								// temp = temp*a(k,k)
								temp = temp * A(k, k);
							}
							// b(k,j) = temp
							B(k, j, write) = temp;
						}
					}
				}
				else { // left, notrans, lower
					for (kernel_inttype j = 0_ki; j < n; j++) {
						for (kernel_inttype k = m - 1_ki; k >= 0_ki; k--) {
							// temp = alpha*b(k,j)
							temp = alpha * B(k, j);
							// b(k,j) = temp
							B(k, j, write) = temp;
							if (nounit) {
								// b(k,j) = b(k,j)*a(k,k)
								B(k, j, write) = B(k, j) * A(k, k);
							}
							for (kernel_inttype i = k + 1_ki; i < m; i++) {
								// b(i,j) = b(i,j) + temp*a(i,k)
								B(i, j, write) = B(i, j) + temp * A(i, k);
							}
						}
					}
				}
			}
			else { // trans
				// Form B = alpha*(A^T)*B
				if (upper) { // left, trans upper
					for (kernel_inttype j = 0_ki; j < n; j++) {
						for (kernel_inttype i = m - 1_ki; i >= 0; i--) {
							// temp = b(i,j)
							temp = B(i, j);
							if (nounit) {
								// temp = temp*a(i,i)
								temp = temp * A(i, i);
							}
							for (kernel_inttype k = 0_ki; k < i; k++) {
								// temp = temp + a(k,i)*b(k,j)
								temp = temp + A(k, i) * B(k, j);
							}
							// b(i,j) = alpha*temp
							B(i, j, write) = alpha * temp;
						}
					}
				}
				else { // left, trans, lower
					for (kernel_inttype j = 0_ki; j < n; j++) {
						for (kernel_inttype i = 0_ki; i < m; i++) {
							// temp = b(i,j)
							temp = B(i, j);
							if (nounit) {
								// temp = temp*a(i,i)
								temp = temp * A(i, i);
							}
							for (kernel_inttype k = i + 1_ki; k < m; k++) {
								// temp = temp + a(k,i)*b(k,j)
								temp = temp + A(k, i) * B(k, j);
							}
							// b(i,j) = alpha*temp
							B(i, j, write) = alpha * temp;
						}
					}
				}
			}
		}      // end of side = left
		else { // side = right
			if (notrans) {
				// Form B = alpha*B*A
				if (upper) { // right, notrans, upper
					for (kernel_inttype j = n - 1_ki; j >= 0_ki; j--) {
						// temp = alpha
						temp = alpha;
						if (nounit) {
							// temp = temp*a(j,j)
							temp = temp * A(j, j);
						}
						for (kernel_inttype i = 0_ki; i < m; i++) {
							// b(i,j) = temp*b(i,j)
							B(i, j, write) = temp * B(i, j);
						}
						for (kernel_inttype k = 0_ki; k < j; k++) {
							// temp = alpha*a(k,j)
							temp = alpha * A(k, j);
							for (kernel_inttype i = 0_ki; i < m; i++) {
								// b(i,j) = b(i,j) + temp*b(i,k)
								B(i, j, write) = B(i, j) + temp * B(i, k);
							}
						}
					}
				}
				else { // right, notrans, lower
					for (kernel_inttype j = 0_ki; j < n; j++) {
						// temp = alpha
						temp = alpha;
						if (nounit) {
							// temp = temp*a(j,j)
							temp = temp * A(j, j);
						}
						for (kernel_inttype i = 0_ki; i < m; i++) {
							// b(i,j) = temp*b(i,j)
							B(i, j, write) = temp * B(i, j);
						}
						for (kernel_inttype k = j + 1_ki; k < n; k++) {
							// temp = alpha*a(k,j)
							temp = alpha * A(k, j);
							for (kernel_inttype i = 0_ki; i < m; i++) {
								// b(i,j) = b(i,j) + temp*b(i,k)
								B(i, j, write) = B(i, j) + temp * B(i, k);
							}
						}
					}
				}
			}
			else {
				// Form B= alpha*B*(A^T)
				if (upper) { // right, trans, upper
					for (kernel_inttype k = 0_ki; k < n; k++) {
						for (kernel_inttype j = 0_ki; j < k; j++) {
							// temp = alpha*a(j,k)
							temp = alpha * A(j, k);
							for (kernel_inttype i = 0_ki; i < m; i++) {
								// b(i,j) = b(i,j) + temp*b(i,k)
								B(i, j, write) = B(i, j) + temp * B(i, k);
							}
						}
						// temp = alpha
						temp = alpha;
						if (nounit) {
							// temp = temp*a(k,k)
							temp = temp * A(k, k);
						}
						// BLAS guards this with "temp.NE.1"
						// but we can't because temp is a vector for us
						for (kernel_inttype i = 0_ki; i < m; i++) {
							// b(i,k) = temp*b(i,k)
							B(i, k, write) = temp * B(i, k);
						}
					}
				}
				else { // right, trans, lower
					for (kernel_inttype k = n - 1_ki; k >= 0_ki; k--) {
						for (kernel_inttype j = k + 1_ki; j < n; j++) {
							// temp = alpha*a(j,k)
							temp = alpha * A(j, k);
							for (kernel_inttype i = 0_ki; i < m; i++) {
								// b(i,j) = b(i,j) + temp*b(i,k)
								B(i, j, write) = B(i, j) + temp * B(i, k);
							}
						}
						// temp = alpha
						temp = alpha;
						if (nounit) {
							// temp = temp*a(k,k)
							temp = temp * A(k, k);
						}
						// BLAS guards this with "temp.NE.1"
						// but we can't because temp is a vector for us
						for (kernel_inttype i = 0_ki; i < m; i++) {
							// b(i,k) = temp*b(i,k)
							B(i, k, write) = temp * B(i, k);
						}
					}
				} // end uplo
			}  // end trans
		} // end side
	}
};
} // namespace perflibs::linalg

#endif

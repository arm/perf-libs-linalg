/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_NAIVE_TRSM
#define PERFLIBS_LINALG_NAIVE_TRSM

#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"

namespace perflibs::linalg {

struct naive_trsm {
	template<typename MatrixTypeA, typename MatrixTypeB, typename Scalar, typename... Args>
	void operator()(char side, char uplo, char transA, char diag, const MatrixTypeA& A, MatrixTypeB& B,
	                compute_position pos, Scalar alpha, Args&&... args) {

		// Let's use some sensible names for matrix dimensions
		auto m = B.cntg();
		auto n = B.strd();

		// These must have been converted to lowercase outside
		auto lside = side == 'l';
		auto upper = uplo == 'u';
		auto notrans = transA == 'n';
		auto nounit = diag == 'n';

		using T1 = decltype(alpha);
		auto alpha_not1 = alpha != one<T1>;
		using T2 = typename MatrixTypeB::value_type;
		T2 temp = 0.0;

		if (lside) {
			// For each column of B
			for (kernel_inttype j = 0_ki; j < n; j++) {

				if (notrans) {
					// Scale the column by alpha if needed
					if (alpha_not1) {
						for (kernel_inttype i = 0_ki; i < m; i++) {
							B(i, j, write) = alpha*B(i, j);
						}
					}
					if (upper) { // left, notrans, upper
						// Working from the bottom of column-vector x (LHS), first find the value of x that
						// is trivial to work out by dividing by the corresponding diagonal A value, then
						// update the rest of the column-vector b (RHS) to subtract the contributions from
						// this new found element of x
						for (kernel_inttype k = m - 1_ki; k >= 0_ki; k--) {
							// Note that we can't easily do e.g. if (B(k, j) != T{0.0}) { then then skip the
							// work below } as is done in BLAS, since B(k, j) is not necessarily a single
							// element!
							if (nounit) {
								// B(k, j) is now really X(k, j)
								B(k, j, write) = B(k, j)/A(k, k);
							}
							// Subtract the contributions of X(k, j)*A(1:k,k) from the remaining values in b
							for (kernel_inttype i = 0_ki; i < k; i++) {
								B(i, j, write) = B(i, j) - B(k, j)*A(i, k);
							}
						}
					}
					else { // left, notrans, lower
						for (kernel_inttype k = 0_ki; k < m; k++) {
							if (nounit) {
								B(k, j, write) = B(k, j)/A(k, k);
							}
							for (kernel_inttype i = k + 1_ki; i < m; i++) {
								B(i, j, write) = B(i, j) - B(k, j)*A(i, k);
							}
						}
					}
				}

				else { // if trans
					if (upper) { // left, trans, upper
						for (kernel_inttype i = 0_ki; i < m; i++) {
							temp = alpha*B(i, j);
							for (kernel_inttype k = 0_ki; k < i; k++) {
								temp = temp - A(k, i)*B(k, j);
							}
							if (nounit) {
								temp = temp/A(i, i);
							}
							B(i, j, write) = temp;
						}
					}
					else { // left, trans, lower
						for (kernel_inttype i = m - 1_ki; i >= 0_ki; i--) {
							temp = alpha*B(i, j);
							for (kernel_inttype k = i + 1_ki; k < m; k++) {
								temp = temp - A(k, i)*B(k, j);
							}
							if (nounit) {
								temp = temp/A(i, i);
							}
							B(i, j, write) = temp;
						}
					}
				}

			} // end loop over n
		} // end of side = left

		else { // side = right
			if (notrans) {
				if (upper) { // right, notrans, upper
					for (kernel_inttype j = 0_ki; j < n; j++) {
						if (alpha_not1) {
							for (kernel_inttype i = 0_ki; i < m; i++) {
								B(i, j, write) = alpha*B(i, j);
							}
						}
						for (kernel_inttype k = 0_ki; k < j; k++) {
							for (kernel_inttype i = 0_ki; i < m; i++) {
								B(i, j, write) = B(i, j) - A(k, j)*B(i, k);
							}
						}
						if (nounit) {
							temp = 1.0/A(j, j);
							for (kernel_inttype i = 0_ki; i < m; i++) {
								B(i, j, write) = temp*B(i, j);
							}
						}
					}
				}
				else { //right, notrans, lower
					for (kernel_inttype j = n - 1_ki; j >= 0_ki; j--) {
						if (alpha_not1) {
							for (kernel_inttype i = 0_ki; i < m; i++) {
								B(i, j, write) = alpha*B(i, j);
							}
						}
						for (kernel_inttype k = j + 1_ki; k < n; k++) {
							for (kernel_inttype i = 0_ki; i < m; i++) {
								B(i, j, write) = B(i, j) - A(k, j)*B(i, k);
							}
						}
						if (nounit) {
							temp = 1.0/A(j, j);
							for (kernel_inttype i = 0_ki; i < m; i++) {
								B(i, j, write) = temp*B(i, j);
							}
						}
					}
				}
			}
			else { // trans
				if (upper) { // right, trans, upper
					for (kernel_inttype k = n - 1_ki; k >= 0_ki; k--) {
						if (nounit) {
							temp = 1.0/A(k, k);
							for (kernel_inttype i = 0_ki; i < m; i++) {
								B(i, k, write) = temp*B(i, k);
							}
						}
						for (kernel_inttype j = 0_ki; j < k; j++) {
							for (kernel_inttype i = 0_ki; i < m; i++) {
								B(i, j, write) = B(i, j) - A(j, k)*B(i, k);
							}
						}
						if (alpha_not1) {
							for (kernel_inttype i = 0_ki; i < m; i++) {
								B(i, k, write) = alpha*B(i, k);
							}
						}
					}
				}
				else { //right, trans, lower
					for (kernel_inttype k = 0_ki; k < n; k++) {
						if (nounit) {
							temp = 1.0/A(k, k);
							for (kernel_inttype i = 0_ki; i < m; i++) {
								B(i, k, write) = temp*B(i, k);
							}
						}
						for (kernel_inttype j = k + 1_ki; j < n; j++) {
							for (kernel_inttype i = 0_ki; i < m; i++) {
								B(i, j, write) = B(i, j) - A(j, k)*B(i, k);
							}
						}
						if (alpha_not1) {
							for (kernel_inttype i = 0_ki; i < m; i++) {
								B(i, k, write) = alpha*B(i, k);
							}
						}
					}
				} // end uplo
			} // end trans
		} // end side

	}
};
}

#endif

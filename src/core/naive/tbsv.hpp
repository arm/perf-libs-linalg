/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_NAIVE_TBSV
#define PERFLIBS_LINALG_NAIVE_TBSV

#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

struct naive_tbsv {
	template<typename MatrixTypeA, typename MatrixTypeB, typename... Args>
	void operator()(MatrixTypeA &A, MatrixTypeB &B,
	                Args &&... args) {

		// Let's use some sensible names for matrix dimensions
		auto n = B.cntg();
		// Triangular, therefore one of kl and ku must be zero
		auto k = A.kl() + A.ku();

		if (A.is_upper()) { // upper, top to bottom
			for (kernel_inttype i = (n-1); i >= 0; i--) {
				B(i,0,write) = B(i,0) / A(i,i);
				for (kernel_inttype j = 1; j < (k+1); j++) {
					if (i-j >= 0)
						B(i-j,0,write) = B(i-j,0) - (A(i-j,i)*B(i,0));
				}
			}
		}
		else { // lower, bottom to top
			for (kernel_inttype i = 0; i < n; i++) {
				B(i,0,write) = B(i,0) / A(i,i);
				for (kernel_inttype j = 1; j < (k+1); j++) {
					if (i+j < n)
						B(i+j,0,write) = B(i+j,0) - (A(i+j,i)*B(i,0));
				}
			}
		}
	}
};
} // namespace perflibs::linalg

#endif

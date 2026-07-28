/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_NAIVE_QR
#define PERFLIBS_LINALG_NAIVE_QR

#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"

namespace perflibs::linalg {

/**
 * A variant of the naive GEMM that is tailored to use inside batch-interleave QR as GEMV
 * In particular alpha = 1, beta = 0 and x[0] = 1
 */
struct naive_gemv_qr {
	template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename... Args>
	void operator()(const MatrixTypeA& a, const MatrixTypeB& x, MatrixTypeC& y, compute_position pos, Args&&... args) {
		PERFLIBS_ASSERT(a.cntg() == x.cntg(), "a and x do not have matching cntg() values");
		PERFLIBS_ASSERT(a.strd() == y.cntg(), "a does not share its strd dim with y's cntg dimension");
		PERFLIBS_ASSERT(x.strd() == y.strd(), "x and y do not have matching strd() values");

		const auto a_strd = a.strd();
		const auto x_strd = x.strd();
		const auto cntg   = a.cntg();

		for(kernel_inttype j = 0; j < x_strd; ++j) {

			for(kernel_inttype i = 0; i < a_strd; ++i) {
				// use write `operator()` to read here since `interleave_batch_matrix` values can
				// only be accessed by reference
				y(i, j, write) = a(0, i, write);
				for(kernel_inttype k = 1; k < cntg; ++k) {
					y(i, j, write) += a(k, i, write) * x(k, j, write);
				}
			}
		}
	}
}; // struct naive_gemv

/**
 * A linalg operator for GER for use in QR
 *
 * Alpha is a vector of ninter scalars and is negated within the function,  A and B are input vectors,
 * and C is the output matrix
 *
 * Furthermore it is assumed that the first entry of A, A(0,0), = 1
 *
 * As ger is effectively a gemm where k==1, cntg for both a and b should be 1 making them vectors
 *
 * c.cntg() is m, therefore equal to a.strd(), and c.strd() is n and therefore equal to b.strd()
 *
 */
struct naive_ger_qr {
	template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename MatrixTypeAlpha,
	         typename... Args>
	void operator()(const MatrixTypeA &a, const MatrixTypeB &b, MatrixTypeC &c, compute_position pos,
	                MatrixTypeAlpha alpha, Args &&...) {
		// cntg  for A and B are effectively 'k', ger is a gemm with no k
		// assert will not be 'on' in the release vesrsion
		PERFLIBS_ASSERT(a.cntg() == 1, "A should be vector");
		PERFLIBS_ASSERT(b.cntg() == 1, "B should be vector");
		PERFLIBS_ASSERT(c.cntg() == a.strd(), "c.cntg() does not match a.strd()");
		PERFLIBS_ASSERT(c.strd() == b.strd(), "c.cntg() does not match b.strd()");
		PERFLIBS_ASSERT(alpha.cntg() == 1, "Alpha should be a scalar");
		PERFLIBS_ASSERT(alpha.strd() == 1, "Alpha should be a scalar");

		for (kernel_inttype j = 0_ki; j < c.strd(); ++j) {
			// the first element of a is always 1 in this use, and alpha needs to be
			// multiplied by -1
			c(0_ki, j) -= alpha(0_ki, 0_ki) * b(0_ki, j);
			for (kernel_inttype i = 1_ki; i < c.cntg(); ++i) {
				// ki simply means the ints are kernel inttypes, can use '0' too,
				// but fixing the types produces a slightly cleaner code gen
				//
				// note that these are all ninter-length vectors under the hood,
				// e.g. alpha looks like a single scalar here but it is in fact
				// a vector of ninter different alphas. Similarly a is a ninter
				// lots of length-c.cntg() vectors
				c(i, j) -= alpha(0_ki, 0_ki) * a(0_ki, i) * b(0_ki, j);
			}
		}
	}
}; // struct naive_ger_qr

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_NAIVE_QR

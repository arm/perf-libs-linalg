/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_NAIVE_GER
#define PERFLIBS_LINALG_NAIVE_GER

#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"

namespace perflibs::linalg {

/**
 * A linalg operator for GER where A and B are input vectors, and C is the output matrix
 *
 * As ger is effectively a gemm where k==1, cntg for both a and b should be 1 making them vectors
 *
 * c.cntg() is m, therefore equal to a.strd(), and c.strd() is n and therefore equal to b.strd()
 */
struct naive_ger {
	template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename ScalarType,
	         typename... Args>
	void operator()(const MatrixTypeA &a, const MatrixTypeB &b, MatrixTypeC &c, compute_position pos,
	                ScalarType alpha, Args &&...) {
		// cntg  for A and B are effectively 'k', ger is a gemm with no k
		// assert will not be 'on' in the release vesrsion
		PERFLIBS_ASSERT(a.cntg() == 1, "A should be vector");
		PERFLIBS_ASSERT(b.cntg() == 1, "B should be vector");
		PERFLIBS_ASSERT(c.cntg() == a.strd(), "c.cntg() does not match a.strd()");
		PERFLIBS_ASSERT(c.strd() == b.strd(), "c.cntg() does not match b.strd()");

		for (kernel_inttype j = 0_ki; j != c.strd(); ++j) {
			for (kernel_inttype i = 0_ki; i != c.cntg(); ++i) {
				// ki simply means the ints are kernel inttypes, can use '0' too,
				// but fixing the types produces a slightly cleaner code gen
				c(i, j, write) += alpha * a(0_ki, i) * b(0_ki, j);
			}
		}
	}
}; // struct naive_ger

/**
 * A linalg operator for GER where Alpha is a vector of ninter scalars,  A and B are input vectors,
 * and C is the output matrix
 *
 * As ger is effectively a gemm where k==1, cntg for both a and b should be 1 making them vectors
 *
 * c.cntg() is m, therefore equal to a.strd(), and c.strd() is n and therefore equal to b.strd()
 */
struct naive_ger_vec_alpha {
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

		for (kernel_inttype j = 0_ki; j != c.strd(); ++j) {
			for (kernel_inttype i = 0_ki; i != c.cntg(); ++i) {
				// ki simply means the ints are kernel inttypes, can use '0' too,
				// but fixing the types produces a slightly cleaner code gen
				//
				// note that these are all ninter-length vectors under the hood,
				// e.g. alpha looks like a single scalar here but it is in fact
				// a vector of ninter different alphas. Similarly a is a ninter
				// lots of length-c.cntg() vectors
				c(i, j, write) += alpha(0_ki, 0_ki) * a(0_ki, i) * b(0_ki, j);
			}
		}
	}
}; // struct naive_ger

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_NAIVE_GER

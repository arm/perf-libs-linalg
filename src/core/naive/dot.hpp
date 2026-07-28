/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_NAIVE_DOT
#define PERFLIBS_LINALG_NAIVE_DOT

#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"

#include "matrix/interleave_batch.hpp"
#include "matrix/adaptors.hpp"

namespace perflibs::linalg {

/**
 * A linalg operator for DOT which works on MatrixTypes. Essentially it is the innermost loop of a GEMM.
 */
struct naive_dot {
	template <typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename... Args>
	void operator()(const MatrixTypeA &a, const MatrixTypeB &b, MatrixTypeC &c, compute_position pos,
	                Args &&...) {
		PERFLIBS_ASSERT(a.strd() == 1, "A must be vector");
		PERFLIBS_ASSERT(b.strd() == 1, "B must be vector");
		PERFLIBS_ASSERT(a.cntg() == b.cntg(), "A and B must be same length");
		PERFLIBS_ASSERT(c.cntg() == 1, "C must be scalar");
		PERFLIBS_ASSERT(c.strd() == 1, "C must be scalar");

		// ki simply means the ints are kernel inttypes, can use '0' too,
		// but fixing the types produces a slightly cleaner code gen
		c(0_ki, 0_ki, write) = scalar{ 0.0 };

		for (kernel_inttype i = 0; i != a.cntg(); ++i) {
			// use write `operator()` to read here since `interleave_batch_matrix` values can
			// only be accessed by reference
			c(0_ki, 0_ki, write) += a(i, 0_ki, write) * b(i, 0_ki, write);
		}
	}
}; // struct naive_dot

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_NAIVE_DOT

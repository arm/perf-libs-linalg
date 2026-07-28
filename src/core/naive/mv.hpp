/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_NAIVE_MV_HPP
#define PERFLIBS_LINALG_NAIVE_MV_HPP

#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"
#include "matrix/adaptors.hpp"

namespace perflibs::linalg {
namespace {

/// This is for debugging and testing. Will produce the correct answer, for y = alpha * a * x + beta * y
struct naive_mv {
	template <typename AType, typename BType, typename CType, typename ScalarType, typename... Args>
	void operator()(AType& a, BType& x, CType& y, compute_position pos, ScalarType alpha, ScalarType beta, Args&&... args) {
		//sanity checks
		PERFLIBS_ASSERT(y.cntg() == a.strd(), "y cntg doesn't match a strd");
		PERFLIBS_ASSERT(x.cntg() == a.cntg(), "x ctng doesn't match a cntg");
		PERFLIBS_ASSERT(x.strd() == 1, "x strd is not 1");
		PERFLIBS_ASSERT(y.strd() == 1, "y strd is not 1");

		if (pos.cntg != 0 || pos.iteration != 0)
			beta = one<ScalarType>;

		if (beta == zero<ScalarType>) {
			set(zero<ScalarType>, y);
		}

		for(kernel_inttype i = 0; i < y.cntg(); ++i) {
			ScalarType tmp  = zero<ScalarType>;
			for(kernel_inttype j = 0; j < x.cntg(); ++j) {
				tmp += a(j, i) * x(j, 0);
			}
			y(i, 0, write) = alpha*tmp + beta*y(i, 0);
		}
	}
}; // struct naive_mv
} // namespace anon
} //namespace perflibs::linalg

#endif // PERFLIBS_LINALG_NAIVE_MV_HPP

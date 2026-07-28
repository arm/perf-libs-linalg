/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_NAIVE_OUT_OF_PLACE_MATMUL_HPP
#define PERFLIBS_LINALG_NAIVE_OUT_OF_PLACE_MATMUL_HPP

#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"

namespace perflibs::linalg::matmul {

/**
 * A naive implementation of out_of_place_matmul which is supposed to work on MatrixTypes
 *
 * Dot products are performed along the cntg dimension of the A and B matrices, as this is their shared dimension ('k' in BLAS terms).
 * These are then accumulated into the corresponding D matrix, where a.strd() == c.cntg() and b.strd() == c.strd()
 * ('m' and 'n' in BLAS terms, respectively), and c.strd() == d.strd() and c.cntg() == d.cntg()
 */
struct naive_out_of_place_matmul {
	template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename MatrixTypeD, typename Scalar, typename... Args>
	void operator()(const MatrixTypeA& a, const MatrixTypeB& b, const MatrixTypeC& c, MatrixTypeD& d, compute_position pos, Scalar alpha, Scalar beta, Args&&... args) {
		PERFLIBS_ASSERT(a.cntg() == b.cntg(), "a and b do not have matching cntg() values");
		PERFLIBS_ASSERT(a.strd() == c.cntg(), "a does not share its strd dim with c's cntg dimension");
		PERFLIBS_ASSERT(b.strd() == c.strd(), "b and c do not have matching strd() values");

		/*
		 * we are going to define tmp like this at the highest level of the loop
		 * because it could be interleave_batch_val, which do have some associated cost
		 * to construct
		 */
		typename MatrixTypeD::value_type tmp;

		const auto a_strd = a.strd();
		const auto b_strd = b.strd();
		const auto cntg   = a.cntg();

		for(kernel_inttype j = 0; j != b_strd; ++j) {

			for(kernel_inttype i = 0; i != a_strd; ++i) {
				tmp = 0.0;
				for(kernel_inttype k = 0; k != cntg; ++k) {
					tmp += a(k, i) * b(k, j);
				}
				d(i, j, write) = alpha * tmp + c(i, j) * beta;
			}
		}
	}
}; // struct naive_out_of_place_matmul

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_NAIVE_OUT_OF_PLACE_MATMUL_HPP

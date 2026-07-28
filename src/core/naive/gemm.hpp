/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_NAIVE_GEMM
#define PERFLIBS_LINALG_NAIVE_GEMM

#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"

namespace perflibs::linalg {

/**
 * A naive implementation of GEMM which is supposed to work on MatrixTypes
 *
 * Dot products are performed along the cntg dimension of the A and B matrix, as this is there shared dimension ('k' in BLAS terms).
 * These are then accumulated into the corresponding C matrix, where a.strd() == c.cntg() and b.strd() == c.strd() ('m' and 'n' in BLAS terms, respectively)
 */
template<typename ArchitectureSpec>
struct naive_gemm {
	zero_mode beta_zero_mode { zero_mode::set };

	template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename Scalar, typename... Args>
	void operator()(const MatrixTypeA& a, const MatrixTypeB& b, MatrixTypeC& c, compute_position pos, Scalar alpha, Scalar beta, Args&&... args) {
		PERFLIBS_ASSERT(a.cntg() == b.cntg(), "a and b do not have matching cntg() values");
		PERFLIBS_ASSERT(a.strd() == c.cntg(), "a does not share its strd dim with c's cntg dimension");
		PERFLIBS_ASSERT(b.strd() == c.strd(), "b and c do not have matching strd() values");

		/*
		 * we are going to define tmp like this at the highest level of the loop
		 * because it could be interleave_batch_val, which do have some associated cost
		 * to construct
		 */
		using c_data_type = typename MatrixTypeC::value_type;
		c_data_type tmp;

		const auto a_strd = a.strd();
		const auto b_strd = b.strd();
		const auto cntg   = a.cntg();

		for(kernel_inttype j = 0_ki; j != b_strd; ++j) {
			for(kernel_inttype i = 0_ki; i != a_strd; ++i) {
				if(alpha != zero<>) {
					tmp = zero<c_data_type>;
					for(kernel_inttype k = 0_ki; k != cntg; ++k) {
						tmp += static_cast<c_data_type>(a(k, i)) * static_cast<c_data_type>(b(k, j));
					}
					if(beta != zero<>) {
						c(i, j, write) = static_cast<c_data_type>(alpha) * tmp + c(i, j) * static_cast<c_data_type>(beta);
					}
					else {
						c(i, j, write) = static_cast<c_data_type>(alpha) * tmp;
					}
				}
				else {
					if(beta != zero<> || beta_zero_mode == zero_mode::scale) {
						c(i, j, write) = c(i, j) * static_cast<c_data_type>(beta);
					}
					else {
						c(i, j, write) = zero<c_data_type>;
					}
				}
			}
		}
	}
}; // struct naive_gemm

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_NAIVE_GEMM

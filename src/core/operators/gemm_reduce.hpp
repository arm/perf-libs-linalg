/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_GEMM_REDUCE_HPP
#define PERFLIBS_LINALG_GEMM_REDUCE_HPP

#include "framework/linalg_util.hpp"
#include "matrix/operations.hpp"

namespace perflibs::linalg {

/**
 * A reduction operator, reduce c1 into c0 using an axpby-like kernel
 */
template<typename AxpbyKernel>
class gemm_reduce {
	AxpbyKernel kernel_;

public:
	PERFLIBS_LINALG_INLINE
	gemm_reduce(AxpbyKernel kernel)
	:	kernel_ { std::move(kernel) }
	{	}

	template<typename C0MatrixType, typename C1MatrixType>
	PERFLIBS_LINALG_INLINE
	void operator() (C0MatrixType& c0, C1MatrixType& c1) const {
		using value_type = typename C0MatrixType::value_type;

		const auto c0_cntg_step = c0.cntg_step();
		const auto c1_cntg_step = c1.cntg_step();

		/*
		 * this behavior could be supported by min-ing the dimensions of the matrix
		 * but should only be added if necessary
		 */
		PERFLIBS_ASSERT(c0.cntg() == c1.cntg(), "reduction matrix cntg dimensions do not match");
		PERFLIBS_ASSERT(c0.strd() == c1.strd(), "reduction matrix strd dimensions do not match");

		// if we can do the whole reduction in a single axpby then do that
		if (c0.strd_step() == c0.cntg()) {
			const auto total_elems = c0.cntg() * c0.strd();

			kernel_(total_elems, one<value_type>, c1.data(), one<value_type>, c0.data(), c1_cntg_step, c0_cntg_step);
		}
		else { //loop over the cols and reduce once by one
			for(auto col_idx = 0_ki; col_idx != c0.strd(); ++col_idx) {
				auto c0_vec = get_strd_panel(c0, col_idx, 1);
				auto c1_vec = get_strd_panel(c1, col_idx, 1);

				kernel_(c0_vec.cntg(), one<value_type>, c1_vec.data(), one<value_type>, c0_vec.data(), c1_cntg_step, c0_cntg_step);
			}
		}
	}
}; // class gemm_reduce

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_GEMM_REDUCE_HPP

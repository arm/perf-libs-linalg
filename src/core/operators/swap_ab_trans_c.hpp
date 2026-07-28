/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SWAP_AB_TRANS_C_HPP
#define PERFLIBS_LINALG_SWAP_AB_TRANS_C_HPP

#include <utility>

#include "framework/compute_position.hpp"

namespace perflibs::linalg {

template<typename Next>
class swap_ab_trans_c {
	Next next_;
public:
	PERFLIBS_LINALG_INLINE
	swap_ab_trans_c(Next next)
	:	next_ { std::move(next) }
	{	}

	/**
	 * LINALG stack operator
	 */
	template<typename AType, typename BType, typename CMatType, typename... Args>
	PERFLIBS_LINALG_INLINE
	void operator()(AType& a, BType& b, CMatType& c, const compute_position& pos, Args&&... args) {
		/*
		 * we must access the parent rather than using C directly because C could be a 1x1 sub_matrix
		 * of the parent which is a vector, in which case the logic cntg() <= 1 is going to go down
		 * the incorrect path
		 */
		const auto c_parent = c.get_parent();

		if(is_cntg_contig(c_parent) && ( c_parent.cntg() <= 1 || !is_strd_contig(c_parent) )) {
			next_(a, b, c, pos, std::forward<Args>(args)...);
		}
		else {
			compute_position new_pos = pos;
			std::swap(new_pos.a_strd, new_pos.b_strd);
			auto c_trans = c.transpose();
			next_(b, a, c_trans, new_pos, std::forward<Args>(args)...);
		}
	}
}; //class swap_ab_trans_c

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_SWAP_AB_TRANS_C_HPP

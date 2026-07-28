/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_PANELLER_HPP
#define PERFLIBS_LINALG_PANELLER_HPP

#include "framework/which.hpp"
#include "framework/compute_position.hpp"

#include "perflibs_assert.hpp"

namespace perflibs::linalg {

/**
 * Stack operator that provides a wrapper around get_strd_panel and get_cntg_panel. Handles panelling over the
 * chosen dimension.
 *
 * \tparam which_dimension the dimension which will be broken up to form a panel
 */
template <which_dimension Dimension, typename Next>
class paneller {
	/**
	 * The next operator in the LINALG stack.
	 */
	Next next_;

public:
	PERFLIBS_LINALG_INLINE
	paneller(Next next)
	:	next_ { std::move(next) }
	{	}

	PERFLIBS_LINALG_INLINE
	paneller(which_dimension_constant<Dimension>, Next next)
	:	next_ { std::move(next) }
	{	}

	/**
	 * Gets the size of the appropriate matrix in the dimension used by the paneller.
	 *
	 * @tparam AType The type of the matrix `a`.
	 * @tparam BType The type of the matrix `b`.
	 * @tparam CType The type of the matrix `c`.
	 * @param [in] a The 'a' matrix as used elsewhere.
	 * @param [in] b The 'b' matrix as used elsewhere.
	 * @param [in] c The 'c' matrix as used elsewhere.
	 * @return the size of the appropriate matrix in the dimension used by the paneller.
	 */
	template<typename AType, typename BType, typename CType>
	PERFLIBS_LINALG_INLINE
	kernel_inttype size(const AType &a, const BType &b, const CType &c) {
		if constexpr (Dimension == which_dimension::a_strd) {
			return a.strd();
		}
		else if constexpr (Dimension == which_dimension::b_strd) {
			return b.strd();
		}
		else if constexpr (Dimension == which_dimension::cntg) {
			return a.cntg(); // (same as b.cntg())
		}
		else {
			// Unreachable.
			PERFLIBS_ASSERT(false,
			                  "Unknown dimension used in paneller -- must be `a_strd`, `b_strd`, or `cntg`.")
		}
	}

	/**
	 * Create a panel and pass it further down the LINALG stack as a subproblem.
	 *
	 * @tparam A The type of the matrix `a`.
	 * @tparam B The type of the matrix `b`.
	 * @tparam C The type of the matrix `c`.
	 * @param [in] a The 'a' matrix as used elsewhere.
	 * @param [in] b The 'b' matrix as used elsewhere.
	 * @param [in] c The 'c' matrix as used elsewhere.
	 * @param [in] panel_pos The start panel_position of the panel.
	 * @param [in] panel_size The panel_size of the panel.
	 * @param [in] compute_pos The compute position.
	 * @param [in] block The block index. If work for a thread has been split up into distinct blocks (e.g. in
	*                    the case of symmetric or triangular matrices), this parameter identifies which block
	*                    we are operating on. If work has not been split into blocks, this should be zero.
	*                    This is placed inside the `compute_position` and passed to the rest of the LINALG
	*                    stack.
	 */
	template <typename AType, typename BType, typename CType, typename... Args>
	PERFLIBS_LINALG_INLINE void operator()(AType &a, BType &b, CType &c, kernel_inttype panel_pos,
	                                  kernel_inttype panel_size, compute_position compute_pos,
	                                  kernel_inttype block, kernel_inttype blocks, Args &&... args) {
		if constexpr (Dimension == which_dimension::a_strd) {
			auto new_a = get_strd_panel(a, panel_pos, panel_size);
			auto new_c = get_cntg_panel(c, panel_pos, panel_size);
			next_(new_a, b, new_c, advance(compute_pos, panel_pos, 0, 0, 0, block, blocks),
			      std::forward<Args>(args)...);
		}
		else if constexpr (Dimension == which_dimension::b_strd) {
			auto new_b = get_strd_panel(b, panel_pos, panel_size);
			auto new_c = get_strd_panel(c, panel_pos, panel_size);
			next_(a, new_b, new_c, advance(compute_pos, 0, panel_pos, 0, 0, block, blocks),
			      std::forward<Args>(args)...);
		}
		else if constexpr (Dimension == which_dimension::cntg) {
			auto new_a = get_cntg_panel(a, panel_pos, panel_size);
			auto new_b = get_cntg_panel(b, panel_pos, panel_size);
			next_(new_a, new_b, c, advance(compute_pos, 0, 0, panel_pos, 0, block, blocks),
			      std::forward<Args>(args)...);
		}
		else {
			// Unreachable.
			PERFLIBS_ASSERT(false, "Unknown dimension used in paneller -- must be `a_strd`, `b_strd`, or `cntg`.")
		}
	}
}; //class paneller

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_PANELLER_HPP

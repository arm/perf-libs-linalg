/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PARTIAL_SEPARATOR_HPP
#define PARTIAL_SEPARATOR_HPP

#include "perflibs_assert.hpp"
#include "perflibs_numeric_utils.hpp"
#include "perflibs_util.hpp"
#include "framework/compute_position.hpp"
#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

/**
 * Given a sub-matrix of a matrix with some form (tri, symm)
 * Then break the sub-matrix up into two small sub matrices
 * One containing an uninterrupted dense block, the other a partial matrix
 *
 * i.e
 *
 *   +-------------
 *   |             \
 *   |              \
 *   |               \
 *   |               |
 *   +---------------+
 *
 * would become
 *
 *   +-------------    ^
 *   |             |   |\
 *   |             |   | \
 *   |             |   |  \
 *   |             |   |  |
 *   +-------------+   +--+
 *
 * and processes separately
 */
template<typename NextFull, typename NextPart>
class partial_separate {
	kernel_inttype a_strd_unroll_;
	kernel_inttype b_strd_unroll_;
	NextFull next_full_;
	NextPart next_part_;
public:
	partial_separate(kernel_inttype a_strd_unroll, kernel_inttype b_strd_unroll, NextFull next_full, NextPart next_part)
	:	a_strd_unroll_ { a_strd_unroll }
	,	b_strd_unroll_ { b_strd_unroll }
	,	next_full_     { next_full     }
	,	next_part_     { next_part     }
	{	}

	template<typename AType, typename BType, typename CMatType, typename... Args>
	inline
	void operator()(AType& a, BType& b, CMatType& c, const compute_position& pos, Args&&... args) {
		/*
		 * We need to sample the non-virtual dimensions of the C matrix, take a horizontal sample at the top and bottom
		 * and a vertical sample at the far right and left.
		 *
		 * When you compare the non-virtual dimensions at the extremes you can then determinen the shape of the matrix
		 *
		 * Left is strd 0, Right is strd max
	 	 * Top is cntg 0, Bottom is cntg max
		 */
		const auto [left_cntg_first, left_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(c, 0, include_diag::no);
		const auto [rght_cntg_first, rght_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(c, c.strd() - 1, include_diag::no);
		PERFLIBS_UNUSED(rght_cntg_last);

		const auto [top_strd_first, top_strd_last ] = get_non_virtual_strd_bounds_for_cntg(c, 0, include_diag::no);
		const auto [btm_strd_first, btm_strd_last ] = get_non_virtual_strd_bounds_for_cntg(c, c.cntg() - 1, include_diag::no);
		PERFLIBS_UNUSED(btm_strd_last);

		if(top_strd_first == 0
		&& btm_strd_first == 0
		&& left_cntg_first == 0
		&& rght_cntg_first == 0) {
			PERFLIBS_ASSERT(c.is_physical(), "C should be physical to compute full block");
			next_full_(a, b, c, pos, std::forward<Args>(args)...);
		}
		else if(top_strd_first == 0
		     && btm_strd_first == 0
		     && left_cntg_first == 0) {
			PERFLIBS_ASSERT(! c.is_physical(), "C should not be physical to compute partial block");
			next_part_(a, b, c, pos, std::forward<Args>(args)...);
		}
		else if(top_strd_first == 0
		     && left_cntg_first == 0
		     && rght_cntg_first == 0) {
			PERFLIBS_ASSERT(! c.is_physical(), "C should not be physical to compute partial block");
			next_part_(a, b, c, pos, std::forward<Args>(args)...);
		}
		//is lower
		else if(top_strd_first == 0
		&& btm_strd_first == 0) {
			const auto cntg_full_sz = c.cntg() - rght_cntg_first;

			//split along strd
			if(top_strd_last > cntg_full_sz) {
				/*    strd
				 *   +-------------+   ^
				 * c |             |   |\
				 * n |             |   | \
				 * t |             |   |  \
				 * g |             |   |  |
				 *   +-------------+   +--+
				 */

				const auto is = interleave_factor(b) * b_strd_unroll_;
				const auto top_strd_last_fix = max(iround_floor(top_strd_last, is), 0);

				{
					auto left_c = get_strd_panel_with_clamp(c, 0, top_strd_last_fix);
					auto left_b = get_strd_panel(b, 0, top_strd_last_fix);


					if(left_c.strd() != 0) {
						PERFLIBS_ASSERT(left_c.is_physical(), "C should be physical to compute full block");
						next_full_(a, left_b, left_c, pos, std::forward<Args>(args)...);
					}
				}
				{
					auto rght_c = get_strd_panel_with_clamp(c, top_strd_last_fix, c.strd() - top_strd_last_fix);
					auto rght_b = get_strd_panel(b, top_strd_last_fix, c.strd() - top_strd_last_fix);

					PERFLIBS_ASSERT(! rght_c.is_physical(), "C should not be physical to compute partial block");

					next_part_(a, rght_b, rght_c,
						advance(pos, 0, top_strd_last_fix, 0),
						std::forward<Args>(args)...);
				}
			}
			else { //split along cntg
				/*    strd
				 *   +-----
				 *   |      \
				 * c |       \
				 * n +--------
				 * t +--------+
				 * g |        |
				 *   |        |
				 *   |        |
				 *   |        |
				 *   +--------+
				 */

				const auto is = interleave_factor(a) * a_strd_unroll_;
				//const auto rght_cntg_first_fix = min(iround(btm_strd_first, is), a.strd());
				const auto rght_cntg_first_fix = min(iround(rght_cntg_first, is), a.strd());

				{
					auto top_c = get_cntg_panel_with_clamp(c, 0, rght_cntg_first_fix);
					auto top_a = get_strd_panel(a, 0, rght_cntg_first_fix);

					PERFLIBS_ASSERT(! top_c.is_physical(), "C should not be physical to compute partial block");

					next_part_(top_a, b, top_c, pos, std::forward<Args>(args)...);
				}
				{
					auto btm_c = get_cntg_panel_with_clamp(c, rght_cntg_first_fix, c.cntg() - rght_cntg_first_fix);
					auto btm_a = get_strd_panel(a, rght_cntg_first_fix, c.cntg() - rght_cntg_first_fix);


					if(btm_c.cntg() != 0) {
						PERFLIBS_ASSERT(btm_c.is_physical(), "C should be physical to compute full block");
						next_full_(btm_a, b, btm_c,
							advance(pos, rght_cntg_first_fix, 0, 0),
							std::forward<Args>(args)...);
					}
				}
			}
		}
		//is upper
		else if(left_cntg_first == 0
		     && left_cntg_first == 0) {
			//split along strd
			if(btm_strd_first > left_cntg_last) {
				/*    strd
				 *   +--+ +-------------+
				 * c |  | |             |
				 * n \  | |             |
				 * t  \ | |             |
				 * g   \| |             |
				 *      v +-------------+
				 */

				const auto is = interleave_factor(b) * b_strd_unroll_;
				const auto btm_strd_first_fix = min(iround(btm_strd_first, is), b.strd());

				{
					auto left_c = get_strd_panel_with_clamp(c, 0, btm_strd_first_fix);
					auto left_b = get_strd_panel(b, 0, btm_strd_first_fix);

					PERFLIBS_ASSERT(! left_c.is_physical(), "C should not be physical to compute partial block");

					next_part_(a, left_b, left_c, pos, std::forward<Args>(args)...);
				}
				{
					auto rght_c = get_strd_panel_with_clamp(c, btm_strd_first_fix, c.strd() - btm_strd_first_fix);
					auto rght_b = get_strd_panel(b, btm_strd_first_fix, c.strd() - btm_strd_first_fix);


					if(rght_c.strd() != 0) {
						PERFLIBS_ASSERT( rght_c.is_physical(), "C should be physical to compute full block");
						next_full_(a, rght_b, rght_c,
							advance(pos, 0, btm_strd_first_fix, 0),
							std::forward<Args>(args)...);
					}
				}
			}
			//split along cntg
			else {
				/*    strd
				 *   +--------+
				 *   |        |
				 * c |        |
				 * n |        |
				 * t |        |
				 * g +--------+
				 *   ---------+
				 *   \        |
				 *    \       |
				 *     -------+
				 */

				const auto is = interleave_factor(a) * a_strd_unroll_;
				const auto left_cntg_last_fix = max(iround(left_cntg_last, is), 0);

				{
					auto top_c = get_cntg_panel_with_clamp(c, 0, left_cntg_last_fix);
					auto top_a = get_strd_panel(a, 0, left_cntg_last_fix);

					if(top_c.cntg() != 0) {
						PERFLIBS_ASSERT( top_c.is_physical(), "C should be physical to compute full block");
						next_full_(top_a, b, top_c, pos, std::forward<Args>(args)...);
					}
				}
				{
					auto btm_c = get_cntg_panel_with_clamp(c, left_cntg_last_fix, c.cntg() - left_cntg_last_fix);
					auto btm_a = get_strd_panel(a, left_cntg_last_fix, c.cntg() - left_cntg_last_fix);

					PERFLIBS_ASSERT( ! btm_c.is_physical(), "C should not be physical to compute partial block");

					next_part_(btm_a, b, btm_c,
						advance(pos, left_cntg_last_fix, 0, 0),
						std::forward<Args>(args)...);
				}
			}
		}
		else {
			PERFLIBS_ASSERT(false, "unknown separation strategy");
		}
	}
}; //class partial_separator

} //namespace perflibs::linalg

#endif //PARTIAL_SEPARATOR_HPP

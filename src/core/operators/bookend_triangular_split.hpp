/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_OPERATORS_BOOKEND_TRIANGULAR_SPLIT_HPP
#define PERFLIBS_LINALG_OPERATORS_BOOKEND_TRIANGULAR_SPLIT_HPP

#include "framework/2d_split.hpp"
#include "framework/linalg_util.hpp"
#include "framework/parallel.hpp"
#include "framework/compute_position.hpp"

namespace perflibs::linalg {

/*
 * Bookend triangular split is a 2D parallelism system for matrices with TriangularForm.
 *
 * It works by splitting the threads across two dimension, forming nteams of nthreads.
 * where nteams * nthreads <= max_threads
 *
 * One of the dimensions is then split in to 2xnteams panels of approx even size
 *
 * Each team of threads is then given two of those panels to process; the first team is given
 * the largest and the smallest panel. The second team is given the 2nd largest
 * and 2nd smallest columns. The last thread is given the central most 2 panels.
 *
 * This means each team has the same amount of work to do. The two panels are denoted
 * as the right and left hand side.
 *
 * The "other" dimension of these panels is then added together as though it was one
 * long panel. And this space is then split across the threads of that team.
 *
 * Note: This means that 1 or more threads will have a section of the lhs and rhs panel to
 * compute
 */
template<typename Next>
class bookend_triangular_split {
	kernel_inttype max_threads_;
	kernel_inttype a_strd_unroll_;
	kernel_inttype b_strd_unroll_;
	Next next_;

public:
	PERFLIBS_LINALG_INLINE
	bookend_triangular_split(kernel_inttype max_threads, kernel_inttype a_strd_unroll, kernel_inttype b_strd_unroll, Next next)
	:	max_threads_     { max_threads     }
	,	a_strd_unroll_   { a_strd_unroll   }
	,	b_strd_unroll_   { b_strd_unroll   }
	,	next_            { std::move(next) }
	{	}

	PERFLIBS_LINALG_INLINE
	void operator() (const auto& a, const auto& b, const auto& c, compute_position pos, auto... args) {
		//first returned value is teams_in_a_strd, but we'll recalculate that using the adjusted version of teams_in_b_strd
		const auto [ _, teams_in_b_strd ]                 = split_2d(max_threads_, a.strd(), b.strd());

		//as each team in b_strd needs two panels, we can't have more than b.strd/2 threads
		const auto teams_in_b_strd_adj                    = min(b.strd() / 2_ki, teams_in_b_strd);

		const auto b_strd_split                           = make_parallel_split(b.strd(), b_strd_unroll_, teams_in_b_strd_adj * 2_ki);

		const auto teams_in_a_strd                        = max_threads_ / teams_in_b_strd_adj;

		const auto b_strd_thread_id                       = pos.thread_num / teams_in_a_strd;
		const auto a_strd_thread_id                       = pos.thread_num % teams_in_a_strd;

		if(b_strd_thread_id * 2_ki  >= b_strd_split.threads)
			return;

		const auto [ lhs_b_strd_first, lhs_b_strd_size ]  = work_distribution(b_strd_thread_id,                            b_strd_split);
		const auto [ rhs_b_strd_first, rhs_b_strd_size ]  = work_distribution(b_strd_split.threads - b_strd_thread_id - 1, b_strd_split);

		const auto c_lhs                                  = get_strd_panel(c, lhs_b_strd_first, lhs_b_strd_size);
		const auto c_rhs                                  = get_strd_panel(c, rhs_b_strd_first, rhs_b_strd_size);

		const auto c_lhs_high_strd                        = c.is_lower() ? 0_ki : c_lhs.strd() - 1_ki;

		const auto [ c_lhs_cntg_first, c_lhs_cntg_last ]  = get_non_virtual_cntg_bounds_for_strd(c_lhs, c_lhs_high_strd);
		const auto [ c_rhs_cntg_first, c_rhs_cntg_last ]  = get_non_virtual_cntg_bounds_for_strd(c_rhs, c_lhs_high_strd);

		//Future optimization: It should be possible to remove this cropping step
		const auto c_lhs_crop                             = get_cntg_panel(c_lhs, c_lhs_cntg_first, c_lhs_cntg_last - c_lhs_cntg_first);
		const auto c_rhs_crop                             = get_cntg_panel(c_rhs, c_rhs_cntg_first, c_rhs_cntg_last - c_rhs_cntg_first);

		const auto cntg_total                             = c_lhs_crop.cntg() + c_rhs_crop.cntg();
		const auto a_strd_split                           = make_parallel_split(cntg_total, a_strd_unroll_, teams_in_a_strd);

		if(a_strd_thread_id >= a_strd_split.threads)
			return;

		const auto [ a_strd_first, a_strd_size ]          = work_distribution(a_strd_thread_id, a_strd_split);

		const auto lhs_a_strd_first                       = min(a_strd_first, c_lhs_crop.cntg());
		const auto lhs_a_strd_last                        = min(lhs_a_strd_first + a_strd_size, c_lhs_crop.cntg());
		const auto lhs_a_strd_size                        = lhs_a_strd_last - lhs_a_strd_first;

		const auto rhs_a_strd_first                       = max(a_strd_first - c_lhs_crop.cntg(), 0_ki);
		const auto rhs_a_strd_size                        = a_strd_size - lhs_a_strd_size;

		const auto c_rhs_thread                           = get_cntg_panel(c_rhs_crop, rhs_a_strd_first, rhs_a_strd_size);
		const auto c_lhs_thread                           = get_cntg_panel(c_lhs_crop, lhs_a_strd_first, lhs_a_strd_size);

		auto invoke_next_advance_ab = [&](const auto& c_block) mutable {
			const auto abs_a_strd = c_block.absolute_cntg() - c.absolute_cntg();
			const auto abs_b_strd = c_block.absolute_strd() - c.absolute_strd();

			next_(
				get_strd_panel(a, abs_a_strd, c_block.cntg()),
				get_strd_panel(b, abs_b_strd, c_block.strd()),
				c_block,
				advance(pos, abs_a_strd, abs_b_strd, 0_ki),
				args...
			);
		};

		invoke_next_advance_ab(c_lhs_thread);
		invoke_next_advance_ab(c_rhs_thread);
	}
}; // class bookend_triangular_split

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_OPERATORS_BOOKEND_TRIANGULAR_SPLIT_HPP

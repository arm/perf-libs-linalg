/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_OPERATORS_RESIDENTS_HPP
#define PERFLIBS_LINALG_OPERATORS_RESIDENTS_HPP

#include "matrix/matrix.hpp"

#include "framework/which.hpp"
#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"

#include <utility>

namespace perflibs::linalg {

/**
 * Loop over blocks the dimensions of one of the three input matrices,
 *  The idea being that matrix will then be blocked into a level of the cache topology.
 *
 * The next level in the stack in invoked with a block of one matrix and panel of the other two
 *
 * Which matrix is blocked over is specified by the 'WhichMatrix' template parameter,
 *  and the dimensions of the blocks and the order of the loops are specified to the constructor at runtime.
 */
template <which_matrix WhichMatrix, typename Next>
class resident {
	/// the next step in the chain, this could be the kernel, or a further blocking algorithm
	bool is_neg_cntg_block_size_;
	bool is_neg_strd_block_size_;
	kernel_inttype cntg_block_size_;
	kernel_inttype strd_block_size_;
	bool cntg_first_;
	Next next_;
public:
	/**
	 * @param cntg_block_size the block dimension used in the contiguous dimension
	 * @param strd_block_size the block dimension used in the strided dimension
	 * @param cntg_first if true, the cntg blocking is the outer loop
	 * @param next - the next step in the stack
	 */
	PERFLIBS_LINALG_INLINE
	resident(kernel_inttype cntg_block_size, kernel_inttype strd_block_size, bool cntg_first, Next next)
	:	is_neg_cntg_block_size_ {cntg_block_size < 0 }
	,	is_neg_strd_block_size_ {strd_block_size < 0 }
	,	cntg_block_size_        { is_neg_cntg_block_size_ ? -cntg_block_size : cntg_block_size }
	,	strd_block_size_        { is_neg_strd_block_size_ ? -strd_block_size : strd_block_size }
	,	cntg_first_             { cntg_first      }
	,	next_                   { std::move(next) }
	{	}

	PERFLIBS_LINALG_INLINE
	resident(which_matrix_constant<WhichMatrix>, kernel_inttype cntg_block_size, kernel_inttype strd_block_size, bool cntg_first, Next next)
	: resident(cntg_block_size, strd_block_size, cntg_first, std::move(next)) { }

	PERFLIBS_LINALG_INLINE
	void operator()(const auto& a, const auto& b, const auto& c, compute_position pos, const auto&... args) {
		if constexpr (WhichMatrix == which_matrix::a) {
			const kernel_inttype cntg_size = a.cntg();
			const kernel_inttype strd_size = a.strd();

			if(empty(a)) {
				return;
			}
			else if(cntg_size <= cntg_block_size_ && strd_size <= strd_block_size_) {
				//if only one iteration, don't bother calling sub_matrix
				next_(a, b, c, pos, args...);
			}
			/*
			 * Cntg first dictates whether we iterate over the cntg or strd dimension first
			 */
			else if (cntg_first_) {
				for (kernel_inttype j = 0; j < cntg_size; j += cntg_block_size_) {
					const kernel_inttype cntg_block_size = min(cntg_size - j, cntg_block_size_);

					const kernel_inttype cntg_block_start = is_neg_cntg_block_size_
					                                      ? ( cntg_size - cntg_block_size - j )
					                                      : j;

					for (kernel_inttype i = 0; i < strd_size; i += strd_block_size_) {
						const kernel_inttype strd_block_size = min(strd_size - i, strd_block_size_);

						const kernel_inttype strd_block_start = is_neg_strd_block_size_
						                                      ? ( strd_size - strd_block_size - i )
						                                      : i;

						auto a_block = a.sub_matrix(cntg_block_start, cntg_block_size, strd_block_start, strd_block_size);
						auto b_panel = get_cntg_panel(b, cntg_block_start, cntg_block_size);
						auto c_panel = get_cntg_panel(c, strd_block_start, strd_block_size);

						next_(a_block, b_panel, c_panel, advance(pos, i, 0, j), args...);
					}
				}
			}
			else {
				for (kernel_inttype i = 0; i < strd_size; i += strd_block_size_) {
					const kernel_inttype strd_block_size = min(strd_size - i, strd_block_size_);

					const kernel_inttype strd_block_start = is_neg_strd_block_size_
					                                      ? ( strd_size - strd_block_size - i )
					                                      : i;

					for (kernel_inttype j = 0; j < cntg_size; j += cntg_block_size_) {
						const kernel_inttype cntg_block_size = min(cntg_size - j, cntg_block_size_);

						const kernel_inttype cntg_block_start = is_neg_cntg_block_size_
						                                      ? ( cntg_size - cntg_block_size - j )
						                                      : j;

						auto a_block = a.sub_matrix(cntg_block_start, cntg_block_size, strd_block_start, strd_block_size);
						auto b_panel = get_cntg_panel(b, cntg_block_start, cntg_block_size);
						auto c_panel = get_cntg_panel(c, strd_block_start, strd_block_size);

						next_(a_block, b_panel, c_panel, advance(pos, i, 0, j), args...);
					}
				}
			}
		}
		else if constexpr (WhichMatrix == which_matrix::b) {
			const kernel_inttype cntg_size = b.cntg();
			const kernel_inttype strd_size = b.strd();

			if(empty(b)) {
				return;
			}
			else if(cntg_size <= cntg_block_size_ && strd_size <= strd_block_size_) {
				next_(a, b, c, pos, args...);
			}
			/*
			 * Cntg first dictates whether we iterate over the cntg or strd dimension first
			 */
			else if (cntg_first_) {
				for (kernel_inttype j = 0; j < cntg_size; j += cntg_block_size_) {
					const kernel_inttype cntg_block_size = min(cntg_size - j, cntg_block_size_);

					const kernel_inttype cntg_block_start = is_neg_cntg_block_size_
					                                      ? ( cntg_size - cntg_block_size - j )
					                                      : j;

					for (kernel_inttype i = 0; i < strd_size; i += strd_block_size_) {
						const kernel_inttype strd_block_size = min(strd_size - i, strd_block_size_);

						const kernel_inttype strd_block_start = is_neg_strd_block_size_
															  ? ( strd_size - strd_block_size - i )
															  : i;

						auto a_panel = get_cntg_panel(a, cntg_block_start, cntg_block_size);
						auto b_block = b.sub_matrix(cntg_block_start, cntg_block_size, strd_block_start, strd_block_size);
						auto c_panel = get_strd_panel(c, strd_block_start, strd_block_size);

						next_(a_panel, b_block, c_panel, advance(pos, 0, i, j), args...);
					}
				}
			}
			else {
				for (kernel_inttype i = 0; i < strd_size; i += strd_block_size_) {
					const kernel_inttype strd_block_size = min(strd_size - i, strd_block_size_);

					const kernel_inttype strd_block_start = is_neg_strd_block_size_
														  ? ( strd_size - strd_block_size - i )
														  : i;

					for (kernel_inttype j = 0; j < cntg_size; j += cntg_block_size_) {
						const kernel_inttype cntg_block_size = min(cntg_size - j, cntg_block_size_);

						const kernel_inttype cntg_block_start = is_neg_cntg_block_size_
						                                      ? ( cntg_size - cntg_block_size - j )
						                                      : j;

						auto a_panel = get_cntg_panel(a, cntg_block_start, cntg_block_size);
						auto b_block = b.sub_matrix(cntg_block_start, cntg_block_size, strd_block_start, strd_block_size);
						auto c_panel = get_strd_panel(c, strd_block_start, strd_block_size);

						next_(a_panel, b_block, c_panel, advance(pos, 0, i, j), args...);
					}
				}
			}
		}
		else if constexpr (WhichMatrix == which_matrix::c) {
			/*
			 * Resident C is currently unsupported but is viable blocking form
			 */
			PERFLIBS_ASSERT(false, "Resident C currently unsupported");
		}
	}
}; // class resident

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_OPERATORS_RESIDENTS_HPP

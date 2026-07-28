/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_CROP_HPP
#define PERFLIBS_LINALG_CROP_HPP

#include "perflibs_numeric_utils.hpp"
#include "perflibs_util.hpp"
#include "framework/compute_position.hpp"

namespace perflibs::linalg {
namespace {

/**
 * This tool fixes the sizees of A and B so that they are just big enough for the amount of work actually required for them
 */
template<typename Next>
class crop {
	kernel_inttype a_strd_unroll_;
	kernel_inttype b_strd_unroll_;
	Next next_;
public:
	crop(kernel_inttype a_strd_unroll, kernel_inttype b_strd_unroll, Next next)
	:	a_strd_unroll_ { a_strd_unroll   }
	,	b_strd_unroll_ { b_strd_unroll   }
	,	next_          { std::move(next) }
	{	}

	/**
	 * LINALG stack operator
	 */
	PERFLIBS_LINALG_INLINE
	void operator()(const auto& a, const auto& b, const auto& c, const compute_position& pos, auto&&... args) {
		//get the dimensions of C which actually contain something useful
		const auto [c_cntg_first, c_cntg_last] = get_non_virtual_cntg_bounds_for_strd(c);
		const auto [c_strd_first, c_strd_last] = get_non_virtual_strd_bounds_for_cntg(c);

		//if the matrices are packed, by what factor, we need to ensure we are cropping
		//to workable numbers
		const auto c_cntg_unroll = interleave_factor(a) * a_strd_unroll_;
		const auto c_strd_unroll = interleave_factor(b) * b_strd_unroll_;

		const auto c_cntg_first_flr = max(iround_floor(c_cntg_first, c_cntg_unroll), 0);
		const auto c_cntg_last_ceil = min(iround      (c_cntg_last, c_cntg_unroll),  c.cntg());

		const auto c_strd_first_flr = max(iround_floor(c_strd_first, c_strd_unroll), 0);
		const auto c_strd_last_ceil = min(iround      (c_strd_last, c_strd_unroll),  c.strd());

		const auto c_cntg_size = c_cntg_last_ceil - c_cntg_first_flr;
		const auto c_strd_size = c_strd_last_ceil - c_strd_first_flr;

		if(empty(c)) {
			return;
		}
		if(c_cntg_first_flr == 0 && c_cntg_size == c.cntg()
		&& c_strd_first_flr == 0 && c_strd_size == c.strd()) {
			next_(a, b, c,
				advance(pos, c_cntg_first_flr, c_strd_first_flr, 0),
				std::forward<decltype(args)>(args)...);
		}
		else {
			auto a_panel = get_strd_panel(a, c_cntg_first_flr, c_cntg_size);
			auto b_panel = get_strd_panel(b, c_strd_first_flr, c_strd_size);

			auto c_block = c.sub_matrix(
				c_cntg_first_flr, c_cntg_size,
				c_strd_first_flr, c_strd_size);

			next_(a_panel, b_panel, c_block,
				advance(pos, c_cntg_first_flr, c_strd_first_flr, 0),
				std::forward<decltype(args)>(args)...);
		}
	}
}; // class crop

} // namespace anon
} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_CROP_HPP

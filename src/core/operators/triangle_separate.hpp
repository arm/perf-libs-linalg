/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_TRIANGLE_SEPARATE_HPP
#define PERFLIBS_LINALG_TRIANGLE_SEPARATE_HPP

#include "operators/paneller.hpp"
#include "framework/compute_position.hpp"
#include "framework/which.hpp"
#include "matrix/operations.hpp"

#include "perflibs_unused.hpp"
#include "perflibs_util.hpp"

namespace perflibs::linalg {

template<which_matrix WhichMatrix, typename AMatType, typename BMatType, typename CMatType>
PERFLIBS_LINALG_INLINE
kernel_inttype get_cntg(const AMatType& a, const BMatType& b, const CMatType& c) {
	if constexpr(WhichMatrix == which_matrix::a) {
		return a.cntg();
	}
	else if constexpr(WhichMatrix == which_matrix::b) {
		return b.cntg();
	}
	else {
		return c.cntg();
	}
}

template<which_matrix WhichMatrix, typename AMatType, typename BMatType, typename CMatType>
PERFLIBS_LINALG_INLINE
kernel_inttype get_strd(const AMatType& a, const BMatType& b, const CMatType& c) {
	if constexpr(WhichMatrix == which_matrix::a) {
		return a.strd();
	}
	else if constexpr(WhichMatrix == which_matrix::b) {
		return b.strd();
	}
	else {
		return c.strd();
	}
}

template<which_matrix WhichMatrix, typename AMatType, typename BMatType, typename CMatType>
PERFLIBS_LINALG_INLINE
bool is_upper(const AMatType& a, const BMatType& b, const CMatType& c) {
	if constexpr(WhichMatrix == which_matrix::a) {
		return a.is_upper();
	}
	else if constexpr(WhichMatrix == which_matrix::b) {
		return b.is_upper();
	}
	else {
		return c.is_upper();
	}
}

template<which_matrix WhichMatrix, which_dimension WhichDimension, typename AMatType, typename BMatType, typename CMatType>
PERFLIBS_LINALG_INLINE
std::pair<kernel_inttype, kernel_inttype> get_non_virtual_bounds(const AMatType& a, const BMatType& b, const CMatType& c, kernel_inttype pos) {
	if constexpr(WhichMatrix == which_matrix::a) {
		if constexpr(WhichDimension == which_dimension::cntg)
			return get_non_virtual_strd_bounds_for_cntg(a, pos);

		return get_non_virtual_cntg_bounds_for_strd(a, pos);
	}

	if constexpr(WhichMatrix == which_matrix::b) {
		if constexpr(WhichDimension == which_dimension::cntg)
			return get_non_virtual_strd_bounds_for_cntg(b, pos);

		return get_non_virtual_cntg_bounds_for_strd(b, pos);
	}

	//Matrix C
	if constexpr(WhichDimension == which_dimension::a_strd)
		return get_non_virtual_strd_bounds_for_cntg(c, pos);

	//b_strd
	return get_non_virtual_cntg_bounds_for_strd(c, pos);
}


/**
 * This tool fixes the sizees of A and B so that they are just big enough for the amount of work actually required for them
 */
template<which_matrix WhichMatrix, typename NextTri, typename NextRect>
class triangle_separate {
	NextTri  next_tri_;
	NextRect next_rect_;
public:
	PERFLIBS_LINALG_INLINE
	triangle_separate(NextTri next_tri, NextRect next_rect)
	:	next_tri_  { std::move(next_tri) }
	,	next_rect_ { std::move(next_rect) }
	{	}

	PERFLIBS_LINALG_INLINE
	triangle_separate(which_matrix_constant<WhichMatrix>, NextTri next_tri, NextRect next_rect)
	:	triangle_separate { std::move(next_tri), std::move(next_rect) }
	{	}

	/**
	 * LINALG stack operator
	 */
	template<typename AType, typename BType, typename CMatType, typename... Args>
	PERFLIBS_LINALG_INLINE
	void operator()(AType& a, BType& b, CMatType& c, const compute_position& pos, Args&&... args) {

		constexpr which_dimension split_dimension =
			corresponding_dimension_v<WhichMatrix, which_dimension::cntg>;

		paneller<split_dimension, NextTri> tri_pan   { next_tri_ };
		paneller<split_dimension, NextRect> rect_pan { next_rect_ };

		const kernel_inttype tri_cntg = get_cntg<WhichMatrix>(a, b, c);
		const kernel_inttype tri_strd = get_strd<WhichMatrix>(a, b, c);

		if(is_upper<WhichMatrix>(a, b, c)) {
			const auto [ tri_first, rect_last ] =
				get_non_virtual_bounds<WhichMatrix, which_dimension::cntg>(a, b, c, 0);

			const kernel_inttype tri_last = min(tri_first + tri_cntg, tri_strd);

			tri_pan (a, b, c, tri_first, tri_last - tri_first, pos, 0, 0, args...);
			rect_pan(a, b, c, tri_last,  rect_last - tri_last, pos, 0, 0, args...);
		}
		else {
			auto [ rect_first, tri_first ] =
				get_non_virtual_bounds<WhichMatrix, which_dimension::cntg>(a, b, c, 0);

			PERFLIBS_UNUSED(rect_first);

			--tri_first;

			const kernel_inttype tri_last = min(tri_first + tri_cntg, tri_strd);

			tri_pan (a, b, c, tri_first, tri_last - tri_first, pos, 0, 0, args...);
			rect_pan(a, b, c, 0,         tri_first,            pos, 0, 0, args...);
		}
	}
}; //class triangle_separate

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_TRIANGLE_SEPARATE_HPP

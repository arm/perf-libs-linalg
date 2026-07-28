/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_OPERATORS_REFLECT_AND_TRANSPOSE_HPP
#define PERFLIBS_LINALG_OPERATORS_REFLECT_AND_TRANSPOSE_HPP

namespace perflibs::linalg {

template<typename Next>
struct reflect_and_transpose {
	Next next_;

	reflect_and_transpose(Next next)
	:	next_{ next }
	{	}

	template<typename AType, typename BType, typename CType, typename... Args>
	PERFLIBS_LINALG_INLINE
	void operator()(const AType& a, const BType& b, const CType& c, compute_position pos, Args&&... args) {
		auto a_reflect = a.reflect().reflect_transpose();
		auto b_adjust  = b.get_parent().sub_matrix(a_reflect.absolute_cntg(), a_reflect.cntg(), b.absolute_strd(), b.strd());
		auto c_adjust  = c.get_parent().sub_matrix(a_reflect.absolute_strd(), a_reflect.strd(), c.absolute_strd(), c.strd());

		next_(a_reflect, b_adjust, c_adjust, pos, args...);
	}
}; // reflect_and_transpose

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_OPERATORS_REFLECT_AND_TRANSPOSE_HPP

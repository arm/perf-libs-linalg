/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_AUTO_RESIDENTS_HPP
#define PERFLIBS_LINALG_AUTO_RESIDENTS_HPP

#include "perflibs_assert.hpp"
#include "perflibs_numeric_utils.hpp"
#include "perflibs_util.hpp"
#include "framework/which.hpp"
#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

template<which_dimension Dimension, typename Next>
class auto_resident {
	kernel_inttype target_size_;
	Next           next_;
public:
	PERFLIBS_LINALG_INLINE
	auto_resident(kernel_inttype target_size, Next next)
	:	target_size_ { target_size     }
	,	next_        { std::move(next) }
	{	}

	PERFLIBS_LINALG_INLINE
	auto_resident(which_dimension_constant<Dimension>, kernel_inttype target_size, Next next)
	:	auto_resident { target_size, std::move(next) }
	{	}

	template <typename AType, typename BType, typename CType, typename... Args>
	PERFLIBS_LINALG_INLINE
	void operator()(AType &a, BType &b, CType &c, compute_position pos, Args... args) {
		using value_type = typename AType::value_type;

		const auto target_elems = iround_div(target_size_, sizeof(value_type));

		if constexpr(Dimension == which_dimension::a_strd) {
			const kernel_inttype ideal_strd = iround_div(target_elems, a.cntg());

			auto r = resident<which_matrix::a, Next>{ a.cntg(), ideal_strd, true, next_ };
			r(a, b, c, pos, args...);
		}
		else {
			PERFLIBS_ASSERT(false, "not implemented");
		}
	}
}; //class auto_resident

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_AUTO_RESIDENTS_HPP

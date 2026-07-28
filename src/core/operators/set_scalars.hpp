/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SET_SCALARS_HPP
#define PERFLIBS_LINALG_SET_SCALARS_HPP

#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

template<typename Scalar, typename Next>
class set_scalar {
	Scalar scalar_;
	bool apply_;
	Next next_;
public:
	PERFLIBS_LINALG_INLINE
	set_scalar(Scalar scalar, bool apply, Next next)
	:	scalar_ { scalar }
	,	apply_  { apply  }
	,	next_   { next   }
	{	}

	/**
	 * LINALG stack operator
	 */
	template<typename AType, typename BType, typename CMatType, typename ScalarType, typename... Args>
	PERFLIBS_LINALG_INLINE
	void operator()(AType& a, BType& b, CMatType& c, compute_position pos, ScalarType alpha, ScalarType beta, Args... args) {
		if (apply_) {
			pos.cntg=0;
			next_(a, b, c, pos, alpha, scalar_, std::forward<Args>(args)...);
		}
		else {
			next_(a, b, c, pos, alpha, beta   , std::forward<Args>(args)...);
		}
	}
}; //class set_scalar

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_SET_SCALARS_HPP

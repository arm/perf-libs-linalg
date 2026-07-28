/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_CONJ_ALPHA_HPP
#define PERFLIBS_LINALG_CONJ_ALPHA_HPP

#include "perflibs_complex.hpp"
#include "framework/compute_position.hpp"

namespace perflibs::linalg {

/**
 * Operator which takes the conjugate of alpha. This is useful for
 * routines such as HER2K which require both alpha and its conjugate.
 */
template<typename Next>
class conj_alpha {

	Next next_;

public:

	conj_alpha(Next next)
	:	next_ { std::move(next) }
	{	}

	template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename Scalar, typename... Args>
	void operator()(MatrixTypeA& a, MatrixTypeB& b, MatrixTypeC& c, compute_position pos, Scalar alpha, Args&&... args) {
		next_(a, b, c, pos, perflibs::conj(alpha), std::forward<Args>(args)...);
	}
};

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_CONJ_ALPHA_HPP

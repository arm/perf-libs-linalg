/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_IF_THEN_ELSE_HPP
#define PERFLIBS_LINALG_IF_THEN_ELSE_HPP

#include "framework/compute_position.hpp"
#include "matrix/type_traits.hpp"

namespace perflibs::linalg {

/**
 * LINALG stack operator for encoding conditional execution into the stack. Example usage:
 *
 * auto driver = ...
 *     if_then_else(
 *        [](auto a, auto b, auto c, compute_position pos, auto... args) {
 *            return pos.block == 0;
 *        },
 *        if_true (
 *            ...
 *        ),
 *        if_false (
 *            ...
 *        ),
 *     )
 */
template<typename Predicate, typename NextTrueBranch, typename NextFalseBranch>
class if_then_else {

	/**
	 * A predicate function. This function should accept all of the parameters that are passed down the LINALG
	 * stack, and should return a boolean. If the function returns true, we continue our execution via
	 * `next_true_branch_`. Otherwise, we continue our execution via `next_false_branch_`.
	 */
	Predicate predicate_;

	/**
	 * The LINALG stack to execute if the predicate returns true.
	 */
	NextTrueBranch next_true_branch_;

	/**
	 * The LINALG stack to execute if the predicate returns false.
	 */
	NextFalseBranch next_false_branch_;

public:
	if_then_else(Predicate predicate, NextTrueBranch next_true_branch, NextFalseBranch next_false_branch)
	:	predicate_        { predicate }
	,	next_true_branch_ { next_true_branch }
	,	next_false_branch_{ next_false_branch }
	{	}

	template<typename... Args>
	void operator()(const Args&... args) {
		if (predicate_(args...))
			next_true_branch_(args...);
		else
			next_false_branch_(args...);
	}
}; // if_then_else

} // namespace perflibs::linalg

#endif /* ifndef PERFLIBS_LINALG_IF_THEN_ELSE_HPP */

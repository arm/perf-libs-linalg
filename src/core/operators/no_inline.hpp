/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_NO_INLINE_HPP
#define PERFLIBS_LINALG_NO_INLINE_HPP

#include <utility>

namespace perflibs::linalg {
namespace {

/**
 * Prevents a stack of operators being inlined into the callee.
 *
 * This operator is useful when trying to find the right compromise between compile times and performance.
 * It helps partition your code into logical units for the optimizer, reducing the amount of work it needs to
 * do, but also helping it make better decisions about inlining down the stack.
 */
template<typename Next>
struct no_inline {
	Next next_;

	no_inline(Next next) : next_{ next } {};

	template<typename... Args>
	__attribute__((noinline)) void operator()(Args&&...args) {
		next_(std::forward<Args>(args)...);
	}
};
} // namespace
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_NO_INLINE_HPP

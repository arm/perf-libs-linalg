/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_PACKAGE_BATCH_INTERLEAVE_SOLVE_STRATEGIES_HPP
#define PERFLIBS_LINALG_PACKAGE_BATCH_INTERLEAVE_SOLVE_STRATEGIES_HPP

#include "framework/compute.hpp"
#include "packages/batch_interleave/solve/problem_context_bases.hpp"
#include "packages/batch_interleave/solve/strategies/batch_interleave_solve_kernel.hpp"

namespace perflibs::linalg {

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<
	batch_interleave::batch_problem<batch_interleave::solve::triangular_solve<T...>>,
	ArchitectureSpec
>> = std::tuple {
	batch_interleave::solve::batch_interleave_solve_kernel{}
};

namespace batch_interleave::solve {

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<
	batch_problem<triangular_solve<T...>>,
	ArchitectureSpec
>& pctx) {
	return compute_impl(pctx);
}

} // namespace batch_interleave::solve

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_PACKAGE_BATCH_INTERLEAVE_SOLVE_STRATEGIES_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_SOLVE_FWD_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_SOLVE_FWD_HPP

#include "framework/compute.hpp"
#include "packages/batch_interleave/solve/problem_context_bases.hpp"
#include "spec/problem_context.hpp"

// DO NOT include strategies.hpp or anything from the strategies dir

namespace perflibs::linalg::batch_interleave::solve {

class batch_interleave_solve_kernel;

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<
	batch_problem<triangular_solve<T...>>,
	ArchitectureSpec
>& pctx);

} // namespace perflibs::linalg::batch_interleave::solve

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_SOLVE_FWD_HPP

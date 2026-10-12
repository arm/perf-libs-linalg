/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_FWD_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_FWD_HPP

#include "packages/batch_interleave/matmul/problem_context_bases.hpp"
#include "spec/problem_context.hpp"
#include "framework/compute.hpp"

// DO NOT include strategies.hpp or anything from the strategies dir

namespace perflibs::linalg::batch_interleave::matmul {

class batch_interleave_matmul2_kernel;
class batch_interleave_matmul3_naive_inner_product;
class batch_interleave_matmul3_kernel;

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<
	batch_problem<::perflibs::linalg::matmul::matmul2<T...>>,
	ArchitectureSpec
>& pctx);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<
	batch_problem<::perflibs::linalg::matmul::matmul3<T...>>,
	ArchitectureSpec
>& pctx);

} // namespace perflibs::linalg::batch_interleave::matmul

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_FWD_HPP

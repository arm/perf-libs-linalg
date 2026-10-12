/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_PACKAGE_BATCH_INTERLEAVE_MATMUL_STRATEGIES_HPP
#define PERFLIBS_LINALG_PACKAGE_BATCH_INTERLEAVE_MATMUL_STRATEGIES_HPP

#include "framework/compute.hpp"

#include "packages/batch_interleave/matmul/problem_context_bases.hpp"
#include "packages/batch_interleave/matmul/strategies/batch_interleave_matmul2_kernel.hpp"
#include "packages/batch_interleave/matmul/strategies/batch_interleave_matmul3_kernel.hpp"
#include "packages/batch_interleave/matmul/strategies/batch_interleave_matmul3_naive_inner_product.hpp"

namespace perflibs::linalg {

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<
	batch_interleave::batch_problem<matmul::matmul2<T...>>,
	ArchitectureSpec
>> = std::tuple {
	batch_interleave::matmul::batch_interleave_matmul2_kernel{}
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<
	batch_interleave::batch_problem<matmul::matmul3<T...>>,
	ArchitectureSpec
>> = std::tuple {
	batch_interleave::matmul::batch_interleave_matmul3_kernel{},
	batch_interleave::matmul::batch_interleave_matmul3_naive_inner_product{}
};

namespace batch_interleave::matmul {

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<
	batch_problem<::perflibs::linalg::matmul::matmul2<T...>>,
	ArchitectureSpec
>& pctx) {
	return compute_impl(pctx);
}

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<
	batch_problem<::perflibs::linalg::matmul::matmul3<T...>>,
	ArchitectureSpec
>& pctx) {
	return compute_impl(pctx);
}

} // namespace batch_interleave::matmul

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_PACKAGE_BATCH_INTERLEAVE_MATMUL_STRATEGIES_HPP

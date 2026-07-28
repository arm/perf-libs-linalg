/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_PACKAGE_SOLVE_STRATEGIES_HPP
#define PERFLIBS_LINALG_PACKAGE_SOLVE_STRATEGIES_HPP

#include "framework/compute.hpp"

#include "packages/solve/fwd.hpp"
#include "packages/solve/problem_context_bases.hpp"
#include "packages/solve/strategies/compressed_triangular_solve_vector.hpp"
#include "packages/solve/strategies/triangular_solve_matrix_large.hpp"
#include "packages/solve/strategies/triangular_solve_matrix_small.hpp"
#include "packages/solve/strategies/triangular_solve_matrix_reference.hpp"
#include "packages/solve/strategies/triangular_solve_vector.hpp"
#include "packages/solve/strategies/triangular_solve_vector_reference.hpp"

namespace perflibs::linalg {

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<solve::solve<T...>, ArchitectureSpec>> = std::tuple {
	solve::compressed_triangular_solve_vector{},
	solve::triangular_solve_vector{},
	solve::triangular_solve_vector_reference{},
	solve::triangular_solve_matrix_small{},
	solve::triangular_solve_matrix_large{},
	solve::triangular_solve_matrix_reference{},
};

namespace solve {

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<solve<T...>, ArchitectureSpec>& pctx) {
	return compute_impl(pctx);
}

} // namespace solve

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_PACKAGE_SOLVE_STRATEGIES_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_MATRIX_SMALL_HPP
#define PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_MATRIX_SMALL_HPP

#include "packages/solve/problem_context_bases.hpp"
#include "packages/solve/strategies/strategy_helper.hpp"

#include "spec/problem_context.hpp"
#include "spec/strategy_tag.hpp"

#include "operators/gemm_exec.hpp"
#include "operators/triangular_solve_resident.hpp"

#include <string_view>

namespace perflibs::linalg::solve {

class triangular_solve_matrix_large;

class triangular_solve_matrix_small {
public:
	static constexpr std::string_view name() { return "triangular_solve_matrix_small"; }

	// Actual computation. this function will be called only if
	// the triangular_solve_matrix_small strategy is valid to solve
	// the problem described by the ProblemContext
	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<triangular_solve_matrix_large>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		using architecture_spec_type = typename ProblemContext::architecture_spec_type;

		if( !this->can_compute(pctx) ) return false;

		// As we only get the kernel from the spec, we use triangular_solve_matrix_large at the moment
		const auto spec = get_spec(spec::strategy_tag<triangular_solve_matrix_large>{}, pctx);

		const bool is_lside = pctx.b.cntg_step() == 1;

		scale(pctx.alpha, pctx.b);
		auto driver = triangular_solve_resident{ 4, is_lside, gemm_exec<architecture_spec_type>{},
		                                         trsm_kernel_exec{ spec.kernel } };

		driver(pctx.a, pctx.b);

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<triangular_solve_matrix_large>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return true; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class triangular_solve_matrix_small
} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_MATRIX_SMALL_HPP

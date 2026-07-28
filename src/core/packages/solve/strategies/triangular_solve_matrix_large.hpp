/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_MATRIX_LARGE_HPP
#define PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_MATRIX_LARGE_HPP


#include "packages/solve/problem_context_bases.hpp"
#include "packages/solve/strategies/strategy_helper.hpp"

#include "spec/problem_context.hpp"
#include "spec/strategy_tag.hpp"

#include "operators/parallelize_trsm.hpp"
#include "operators/gemm_exec.hpp"
#include "operators/triangular_solve_resident.hpp"

#include <string_view>

namespace perflibs::linalg::solve {

class triangular_solve_matrix_large {
public:
	static constexpr std::string_view name() { return "triangular_solve_matrix_large"; }

	// Actual computation. this function will be called only if
	// the triangular_solve_matrix_large strategy is valid to solve
	// the problem described by the ProblemContext
	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<triangular_solve_matrix_large>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if( !this->can_compute(pctx) ) return false;

		const auto spec = get_spec(spec::strategy_tag<triangular_solve_matrix_large>{}, pctx);

		const bool is_lside = pctx.b.cntg_step() == 1;

		auto driver =
			parallelize_trsm          { spec.max_threads, spec.kernel_unroll,
			triangular_solve_resident { spec.cntg_recursive_block_sizes[5], is_lside, gemm_exec{ pctx.architecture_spec },
			triangular_solve_resident { spec.cntg_recursive_block_sizes[4], is_lside, gemm_exec{ pctx.architecture_spec },
			triangular_solve_resident { spec.cntg_recursive_block_sizes[3], is_lside, gemm_exec{ pctx.architecture_spec },
			triangular_solve_resident { spec.cntg_recursive_block_sizes[2], is_lside, gemm_exec{ pctx.architecture_spec },
			triangular_solve_resident { spec.cntg_recursive_block_sizes[1], is_lside, gemm_exec{ pctx.architecture_spec },
			triangular_solve_resident { spec.cntg_recursive_block_sizes[0], is_lside, gemm_exec{ pctx.architecture_spec },
			                            trsm_kernel_exec{spec.kernel} } } } } } } };

		scale(pctx.alpha, pctx.b);
		driver(pctx.a, pctx.b);

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<triangular_solve_matrix_large>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const { return true; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class triangular_solve_matrix_large
} // namespace perflibs::linalg::solve

#endif //PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_MATRIX_LARGE_HPP

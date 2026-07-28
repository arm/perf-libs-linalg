/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_VECTOR_HPP
#define PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_VECTOR_HPP

#include "packages/solve//problem_context_bases.hpp"
#include "packages/solve/strategies/strategy_helper.hpp"

#include "spec/problem_context.hpp"
#include "spec/strategy_tag.hpp"

#include "operators/gemm_exec.hpp"
#include "operators/triangular_solve_resident.hpp"

#include "matrix/matrix.hpp"
#include "framework/alloc.hpp"

#include <string_view>

namespace perflibs::linalg::solve {

class triangular_solve_vector {
	template<typename ADataType, typename BDataType, typename ScalarType, typename ArchitectureSpec>
	using pctx_t = spec::problem_context<
		solve<
			triangular_matrix<matrix_base<const ADataType>>,
			general_matrix<   matrix_base<      BDataType>>,
			ScalarType
		>,
		ArchitectureSpec
	>;

public:
	static constexpr std::string_view name() { return "triangular_solve_vector"; }

	template<typename ADataType, typename BDataType, typename ScalarType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE

	bool operator() (const pctx_t<ADataType, BDataType, ScalarType, ArchitectureSpec>& pctx) const {
		if( !this->can_compute(pctx) ) return false;

		const auto spec = get_spec(spec::strategy_tag<triangular_solve_vector>{}, pctx);

		constexpr auto is_lside = true;

		auto driver =
			triangular_solve_resident{ spec.cntg_recursive_block_sizes[3], is_lside, gemm_exec{ pctx.architecture_spec },
			triangular_solve_resident{ spec.cntg_recursive_block_sizes[2], is_lside, gemm_exec{ pctx.architecture_spec },
			triangular_solve_resident{ spec.cntg_recursive_block_sizes[1], is_lside, gemm_exec{ pctx.architecture_spec },
			triangular_solve_resident{ spec.cntg_recursive_block_sizes[0], is_lside, gemm_exec{ pctx.architecture_spec },
			                           trsv_kernel_exec{spec.kernel, spec.kernel_axpby, spec.kernel_dot} }}}};

		// TRSM
		if (pctx.alpha != one<ScalarType>) {
			scale(pctx.alpha, pctx.b);
		}

		if (!is_cntg_contig(pctx.b)) {
			// dot and axpby kernels require data to be contiguous so pack the data in b
			auto b_buffer = get_memory<ScalarType, memory_bank::triangular_solve>(pctx.b.cntg());

			general_matrix packed_b { matrix_base { b_buffer, pctx.b.cntg(), pctx.b.strd(), 1, pctx.b.cntg() } };

			//TODO: can this be done with a copy operator in the LINALG stack?
			copy(pctx.b, packed_b);
			driver(pctx.a, packed_b);
			copy(packed_b, pctx.b);

			return_memory<ScalarType, memory_bank::level2>(b_buffer);
		}
		else {
			driver(pctx.a, pctx.b);
		}
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<triangular_solve_vector>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		return is_left(pctx.side) && pctx.b.strd() == 1 && pctx.alpha == one<typename ProblemContext::scalar_type>;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class triangular_solve_vector
} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_VECTOR_HPP

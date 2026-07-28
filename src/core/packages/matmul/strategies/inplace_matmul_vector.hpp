/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_INPLACE_MATMUL_STRATEGIES_MATRIX_VECTOR_HPP
#define PERFLIBS_LINALG_INPLACE_MATMUL_STRATEGIES_MATRIX_VECTOR_HPP

#include "spec/strategy_tag.hpp"

#include "operators/residents.hpp"
#include "operators/copy_matrix.hpp"
#include "operators/parallelize.hpp"
#include "operators/gemm_exec.hpp"

#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"

#include "blas/mv.hpp"
#include "blas/mv_common.hpp"

#include <type_traits>

namespace perflibs::linalg::matmul {

class inplace_matmul_vector {

public:
	static constexpr std::string_view name() { return "inplace_matmul_vector"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<inplace_matmul_vector>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if ( !this->can_compute(pctx) ) return false;

		const auto spec = get_spec(spec::strategy_tag<inplace_matmul_vector>{}, pctx);

		//can't have more threads than half the work load, an must have more than one
		const kernel_inttype num_threads =
			max(1, min( pctx.a.strd() / 2, spec.max_threads));

		using b_value_type = typename ProblemContext::b_matrix_type::value_type;

		compute_position pos { 0, 0, 0, 0 };

		if (is_cntg_contig(pctx.a)) {
			auto *buffer = get_memory<b_value_type, memory_bank::level2>(pctx.b.cntg());
			auto driver  =
				copy_matrix            { b_matrix, buffer_pool { buffer }, general_cntg_contig_generator { true },
				parallelize            { triangular_parallel_strat, a_strd, num_threads,
				/*
				 * Block over the cntg dimension further, this ensures that GEMV will be called
				 * when there are few threads
				 */
				resident               { a_matrix, pctx.a.cntg(), spec.cntg_block_size, true,
				exp::symv_square_panel { cntg,
				/*
				 * Process the square using dot
				 */
				exp::trmv_vector_split { exp::dot_exec { spec.kernel_dot } },
				/*
				 * Process the panel using GEMV
				 */
				gemm_exec              { pctx.architecture_spec }}}}};

			driver(pctx.a, pctx.b, pctx.b, pos, pctx.alpha, pctx.beta);
			return_memory<b_value_type, memory_bank::level2>(buffer);
		}
		else {
			auto *buffer = get_memory<b_value_type, memory_bank::level2>(pctx.b.cntg());
			auto driver  =
				copy_matrix            { c_matrix, buffer_pool { buffer }, general_cntg_contig_generator { true },
				parallelize            { triangular_parallel_strat, a_strd, num_threads,
				/*
				 * Block over the cntg dimension further, this ensures that GEMV will be called
				 * when there are few threads
				 */
				resident               { a_matrix, pctx.a.cntg(), spec.cntg_block_size, true,
				exp::symv_square_panel { cntg,
				/*
				 * Process the square using axpby
				 */
				exp::mv_reflect        { exp::axpby_exec { spec.kernel_axpby }, no_op {} },
				/*
				 * Process the panel using GEMV
				 */
				gemm_exec              { pctx.architecture_spec }}}}};

			driver(pctx.a, pctx.b, pctx.b, pos, pctx.alpha, pctx.beta);
			return_memory<b_value_type, memory_bank::level2>(buffer);
		}
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<inplace_matmul_vector>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		return pctx.b.strd() == 1 && pctx.b.strd_step() == 0;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; //class inplace_matrix_vector

} // perflibs::linalg::strat

#endif //PERFLIBS_LINALG_INPLACE_MATMUL_STRATEGIES_MATRIX_VECTOR_HPP

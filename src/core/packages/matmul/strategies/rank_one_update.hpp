/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_RANK_ONE_UPDATE_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_RANK_ONE_UPDATE_HPP

#include "spec/problem_context.hpp"
#include "spec/strategy_tag.hpp"

#include "framework/buffer_pool.hpp"
#include "framework/alloc.hpp"
#include "framework/compute_position.hpp"
#include "framework/linalg_util.hpp"

#include "operators/drop_imag_diagonal.hpp"
#include "operators/outer_product.hpp"
#include "operators/copy_matrix.hpp"
#include "operators/parallelize.hpp"
#include "operators/residents.hpp"
#include "operators/sequence.hpp"
#include "operators/no_op.hpp"
#include "operators/crop.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class rank_one_update {
public:
	static constexpr std::string_view name() { return "rank_one_update"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<rank_one_update>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		using a_value_type = std::remove_cv_t<typename ProblemContext::a_matrix_type::value_type>;

		if( !this->can_compute(pctx) ) return false;

		// We then get all of the configuration details about how we will execute the routine.
		// TODO: replace the strategy_tag later
		const auto spec  = get_spec(spec::strategy_tag<rank_one_update>{}, pctx);

		const auto buf_size = pctx.a.strd();

		a_value_type *buffer = is_strd_contig(pctx.a)
				? nullptr
				: get_memory<a_value_type, memory_bank::level2>(buf_size * spec.max_threads);

		buffer_pool buffer_pool { buffer, spec.max_threads, buf_size };

		if constexpr (is_hermitian_matrix_v<decltype(pctx.c)>) {
			auto driver =
				parallelize { triangular_parallel_strat, b_strd, spec.max_threads,
				resident    { b_matrix, 1, spec.b_strd_block_size, false,
				crop        { 1, 1,
				copy_matrix { a_matrix, buffer_pool, general_strd_contig_generator{},
				sequence    { outer_product_terminal { spec.kernel_axpby },
				              drop_imag_diagonal     { no_op{} } } } } } };

			// TODO: this is a quick fix and not a "real" solution as it breaks the model,
			// A is conj and the stack should handle it correctly rather than be tricked
			// a better solution is to create a conversion object and use pack rather
			// than copy matrix, the conversion object can than have option to determine
			// what happens when the matrix is copied.

			// set .is_conj to false
			// so that copy_matrix doesn't conjugate the elements of A when
			// copying. Conjugation is done by the axpby kernel.
			const auto a = general_matrix { pctx.a.get_matrix_base(), false };
			driver(a, pctx.b, pctx.c, compute_position{}, pctx.alpha, pctx.beta);
		}
		else {
			auto driver =
				parallelize            { triangular_parallel_strat, b_strd, spec.max_threads,
				resident               { b_matrix, 1, spec.b_strd_block_size, false,
				crop                   { 1, 1,
				copy_matrix            { a_matrix, buffer_pool, general_strd_contig_generator{},
				outer_product_terminal { spec.kernel_axpby } } } } };

			driver(pctx.a, pctx.b, pctx.c, compute_position{}, pctx.alpha, pctx.beta);
		}


		if (buffer != nullptr) {
			return_memory<a_value_type, memory_bank::level2>(buffer);
		}

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<rank_one_update>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		return pctx.a.cntg() == 1 && pctx.b.cntg() == 1 && pctx.beta == one<typename ProblemContext::scalar_type>;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class rank_one_update
} // namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_RANK_ONE_UPDATE_HPP

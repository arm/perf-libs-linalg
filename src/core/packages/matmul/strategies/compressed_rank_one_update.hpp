/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_COMPRESSED_RANK_ONE_UPDATE_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_COMPRESSED_RANK_ONE_UPDATE_HPP

#include "spec/strategy_tag.hpp"

#include "framework/compute_position.hpp"
#include "framework/buffer_pool.hpp"
#include "framework/linalg_util.hpp"
#include "framework/alloc.hpp"

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

class compressed_rank_one_update {
public:
	static constexpr std::string_view name() { return "compressed_rank_one_update"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<compressed_rank_one_update>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		using scalar_type = typename ProblemContext::scalar_type;

		if ( !this->can_compute(pctx) ) return false;

		// We then get all of the configuration details about how we will execute the routine.
		const auto spec = get_spec(spec::strategy_tag<compressed_rank_one_update>{}, pctx);

		// Sanity checks.
		PERFLIBS_ASSERT(pctx.b.cntg() == 1, "X must be an 1xn vector.");
		PERFLIBS_ASSERT(pctx.a.cntg() == 1, "X must be an 1xn vector.");
		PERFLIBS_ASSERT(pctx.c.cntg() == pctx.c.strd(), "A must be a square matrix.");
		PERFLIBS_ASSERT(pctx.c.cntg() == pctx.a.strd(), "A and X have incompatible dimensions.");
		PERFLIBS_ASSERT(pctx.c.cntg_step() == 1, "cntg_step for A must be 1.");

		const auto buf_size = pctx.a.strd();

		scalar_type *buffer = is_strd_contig(pctx.a)
		                    ? nullptr
		                    : get_memory<scalar_type, memory_bank::level2>(buf_size * spec.max_threads);

		buffer_pool buffer_pool { buffer, spec.max_threads, buf_size };

		if constexpr (is_hermitian_matrix_v<typename ProblemContext::c_matrix_type>) {
			auto driver =
				parallelize            { triangular_parallel_strat, b_strd, spec.max_threads,
				resident               { b_matrix, 1, spec.b_strd_block_size, false,
				crop                   { 1, 1,
				copy_matrix            { a_matrix, buffer_pool, general_strd_contig_generator{},
				sequence               { outer_product_terminal { spec.kernel_axpby },
				                         drop_imag_diagonal     { no_op{} } } } } } };

			driver(pctx.a, pctx.b, pctx.c, compute_position{}, pctx.alpha, pctx.beta);
		}
		else {  // symmetric matrix
			auto driver =
				parallelize            { triangular_parallel_strat, b_strd, spec.max_threads,
				resident               { b_matrix, 1, spec.b_strd_block_size, false,
				crop                   { 1, 1,
				copy_matrix            { a_matrix, buffer_pool, general_strd_contig_generator{},
				outer_product_terminal { spec.kernel_axpby } } } } };

			driver(pctx.a, pctx.b, pctx.c, compute_position{}, pctx.alpha, pctx.beta);
		}

		if (buffer != nullptr) {
			return_memory<scalar_type, memory_bank::level2>(buffer);
		}

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<compressed_rank_one_update>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return true; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; //class compressed_symmetric_matrix_vector

} //namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_COMPRESSED_RANK_ONE_UPDATE_HPP

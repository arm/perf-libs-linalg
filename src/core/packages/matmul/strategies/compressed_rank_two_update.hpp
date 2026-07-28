/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_COMPRESSED_RANK_TWO_UPDATE_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_COMPRESSED_RANK_TWO_UPDATE_HPP

#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"

#include "operators/symmetric_rank_two_update.hpp"
#include "operators/hermitian_rank_two_update.hpp"

#include "operators/drop_imag_diagonal.hpp"
#include "operators/parallelize.hpp"
#include "operators/residents.hpp"
#include "operators/sequence.hpp"
#include "operators/no_op.hpp"
#include "operators/crop.hpp"

#include "spec/strategy_tag.hpp"

#include "perflibs_assert.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class compressed_rank_two_update {
public:
	static constexpr std::string_view name() { return "compressed_rank_two_update"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<compressed_rank_two_update>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if ( !this->can_compute(pctx) ) return false;

		// We then get all of the configuration details about how we will execute the routine.
		const auto spec = get_spec(spec::strategy_tag<compressed_rank_two_update>{}, pctx);

		// Sanity checks.
		PERFLIBS_ASSERT(pctx.a.cntg() == 1, "X must be an 1xn vector.");
		PERFLIBS_ASSERT(pctx.c.cntg() == pctx.c.strd(), "A must be a square matrix.");
		PERFLIBS_ASSERT(pctx.c.cntg() == pctx.a.strd(), "A and X have incompatible dimensions.");
		PERFLIBS_ASSERT(pctx.c.cntg_step() == 1, "cntg_step for A must be 1.");

		if constexpr (is_hermitian_matrix_v<typename ProblemContext::c_matrix_type>) {
			auto sequential_driver =
				resident        { b_matrix, 1, spec.b_strd_block_size, false,
				crop            { 1, 1,
				sequence        { hermitian_rank_two_update { spec.kernel_axpby },
				                  drop_imag_diagonal        { no_op{} } } } };

			// Don't produce the parallelised driver unless the open_mp version of the library is built
			if constexpr(omp::is_mp){
				if(spec.max_threads > 1) {
					// Build and run driver.
					auto driver =
						/*
						* Parallelize over the strided dimension -- that is, split unpacked_a into columns and
						* process each column on a separate thread.
						*
						* Perform a hermitian rank two update on the matrix using our optimized axpy
						* kernel.
						*/
						parallelize { triangular_parallel_strat, b_strd, spec.max_threads, sequential_driver };

					driver(pctx.a, pctx.b, pctx.c, compute_position{}, pctx.alpha);
					return true;
				}
			}
			sequential_driver(pctx.a, pctx.b, pctx.c, compute_position{}, pctx.alpha);
		}
		else {
			auto sequential_driver =
				resident                  { b_matrix, 1, spec.b_strd_block_size, false,
				crop                      { 1, 1,
				symmetric_rank_two_update { spec.kernel_axpby } } };

			// Don't produce the parallelised driver unless the open_mp version of the library is built
			if constexpr(omp::is_mp){
				if(spec.max_threads > 1) {
					// Build and run driver.
					auto driver =
						/*
						* Parallelize over the strided dimension -- that is, split unpacked_a into columns and
						* process each column on a separate thread.
						*/
						parallelize { triangular_parallel_strat, b_strd, spec.max_threads, sequential_driver };
					driver(pctx.a, pctx.b, pctx.c, compute_position{}, pctx.alpha);
					return true;
				}
			}
			sequential_driver(pctx.a, pctx.b, pctx.c, compute_position{}, pctx.alpha);
		}

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<compressed_rank_two_update>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return true; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; //class compressed_rank_two_update

} //namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_COMPRESSED_RANK_TWO_UPDATE_HPP

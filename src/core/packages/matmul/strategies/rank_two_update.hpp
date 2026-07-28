/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_RANK_TWO_UPDATE_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_RANK_TWO_UPDATE_HPP

#include "packages/matmul/strategies.hpp"

#include "operators/symmetric_rank_two_update.hpp"
#include "operators/hermitian_rank_two_update.hpp"

#include "framework/linalg_util.hpp"

#include "operators/drop_imag_diagonal.hpp"
#include "operators/parallelize.hpp"
#include "operators/residents.hpp"
#include "operators/sequence.hpp"
#include "operators/no_op.hpp"
#include "operators/crop.hpp"

#include "spec/strategy_tag.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class rank_two_update {
public:
	static constexpr std::string_view name() { return "rank_two_update"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<rank_two_update>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if ( !this->can_compute(pctx) ) return false;

		const auto spec = get_spec(spec::strategy_tag<rank_two_update>{}, pctx);

		if constexpr (is_hermitian_matrix_v<decltype(pctx.c)>) {

			auto sequential_driver =
				resident        { b_matrix, 1, spec.b_strd_block_size, false,
				crop            { 1, 1,
				sequence        { hermitian_rank_two_update { spec.kernel_axpby },
				                  drop_imag_diagonal        { no_op{} } } } };

			// Don't produce the parallelised driver unless the OpenMP version of the library is built
			if constexpr(omp::is_mp){
				if(spec.max_threads > 1) {
					// Build and run driver.
					auto driver =
						/*
						* Parallelize over the strided dimension -- that is, split unpacked_a into columns and
						* process each column on a separate thread.
						*/
						parallelize { triangular_parallel_strat, b_strd, spec.max_threads,
						/*
						* Perform a hermitian rank two update on the matrix using our optimized axpy
						* kernel.
						*/
						sequential_driver };

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

			// Don't produce the parallelised driver unless the OpenMP version of the library is built
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
	requires spec::has_get_spec<spec::strategy_tag<rank_two_update>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		using scalar_type = typename ProblemContext::scalar_type;

		return pctx.a.cntg() == 1 && pctx.c.cntg_step() == 1
		    && pctx.beta == one<scalar_type> && !pctx.a.is_conj();
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; //class rank_two_update
} // namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_RANK_TWO_UPDATE_HPP

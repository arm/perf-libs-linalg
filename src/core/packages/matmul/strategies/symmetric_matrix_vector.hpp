/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SYMMETRIC_MATRIX_VECTOR
#define PERFLIBS_LINALG_SYMMETRIC_MATRIX_VECTOR

#include "perflibs_assert.hpp"
#include "perflibs_numeric_utils.hpp"
#include "perflibs_util.hpp"
#include "blas/mv.hpp"
#include "blas/mv_common.hpp"

#include "framework/alloc.hpp"
#include "framework/buffer_pool.hpp"
#include "framework/compute_position.hpp"

#include "operators/reduce.hpp"
#include "operators/sequence.hpp"
#include "operators/gemm_exec.hpp"
#include "operators/residents.hpp"
#include "operators/parallelize.hpp"
#include "operators/if_then_else.hpp"
#include "operators/outer_product.hpp"
#include "operators/auto_resident.hpp"
#include "operators/reflect_and_transpose.hpp"

#include "spec/strategy_tag.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class symmetric_matrix_vector {
public:
	static constexpr std::string_view name() { return "symmetric_matrix_vector"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<symmetric_matrix_vector>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator()(const ProblemContext& pctx) const {
		using scalar_type = ProblemContext::scalar_type;

		if (! this->can_compute(pctx) ) return false;

		const auto spec = get_spec(spec::strategy_tag<symmetric_matrix_vector>{}, pctx);

		PERFLIBS_ASSERT(pctx.b.strd() == 1, "X must be an nx1 vector.");
		PERFLIBS_ASSERT(pctx.c.strd() == 1, "Y must be an nx1 vector.");
		PERFLIBS_ASSERT(pctx.a.cntg() == pctx.a.strd(), "A must be a square matrix.");
		PERFLIBS_ASSERT(pctx.a.cntg() == pctx.b.cntg(), "A and X have incompatible dimensions.");
		PERFLIBS_ASSERT(pctx.c.cntg() == pctx.a.strd(), "A and Y have incompatible dimensions.");
		PERFLIBS_ASSERT(pctx.b.strd() == pctx.c.strd(), "X and Y have incompatible dimensions.");

		// Nothing to do? Early return.
		if (pctx.a.cntg() == 0)
			return true;

		// Fix if `pctx.c` is entirely NaNs.
		if (pctx.beta == zero<scalar_type>)
			set(zero<scalar_type>, pctx.c);

		// If `alpha` is zero, `symv` reduces to y := beta * y -- so just scale by `beta` and exit early.
		if (pctx.alpha == zero<scalar_type>) {
			scale(pctx.beta, pctx.c);
			return true;
		}

		const kernel_inttype num_threads =
			perflibs::max(1, perflibs::min(pctx.c.cntg() / 2, spec.max_threads));

		scale(pctx.beta, pctx.c);
		const auto beta = one<scalar_type>;

		auto sequential_driver =
			/**
			* Break the matrix up across the cntg, so that `symv_square_and_panel`
			* can do its just effective
			*/
			resident { a_matrix, spec.cntg_block_size, pctx.a.strd(), true,
				/*
				* Separate the heads on the panels which have triangular form, from
				* the rectangular tails which are effectively dense matrices
				*/
			exp::symv_square_panel { which_dimension::a_strd,
				/*
				* Process the triangular head, using dot for one side of the
				* symmetrix divide, and axpby for the other, ensure the algorithms
				* are working on contiguous data
				*/
				exp::mv_reflect {
					exp::axpby_exec { spec.kernel_axpby },
					exp::dot_exec   { spec.kernel_dot   }
				},
				auto_resident {
					a_strd,
					spec.a_cache_block_bytes,
					sequence {
						gemm_exec { pctx.architecture_spec },
						reflect_and_transpose {
							gemm_exec { pctx.architecture_spec }}
					}
				}}};

		if(omp::is_mp && num_threads > 1) {
			// Set up our byte-sizes for the memory we will allocate.

			// Each thread gets `threads_buffer_size` bytes -- enough to house a 'vector' of the same length as
			// `pctx.c` for each thread. These vectors will be accumulated at the very end.
			const auto threads_buffer_size = pctx.c.cntg() * num_threads * sizeof(scalar_type);

			// We also keep a counter for each thread. These counters are only ever used below, in the driver,
			// to scale by beta on the first reduction step.
			const auto counts_buffer_size = num_threads * sizeof(kernel_inttype);

			// Grab a chunk of memory (just raw bytes) of the size we need.
			auto raw_buffer = get_memory<uint8_t, memory_bank::level2>(threads_buffer_size + counts_buffer_size);

			// Region of memory that the threads will index into to get their buffer.
			auto threads_raw_buffer = reinterpret_cast<scalar_type *>(raw_buffer);

			// Region of memory housing the counts for each thread.
			auto counts_raw_buffer = reinterpret_cast<kernel_inttype *>(raw_buffer + threads_buffer_size);
			std::fill(counts_raw_buffer, counts_raw_buffer + num_threads, 0); // zero the counters

			// Wrap the above memory regions in buffer pool objects for ease-of-use.
			buffer_pool threads_buffer_pool { threads_raw_buffer, num_threads, pctx.c.cntg() };
			buffer_pool counts_buffer_pool  { counts_raw_buffer,  num_threads, 1 };

			synchronization synchro{ num_threads };

			auto driver =
				/**
				* We are going to parallelise over the a.cntg() dimension
				*/
				parallelize { triangular_parallel_strat, cntg, num_threads,
				reduce { parallel_reduction_strat, num_threads, threads_buffer_pool, synchro,
					/*
					 * Once we have got all our parallelism set, then we will use the sequential
					 * driver to do the work
					 */
					sequential_driver,
					/*
					 * Reduction step -- accumulate each thread's buffer, scaled by beta for the first thread.
					 */
					if_then_else {
						[&](auto... args) {
							kernel_inttype *count = counts_buffer_pool.get_buffer(omp::get_thread_num());
							return (*count)++ == 0;
						},
						vec_accumulate_scaled(spec.kernel_axpby),
						vec_accumulate(spec.kernel_axpby)
				}}};

			driver(pctx.a, pctx.b, pctx.c, compute_position{}, pctx.alpha, beta);

			return_memory<uint8_t, memory_bank::level2>(raw_buffer);
		}
		else {
			sequential_driver(pctx.a, pctx.b, pctx.c, compute_position{}, pctx.alpha, beta);
		}

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<symmetric_matrix_vector>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		return pctx.b.strd() == 1;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; //class symmetric_matrix_vector

} //namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_SYMMETRIC_MATRIX_VECTOR

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_RANK_K_UPDATE_LARGE_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_RANK_K_UPDATE_LARGE_HPP

#include "spec/strategy_tag.hpp"

#include "framework/alloc.hpp"
#include "framework/buffer_pool.hpp"

#include "operators/pack.hpp"
#include "operators/crop.hpp"
#include "operators/residents.hpp"
#include "operators/kernel_exec.hpp"
#include "operators/copy_matrix.hpp"
#include "operators/tri_resident.hpp"
#include "operators/parallel_spawn.hpp"
#include "operators/partial_separate.hpp"
#include "operators/bookend_triangular_split.hpp"

#include "spec/get_block_sizes.hpp"

#include "perflibs_assert.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class rank_update_generic;

class rank_k_update_large {
public:
	static constexpr std::string_view name() { return "rank_k_update_large"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<rank_update_generic>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if ( !this->can_compute(pctx) ) return false;

		const auto spec         = get_spec(spec::strategy_tag<rank_update_generic>{}, pctx);
		const auto kspec        = spec.kernel_spec;

		using kernel_value_type = typename decltype(kspec)::value_type;

		const auto l1_cntg      = spec::get_l1_cntg(spec);
		const auto l1_strd      = spec::get_l1_strd(spec);
		const auto l2_strd      = spec::get_l2_strd(spec);

		// sanity check
		PERFLIBS_ASSERT(l1_cntg % kspec.cntg_unroll == 0,                         "l1_ctng must be multiple of cntg_unroll");
		PERFLIBS_ASSERT(l1_strd % kspec.a_interleave_spec.strd_interleave() == 0, "l1_strd must be multiple of a_interleaved_rows");
		PERFLIBS_ASSERT(l2_strd % kspec.b_interleave_spec.strd_interleave() == 0, "l2_strd must be multiple of b_interleaved_rows");

		const kernel_inttype a_buf_sz = spec.a_convert.elements_required(l1_cntg, l1_strd);
		const kernel_inttype b_buf_sz = spec.b_convert.elements_required(l1_cntg, l2_strd);
		const kernel_inttype c_buf_sz = l1_strd * l2_strd;

		// the amount of memory required by each thread
		const kernel_inttype buffer_size = a_buf_sz + b_buf_sz + c_buf_sz;

		// allocate all of the memory for all of the treads
		auto buffer = get_memory<kernel_value_type>(buffer_size * spec.max_threads);

		buffer_pool a_buf { buffer,                       spec.max_threads, buffer_size };
		buffer_pool b_buf { buffer + a_buf_sz,            spec.max_threads, buffer_size };
		buffer_pool c_buf { buffer + a_buf_sz + b_buf_sz, spec.max_threads, buffer_size };

		// basic gemm kernel driver which copies the values of C into a buffer, performs the gemm and copies
		// back
		auto gemm_copy_driver =
			copy_matrix{ c_matrix, c_buf, general_cntg_contig_generator{},
			kernel_exec{ kspec.kernel, kspec.apply_beta }};

		// basic gemm kernel driver with no copy
		auto gemm_driver = kernel_exec{ kspec.kernel, kspec.apply_beta };

		auto base_rank_k_update_driver =
			/*
			* Block B into L3 cache
			*/
			resident { b_matrix, l1_cntg, l2_strd, spec.l2_cntg_first,
			/*
			* Pack B in to the transposed interleave format
			*/
			pack     { b_matrix, b_buf, spec.b_convert,
			/*
			* Crop the compute space to move any unnecessary white space introduced by the
			* triangular nature of C (SYMM)
			*
			* in this particular crop we expect to see a.strd be reduced, b should not be affected
			*/
			crop     { 1_ki, kspec.b_interleave_spec.strd_unroll,
			/*
			* Block A into L2 cache
			*/
			resident { a_matrix, l1_cntg, l1_strd, spec.l1_cntg_first,
			/*
			* Again, we do not what to be doing compute on white space
			*
			* B at this point is oversized, although it is packed at this point
			* we can stil reduce it's size so long as it is multiples of it's
			* interleave factor
			*/
			crop     { 1_ki, kspec.b_interleave_spec.strd_unroll,
			/*
			* Pack A into the transposed-interleaved format
			*/
			pack     { a_matrix, a_buf, spec.a_convert,
			/*
			* Here we split the blocks of C into 'Full dense blocks' and
			* portions which contain triangles and parts of triangles
			*
			* the full-dense-blocks (c.is_physical() == true) can be
			* computed directly into the portions containing triangles
			* (partial-blocks) (c.is_physical() == false) are computed into
			* a buffer and then copied back in
			*/
			partial_separate { kspec.a_interleave_spec.strd_unroll, kspec.b_interleave_spec.strd_unroll,
			/*
			* do gemm - see above
			*/
			gemm_driver, gemm_copy_driver }}}}}}};

		auto rank_k_update_driver =
			parallel_spawn           { spec.max_threads,
			bookend_triangular_split { spec.max_threads, 1_ki, 1_ki,
			base_rank_k_update_driver }};

		rank_k_update_driver(pctx.a, pctx.b, pctx.c, { 0, 0, 0 }, pctx.alpha, pctx.beta);

		if constexpr (is_hermitian_matrix_v<decltype(pctx.c)>) {
			zero_diag_imag(pctx.c);
		}

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<rank_update_generic>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		return pctx.alpha != zero<typename ProblemContext::scalar_type>
		    && pctx.beta_zero_mode != zero_mode::scale
		    && pctx.c.strd() > 1 // bookend_triangular_split needs at least 2 cols
		    && pctx.a.cntg_step() >= 1 && pctx.a.strd_step() >= 1
		    && pctx.b.cntg_step() >= 1 && pctx.b.strd_step() >= 1
		    && pctx.c.cntg_step() == 1 && pctx.c.strd_step() >= 1;
 	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class rank_k_update_large
} // namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_RANK_K_UPDATE_LARGE_HPP

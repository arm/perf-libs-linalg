/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_INPLACE_MATMUL_STRATEGIES_MATMUL3_INTERLEAVED_TRI_GEN_GEN_HPP
#define PERFLIBS_LINALG_INPLACE_MATMUL_STRATEGIES_MATMUL3_INTERLEAVED_TRI_GEN_GEN_HPP

#include "packages/matmul/problem_context_bases.hpp"

#include "spec/problem_context.hpp"
#include "spec/strategy_tag.hpp"
#include "spec/get_block_sizes.hpp"

#include "operators/pack.hpp"
#include "operators/no_inline.hpp"
#include "operators/residents.hpp"
#include "operators/kernel_exec.hpp"
#include "operators/set_scalars.hpp"
#include "operators/parallelize.hpp"
#include "operators/tri_resident.hpp"
#include "operators/triangle_separate.hpp"

#include "framework/alloc.hpp"
#include "framework/linalg_util.hpp"
#include "framework/buffer_pool.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

// matmul3_interleaved_tri_gen_gen is symmetric (in a and b) with matmul3_interleaved_gen_tri_gen
class matmul3_interleaved_tri_gen_gen {
public:
	static constexpr std::string_view name() { return "matmul3_interleaved_tri_gen_gen"; }

	template<typename ProblemContext>
	requires
		spec::has_get_spec<spec::strategy_tag<matmul3_interleaved_tri_gen_gen>, ProblemContext> &&
		is_triangular_matrix_v<typename ProblemContext::a_matrix_type>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if (! this->can_compute(pctx) ) return false;

		const auto spec  = get_spec(spec::strategy_tag<matmul3_interleaved_tri_gen_gen>{}, pctx);
		const auto kspec = spec.kernel_spec;

		using kernel_value_type = typename decltype(kspec)::value_type;

		//block_sizes
		const auto l1_cntg = get_l1_cntg(spec);
		const auto l1_strd = get_l1_strd(spec);
		const auto l2_strd = get_l2_strd(spec);

		/*
		* Calculate the size of the buffers for each thread using the conversion objects
		* We are using the size of the blocks as the dimension to allocate, whislt we could
		* min() the size of the blocks with the size of the matrices this creates an issue
		* when the inplace matmul is invoked repeatedly with slightly increasing problem sizes
		* as it will constantly be reallocating.
		*/
		const kernel_inttype a_buf_sz_per_thread     = spec.a_convert.elements_required(l1_cntg, max(l1_strd, l1_cntg));
		const kernel_inttype b_buf_sz_per_thread     = spec.b_convert.elements_required(l1_cntg, l2_strd);

		const kernel_inttype total_buf_sz_per_thread = a_buf_sz_per_thread + b_buf_sz_per_thread;
		const kernel_inttype total_buf_sz            = total_buf_sz_per_thread * spec.max_threads;

		//allocation
		auto a_ptr = get_memory<kernel_value_type>(total_buf_sz);
		auto b_ptr = a_ptr + a_buf_sz_per_thread;

		/*
		* Each thread will have have its own buffer for both the A & B matrix to be packed into
		* These will be stored next to each other in memory like so
		*
		* Th0               Th1                     ThN
		* | a_buf0 | b_buf0 | a_buf1 | b_buf1 | ... | a_bufN | b_bufN |
		*/
		buffer_pool a_buf { a_ptr, spec.max_threads, total_buf_sz_per_thread };
		buffer_pool b_buf { b_ptr, spec.max_threads, total_buf_sz_per_thread };

		kernel_inttype a_strd_unroll = kspec.a_interleave_spec.strd_unroll * kspec.a_interleave_spec.strd_interleave();
		kernel_inttype b_strd_unroll = kspec.b_interleave_spec.strd_unroll * kspec.b_interleave_spec.strd_interleave();

		/*
		* In kernel_exec: compute_position.cntg == 0 signals that the C matrix should be
		* multiplied by `beta`, and multiplied (implicitly) by one otherwise.
		*
		* Care must be taken to multiply each element of C by beta exactly once, and prior to
		* the accumulation of A*B into C:
		*
		* > For problems where a.is_lower() is true, this is done in the first triangle block,
		*   and first rectangular block processed (note that this is precisely when
		*   compute_position.cntg is zero)
		*
		* > For problems where a.is_lower() is false, this is done in each triangle block only
		*
		* The first case is handled "naturally", but for the second case, a small hack is used:
		* the set_scalar operation is used to set compute_position.cntg to be zero for each
		* triangular block.
		*/
		const bool apply_set_scalar = pctx.a.is_lower();

		auto driver =
			/*
			* Each col on the B matrix is in effect a separate operation
			* So we will parallelise over this dimension
			*/
			parallelize       { general_parallel_strat, b_strd, spec.max_threads, spec.b_convert.strd_interleave(),
			/*
			* Block B into L3 cache, loop order is critical as it is inplace
			*/
			resident          { b_matrix, l1_cntg, l2_strd, spec.l2_cntg_first,
			pack              { b_matrix, b_buf, spec.b_convert,
			triangle_separate { a_matrix,

				/*
				* This driver is used to compute the parts of the problem which lie on the diagonal
				*/
				no_inline       {
				pack            { a_matrix, a_buf, spec.a_convert,
				/*
				* A & B are now packed, however we want to take the iteration
				* over this packed data out of the GEMM kernels and into our
				* own hands so we can start elimating data that be in the virtual
				* half of the triangular matrix
				*
				* To do this we will first iterate over interleave rows of of B
				* so for example if the kernel takes 6 rows of B at a time, we'll
				* loop over B in 6s (b_interleave_spec.strd_unroll)...
				*/
				resident        { b_matrix, std::numeric_limits<kernel_inttype>::max(), b_strd_unroll, spec.l1_cntg_first,
				/*
				* ...we then want to loop over the packed A matrix, which was
				* packed from a triangular source matrix. Because A & B share
				* a common dimension (cntg), as we iterate over the triangular
				* A matrix via its shared Dimension with C, we can 'shorten'
				* the dimension shared between A & B, eliminating work that the
				* kernel would otherwise have to do
				*/
				tri_resident    { a_matrix, a_strd, pctx.a.is_lower(), a_strd_unroll, kspec.cntg_unroll,
				/*
				* Ensure C is multiplied by beta exactly once
				*/
				set_scalar      { pctx.beta, apply_set_scalar,
				kernel_exec     { kspec.kernel, kspec.apply_beta }}}}}},

				/*
				* when the problem is rectangular (no triangle section) then we can
				* just do a relatively normal GEMM, block A into L2 and packing
				*/
				no_inline       {
				resident        { a_matrix, l1_cntg, l1_strd, spec.l1_cntg_first,
				pack            { a_matrix, a_buf, spec.a_convert,
				kernel_exec     { kspec.kernel, kspec.apply_beta }}}}

			}}}};

		driver(pctx.a, pctx.b, pctx.c, { 0, 0, 0, 0 }, pctx.alpha, pctx.beta);

		return true;
	}

	template<typename ProblemContext>
	requires
		spec::has_get_spec<spec::strategy_tag<matmul3_interleaved_tri_gen_gen>, ProblemContext> &&
		is_triangular_matrix_v<typename ProblemContext::a_matrix_type>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return true; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }
}; //class matmul3_interleaved_tri_gen_gen
} // perflibs::linalg::matmul

#endif //PERFLIBS_LINALG_INPLACE_MATMUL_STRATEGIES_MATMUL3_INTERLEAVED_TRI_GEN_GEN_HPP

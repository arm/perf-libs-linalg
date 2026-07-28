/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_LARGE_NO_SYNC_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_LARGE_NO_SYNC_HPP

#include "perflibs_numeric_utils.hpp"
#include "perflibs_util.hpp"
#include "framework/2d_split.hpp"
#include "framework/alloc.hpp"
#include "framework/which.hpp"
#include "framework/parallel.hpp"
#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"

#include "operators/pack.hpp"
#include "operators/copy_matrix.hpp"
#include "operators/residents.hpp"
#include "operators/kernel_exec.hpp"
#include "operators/parallelise_2d.hpp"

#include "spec/get_block_sizes.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

/// @tparam UseAccelerator to be run on the Accelerator if there is one (i.e. SME) else Host
template<bool UseAccelerator>
class matmul3_interleaved_large_no_sync {
public:
	static constexpr std::string_view name() { return "matmul3_interleaved_large_no_sync"; }

	using strategy_t = std::conditional_t<UseAccelerator
	                 , matmul3_interleaved_strategy_tag_accelerator
	                 , matmul3_interleaved_strategy_tag>;


	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<strategy_t>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if (! can_compute(pctx) ) return false;

		const auto spec   = get_spec(spec::strategy_tag<strategy_t>{}, pctx);
		const auto& kspec = spec.kernel_spec;

		using kernel_spec_type   = typename std::remove_cvref_t<decltype(kspec)>;
		using a_kernel_data_type = typename kernel_spec_type::a_data_type;
		using b_kernel_data_type = typename kernel_spec_type::b_data_type;
		using c_kernel_data_type = typename kernel_spec_type::c_data_type;
		/*
		 * Is the datatype of kernel the same as the input types? It might be different
		 * if this is a mixed precision GEMM or for a "faked" hgemm
		 */
		constexpr bool is_homogeneous = std::is_same_v<typename ProblemContext::c_matrix_type::value_type, c_kernel_data_type>;

		//block_sizes
		const auto l1_cntg = get_l1_cntg(spec);
		const auto l1_strd = get_l1_strd(spec);
		const auto l2_strd = get_l2_strd(spec);

		/*
		 * Calculate how we are going to use our threads
		 */
		const kernel_inttype a_total_unroll = kspec.a_interleave_spec.strd_interleave() * kspec.a_interleave_spec.strd_unroll;
		const kernel_inttype b_total_unroll = kspec.b_interleave_spec.strd_interleave() * kspec.b_interleave_spec.strd_unroll;

		/*
		 * as this is a homogeneous strategy, if we are using the accelerator,
		 * then we can only use the min of the cores and accelerators
		 */
		const auto max_threads = UseAccelerator
		                      ? min(spec.max_threads, pctx.architecture_spec.get_accelerator_count() )
		                      : spec.max_threads;

		const auto [ a_strd_threads, b_strd_threads ] = split_2d(max_threads, pctx.a.strd(), pctx.b.strd());

		const auto [ a_strd_split, b_strd_split ] = make_parallel_2d_split(
			a_strd_threads, pctx.a.strd(), a_total_unroll,
			b_strd_threads, pctx.b.strd(), b_total_unroll);

		const kernel_inttype total_threads = a_strd_split.threads * b_strd_split.threads;

		/*
		 * Calculate how much memory we need for A, B & (if hetrogenious) C buffers
		 */
		const kernel_inttype a_buf_sz_bytes         = spec.a_convert.bytes_required(l1_cntg, l1_strd);
		const kernel_inttype b_buf_sz_bytes         = spec.b_convert.bytes_required(l1_cntg, l2_strd);
		const kernel_inttype c_buf_sz_elems         = is_homogeneous ? 0 : l1_strd * l2_strd; //0 if homogeneous

		//convert to bytes, and round up to the nearest alignment value, TODO ensure we are also applying disalignment
		const kernel_inttype a_buf_sz_bytes_aligned =                      iround_if_not_zero(a_buf_sz_bytes, spec.a_align);
		const kernel_inttype b_buf_sz_bytes_aligned =                      iround_if_not_zero(b_buf_sz_bytes, spec.b_align);
		const kernel_inttype c_buf_sz_bytes_aligned = is_homogeneous ? 0 : iround_if_not_zero(c_buf_sz_elems * sizeof(c_kernel_data_type), spec.c_align);

		/*
		 * account for threads and alignment (as bytes)
		 *
		 * Note that we specify 2* the alignment value. This is so that we can disalign.
		 * For example if we specify to align to 32, then we want to explicitly disalign to 64,
		 * because otherwise we'd of specified 64
		 */
		const kernel_inttype a_buf_sz_thread_bytes_aligned =                      a_buf_sz_bytes_aligned * total_threads + spec.a_align*2; //2x because align_to also disaligns
		const kernel_inttype b_buf_sz_thread_bytes_aligned =                      b_buf_sz_bytes_aligned * total_threads + spec.b_align*2;
		const kernel_inttype c_buf_sz_thread_bytes_aligned = is_homogeneous ? 0 : c_buf_sz_bytes_aligned * total_threads + spec.c_align*2;

		const kernel_inttype total_buffer_size = a_buf_sz_thread_bytes_aligned
		                                       + c_buf_sz_thread_bytes_aligned
		                                       + b_buf_sz_thread_bytes_aligned;

		//allocation
		auto buffer = get_memory<std::uint8_t>(total_buffer_size);
		///..and partition
		auto a_ptr_bytes     = buffer;
		auto c_ptr_bytes     = a_ptr_bytes + a_buf_sz_thread_bytes_aligned;
		auto b_ptr_bytes     = c_ptr_bytes + c_buf_sz_thread_bytes_aligned;

		//convert to our working data type (rather than as bytes)
		auto a_ptr_elems = align_to(reinterpret_cast<a_kernel_data_type*>(a_ptr_bytes), spec.a_align);
		auto b_ptr_elems = align_to(reinterpret_cast<b_kernel_data_type*>(b_ptr_bytes), spec.b_align);
		auto c_ptr_elems = align_to(reinterpret_cast<c_kernel_data_type*>(c_ptr_bytes), spec.c_align);

		/*
		 * Build our buffer pool objects
		 */
		buffer_pool a_pack_buffer_pool { a_ptr_elems, total_threads, bytes{ a_buf_sz_bytes_aligned } };
		buffer_pool b_pack_buffer_pool { b_ptr_elems, total_threads, bytes{ b_buf_sz_bytes_aligned } };
		buffer_pool c_copy_buffer_pool { c_ptr_elems, total_threads, bytes{ c_buf_sz_bytes_aligned } };

		constexpr auto spawn = std::bool_constant<true>{};
		constexpr auto check_threads = std::bool_constant<false>{};

		auto driver =
			/*
			 * Spawns the threads and divides up the problem space based on the
			 * scheme determined by a_strd_split and b_strd_split
			 */
			parallelise_2d { spawn, check_threads, a_strd_split, b_strd_split,
			/*
			 * Blocks over B (typically packing into L3)
			 */
			resident       { b_matrix, l1_cntg, l2_strd, spec.l2_cntg_first,
			/*
			 * Each thread packs its own chunk of B
			 */
			pack           { b_matrix, b_pack_buffer_pool, spec.b_convert,
			/*
			 * Blocks over A  (typically packing L2)
			 */
			resident       { a_matrix, l1_cntg, l1_strd, spec.l1_cntg_first,
			/*
			 * Each thread packs its own chunk of A
			 */
			pack           { a_matrix, a_pack_buffer_pool, spec.a_convert,
			/*
			 * When the output matrix is not compatible with kernel type we
			 * then do the compute into a compatible temporary matrix type
			 * and then copy it back out into the users output matrix
			 * performing any compatibility conversions.
			 *
			 * Use cases include:
			 *     + SYMM
			 *     + GEMM
			 *     + fp16 when the machine does support fp16
			 */
			copy_matrix { c_matrix, c_copy_buffer_pool, general_cntg_contig_generator{},
			/*
			 * invokes the kernel
			 */
			kernel_exec    { kspec.kernel, kspec.apply_beta }}}}}}};

		compute_position pos { 0, 0, 0, 0 };

		driver(pctx.a, pctx.b, pctx.c, pos, pctx.alpha, pctx.beta);

		return_memory(buffer);

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<strategy_t>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool can_compute(const ProblemContext& pctx) const {
		const bool is_enabled = ( !UseAccelerator )
		                     || ( pctx.architecture_spec.get_accelerator_count() > 0 );

		return is_enabled
		    && pctx.alpha != zero<typename ProblemContext::scalar_type>
		    && pctx.beta_zero_mode != zero_mode::scale
		    && pctx.a.cntg_step() >= 1 && pctx.a.strd_step() >= 1
		    && pctx.b.cntg_step() >= 1 && pctx.b.strd_step() >= 1
		    && pctx.c.cntg_step() == 1 && pctx.c.strd_step() >= 1;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; //class matmul3_interleaved_large_no_sync

} //namespace perflibs::linalg::matmul

#endif //PERFLIBS_LINALG_MATMUL_STRATEGIES_LARGE_NO_SYNC_HPP

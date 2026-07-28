/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_LARGE_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_LARGE_HPP

#include "perflibs_numeric_utils.hpp"
#include "perflibs_util.hpp"
#include "framework/alloc.hpp"
#include "framework/which.hpp"
#include "framework/parallel.hpp"
#include "framework/2d_split.hpp"
#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"
#include "framework/synchronization.hpp"
#include "framework/active_buffer_pool.hpp"

#include "operators/no_op.hpp"
#include "operators/pack.hpp"
#include "operators/bcms.hpp"
#include "operators/copy_matrix.hpp"
#include "operators/residents.hpp"
#include "operators/kernel_exec.hpp"
#include "operators/parallelise_2d.hpp"
#include "operators/parallelise_3d.hpp"
#include "operators/gemm_reduce.hpp"

#include "nonowning_vector.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

template<bool SpawnThreads = true>
class matmul3_interleaved_large {
	template<typename SpecType, typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool impl_2d(const SpecType& spec, const ProblemContext& pctx) const {
		const auto& kspec = spec.kernel_spec;

		using architecture_spec = typename SpecType::architecture_spec_type;

		using kernel_spec_type   = typename std::remove_cvref_t<decltype(kspec)>;
		using a_kernel_data_type = typename kernel_spec_type::a_data_type;
		using b_kernel_data_type = typename kernel_spec_type::b_data_type;
		using c_kernel_data_type = typename kernel_spec_type::c_data_type;


		constexpr bool is_homogeneous       = std::is_same_v<typename ProblemContext::c_matrix_type::value_type, c_kernel_data_type>;

		// block_sizes
		const auto l1_cntg = get_l1_cntg(spec);
		const auto l1_strd = get_l1_strd(spec);
		const auto l2_strd = get_l2_strd(spec);

		/*
		 * Calculate how we are going to use our threads
		 */
		const kernel_inttype a_total_unroll = kspec.a_interleave_spec.strd_interleave() * kspec.a_interleave_spec.strd_unroll;
		const kernel_inttype b_total_unroll = kspec.b_interleave_spec.strd_interleave() * kspec.b_interleave_spec.strd_unroll;

		const auto [ a_strd_threads, b_strd_threads ] = split_2d(spec.max_threads, pctx.a.strd(), pctx.b.strd());

		const auto [ a_strd_split, b_strd_split ] = make_parallel_2d_split(
			a_strd_threads, pctx.a.strd(), a_total_unroll,
			b_strd_threads, pctx.b.strd(), b_total_unroll);

		const kernel_inttype total_threads = a_strd_split.threads * b_strd_split.threads;


		/*
		 * Calculate how much memory we need for A, B & (if hetrogenious) C buffers
		 */
		const kernel_inttype a_buf_sz_bytes =                         iround_if_not_zero(spec.a_convert.bytes_required(l1_cntg, l1_strd), spec.a_align);
		const kernel_inttype b_buf_sz_bytes =                         iround_if_not_zero(spec.b_convert.bytes_required(l1_cntg, l2_strd), spec.b_align);
		const kernel_inttype c_buf_sz_bytes = is_homogeneous ? 0_ki : iround_if_not_zero(l1_strd * l2_strd * sizeof(c_kernel_data_type), spec.c_align);

		/*
		 * account for threads and alignment (as bytes)
		 *
		 * Note that we specify 2* the alignment value. This is so that we can disalign.
		 * For example if we specify to align to 32, then we want to explicitly disalign to 64,
		 * because otherwise we'd of specified 64
		 *
		 * The A and C buffers are sized for all threads here. In contrast, the B buffer is
		 * sized for a single team of threads, and BCMS multiplies the B buffer size by the
		 * number of teams internally.
		 */
		const kernel_inttype a_buf_sz_thread_bytes =                         spec.a_align*2_ki + a_buf_sz_bytes * total_threads;
		const kernel_inttype b_buf_sz_team_bytes   =                         spec.b_align*2_ki + b_buf_sz_bytes;
		const kernel_inttype c_buf_sz_thread_bytes = is_homogeneous ? 0_ki : spec.c_align*2_ki + c_buf_sz_bytes * total_threads;

		/*
		 * adding threading synchronization and booking keeping objects
		 *
		 * Note that the synchronization object must be aligned as it will cause
		 * a SIGBUS error on the a64fx
		 */
		const kernel_inttype bcms_buffer_size_bytes = bcms_total_size<b_kernel_data_type>(
			b_strd_split.threads, //number of teams
			a_strd_split.threads, //threads per team
			b_buf_sz_team_bytes,  //size (bytes) per team
			architecture_spec::cacheline_len);


		const kernel_inttype total_buffer_size = a_buf_sz_thread_bytes
						                       + c_buf_sz_thread_bytes
						                       + bcms_buffer_size_bytes;

		//allocation
		auto buffer = get_memory<std::uint8_t>(total_buffer_size);
		///..and partition
		auto a_ptr_bytes     = buffer;
		auto c_ptr_bytes     = a_ptr_bytes + a_buf_sz_thread_bytes;
		auto b_ptr_bytes     = c_ptr_bytes + c_buf_sz_thread_bytes;

		auto b_thread_ptr_elems = setup_static_bcms<b_kernel_data_type>(
			b_strd_split.threads, //number of teams
			a_strd_split.threads, //threads per team
			b_ptr_bytes,          //pointer to buffer
			b_buf_sz_team_bytes,  //size (bytes) per team
			architecture_spec::cacheline_len);

		//convert to our working data type (rather than as bytes)
		auto a_ptr_elems = align_to(reinterpret_cast<a_kernel_data_type*>(a_ptr_bytes), spec.a_align);
		//TODO is B aligned??
		//auto b_ptr_elems = align_to(reinterpret_cast<kernel_value_type*>(b_ptr_bytes), spec.b_align);
		auto c_ptr_elems = align_to(reinterpret_cast<c_kernel_data_type*>(c_ptr_bytes), spec.c_align);

		/*
		 * Build our buffer pool objects
		 */
		buffer_pool a_pack_buffer_pool { a_ptr_elems,         total_threads, bytes{ a_buf_sz_bytes } };
		buffer_pool c_copy_buffer_pool { c_ptr_elems,         total_threads, bytes{ c_buf_sz_bytes } };
		buffer_pool b_bcms_buffer_pool { b_thread_ptr_elems,  total_threads, bytes{ sizeof(bcms_thread_record<b_kernel_data_type>) } };

		using spawn_t = std::bool_constant<SpawnThreads>;
		using check_threads_t = std::bool_constant<true>;

		auto driver =
			/*
			 * Spawns the threads and divides up the problem space based on the
			 * scheme determined by a_strd_split and b_strd_split
			 */
			parallelise_2d { spawn_t{}, check_threads_t{}, a_strd_split, b_strd_split,
			/*
			 * Blocks over B (typically packing into L3)
			 */
			resident       { b_matrix, l1_cntg, l2_strd, spec.l2_cntg_first,
			/*
			 * Thread belonging to the same team co-operatively Pack their current
			 * B submatrix
			 */
			bcms           { b_matrix, b_bcms_buffer_pool, spec.b_convert,
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

		/*
		 * The driver returns a boolean signaling whether or not it was successfully
		 * able to acquire all of the threads we wanted to when setting up our memory
		 * and syncro objects.
		 *
		 * As those objects are dependent on getting the right number of threads, it is
		 * critical to bail out and try a different strategy
		 */
		const auto res = driver(pctx.a, pctx.b, pctx.c, pos, pctx.alpha, pctx.beta);

		return_memory(buffer);

		return res;
	}

	template<typename SpecType, typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool impl_3d(
		kernel_inttype cntg_threads,
		const SpecType& spec,
		const ProblemContext& pctx) const {

		const auto& kspec = spec.kernel_spec;

		using architecture_spec = typename SpecType::architecture_spec_type;
		using c_value_type      = typename ProblemContext::c_matrix_type::value_type;

		using kernel_spec_type   = typename std::remove_cvref_t<decltype(kspec)>;
		using a_kernel_data_type = typename kernel_spec_type::a_data_type;
		using b_kernel_data_type = typename kernel_spec_type::b_data_type;

		// block_sizes
		const auto l1_cntg = get_l1_cntg(spec);
		const auto l1_strd = get_l1_strd(spec);
		const auto l2_strd = get_l2_strd(spec);

		/*
		 * Calculate how we are going to use our threads
		 */
		const kernel_inttype a_total_unroll = kspec.a_interleave_spec.strd_interleave() * kspec.a_interleave_spec.strd_unroll;
		const kernel_inttype b_total_unroll = kspec.b_interleave_spec.strd_interleave() * kspec.b_interleave_spec.strd_unroll;

		const auto [ a_strd_threads, b_strd_threads ] = split_2d(spec.max_threads / cntg_threads, pctx.a.strd(), pctx.b.strd());

		const auto [ a_strd_split, b_strd_split, cntg_split ] = make_parallel_3d_split(
			a_strd_threads, pctx.a.strd(), a_total_unroll,
			b_strd_threads, pctx.b.strd(), b_total_unroll,
			  cntg_threads, pctx.a.cntg(), kspec.cntg_unroll);


		const kernel_inttype total_threads = a_strd_split.threads * b_strd_split.threads * cntg_split.threads;

		/*
		 * Calculate how much memory we need for A, B & (if hetrogenious) C buffers
		 */
		const kernel_inttype a_buf_sz_elems = spec.a_convert.elements_required(l1_cntg, l1_strd); //size in elems
		const kernel_inttype b_buf_sz_elems = spec.b_convert.elements_required(l1_cntg, l2_strd);

		const kernel_inttype c_buf_sz_elems = l1_strd * l2_strd; //this isn't used if all types match
		const kernel_inttype kblk_buf_sz_elems = pctx.c.cntg() * pctx.c.strd();

		const kernel_inttype cacheline_len  = architecture_spec::cacheline_len;

		//construct allocators
		basic_buffer_pool_allocator<a_kernel_data_type, memory_bank::main_a> a_alloc    { a_buf_sz_elems,    bytes{ spec.a_align }, bytes{ 1 } };
		basic_buffer_pool_allocator<c_value_type, memory_bank::main_c>       c_alloc    { c_buf_sz_elems,    bytes{ spec.c_align }, bytes{ 1 } };
		basic_buffer_pool_allocator<c_value_type, memory_bank::main>         kblk_alloc { kblk_buf_sz_elems, bytes{ spec.c_align }, bytes{ 1 } };

		//a_strd_split.threads threads in a team,  b_strd_split.threads teams
		bcms_team_buffer_pool_allocator<b_kernel_data_type, memory_bank::bcms_team_record> b_bcms_team_alloc {
			a_strd_split.threads,   // number of threads in a team
			bytes{ cacheline_len }, // alignment of synchro object
			b_buf_sz_elems,         // number of elements in buffer
			bytes{ spec.b_align },  // alignment of buffer
			bytes{ 1 }              // disalignment of buffer
		};

		active_buffer_pool b_bcms_team_buffer_pool { total_threads, a_strd_split.threads, b_bcms_team_alloc };

		bcms_buffer_pool_allocator<decltype(b_bcms_team_buffer_pool), memory_bank::main_b> b_alloc { std::move(b_bcms_team_buffer_pool) };

		auto axpby_kernel = get_spec(axpby_kernel_tag { spec::strategy_tag<matmul3_interleaved_large>{} }, pctx);

		using check_threads_t = std::bool_constant<true>;

		auto driver =
			/*
			 * Spawns the threads and divides up the problem space based on the
			 * scheme determined by a_strd_split and b_strd_split
			 */
			parallelise_3d { check_threads_t{}, a_strd_split, b_strd_split, cntg_split, active_buffer_pool{ total_threads, cntg_threads, kblk_alloc },
				/*
				 * Blocks over B (typically packing into L3)
				 */
				resident       { b_matrix, l1_cntg, l2_strd, spec.l2_cntg_first,
				/*
				 * Thread belonging to the same team co-operatively Pack their current
				 * B submatrix
				 */
				bcms           { b_matrix, active_buffer_pool { total_threads, 1, std::move(b_alloc) }, spec.b_convert,
				/*
				 * Blocks over A  (typically packing L2)
				 */
				resident       { a_matrix, l1_cntg, l1_strd, spec.l1_cntg_first,
				/*
				 * Each thread packs its own chunk of A
				 */
				pack           { a_matrix, active_buffer_pool { total_threads, 1, a_alloc }, spec.a_convert,
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
				copy_matrix    { c_matrix, active_buffer_pool { total_threads, 1, c_alloc }, general_cntg_contig_generator{},
				/*
				 * invokes the kernel
				 */
				kernel_exec    { kspec.kernel, kspec.apply_beta }}}}}},
			gemm_reduce { axpby_kernel }};

		compute_position pos { 0, 0, 0, 0 };

		/*
		 * The driver returns a boolean signaling whether or not it was successfully
		 * able to acquire all of the threads we wanted to when setting up our memory
		 * and syncro objects.
		 *
		 * As those objects are dependent on getting the right number of threads, it is
		 * critical to bail out and try a different strategy
		 */
		return driver(pctx.a, pctx.b, pctx.c, pos, pctx.alpha, pctx.beta);
	}

public:
	static constexpr std::string_view name() { return "matmul3_interleaved_large"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matmul3_interleaved_strategy_tag>, ProblemContext>
	      && spec::has_get_spec<axpby_kernel_tag<spec::strategy_tag<matmul3_interleaved_large>>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if (! can_compute(pctx) ) return false;

		const auto spec         = get_spec(spec::strategy_tag<matmul3_interleaved_strategy_tag>{}, pctx);
		using spec_type         = decltype(spec);
		using architecture_spec = typename spec_type::architecture_spec_type;

		const auto nodes        = spec::numa_nodes(spec.max_threads, architecture_spec{});
		const auto cntg_threads = min(min(spec.max_threads, pctx.a.cntg()), nodes);

		//try 3D -- could fail on allocation
		if(cntg_threads >= 1 && impl_3d(cntg_threads, spec, pctx)) {
			return true;
		}

		return impl_2d(spec, pctx);
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	/**
	 * Because we have copy_matrix{c}, we should be able to handle any input matrix
	 * so always true
	 */
	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matmul3_interleaved_strategy_tag>, ProblemContext>
	      && spec::has_get_spec<axpby_kernel_tag<spec::strategy_tag<matmul3_interleaved_large>>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool can_compute(const ProblemContext& pctx) const {
		return pctx.alpha != zero<typename ProblemContext::scalar_type>
		    && pctx.beta_zero_mode != zero_mode::scale
		    && pctx.a.cntg_step() >= 1 && pctx.a.strd_step() >= 1
		    && pctx.b.cntg_step() >= 1 && pctx.b.strd_step() >= 1
		    && pctx.c.cntg_step() == 1 && pctx.c.strd_step() >= 1;
	}


	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; //class matmul3_interleaved_large

} //namespace perflibs::linalg::matmul

#endif //PERFLIBS_LINALG_MATMUL_STRATEGIES_LARGE_HPP

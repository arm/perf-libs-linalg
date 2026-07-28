/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_REDUCE_HPP
#define PERFLIBS_LINALG_REDUCE_HPP

#include "framework/buffer_pool.hpp"
#include "framework/compute_position.hpp"
#include "framework/parallel.hpp"
#include "framework/synchronization.hpp"
#include "framework/which.hpp"
#include "matrix/type_traits.hpp"

namespace perflibs::linalg {

/**
 * LINALG stack operator for performing a reduction operation.
 */
template<which_reduction_strategy ReductionStrategy, typename Buffer, typename NextMainWorkload, typename NextReduce>
class reduce {

	/**
	 * The total number of threads.
	 */
	const kernel_inttype threads_;

	/**
	 * The thread buffer used for the parallel reduction.
	 */
	Buffer buffer_;

	/**
	 * Synchro object for synchronization between threads.
	 */
	synchronization &synchro_;

	/**
	 * The next operator in the LINALG stack (main workload).
	 */
	NextMainWorkload next_main_workload_;

	/**
	 * The next operator in the LINALG stack (reduction).
	 */
	NextReduce next_reduce_;

	/**
	 * Do a serial reduction.
	 *
	 * @tparam CType The type of the parameter `c`.
	 * @param c [out] The matrix where the output is written.
	 * @param pos [in] The compute position.
	 * @param thread_num [in] The thread number for the current thread.
	 */
	template<typename CMatrixType, typename... Args>
	PERFLIBS_LINALG_INLINE
	void do_serial_reduction(const CMatrixType& c, const compute_position& pos, const Args&... args) {
		// Only the master thread does any work in a serial reduction.
		if (pos.thread_num != 0)
			return;

		for (kernel_inttype i = 0; i < buffer_.get_nbuffers(); i++) {
			auto c_thread_local_buffer = buffer_.get_buffer(i);
			CMatrixType thread_local_vector{ c_thread_local_buffer, c.cntg(), c.strd(), 1, c.cntg() };

			next_reduce_(c, thread_local_vector, pos, args...);
		}
	}

	/**
	 * Do a parallel reduction.
	 *
	 * @tparam CType The type of the parameter `c`.
	 * @param c [out] The matrix where the output is written.
	 * @param pos [in] The compute position.
	 * @param thread_num [in] The thread number for the current thread.
	 */
	template<typename CMatrixType, typename... Args>
	PERFLIBS_LINALG_INLINE
	void do_parallel_reduction(const CMatrixType& c, const compute_position& pos, const Args&... args) {
		auto split = make_parallel_split(c.cntg(), 1, threads_, c.cntg());

		const auto [chunk_start, chunk_size] = work_distribution(pos.thread_num, split, 1);
		auto sub_c = get_cntg_panel(c, chunk_start, chunk_size);

		for (kernel_inttype i = 0; i < buffer_.get_nbuffers(); i++) {
			auto c_thread_local_buffer = buffer_.get_buffer(i);
			general_matrix thread_local_vector { matrix_base { c_thread_local_buffer, c.cntg(), c.strd(), 1, c.cntg() } };
			auto sub_thread_local_vector = get_cntg_panel(thread_local_vector, chunk_start, chunk_size);

			next_reduce_(sub_c, sub_thread_local_vector, pos, args...);
		}
	}

public:
	PERFLIBS_LINALG_INLINE
	reduce(kernel_inttype physical_threads, Buffer& buffer, synchronization& synchro, NextMainWorkload next_main_workload, NextReduce next_reduce)
	:	threads_            { physical_threads }
	,	buffer_             { buffer }
	,	synchro_            { synchro }
	,	next_main_workload_ { std::move(next_main_workload) }
	,	next_reduce_        { std::move(next_reduce)        }
	{	}

	PERFLIBS_LINALG_INLINE
	reduce(which_reduction_strat_constant<ReductionStrategy>, kernel_inttype physical_threads, Buffer& buffer, synchronization& synchro, NextMainWorkload next_main_workload, NextReduce next_reduce)
	:	reduce { physical_threads, buffer, synchro, std::move(next_main_workload), std::move(next_reduce) }
	{	}

	/**
	 * Run the main workload for all threads. Run the reduction on the master thread if a serial reduction
	 * is chosen, or across all the threads if a parallel reduction is chosen.
	 *
	 * @tparam AType The type of the matrix `a`.
	 * @tparam BType The type of the matrix `b`.
	 * @tparam CType The type of the matrix `c`.
	 * @param [in,out] a Passed to the main workload operator which may modify the referenced object.
	 * @param [in,out] b Passed to the main workload operator which may modify the referenced object.
	 * @param [out] c The result of the reduction is written into this LINALG matrix. Not passed to the main
	 *                workload operator.
	 * @param [in] pos The compute position.
	 */
	template<typename AMatrixType, typename BMatrixType, typename CMatrixType, typename... Args>
	void operator()(const AMatrixType &a, const BMatrixType &b, const CMatrixType &c, const compute_position& pos, const Args&... args) {
		using scalar_type = typename CMatrixType::value_type;

		const bool is_first_block_in_thread = pos.block == 0;
		const bool is_last_block_in_thread = pos.block == pos.blocks - 1;

		auto c_thread_local_buffer = buffer_.get_buffer(pos.thread_num);

		general_matrix c_thread_local { matrix_base { c_thread_local_buffer, c.cntg(), c.strd(), 1, c.cntg() } };

		if (is_first_block_in_thread)
			set(zero<scalar_type>, c_thread_local);

		// Run the main workload for this thread.
		next_main_workload_(a, b, c_thread_local, pos, args...);

		if (!is_last_block_in_thread)
			return; // this thread has more work to do before the reduction step.

		if (threads_ > 1)
			synchro_();

		if constexpr (ReductionStrategy == which_reduction_strategy::serial) {
			do_serial_reduction(c, pos, args...);
		}
		else {
			do_parallel_reduction(c, pos, args...);
		}
	}

}; // class reduce

} // namespace perflibs::linalg

#endif /* ifndef PERFLIBS_LINALG_REDUCE_HPP */

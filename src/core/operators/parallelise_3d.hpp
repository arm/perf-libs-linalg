/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_PARALLELISE_3D_HPP
#define PERFLIBS_LINALG_PARALLELISE_3D_HPP

#include "framework/linalg_util.hpp"
#include "framework/parallel.hpp"
#include "framework/compute_position.hpp"

#include "matrix/operations.hpp"

namespace perflibs::linalg {

PERFLIBS_LINALG_INLINE
auto make_parallel_3d_split(
	kernel_inttype a_strd_threads, kernel_inttype a_strd, kernel_inttype a_strd_unroll,
	kernel_inttype b_strd_threads, kernel_inttype b_strd, kernel_inttype b_strd_unroll,
	kernel_inttype   cntg_threads, kernel_inttype   cntg, kernel_inttype   cntg_unroll) {

	return std::tuple {
		make_parallel_split(a_strd, a_strd_unroll, a_strd_threads, a_strd),
		make_parallel_split(b_strd, b_strd_unroll, b_strd_threads, b_strd),
		make_parallel_split(  cntg,   cntg_unroll,   cntg_threads,   cntg)
	};
}

/**
 * LINALG stack operator which parallelise over all 3 BLAS problem dimensions
 *
 * Race conditions caused by parallelising over `cntg` (shared dimension of A & B)
 * are handled by creating a duplication of the output matrix C and then performing
 * a reduction using AXPBY.
 * CheckCount checks whether or not the requested number of threads can be spawned
 */
template<bool CheckCount, typename BufferType, typename Next, typename NextReduce>
class parallelise_3d {
	parallel_split a_strd_split_;
	parallel_split b_strd_split_;
	parallel_split   cntg_split_;

	BufferType     buffer_;
	Next           next_;
	NextReduce     next_reduce_;

public:
	PERFLIBS_LINALG_INLINE
	parallelise_3d(parallel_split a_strd_split, parallel_split b_strd_split, parallel_split cntg_split, BufferType buffer, Next next, NextReduce next_reduce)
	:	a_strd_split_ { std::move(a_strd_split) }
	,	b_strd_split_ { std::move(b_strd_split) }
	,	cntg_split_   { std::move(cntg_split)   }
	,	buffer_       { std::move(buffer)       }
	,	next_         { std::move(next)         }
	,	next_reduce_  { std::move(next_reduce)  }
	{	}

	PERFLIBS_LINALG_INLINE
	parallelise_3d(std::bool_constant<CheckCount>, parallel_split a_strd_split, parallel_split b_strd_split, parallel_split cntg_split, BufferType buffer, Next next, NextReduce next_reduce)
	:	parallelise_3d(std::move(a_strd_split), std::move(b_strd_split), std::move(cntg_split), std::move(buffer), std::move(next), std::move(next_reduce))
	{	}


	template <typename AMatrixType, typename BMatrixType, typename CMatrixType, typename ScalarType, typename... Args>
	PERFLIBS_LINALG_INLINE
	bool operator()(const AMatrixType &a, const BMatrixType &b, const CMatrixType &c, compute_position pos, ScalarType alpha, ScalarType beta, Args... args) {
		const kernel_inttype total_threads =   cntg_split_.threads
		                                   * a_strd_split_.threads
		                                   * b_strd_split_.threads;

		const kernel_inttype a_b_threads = a_strd_split_.threads * b_strd_split_.threads;

		synchronization synchron { total_threads };

		bool success = true;

		parallel(total_threads, [&](const auto thread_num) {

			// Do not use this strategy if the number of
			// Threads requested cannot be spawned.
			if constexpr (CheckCount) {
				if (total_threads > omp::get_num_threads()) {
					if (thread_num == 0) {
						success = false;
					}
					return;
				}
			}

			const auto   cntg_thread_num = thread_num / a_b_threads;
			const auto a_strd_thread_num = ( thread_num % a_b_threads ) % a_strd_split_.threads;
			const auto b_strd_thread_num = ( thread_num % a_b_threads ) / a_strd_split_.threads;


			const auto [ a_strd_start, a_strd_thread_work ] = work_distribution(a_strd_thread_num, a_strd_split_, a_strd_split_.interleaved_rows);
			const auto [ b_strd_start, b_strd_thread_work ] = work_distribution(b_strd_thread_num, b_strd_split_, b_strd_split_.interleaved_rows);
			const auto [   cntg_start,   cntg_thread_work ] = work_distribution(  cntg_thread_num,   cntg_split_,   cntg_split_.interleaved_rows);

			auto thread_a = a.sub_matrix(cntg_start, cntg_thread_work, a_strd_start, a_strd_thread_work);
			auto thread_b = b.sub_matrix(cntg_start, cntg_thread_work, b_strd_start, b_strd_thread_work);

			if(cntg_start == 0) {
				//we are in the thread team that is going to operate directly into the user C matrix
				auto thread_c = c.sub_matrix(a_strd_start, a_strd_thread_work, b_strd_start, b_strd_thread_work);

				next_(thread_a, thread_b, thread_c, advance(pos, a_strd_start, b_strd_start, 0, 0, 0, 0, thread_num), alpha, beta, args...);

				synchron();

				// iterate over every other C-teams C buffer and reduce it into the users output matrix
				for(kernel_inttype i = 1; i < cntg_split_.threads; ++i) {
					auto buffer = buffer_.get_buffer(i * a_b_threads + thread_num);

					general_matrix team_local_c { matrix_base { buffer, c.cntg(), c.strd(), 1, c.cntg() } };

					auto thread_local_c  = team_local_c.sub_matrix(
						a_strd_start, a_strd_thread_work,
						b_strd_start, b_strd_thread_work);

					next_reduce_(thread_c, thread_local_c);
				}
			}
			else {
				auto buffer = buffer_.get_buffer(thread_num);

				//TODO apply some sort of ACTD to strd_step
				general_matrix team_local_c { matrix_base { buffer, c.cntg(), c.strd(), 1, c.cntg() } };

				auto thread_local_c  = team_local_c.sub_matrix(
					a_strd_start, a_strd_thread_work,
					b_strd_start, b_strd_thread_work);


				set(zero<ScalarType>, thread_local_c);

				next_(thread_a, thread_b, thread_local_c, advance(pos, a_strd_start, b_strd_start, cntg_start, 0, 0, 0, thread_num), alpha, one<ScalarType>, args...);

				synchron();
			}
		});
		return success;
	}
}; //class parallelise_3d

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_PARALLELISE_3D_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_PARALLELISE_2D_HPP
#define PERFLIBS_LINALG_PARALLELISE_2D_HPP

#include "framework/linalg_util.hpp"
#include "framework/parallel.hpp"
#include "framework/compute_position.hpp"

#include "matrix/operations.hpp"

namespace perflibs::linalg {

PERFLIBS_LINALG_INLINE
std::tuple<parallel_split, parallel_split> make_parallel_2d_split(
	kernel_inttype a_strd_threads, kernel_inttype a_strd, kernel_inttype a_strd_unroll,
	kernel_inttype b_strd_threads, kernel_inttype b_strd, kernel_inttype b_strd_unroll) {

	return {
		make_parallel_split(a_strd, a_strd_unroll, a_strd_threads, a_strd),
		make_parallel_split(b_strd, b_strd_unroll, b_strd_threads, b_strd)
	};
}

/**
 * Parallelise over the problem space over the dimensions a_strd and b_strd allocating
 * a thread to each block.
 *
 * If Spawn is true then parallel_2d will allocate the threads itself.
 *
 * @tparam Spawn see above
 * @tparam CheckCount checks whether or not the requested number of threads has been spawned
 */
template<bool Spawn, bool CheckCount, typename Next>
class parallelise_2d {
	constexpr static bool spawn_ { Spawn };

	parallel_split a_strd_split_;
	parallel_split b_strd_split_;
	Next           next_;

public:
	PERFLIBS_LINALG_INLINE
	parallelise_2d(parallel_split a_strd_split, parallel_split b_strd_split, Next next)
	:	a_strd_split_ { a_strd_split    }
	,	b_strd_split_ { b_strd_split    }
	,	next_         { std::move(next) }
	{	}

	PERFLIBS_LINALG_INLINE
	parallelise_2d(std::bool_constant<Spawn>, std::bool_constant<CheckCount>, parallel_split a_strd_split, parallel_split b_strd_split, Next next)
	:	parallelise_2d(std::move(a_strd_split), std::move(b_strd_split), std::move(next))
	{	}

	template <typename AType, typename BType, typename CType, typename... Args>
	PERFLIBS_LINALG_INLINE
	bool operator()(AType &a, BType &b, CType &c, compute_position pos, Args... args) {
		const kernel_inttype total_threads = a_strd_split_.threads * b_strd_split_.threads;

		if(! omp::is_mp || total_threads == 1) {
			//do nothing, pass straight through
			next_(a, b, c, pos, args...);
			return true;
		}
		else {
			bool success = true;

			auto work_load = [&](auto thread_num) {
				/*
				 * If CheckCount is set then we check that the number of threads we were
				 * allocated is at least as great as the number we need
				 *
				 * It is likely that the amount we were allocate will exceed the amount
				 * needed as we request the historic max threads. It is permissible for the
				 * OpenMP runtime to allocate us less threads, in which case we bail out
				 */
				if constexpr(CheckCount) {
					if(total_threads > omp::get_num_threads()) {
						if(thread_num == 0) {
							success = false;
						}
						return;
					}
				}

				const auto b_strd_thread_num = thread_num / a_strd_split_.threads;
				const auto a_strd_thread_num = thread_num - ( a_strd_split_.threads * b_strd_thread_num); //avoid using mod

				const auto [ a_strd_start, a_strd_thread_work ] = work_distribution(a_strd_thread_num, a_strd_split_, a_strd_split_.interleaved_rows);
				const auto [ b_strd_start, b_strd_thread_work ] = work_distribution(b_strd_thread_num, b_strd_split_, b_strd_split_.interleaved_rows);

				auto thread_a = get_strd_panel(a, a_strd_start, a_strd_thread_work);
				auto thread_b = get_strd_panel(b, b_strd_start, b_strd_thread_work);
				auto thread_c = c.sub_matrix(a_strd_start, a_strd_thread_work, b_strd_start, b_strd_thread_work);

				next_(thread_a, thread_b, thread_c, advance(pos, a_strd_start, b_strd_start, 0, 0, 0, 0, thread_num), args...);
			};

			if constexpr(spawn_) {
				parallel(total_threads, work_load);
			}
			else {
				work_load(omp::get_thread_num());
			}

			//will always return true if CheckCount is false
			return (!CheckCount) || success;
		}
	}
}; //class parallelise_2d

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_PARALLELISE_2D_HPP

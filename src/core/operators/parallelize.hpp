/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_PARALLELIZE_HPP
#define PERFLIBS_LINALG_PARALLELIZE_HPP

#include "perflibs_numeric_utils.hpp"
#include "perflibs_util.hpp"
#define PERFLIBS_LINALG_DIAGNOSTIC(fmt, ...) printf("[DIAGNOSTIC] " fmt "\n", ##__VA_ARGS__);

#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"
#include "framework/parallel.hpp"
#include "framework/which.hpp"

#include "operators/paneller.hpp"

#include "matrix/type_traits.hpp"

#include "perflibs_assert.hpp"
#include "timer.hpp"

#include <tuple>

namespace perflibs::linalg {
namespace {

template <which_parallel_strat Strategy, which_dimension Dimension, typename Next,
         bool Diagnostic = false>
class parallelize {

	/**
	 * The maximum number of threads to use.
	 */
	kernel_inttype max_threads_;

	kernel_inttype interleave_factor_ { 1 };

	/**
	 * The next operator in the LINALG stack.
	 */
	Next next_;

	/**
	 * Split over a general matrix and parallelize.
	 * @tparam A The type of the matrix `a`.
	 * @tparam B The type of the matrix `b`.
	 * @tparam C The type of the matrix `c`.
	 * @param [in] a A secondary matrix to parallelize over. This matrix will be split in-line with `c` to
	 *               create a subproblem.
	 * @param [in] b A secondary matrix to parallelize over. This matrix will be split in-line with `c` to
	 *               create a subproblem.
	 * @param [in, out] c The general matrix to parallelize over.
	*/
	template <typename AType, typename BType, typename CType, typename... Args>
	PERFLIBS_LINALG_INLINE void parallelize_general(const AType &a, const BType &b, const CType &c, compute_position pos,
	                                           Args... args) {
		paneller<Dimension, Next> pan(next_);
		const auto parallelize_dimension_size = pan.size(a, b, c);


		// Split up the matrix into panels and parallelize accordingly.
		auto split = make_parallel_split(parallelize_dimension_size,
			interleave_factor_, max_threads_, parallelize_dimension_size);

		parallel(split.threads, [=](auto thread_num) mutable {
			pos.thread_num = thread_num;

			if constexpr (Diagnostic) {
				auto start_time = perflibs::timer_start();

				const auto[chunk_start, chunk_size] = work_distribution(thread_num, split, 1);
				pan(a, b, c, chunk_start, chunk_size, pos, 0, 1, args...); // zereoth block of one

				auto time_delta = perflibs::timer_end(std::move(start_time));

				PERFLIBS_LINALG_DIAGNOSTIC("thread %d took %f seconds to complete. Chunk start %d Chunk size %d",
					thread_num, time_delta, int(chunk_start), int(chunk_size));
			}
			else {
				const auto[chunk_start, chunk_size] = work_distribution(thread_num, split, 1);


				pan(a, b, c, chunk_start, chunk_size, pos, 0, 1, args...); // zereoth block of one
			}
		});
	}

	/**
	 * Split over a triangular matrix and parallelize.
	 *
	 * To split over a triangular matrix with this technique, we do the following -- suppose we have a
	 * triangular matrix:
	 *
	 *     A B C D E
	 *       B C D E
	 *         C D E
	 *           D E
	 *             E
	 *
	 * We combine 'opposite' panels like so to ensure there is an even amount of work being done per thread:
	 *
	 *     A   B
	 *     E   B   C
	 *     E + D + C
	 *     E   D   C
	 *     E   D
	 *     E   D
	 *
	 * @tparam A The type of the matrix `a`.
	 * @tparam B The type of the matrix `b`.
	 * @tparam C The type of the matrix `c`.
	 * @param [in] a A secondary matrix to parallelize over. This matrix will be split in-line with `c` to
	 *               create a subproblem.
	 * @param [in] b A secondary matrix to parallelize over. This matrix will be split in-line with `c` to
	 *               create a subproblem.
	 * @param [in, out] c The primary triangular matrix to parallelize over.
	*/
	template <typename AType, typename BType, typename CType, typename... Args>
	PERFLIBS_LINALG_INLINE void parallelize_triangular_split(AType &a, BType &b, CType &c,
	                                                        compute_position pos, Args... args) {

		static_assert(
			( Dimension == which_dimension::a_strd && ( is_triangular_form_v<AType> || is_triangular_form_v<CType> ) ) ||
			( Dimension == which_dimension::b_strd && ( is_triangular_form_v<BType> || is_triangular_form_v<CType> ) ) ||
			( Dimension == which_dimension::cntg   && ( is_triangular_form_v<AType> || is_triangular_form_v<BType> ) ),
			"Dimension  over which to parallelise does not belong to a triangular matrix"
		);

		paneller<Dimension, Next> pan(next_);
		const auto parallelize_dimension_size = pan.size(a, b, c);

		// If the matrix has size one in the dimension we're parallelising over, get straight on with it.
		if (parallelize_dimension_size <= 1) {
			// zeroeth block of one.
			next_(a, b, c, advance(pos, 0, 0, 0, 0, 0, 1), args...);
			return;
		}

		const kernel_inttype half_work = parallelize_dimension_size / 2;
		const kernel_inttype threads_to_use = perflibs::min(half_work, max_threads_);

		// NOTE: left_work may be greater than right work if the number of columns/rows in the matrix is odd.
		//       We pass right_split.threads to make_parallel_split for left_split because we want to use the
		//       same number of threads for both left and right work.
		const kernel_inttype left_work = half_work + kernel_inttype{ parallelize_dimension_size % 2 != 0 };
		const auto right_split = make_parallel_split(half_work, 1, threads_to_use, parallelize_dimension_size);
		const auto left_split = make_parallel_split(left_work, 1, right_split.threads, parallelize_dimension_size);

		// Spin off threads.
		parallel(threads_to_use, [=, &args...](auto thread_num) mutable {
			/*
			 * left_start and left_size describe a panel on the leftmost side of the matrix.
			 * right_start and right_size describe a panel on the rightmost side of the matrix.
			 * These panels should have an approximately equal number number of elements (combined) for each
			 * thread, therefore splitting up work evenly.
			 */
			auto [left_start, left_size] = work_distribution(thread_num, left_split, 1);
			auto [right_start, right_size] = work_distribution(threads_to_use - thread_num - 1, right_split, 1);
			right_start += left_work;

			pos.thread_num = thread_num;

			if constexpr (Diagnostic) {
				auto start_time = perflibs::timer_start();
				if (left_start + left_size == right_start) {
					pan(a, b, c, left_start, left_size + right_size, pos, 0, 1, args...); // zeroeth block of one
				}
				else {
					pan(a, b, c, left_start, left_size, pos, 0, 2, args...);   // zeroeth block of two
					pan(a, b, c, right_start, right_size, pos, 1, 2, args...); // first block of two
				}
				auto time_delta = perflibs::timer_end(std::move(start_time));
				PERFLIBS_LINALG_DIAGNOSTIC("thread %d took %f seconds to complete.", thread_num, time_delta);
			}
			else {
				if (left_start + left_size == right_start) {
					pan(a, b, c, left_start, left_size + right_size, pos, 0, 1, args...); // zeroeth block of one
				}
				else {
					pan(a, b, c, left_start, left_size, pos, 0, 2, args...);   // zeroeth block of two
					pan(a, b, c, right_start, right_size, pos, 1, 2, args...); // first block of two
				}
			}
		});
	}

public:
	parallelize(kernel_inttype max_threads, Next next)
	:	max_threads_{ max_threads     }
	,	next_       { std::move(next) }
	{	}

	parallelize(kernel_inttype max_threads, kernel_inttype interleave_factor, Next next)
	:	max_threads_       { max_threads     }
	,	interleave_factor_ { interleave_factor }
	,	next_              { std::move(next) }
	{	}

	parallelize(Next next)
	:	parallelize(omp::get_max_threads(), std::move(next)) { }

	parallelize(which_parallel_strat_constant<Strategy>, which_dimension_constant<Dimension>, kernel_inttype max_threads, Next next)
	:	parallelize(max_threads, std::move(next)) { }

	parallelize(which_parallel_strat_constant<Strategy>, which_dimension_constant<Dimension>, kernel_inttype max_threads, kernel_inttype interleave_factor, Next next)
	:	parallelize(max_threads, interleave_factor, std::move(next)) { }

	/**
	 * Split into subproblems over `c` and parallelize.
	 *
	 * We decide how to parallelize over a matrix based on the specific type of `a`.
	 *
	 * @tparam A The type of the matrix `a`.
	 * @tparam B The type of the matrix `b`.
	 * @tparam C The type of the matrix `c`.
	 * @param [in] a A secondary matrix to parallelize over. This matrix will be split in-line with `c` to
	 *               create a subproblem.
	 * @param [in] b A secondary matrix to parallelize over. This matrix will be split in-line with `c` to
	 *               create a subproblem.
	 * @param [in, out] c The primary matrix to parallelize over.
	 */
	template <typename AType, typename BType, typename CType, typename... Args>
	PERFLIBS_LINALG_INLINE void operator()(const AType &a, const BType &b, const CType &c, compute_position pos, Args &&... args) {
		PERFLIBS_ASSERT(max_threads_ >= 1, "max_threads_ is less than or equal to zero!");

		if (!omp::is_mp || max_threads_ == 1) {
			next_(a, b, c, advance(pos, 0, 0, 0, 0, 0, 1), args...);
		}
		else if constexpr (Strategy == which_parallel_strat::general) {
			parallelize_general(a, b, c, pos, std::forward<Args>(args)...);
		}
		else if constexpr (Strategy == which_parallel_strat::triangular) {
			parallelize_triangular_split(a, b, c, pos, std::forward<Args>(args)...);
		}
		else {
			// Unreachable.
			PERFLIBS_ASSERT(false, "Unknown `which_parallel_strat` selected.")
		}
	}

}; // class parallelize

/* Helper functions for parallelization. */

// Parallelize over the destination matrix's `cntg` dimensions.
template <which_parallel_strat Strategy, typename Next>
using parallelize_dest_cntg = parallelize<Strategy, which_dimension::a_strd, Next>;

// Parallelize over the destination matrix's `strd` dimensions.
template <which_parallel_strat Strategy, typename Next>
using parallelize_dest_strd = parallelize<Strategy, which_dimension::b_strd, Next>;

// Parallelize over the `cntg` dimension.
template <which_parallel_strat Strategy, typename Next>
using parallelize_cntg = parallelize<Strategy, which_dimension::cntg, Next>;

}
} // namespace perflibs::linalg

#endif /* ifndef PERFLIBS_LINALG_PARALLELIZE_HPP */

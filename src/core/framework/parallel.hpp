/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_PARALLEL
#define PERFLIBS_LINALG_PARALLEL

#include "perflibs_assert.hpp"
#include "detect/omp.hpp"
#include "perflibs_unused.hpp"
#include "perflibs_util.hpp"
#include "perflibs_numeric_utils.hpp"
#include "linalg_util.hpp"
#include "dyn_array.hpp"

namespace perflibs::linalg {

/**
 * We keep track of the global maximum number of threads that we've
 * needed since library startup. For any parallel region, we request
 * this number of threads so that the OpenMP runtime never shrinks
 * the thread pool, avoiding the overhead of unnecessarily spawning
 * threads in the future.
 */
class parallel_state_t {
	int max_threads;
public:
	parallel_state_t() : max_threads { 1 } {}

	/**
	 * Return the number of threads to spawn for a parallel region given
	 * the number of threads that we want to use, updating the global
	 * state (i.e. max threads we've used so far).
	 *
	 * Note: this is not an atomic operation and is therefore not thread
	 * safe. If multiple threads attempt to perform an update then we
	 * may end up with a garbage value in max_threads, but this will not
	 * propagate: our bounds checks ensures that the returned value lies
	 * in the range [ nt, omp_get_max_threads() ].
	 *
	 * @param[in] nt The number of threads that we want to use.
	 * @returns      The number of threads to spawn.
	 */
	PERFLIBS_LINALG_INLINE int request(const int nt) {
		// Operate on a thread-local copy of global max
		const int current_max = max_threads;

		if (nt > current_max) {
			const int new_max = omp::bounded_get_max_threads(nt);
			max_threads = new_max; // update global state
			return new_max;
		}
		return omp::bounded_get_max_threads(current_max);
	}
}; // class parallel_state_t

// inline: ensure all TUs share the same parallel_state object
inline parallel_state_t parallel_state;

/**
 * OpenMP parallel loop abstraction
 *
 * @param nt number of threads
 * @param f the callable object to execute in parallel, takes form  f(int) -> (ignored)
 */
template<typename Func>
PERFLIBS_LINALG_INLINE
void parallel(int nt, Func f) {
	PERFLIBS_ASSERT(nt > 0, "number of threads must be positive");

	if constexpr(omp::is_mp) {
		if(nt > 1) {
			// Decide how many threads to spawn
			const int num_spawn = parallel_state.request(nt);
			PERFLIBS_UNUSED(num_spawn); // suppress erroneous warning from clang

			#pragma omp parallel for num_threads(num_spawn) firstprivate(f)
			for(int i=0; i<nt; ++i) {
				f(i);
			}
			return;
		}
	}

	/*
	 * we need to use a loop here in case the workload has been
	 * split up despite not having multiple threads, this also
	 * handles the nt < 1 case correctly
	 */
	for(int i=0; i<nt; ++i) {
		f(i);
	}
}


/**
 * A version of parallel where the number of iteration may differ from the number of threads
 * in this case, it is up to openMP to default schedule how it wish to divide this work up
 *
 * This version of parallel should be avoided where possible in preference to calculating how
 * many thread you actually want to use, and then allocate the work load based on threadid
 *
 * @param iterations [in] the number of iterations to perform
 * @param nt [in] number of threads
 * @param f [in] the callable object to execute in parallel, takes form  f(int) -> (ignored)
 */
template<typename Func>
void parallel(int iterations, int nt, Func f) {
	PERFLIBS_ASSERT(nt > 0, "number of threads must be positive");

	if (nt > 1) {
		#pragma omp parallel for num_threads(nt) firstprivate(f)
		for(int i=0;i<iterations;++i) {
			f(i);
		}
	}
	else {
		for(int i=0;i<iterations;++i) {
			f(i);
		}
	}
}



/**
 * Performs a reduction operation in parallel on a function that takes as a
 * single argument the thread ID on which it operates, and returns a value of
 * type T to accumulate into.
 * @tparam T  The data type of the result returned
 * @tparam F  The type of function to operate on. This must take a single
 *            integer as a parameter, and return a value of type \c T
 * @param nt   [in] The number of threads to use
 * @param f    [in] Function which performs part of the parallel computation for a single thread
 * @return The value of performing \c f on each thread and summing the values together
 */
template<typename T, typename F>
T reduce_add_parallel(int nt, F f) {
	PERFLIBS_ASSERT(nt > 0, "number of threads must be positive");
	if (nt == 1)
		return f(0);

	T result = zero<T>;
	// Try and use an array rather than a vector. dyn_array uses the appropriate
	// data type depending on whether capacity requirements are larger or
	// smaller than the static size of the structure. Set the static size to 256
	// (number of processors on dual-socket Altra Max), and request for the size
	// to be the number of threads passed in.
	perflibs::dyn_array<T, 256> tmp_result((size_t)nt);

	// Decide how many threads to spawn
	const int num_spawn = parallel_state.request(nt);
	PERFLIBS_UNUSED(num_spawn); // suppress erroneous warning from clang

	#pragma omp parallel for default(none) num_threads(num_spawn) firstprivate(f, nt) shared(tmp_result)
	for (int i = 0; i < nt; ++i) {
		tmp_result[i] = f(i);
	}
	for (int i = 0; i < nt; ++i) {
		result += tmp_result[i];
	}
	return result;
}

struct parallel_split {
	///how many threads are we going to use
	kernel_inttype threads;

	kernel_inttype work_per_thread;

	kernel_inttype total_actual_work;
	//the amount to add to each thread that needs additional work
	kernel_inttype modifier;
	// the number of threads we need to add additional work to
	kernel_inttype modified_threads;

	kernel_inttype interleaved_rows;
}; //struct parallel_split


/**
 * Produces an even split of a work size across a specified number of threads
 * @param dimension the worksize
 * @param interleave specifies where there is a degree of indivisibility to the worksize
 * @param max_threads the number of thread to split the work across
 * @param _ legacy parameter that once required but now defunct. To be removed in future
 */
namespace {

PERFLIBS_LINALG_INLINE
parallel_split make_parallel_split(kernel_inttype dimension, kernel_inttype interleave, kernel_inttype max_threads, kernel_inttype _ = 1) {
	PERFLIBS_UNUSED(_);

	if(max_threads == 1 || dimension == 0) return { 1, dimension, dimension, 0, 1 };

	const kernel_inttype actual_work      = iround_div(dimension, interleave);
	const kernel_inttype threads          = min(max_threads, actual_work);
	const kernel_inttype work_per_thread  = actual_work / threads;
	const kernel_inttype modified_threads = actual_work - ( threads * work_per_thread); //avoid using mod

	return { threads, work_per_thread, dimension, 1_ki, modified_threads, interleave };
}

enum split_options {
	min_work, chunk_spread, chunk_fill
};

PERFLIBS_LINALG_INLINE
parallel_split make_parallel_split_min_work(kernel_inttype dimension, kernel_inttype max_threads, kernel_inttype first_chunk_size, split_options split_option) {
	const auto min_work_workloads = split_option == split_options::min_work
	                              ? iround_floor(dimension, min_work) / first_chunk_size
	                              : iround_div(dimension, first_chunk_size); //fill & spread

	const auto min_work_threads   = max(min(max_threads, min_work_workloads), 1);
	auto split                    = make_parallel_split(dimension, 1, min_work_threads);

	if(split_option == split_options::chunk_fill && min_work_workloads <= max_threads ) {
		split.modifier         = first_chunk_size - split.work_per_thread;
		split.modified_threads = (split.work_per_thread * split.threads ) / split.modifier;
	}
	return split;
}

} // namespace anon

PERFLIBS_LINALG_INLINE static
std::pair<kernel_inttype, kernel_inttype> work_distribution(kernel_inttype thread_num, const parallel_split& split, kernel_inttype _ = 1) {
	PERFLIBS_UNUSED(_);

	kernel_inttype start       = split.work_per_thread * thread_num;
	kernel_inttype thread_work = split.work_per_thread;

	if(thread_num < split.modified_threads) {
		thread_work += split.modifier;
		start       += thread_num * split.modifier;
	}
	else {
		start += split.modified_threads;
	}

	const kernel_inttype actual_start = start * split.interleaved_rows;

	if(thread_num == (split.threads - 1)) {
		const kernel_inttype actual_work = split.total_actual_work - actual_start;

		return { actual_start, actual_work };
	}
	else {
		const kernel_inttype actual_work = thread_work * split.interleaved_rows;

		return { actual_start, actual_work };
	}
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_PARALLEL

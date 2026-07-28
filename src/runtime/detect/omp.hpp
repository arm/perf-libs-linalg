/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_OMP_HPP
#define PERFLIBS_OMP_HPP

#include "cpu_topology.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace perflibs {

namespace linalg {

namespace omp {

#ifdef _OPENMP
inline constexpr bool is_mp = true;
#else
inline constexpr bool is_mp = false;
#endif

/**
 * Return whether the next parallel level will be a parallel region in which we want to
 * use multithreading.
 *
 * Note: OMP_NESTED is deprecated, and since we have moved to a single build of GCC we
 * have elected to ignore it completely. This changes behavior compared with pre 24.04
 * GCC builds of the library. Refer any users to documentation if we have queries about
 * this.
 */
inline bool get_nested() {
#ifdef _OPENMP
	// General note: Levels are indexed from zero; here, return true if the current level is smaller than the max.
	return omp_get_active_level() < omp_get_max_active_levels();
#else
	return false;
#endif
}

/**
 * Use this function to get the max number of threads that may be spawned in
 * the next parallel region. This is not guaranteed to be the number actually spawned.
 */
inline int get_max_threads() {
#ifdef _OPENMP
	return get_nested() ? omp_get_max_threads() : 1;
#else
	return 1;
#endif
}

/** Returns the physical number of cores */
inline int get_num_procs() {
#ifdef _OPENMP
	return get_nested() ? omp_get_num_procs() : 1;
#else
	return 1;
#endif
}

/** Gets max cores regardless of hyperthreading, taking into account OMP_NUM_THREADS. */
inline int get_max_cores() {
#ifdef _OPENMP
	const auto total_cores = ::perflibs::cpu_topology::get_total_cores();
	const auto max_threads = get_max_threads();
	return total_cores < max_threads ? static_cast<int>(total_cores) : max_threads;
#else
	return 1;
#endif
}

/*
 * If openmp build then it returns the result of
 *     omp_get_num_threads()
 * else
 *     returns 1 (serial build)
 */
inline int get_num_threads() {
#ifdef _OPENMP
	return omp_get_num_threads();
#else
	return 1;
#endif
}

/**
 * Returns the min of the specified max value and
 * omp::get_max_threads.
 *
 * Used for thread throttling where you want to specify a max threads
 * for the algorithm and min with the max threads the users has given
 */
inline int bounded_get_max_threads(int max) {
	// Avoid calling get_max_threads when we can return immediately.
	if (max <= 1)
		return 1;

	const auto max_threads = get_max_threads();
	return max < max_threads ? max : max_threads;
}

inline void set_num_threads(int nthreads) {
#ifdef _OPENMP
	omp_set_num_threads(nthreads);
#endif
}

/*
 * If openmp build then it returns the result of
 *     omp_in_parallel()
 * else
 *     returns false (serial build)
 */
inline bool in_parallel() {
#ifdef _OPENMP
	return omp_in_parallel();
#else
	return false;
#endif
}

/*
 * If openmp build then it returns the result of
 *     omp_get_thread_num()
 * else
 *     returns 0 (ie the first of 1 threads)
 */
inline int get_thread_num() {
#ifdef _OPENMP
	return omp_get_thread_num();
#else
	return 0;
#endif
}

} // namespace omp

} // namespace linalg

} // namespace perflibs

#endif

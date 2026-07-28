/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FRAMEWORK_2D_SPLIT_HPP
#define PERFLIBS_LINALG_FRAMEWORK_2D_SPLIT_HPP

#include "framework/linalg_util.hpp"
#include "perflibs_assert.hpp"
#include <cmath>

namespace perflibs::linalg {

namespace { //<anon>

/**
 * Given two dimensions and a maximum number of threads to utilize, calculate the best
 * combination of threads that fit in (multiplied together) max_threads.
 *
 * This algorithm assumes that work in either of the dimensions is equally difficult
 * to compute
 *
 * @returns [m_nthreads, n_nthreads] A pair of the threads that should be used in each dimension
 */
PERFLIBS_LINALG_INLINE
std::pair<kernel_inttype, kernel_inttype> split_2d(kernel_inttype max_threads, kernel_inttype m, kernel_inttype n) {
	//if we've only got one thread or a zero dims, return the trivial split
	if (max_threads == 1 || m == 0 || n == 0) {
		return { 1, 1 };
	}

	PERFLIBS_ASSERT(m > 0, "m should be positive in split_2d\n");
	PERFLIBS_ASSERT(n > 0, "n should be positive in split_2d\n");

	/*
	 * We want the same ratio of threads in M & N to the ratio of m and n problem size
	 *
	 * Therefore:	mt/nt == m/n	where mt*nt == max_threads
	 *
	 *			 max_threads/nt = mt	&	(max_threads/nt) * (m/n) = nt
	 *		  nt^2 = max_threads * (m/n)
	 *		  nt = sqrt( max_threads * (m/n) )
	 */
	//ratio of m to n in problem dimensions
	double ratio = m / static_cast<double>(n);

	// nt = sqrt(max_threads * (m / n) )
	const kernel_inttype adjusted = round(std::sqrt(max_threads * ratio));

	//find the nearest factor of max_threads
	for(kernel_inttype i = 0; i!= adjusted; ++i) {
		//try down
		const kernel_inttype adj_down = adjusted - i;
		if(max_threads % adj_down == 0)
			return { adj_down, max_threads / adj_down };

		//try up
		const kernel_inttype adj_up = adjusted + i;
		if(max_threads % adj_up == 0)
			return { adj_up, max_threads / adj_up };
	}

	// No exact factorization match found; fall back by biasing maxes toward the larger dimension.
	if(m > n) return{ min(m, max_threads), 1 };
	else      return{ 1, min(n, max_threads) };
}

} //namespace //<anon>

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_FRAMEWORK_2D_SPLIT_HPP

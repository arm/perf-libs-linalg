/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_REDUCTION_KERNEL_HPP
#define PERFLIBS_LINALG_FACTORIZATION_REDUCTION_KERNEL_HPP

#include "matrix/matrix.hpp"

#include "perflibs_util.hpp"

namespace perflibs::linalg::factorization {

/*
 * Performs reduction across thread workspaces and stores results in the output matrix.
 *
 * This function must be called from within an OpenMP parallel region.
 * It takes the thread-local results from each thread's workspace section and
 * combines them into a final result in the output matrix.
 *
 * MatrixType: Type of the output matrix
 *
 * thread_workspace: Pointer to the start of the thread workspace memory
 * output_matrix: Matrix where the reduction results will be stored
 * n: Number of elements to process
 * num_threads: Number of threads participating in the reduction
 */
template<typename ArchitectureSpec, typename MatrixType>
PERFLIBS_LINALG_INLINE void vectors_reduction(const typename MatrixType::value_type *thread_workspace,
                                         MatrixType& output_matrix, kernel_inttype n,
                                         kernel_inttype num_threads) {

	#pragma omp for schedule(static)
	for (auto j = 0_ki; j < n; ++j) {
		typename MatrixType::value_type sum = 0.0;
		#pragma omp simd
		for (auto t = 0_ki; t < num_threads; ++t) {
			sum += thread_workspace[t * n + j];
		}
		output_matrix(j, 0, write) = sum;
	}
}
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_REDUCTION_KERNEL_HPP

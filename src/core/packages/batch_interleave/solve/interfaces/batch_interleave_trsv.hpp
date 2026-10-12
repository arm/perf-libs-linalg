/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_SOLVE_INTERFACES_BATCH_INTERLEAVE_TRSV_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_SOLVE_INTERFACES_BATCH_INTERLEAVE_TRSV_HPP

#include "packages/batch_interleave/solve/interfaces/batch_interleave_trsm.hpp"

namespace perflibs::linalg::batch_interleave::solve {

/** In-place triangular vector solve over batches of interleaved vectors. */
template<typename IntType, typename T, typename ArchitectureSpec>
pl_linalg_status_t batch_interleave_trsv(IntType ninter, IntType nbatch, char uplo, char transa, char diag,
                                     IntType n, const T *a, IntType batch_step_a, IntType a_row_step,
                                     IntType a_column_step, T *x, IntType batch_step_x, IntType x_step) {
	const auto x_column_step = ninter * x_step;
	return batch_interleave_trsm<IntType, T, ArchitectureSpec>(
		ninter, nbatch, 'L', uplo, transa, diag, n, 1, one<T>, a, batch_step_a, a_row_step,
		a_column_step, x, batch_step_x, x_step, x_column_step);
}

} // namespace perflibs::linalg::batch_interleave::solve

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_SOLVE_INTERFACES_BATCH_INTERLEAVE_TRSV_HPP

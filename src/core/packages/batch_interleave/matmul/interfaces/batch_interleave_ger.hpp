/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_INTERFACES_BATCH_INTERLEAVE_GER_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_INTERFACES_BATCH_INTERLEAVE_GER_HPP

#include "pl_linalg_status.h"
#include "detect/system.hpp"
#include "packages/batch_interleave/matmul/strategies.hpp"

namespace perflibs::linalg::batch_interleave::matmul {

/** Rank-one update over batches of interleaved matrices. */
template<typename IntType, typename T, typename ArchitectureSpec>
pl_linalg_status_t batch_interleave_ger(IntType ninter, IntType nbatch, IntType m, IntType n, T alpha,
                                    const T *x, IntType batch_step_x, IntType x_step,
                                    const T *y, IntType batch_step_y, IntType y_step,
                                    T *a, IntType batch_step_a, IntType a_row_step, IntType a_column_step) {

	if (ninter == 0 || nbatch == 0 || m == 0 || n == 0 || alpha == zero<T>) {
		return PL_LINALG_STATUS_SUCCESS;
	}

	if (m < 0 || n < 0 || ninter < 0 || nbatch < 0 || x_step < ninter || y_step < ninter ||
	    a_row_step < ninter || a_column_step < ninter || batch_step_x < kernel_inttype { x_step } * m ||
	    batch_step_y < kernel_inttype { y_step } * n || batch_step_a < kernel_inttype { ninter } * m * n) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	// matmul3 stores the reduction dimension first; GER has a single outer product.
	const auto x_matrix = general_matrix {
		interleave_batch_matrix { x, ninter, 1, m, 0, x_step }
	};
	const auto y_matrix = general_matrix {
		interleave_batch_matrix { y, ninter, 1, n, 0, y_step }
	};
	const auto a_matrix = general_matrix {
		interleave_batch_matrix { a, ninter, m, n, a_row_step, a_column_step }
	};

	const spec::problem_context pctx {
		matmul3_problem<T> {
			strided_batch_view { x_matrix, nbatch, batch_step_x },
			strided_batch_view { y_matrix, nbatch, batch_step_y },
			strided_batch_view { a_matrix, nbatch, batch_step_a },
			alpha,
			one<T>,
			zero_mode::set
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	compute(pctx);
	return PL_LINALG_STATUS_SUCCESS;
}

} // namespace perflibs::linalg::batch_interleave::matmul

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_INTERFACES_BATCH_INTERLEAVE_GER_HPP

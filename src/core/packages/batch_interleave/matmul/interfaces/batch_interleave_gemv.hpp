/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_INTERFACES_BATCH_INTERLEAVE_GEMV_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_INTERFACES_BATCH_INTERLEAVE_GEMV_HPP

#include "pl_linalg_status.h"

#include "detect/system.hpp"
#include "framework/linalg_blas_types.hpp"
#include "matrix/adaptors.hpp"
#include "matrix/interleave_batch_matrix.hpp"
#include "matrix/strided_batch_view.hpp"
#include "packages/batch_interleave/matmul/strategies.hpp"
#include "spec/problem_context.hpp"

namespace perflibs::linalg::batch_interleave::matmul {

/** General matrix-vector multiplication over batches of interleaved operands. */
template<typename IntType, typename T, typename ArchitectureSpec>
pl_linalg_status_t batch_interleave_gemv(IntType ninter, IntType nbatch, char transa, IntType m, IntType n,
                                     T alpha, const T *a, IntType batch_step_a, IntType a_row_step,
                                     IntType a_column_step, const T *x, IntType batch_step_x,
                                     IntType x_step, T beta, T *y, IntType batch_step_y, IntType y_step) {
	if (m == 0 || n == 0) {
		return PL_LINALG_STATUS_SUCCESS;
	}

	const auto a_trans = c_to_trans(transa);
	if (!a_trans) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	if (ninter < 0 || nbatch < 0 || m < 0 || n < 0 || batch_step_a < ninter || batch_step_x < ninter ||
	    batch_step_y < ninter || a_row_step < ninter || a_column_step < ninter || x_step < ninter ||
	    y_step < ninter) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	if (ninter == 0 || nbatch == 0) {
		return PL_LINALG_STATUS_SUCCESS;
	}

	const auto is_a_trans       = is_trans(a_trans);
	const auto output_size      = is_a_trans ? n : m;
	const auto reduction_size   = is_a_trans ? m : n;
	const auto a_reduction_step = is_a_trans ? a_row_step : a_column_step;
	const auto a_output_step    = is_a_trans ? a_column_step : a_row_step;

	const auto a_matrix = general_matrix {
		interleave_batch_matrix { a, ninter, reduction_size, output_size, a_reduction_step, a_output_step }
	};
	const auto b_matrix = general_matrix {
		interleave_batch_matrix { x, ninter, reduction_size, 1, x_step, 0 }
	};
	const auto c_matrix = general_matrix {
		interleave_batch_matrix { y, ninter, output_size, 1, y_step, 0 }
	};

	const spec::problem_context pctx {
		matmul3_problem<T> {
			strided_batch_view { a_matrix, nbatch, batch_step_a },
			strided_batch_view { b_matrix, nbatch, batch_step_x },
			strided_batch_view { c_matrix, nbatch, batch_step_y },
			alpha,
			beta,
			zero_mode::set
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	compute(pctx);
	return PL_LINALG_STATUS_SUCCESS;
}

} // namespace perflibs::linalg::batch_interleave::matmul

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_INTERFACES_BATCH_INTERLEAVE_GEMV_HPP

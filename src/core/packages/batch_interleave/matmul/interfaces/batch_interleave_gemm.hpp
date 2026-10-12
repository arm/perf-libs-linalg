/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_INTERFACES_BATCH_INTERLEAVE_GEMM_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_INTERFACES_BATCH_INTERLEAVE_GEMM_HPP

#include "pl_linalg_status.h"

#include "detect/system.hpp"
#include "framework/linalg_blas_types.hpp"
#include "matrix/adaptors.hpp"
#include "matrix/interleave_batch_matrix.hpp"
#include "matrix/strided_batch_view.hpp"
#include "packages/batch_interleave/matmul/strategies.hpp"
#include "spec/problem_context.hpp"

namespace perflibs::linalg::batch_interleave::matmul {

/** General matrix multiplication over batches of interleaved matrices. */
template<typename IntType, typename T, typename ArchitectureSpec>
pl_linalg_status_t batch_interleave_gemm(IntType ninter, IntType nbatch, char transa, char transb, IntType m,
                                     IntType n, IntType k, T alpha, const T *a, IntType batch_step_a,
                                     IntType a_row_step, IntType a_column_step, const T *b,
                                     IntType batch_step_b, IntType b_row_step, IntType b_column_step, T beta,
                                     T *c, IntType batch_step_c, IntType c_row_step, IntType c_column_step) {

	const auto a_trans = c_to_trans(transa);
	const auto b_trans = c_to_trans(transb);
	if (!a_trans || !b_trans) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	// Preserve the existing interleave-batch quick-return behavior.
	if (ninter == 0 || nbatch == 0 || m == 0 || n == 0) {
		return PL_LINALG_STATUS_SUCCESS;
	}

	if (ninter < 0 || nbatch < 0 || m < 0 || n < 0 || k < 0 || batch_step_a < ninter ||
	    batch_step_b < ninter || batch_step_c < ninter || a_row_step < ninter || b_row_step < ninter ||
	    c_row_step < ninter || a_column_step < ninter || b_column_step < ninter || c_column_step < ninter) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	// matmul3 stores the reduction dimension as the contiguous dimension.
	const auto a_contiguous_step = a_trans == PERFLIBS_NOTRANS ? a_column_step : a_row_step;
	const auto a_strided_step = a_trans == PERFLIBS_NOTRANS ? a_row_step : a_column_step;
	const auto b_contiguous_step = is_trans(b_trans) ? b_column_step : b_row_step;
	const auto b_strided_step = is_trans(b_trans) ? b_row_step : b_column_step;

	const spec::problem_context pctx {
		matmul3_problem<T> {
			strided_batch_view {
				general_matrix { interleave_batch_matrix { a, ninter, k, m, a_contiguous_step, a_strided_step } },
				nbatch, batch_step_a
			},
			strided_batch_view {
				general_matrix { interleave_batch_matrix { b, ninter, k, n, b_contiguous_step, b_strided_step } },
				nbatch, batch_step_b
			},
			strided_batch_view {
				general_matrix { interleave_batch_matrix { c, ninter, m, n, c_row_step, c_column_step } },
				nbatch, batch_step_c
			},
			alpha, beta, zero_mode::set
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	compute(pctx);
	return PL_LINALG_STATUS_SUCCESS;
}

} // namespace perflibs::linalg::batch_interleave::matmul

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_INTERFACES_BATCH_INTERLEAVE_GEMM_HPP

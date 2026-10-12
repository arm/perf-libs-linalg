/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_SOLVE_INTERFACES_BATCH_INTERLEAVE_TRSM_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_SOLVE_INTERFACES_BATCH_INTERLEAVE_TRSM_HPP

#include "pl_linalg_status.h"

#include "detect/system.hpp"
#include "framework/linalg_blas_types.hpp"
#include "matrix/adaptors.hpp"
#include "matrix/interleave_batch_matrix.hpp"
#include "matrix/strided_batch_view.hpp"
#include "packages/batch_interleave/solve/strategies.hpp"
#include "spec/problem_context.hpp"

namespace perflibs::linalg::batch_interleave::solve {

/** In-place triangular solve over batches of interleaved matrices. */
template<typename IntType, typename T, typename ArchitectureSpec>
pl_linalg_status_t batch_interleave_trsm(IntType ninter, IntType nbatch, char side, char uplo, char transa,
                                     char diag, IntType m, IntType n, T alpha, const T *a,
                                     IntType batch_step_a, IntType a_row_step, IntType a_column_step,
                                     T *b, IntType batch_step_b, IntType b_row_step, IntType b_column_step) {
	if (m == 0 || n == 0 || ninter == 0 || nbatch == 0) {
		return PL_LINALG_STATUS_SUCCESS;
	}

	const bool is_left = option_matches(side, 'L');
	if (!is_left && !option_matches(side, 'R')) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}
	if (!option_matches(uplo, 'U', 'L') || !option_matches(transa, 'N', 'T') ||
	    !option_matches(diag, 'U', 'N') || m < 0 || n < 0 || ninter < 0 || nbatch < 0) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	const auto a_size = is_left ? m : n;
	if (a_row_step < ninter || a_column_step < ninter ||
	    batch_step_a < kernel_inttype { ninter } * a_size * a_size || b_row_step < ninter ||
	    b_column_step < ninter || batch_step_b < kernel_inttype { ninter } * m * n) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	const auto a_trans = c_to_trans(transa);
	const bool is_a_trans = is_trans(a_trans) ^ !is_left;
	const auto a_uplo = is_a_trans ? c_to_uplo(uplo) : lower_flip(c_to_uplo(uplo));
	const auto a_contiguous_step = is_a_trans ? a_row_step : a_column_step;
	const auto a_strided_step = is_a_trans ? a_column_step : a_row_step;
	const auto b_contiguous_step = is_left ? b_row_step : b_column_step;
	const auto b_strided_step = is_left ? b_column_step : b_row_step;
	const auto b_strided_size = is_left ? n : m;

	const auto a_matrix = triangular_matrix {
		a_uplo, c_to_diag(diag),
		interleave_batch_matrix { a, ninter, a_size, a_size, a_contiguous_step, a_strided_step }
	};
	const auto b_matrix = general_matrix {
		interleave_batch_matrix { b, ninter, a_size, b_strided_size, b_contiguous_step, b_strided_step }
	};
	const auto a_batch = strided_batch_view { a_matrix, nbatch, batch_step_a };
	const auto b_batch = strided_batch_view { b_matrix, nbatch, batch_step_b };

	const spec::problem_context pctx {
		solve_problem<T> {
			a_batch,
			b_batch,
			alpha
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	compute(pctx);
	return PL_LINALG_STATUS_SUCCESS;
}

} // namespace perflibs::linalg::batch_interleave::solve

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_SOLVE_INTERFACES_BATCH_INTERLEAVE_TRSM_HPP

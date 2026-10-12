/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_INTERFACES_BATCH_INTERLEAVE_SCAL_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_INTERFACES_BATCH_INTERLEAVE_SCAL_HPP

#include "pl_linalg_status.h"
#include "detect/system.hpp"
#include "packages/batch_interleave/matmul/strategies.hpp"

namespace perflibs::linalg::batch_interleave::matmul {

/** Scale batches of interleaved vectors by a shared scalar. */
template<typename IntType, typename T, typename ArchitectureSpec>
pl_linalg_status_t batch_interleave_scal(IntType ninter, IntType nbatch, IntType n, T alpha,
                                    T *x, IntType batch_step_x, IntType x_step) {

	if (n < 0 || ninter < 0 || nbatch < 0 || x_step < ninter ||
	    batch_step_x < kernel_inttype { x_step } * n) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	if (ninter == 0 || nbatch == 0 || n == 0) {
		return PL_LINALG_STATUS_SUCCESS;
	}

	// A and B provide only the matmul3 dimensions; the matmul3 alpha is zero, so their data is unused.
	const auto a_matrix = general_matrix {
		interleave_batch_matrix<const T> { x, ninter, 1, n, 0, x_step }
	};
	const auto b_matrix = general_matrix {
		interleave_batch_matrix<const T> { x, ninter, 1, 1, 0, 0 }
	};
	const auto x_matrix = general_matrix {
		interleave_batch_matrix { x, ninter, n, 1, x_step, 0 }
	};

	const spec::problem_context pctx {
		matmul3_problem<T> {
			strided_batch_view { a_matrix, nbatch, batch_step_x },
			strided_batch_view { b_matrix, nbatch, batch_step_x },
			strided_batch_view { x_matrix, nbatch, batch_step_x },
			zero<T>,
			alpha,
			zero_mode::scale
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	compute(pctx);
	return PL_LINALG_STATUS_SUCCESS;
}

} // namespace perflibs::linalg::batch_interleave::matmul

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_INTERFACES_BATCH_INTERLEAVE_SCAL_HPP

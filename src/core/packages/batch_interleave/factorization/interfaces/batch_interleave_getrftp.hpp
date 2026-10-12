/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_INTERFACES_BATCH_INTERLEAVE_GETRFTP_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_INTERFACES_BATCH_INTERLEAVE_GETRFTP_HPP

#include "pl_linalg_status.h"

#include "detect/system.hpp"
#include "matrix/adaptors.hpp"
#include "matrix/interleave_batch_matrix.hpp"
#include "matrix/strided_batch_view.hpp"
#include "packages/batch_interleave/factorization/strategies.hpp"
#include "spec/problem_context.hpp"

namespace perflibs::linalg::batch_interleave::factorization {

/**
 * Interface for LU factorization with threshold partial pivoting of
 * interleaved matrix batches.
 */
template<typename IntType, typename T, typename ArchitectureSpec>
pl_linalg_status_t batch_interleave_getrftp(
	IntType ninter, IntType nbatch, IntType m, IntType n, T *a,
	IntType batch_step_a, IntType a_contiguous_step, IntType a_strided_step,
	T threshold, IntType *pivot, IntType batch_step_pivot, IntType pivot_step) {

	if (ninter > max_interleave_size) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	// Preserve the existing LAPACK-style quick return.
	if (nbatch == 0 || ninter == 0 || m == 0 || n == 0) {
		return PL_LINALG_STATUS_SUCCESS;
	}

	if (m < 0 || n < 0 || ninter < 0 || nbatch < 0) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}
	if (pivot_step < ninter || a_contiguous_step < ninter || a_strided_step < ninter) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	const IntType min_mn = min(m, n);
	if (batch_step_pivot < pivot_step * min_mn || batch_step_a < ninter * m * n) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	const auto a_matrix = general_matrix {
		interleave_batch_matrix { a, ninter, m, n, a_contiguous_step, a_strided_step }
	};
	const auto pivot_matrix = general_matrix {
		interleave_batch_matrix { pivot, ninter, min_mn, 1, pivot_step, 0 }
	};

	const spec::problem_context pctx {
		threshold_pivoted_lu_problem<T, IntType> {
			strided_batch_view { a_matrix, nbatch, batch_step_a },
			strided_batch_view { pivot_matrix, nbatch, batch_step_pivot },
			threshold
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	compute(pctx);
	return PL_LINALG_STATUS_SUCCESS;
}

} // namespace perflibs::linalg::batch_interleave::factorization

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_INTERFACES_BATCH_INTERLEAVE_GETRFTP_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_INTERFACES_BATCH_INTERLEAVE_POTRF_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_INTERFACES_BATCH_INTERLEAVE_POTRF_HPP

#include "pl_linalg_status.h"

#include "detect/system.hpp"
#include "framework/linalg_blas_types.hpp"
#include "matrix/adaptors.hpp"
#include "matrix/interleave_batch_matrix.hpp"
#include "matrix/strided_batch_view.hpp"
#include "packages/batch_interleave/factorization/strategies.hpp"

namespace perflibs::linalg::batch_interleave::factorization {

/**
 * Interface for Cholesky factorization of interleaved matrix batches.
 */
template<typename IntType, typename T, typename ArchitectureSpec>
pl_linalg_status_t batch_interleave_potrf(
	IntType ninter, IntType nbatch, char uplo, IntType n,
	T *A, IntType bstrd_A, IntType istrd_A, IntType jstrd_A) {
	if (ninter > max_interleave_size) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	// Preserve the existing interleave-batch quick-return behavior.
	if (nbatch == 0 || ninter == 0 || n == 0) {
		return PL_LINALG_STATUS_SUCCESS;
	}

	if (n < 0 || ninter < 0 || nbatch < 0) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}
	if (istrd_A < ninter || jstrd_A < ninter) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}
	if (bstrd_A < ninter * n * n) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	if (!option_matches(uplo, 'U', 'L')) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	const spec::problem_context pctx {
		cholesky_problem<T> {
			strided_batch_view {
				symmetric_matrix {
					c_to_uplo(uplo),
					interleave_batch_matrix {
						A,
						ninter,
						n,
						n,
						istrd_A,
						jstrd_A
					}
				},
				nbatch,
				bstrd_A
			}
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	compute(pctx);
	return PL_LINALG_STATUS_SUCCESS;
}

} // namespace perflibs::linalg::batch_interleave::factorization

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_INTERFACES_BATCH_INTERLEAVE_POTRF_HPP

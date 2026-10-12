/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_INTERFACES_BATCH_INTERLEAVE_ORGQR_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_INTERFACES_BATCH_INTERLEAVE_ORGQR_HPP

#include "pl_linalg_status.h"

#include "detect/system.hpp"
#include "matrix/adaptors.hpp"
#include "matrix/interleave_batch_matrix.hpp"
#include "matrix/strided_batch_view.hpp"
#include "packages/batch_interleave/factorization/strategies.hpp"
#include "spec/problem_context.hpp"

namespace perflibs::linalg::batch_interleave::factorization {

/**
 * Interface for forming Q matrices from interleaved QR factorizations.
 */
template<typename IntType, typename T, typename ArchitectureSpec>
pl_linalg_status_t batch_interleave_orgqr(
	IntType ninter, IntType nbatch, IntType m, IntType n,
	const IntType *reflector_count, T *a, IntType batch_step_a,
	IntType a_contiguous_step, IntType a_strided_step, const T *tau,
	IntType batch_step_tau, IntType tau_step) {

	if (ninter > max_interleave_size) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	// Preserve the existing LAPACK-style quick return.
	if (m == 0 || n == 0) {
		return PL_LINALG_STATUS_SUCCESS;
	}

	if (m < 0 || n < 0 || ninter < 0 || nbatch < 0 || n > m) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}
	if (tau_step < ninter || a_contiguous_step < ninter || a_strided_step < ninter) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}
	if (batch_step_tau < tau_step * n || batch_step_a < ninter * m * n) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}
	for (IntType index = 0; index < ninter * nbatch; index++) {
		if (reflector_count[index] < 0 || reflector_count[index] > n) {
			return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
		}
	}

	spec::problem_context pctx {
		generate_q_from_qr_problem<T, IntType> {
			strided_batch_view {
				general_matrix {
					interleave_batch_matrix {
						a,
						ninter,
						m,
						n,
						a_contiguous_step,
						a_strided_step
					}
				},
				nbatch,
				batch_step_a
			},
			strided_batch_view {
				general_matrix {
					interleave_batch_matrix {
						reflector_count,
						ninter,
						1,
						1,
						0,
						0
					}
				},
				nbatch,
				ninter
			},
			strided_batch_view {
				general_matrix {
					interleave_batch_matrix {
						tau,
						ninter,
						n,
						1,
						tau_step,
						0
					}
				},
				nbatch,
				batch_step_tau
			}
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	compute(pctx);
	return PL_LINALG_STATUS_SUCCESS;
}

} // namespace perflibs::linalg::batch_interleave::factorization

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_INTERFACES_BATCH_INTERLEAVE_ORGQR_HPP

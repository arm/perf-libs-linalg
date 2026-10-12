/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_INTERFACES_BATCH_INTERLEAVE_ORMQR_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_INTERFACES_BATCH_INTERLEAVE_ORMQR_HPP

#include "pl_linalg_status.h"

#include "detect/system.hpp"
#include "framework/linalg_blas_types.hpp"
#include "matrix/adaptors.hpp"
#include "matrix/interleave_batch_matrix.hpp"
#include "matrix/strided_batch_view.hpp"
#include "packages/batch_interleave/factorization/strategies.hpp"
#include "spec/problem_context.hpp"

namespace perflibs::linalg::batch_interleave::factorization {

/**
 * Interface for applying Q factors from interleaved QR factorizations.
 */
template<typename IntType, typename T, typename ArchitectureSpec>
pl_linalg_status_t batch_interleave_ormqr(
	IntType ninter, IntType nbatch, char side, char trans_q, IntType m, IntType n,
	const IntType *reflector_count, const T *a, IntType batch_step_a,
	IntType a_contiguous_step, IntType a_strided_step, const T *tau,
	IntType batch_step_tau, IntType tau_step, T *c, IntType batch_step_c,
	IntType c_contiguous_step, IntType c_strided_step) {

	if (ninter > max_interleave_size) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	if (m == 0 || n == 0) {
		return PL_LINALG_STATUS_SUCCESS;
	}

	if (!option_matches(side, 'L', 'R') || !option_matches(trans_q, 'N', 'T')) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}
	if (m < 0 || n < 0 || ninter < 0 || nbatch < 0) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	const bool left = option_matches(side, 'L');
	const bool transpose_q = option_matches(trans_q, 'T');
	const IntType q_order = left ? m : n;
	if (tau_step < ninter || a_contiguous_step < ninter || a_strided_step < ninter
	 || c_contiguous_step < ninter || c_strided_step < c_contiguous_step * m) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}
	if (batch_step_tau < tau_step * q_order
	 || batch_step_a < ninter * q_order * q_order
	 || batch_step_c < ninter * m * n) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}
	for (IntType index = 0; index < ninter * nbatch; index++) {
		if (reflector_count[index] < 0 || reflector_count[index] > q_order) {
			return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
		}
	}

	const auto a_matrix = general_matrix {
		interleave_batch_matrix { a, ninter, q_order, q_order, a_contiguous_step, a_strided_step }
	};
	const auto reflector_count_matrix = general_matrix {
		interleave_batch_matrix { reflector_count, ninter, 1, 1, 0, 0 }
	};
	const auto tau_matrix = general_matrix {
		interleave_batch_matrix { tau, ninter, q_order, 1, tau_step, 0 }
	};
	const auto c_matrix = general_matrix {
		interleave_batch_matrix { c, ninter, m, n, c_contiguous_step, c_strided_step }
	};

	const spec::problem_context pctx {
		apply_q_from_qr_problem<T, IntType> {
			left ? transpose_q : !transpose_q,
			strided_batch_view { a_matrix, nbatch, batch_step_a },
			strided_batch_view { reflector_count_matrix, nbatch, ninter },
			strided_batch_view { tau_matrix, nbatch, batch_step_tau },
			strided_batch_view { left ? c_matrix : c_matrix.transpose(), nbatch, batch_step_c }
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	compute(pctx);
	return PL_LINALG_STATUS_SUCCESS;
}

} // namespace perflibs::linalg::batch_interleave::factorization

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_INTERFACES_BATCH_INTERLEAVE_ORMQR_HPP

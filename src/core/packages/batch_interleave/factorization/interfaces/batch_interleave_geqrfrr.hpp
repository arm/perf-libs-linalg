/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_INTERFACES_GEQRFRR_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_INTERFACES_GEQRFRR_HPP

#include "pl_linalg_status.h"

#include "packages/batch_interleave/factorization/strategies.hpp"

#include "spec/problem_context.hpp"
#include "detect/system.hpp"
#include "framework/linalg_util.hpp"
#include "matrix/adaptors.hpp"
#include "matrix/interleave_batch_matrix.hpp"
#include "matrix/strided_batch_view.hpp"

namespace perflibs::linalg::batch_interleave::factorization {

/**
 * Interface for interleaved-batch QR factorization with column pivoting.
 *
 * Performs argument validation and delegates to the compute pipeline.
 *
 * @tparam IntType        integer type for dimensions, strides, pivots and rank
 * @tparam T              floating-point scalar type
 * @tparam ArchitectureSpec  architecture-specific spec
 */
template<typename IntType, typename T, typename ArchitectureSpec>
pl_linalg_status_t batch_interleave_geqrfrr(
        IntType  ninter,  IntType  nbatch,
        IntType  m,       IntType  n,
        T       *A,       IntType  bstrd_A,  IntType  istrd_A,  IntType  jstrd_A,
        IntType *pvt,     IntType  bstrd_pvt, IntType  istrd_pvt,
        T       *tau,     IntType  bstrd_tau, IntType  istrd_tau,
        IntType *rank) {

	// at present the implementation is limited to a maximum of 64 interleaved vectors
	if (ninter > max_interleave_size) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	if (m < 0 || n < 0 || ninter < 0 || nbatch < 0) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	const IntType min_mn      = perflibs::min(m, n);
	const IntType min_strd_A  = min_mn == 0 ? 0 : ninter;

	if (istrd_pvt < ninter || istrd_tau < ninter ||
	    istrd_A   < min_strd_A || jstrd_A  < min_strd_A) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	if (bstrd_pvt < istrd_pvt * n ||
	    bstrd_tau < istrd_tau * n ||
	    bstrd_A   < ninter * m * n) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	spec::problem_context pctx {
		pivoted_qr_problem<T, IntType> {
			strided_batch_view {
				general_matrix {
					interleave_batch_matrix {
						A,
						ninter,
						m,
						n,
						istrd_A,
						jstrd_A
					}
				},
				nbatch,
				bstrd_A
			},
			strided_batch_view {
				general_matrix {
					interleave_batch_matrix {
						pvt,
						ninter,
						n,
						1,
						istrd_pvt,
						0,
					}
				},
				nbatch,
				bstrd_pvt
			},
			strided_batch_view {
				general_matrix {
					interleave_batch_matrix {
						tau,
						ninter,
						min_mn,
						1,
						istrd_tau,
						0,
					}
				},
				nbatch,
				bstrd_tau
			},
			strided_batch_view {
				general_matrix {
					interleave_batch_matrix {
						rank,
						ninter,
						1,
						1,
						1,
						0,
					}
				},
				nbatch,
				ninter
			}
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	compute(pctx);

	return PL_LINALG_STATUS_SUCCESS;
}

} // namespace perflibs::linalg::batch_interleave::factorization

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_INTERFACES_GEQRFRR_HPP

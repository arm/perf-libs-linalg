/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_INTERFACES_DOT_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_INTERFACES_DOT_HPP

#include "pl_linalg_status.h"

#include "packages/batch_interleave/matmul/strategies.hpp"

#include "spec/problem_context.hpp"
#include "detect/system.hpp"
#include "framework/linalg_util.hpp"
#include "matrix/adaptors.hpp"
#include "matrix/interleave_batch_matrix.hpp"

namespace perflibs::linalg::batch_interleave::matmul {

template<typename IntType, typename T, typename ArchitectureSpec>
pl_linalg_status_t batch_interleave_dot(IntType ninter, IntType nbatch, IntType n,
                                    const T *x, IntType batch_step_x, IntType incx,
                                    const T *y, IntType batch_step_y, IntType incy,
                                    T *z, IntType batch_step_z) {

	// at present the implementation is limited to a maximum of 64 interleaved vectors
	if (ninter < 0 || ninter > max_interleave_size || nbatch < 0) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	if (ninter == 0 || nbatch == 0) {
		return PL_LINALG_STATUS_SUCCESS;
	}

	// to match BLAS DOT we must return 0 if we do nothing
	// note BLAS DOT does no argument checking and just returns 0 for n<=0
	if (n <= 0) {
		if (batch_step_z < 0)
			return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;

		for (IntType i = 0; i != nbatch; ++i) {
			for (IntType j = i * batch_step_z; j < (i * batch_step_z) + ninter; ++j) {
				z[j] = zero<T>;
			}
		}
		return PL_LINALG_STATUS_SUCCESS;
	}
	if (incx < ninter || incy < ninter) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}
	if (batch_step_x < incx * n || batch_step_y < incy * n || batch_step_z < ninter) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	const auto alpha = one<T>;
	const auto beta = zero<T>;

	spec::problem_context pctx {
		matmul3_problem<T> {
			strided_batch_view {
				general_matrix {
					interleave_batch_matrix {
						x,
						ninter,
						n,
						1,
						incx,
						0
					}
				},
				nbatch,
				batch_step_x
			},
			strided_batch_view {
				general_matrix {
					interleave_batch_matrix {
						y,
						ninter,
						n,
						1,
						incy,
						0
					}
				},
				nbatch,
				batch_step_y
			},
			strided_batch_view {
				general_matrix {
					interleave_batch_matrix {
						z,
						ninter,
						1,
						1,
						0,
						0
					}
				},
				nbatch,
				batch_step_z
			},
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

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_INTERFACES_DOT_HPP

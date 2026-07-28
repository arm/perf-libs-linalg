/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_ASUM_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_ASUM_HPP

#include "packages/misc/strategies.hpp"
#include "matrix/matrix.hpp"

#include "spec/problem_context_helpers.hpp"
#include "framework/linalg_util.hpp"

#include "perflibs_complex.hpp"

namespace perflibs::linalg {

template<bool ParamCheck, typename IntType, typename T, typename RealType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
RealType asum(const IntType *n, const T *x, const IntType *incx) {

	// Special case for empty vectors, to avoid the overhead of the strategy system in this case
	if (*n <= 0 || *incx <= 0) return RealType(0);
	if (*n == 1) return static_cast<RealType>(perflibs::sum_abs(x[0]));

	RealType out;
	spec::problem_context pctx {
		misc::l1_norm {
			general_matrix { matrix_base { x, *n, 1, *incx, 0 } },
			out
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	}; // pctx

	compute(pctx);

	return out;
}

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_BLAS_INTERFACES_ASUM_HPP

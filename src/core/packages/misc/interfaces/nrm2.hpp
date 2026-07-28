/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_NRM2_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_NRM2_HPP

#include "packages/misc/strategies.hpp"
#include "matrix/matrix.hpp"

#include "spec/problem_context_helpers.hpp"
#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

template<bool ParamCheck, typename IntType, typename T, typename RealType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
RealType nrm2(const IntType *n, const T *x, const IntType *incx) {

	if (*n < 1) return RealType(0);
	if (*n == 1) return static_cast<RealType>(std::abs(x[0]));

	RealType out;
	// The fallback kernel expects a pointer to the first element of the vector,
	// even for negative increments (where the first element is not at the base pointer).
	if (*incx < 0) x += (*incx * -1) * (*n - 1);

	spec::problem_context pctx {
		misc::l2_norm {
			general_matrix { matrix_base { x, *n, 1, *incx, 0 } },
			out
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	}; //pctx

	compute(pctx);

	return out;
}

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_BLAS_INTERFACES_NRM2_HPP

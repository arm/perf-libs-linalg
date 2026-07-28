/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_SWAP_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_SWAP_HPP

#include "packages/misc/strategies.hpp"

#include "spec/problem_context_helpers.hpp"
#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

template<bool ParamCheck, typename IntType, typename T, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void swap(const IntType *n, T *x, const IntType *incx, T *y, const IntType *incy) {

	if (*n <= 0) return;

	// The fallback kernel expects pointers to the first elements of each input vector x and y,
	// even for negative increments (where the first element is not at the base pointer).
	if (*incx < 0) x += (*incx * -1) * (*n - 1);
	if (*incy < 0) y += (*incy * -1) * (*n - 1);

	spec::problem_context pctx {
		misc::swap {
			general_matrix { matrix_base { x, *n, 1, *incx, 0 } },
			general_matrix { matrix_base { y, *n, 1, *incy, 0 } },
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	}; //pctx

	compute(pctx);
}

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_BLAS_INTERFACES_SWAP_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_ROT_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_ROT_HPP

#include "packages/misc/strategies.hpp"

namespace perflibs::linalg {

template<bool ParamCheck, typename IntType, typename T1, typename T2, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void rot(
	const IntType *n,
	T1 *x, const IntType *incx,
	T1 *y, const IntType *incy,
	const remove_complex_t<promote_t<T1, T2>> *c,
	const T2 *s) {

	if (*incx < 0) x += (*incx * -1) * (*n - 1);
	if (*incy < 0) y += (*incy * -1) * (*n - 1);

	spec::problem_context pctx {
		misc::rot{
			*n, x, *incx, y, *incy, *c, *s
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	};

	compute(pctx);
} // void rot

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_BLAS_INTERFACES_ROT_HPP

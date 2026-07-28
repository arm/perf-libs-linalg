/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_ROTM_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_ROTM_HPP

#include "packages/misc/strategies.hpp"

namespace perflibs::linalg {

template<bool ParamCheck, typename IntType, typename T, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void rotm(
	const IntType *n,
	T *x, const IntType *incx,
	T *y, const IntType *incy,
	const T *param) {

	spec::problem_context pctx {
		misc::rotm {
			(kernel_inttype) *n, x, (kernel_inttype) *incx, y, (kernel_inttype) *incy, param
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	};

	compute(pctx);
} // void rotm

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_BLAS_INTERFACES_ROTM_HPP

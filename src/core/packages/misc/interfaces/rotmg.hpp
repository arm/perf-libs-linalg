/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_ROTMG_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_ROTMG_HPP

#include "packages/misc/strategies.hpp"

namespace perflibs::linalg {

template<bool ParamCheck, typename T, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void rotmg(T *d1, T *d2, T *x, const T *y, T *param) {

	spec::problem_context pctx {
		misc::rotmg {
			d1, d2, x, y, param
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	};

	compute(pctx);
} // void rotmg

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_BLAS_INTERFACES_ROTMG_HPP

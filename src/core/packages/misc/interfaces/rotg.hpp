/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_ROTG_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_ROTG_HPP

#include "packages/misc/strategies.hpp"

namespace perflibs::linalg {

template<bool ParamCheck, typename T, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void rotg(
	T *a,
	T *b,
	perflibs::remove_complex_t<promote_t<T>> *c,
	T *s) {

	spec::problem_context pctx {
		misc::rotg {
			a, b, c, s
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	};
	compute(pctx);
} // void rotg

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_BLAS_INTERFACES_ROTG_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_IAMAX_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_IAMAX_HPP

#include "packages/misc/strategies.hpp"
#include "matrix/matrix.hpp"

#include "spec/problem_context_helpers.hpp"
#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

template<bool ParamCheck, typename IntType, typename T, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
IntType iamax(const IntType *n, const T *x, const IntType *incx) {
	kernel_inttype out;

	spec::problem_context pctx {
		misc::find_index {
			general_matrix { matrix_base { x, *n, 1, *incx, 0 } },
			out,
			misc::find_operation::absolute_max
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	}; //pctx

	compute(pctx);

	return out;
};

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_IAMAX_HPP

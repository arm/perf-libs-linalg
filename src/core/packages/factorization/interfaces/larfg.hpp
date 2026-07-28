/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_INTERFACES_LARFG_HPP
#define PERFLIBS_LINALG_FACTORIZATION_INTERFACES_LARFG_HPP

#include "packages/factorization/strategies.hpp"
#include "packages/factorization/problem_context_bases.hpp"

#include "spec/strategy_tag.hpp"
#include "matrix/matrix.hpp"
#include "framework/compute.hpp"
#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

template<bool Paramcheck, typename IntType, typename Atype, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void larfg(const IntType *n, Atype *alpha, Atype *x, const IntType *incx, Atype *tau){

	// Quick return if possible
	if (*n <= 0) {
		*tau = 0;
		return;
	}

	const     kernel_inttype x_cntg      = *n - 1;
	const     kernel_inttype x_strd      = 1;
	const     kernel_inttype x_cntg_step = *incx;
	constexpr kernel_inttype x_strd_step = 1;

	auto pctx = spec::problem_context{
		factorization::generate_reflector{
			*alpha,
			general_matrix{ matrix_base{ x, x_cntg, x_strd, x_cntg_step, x_strd_step } },
			*tau
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	};
	factorization::compute(pctx);
}
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_FACTORIZATION_INTERFACES_LARFG_HPP

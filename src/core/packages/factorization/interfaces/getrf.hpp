/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_INTERFACES_GETRF_HPP
#define PERFLIBS_LINALG_FACTORIZATION_INTERFACES_GETRF_HPP

#include "packages/factorization/strategies.hpp"
#include "packages/factorization/problem_context_bases.hpp"

#include "spec/strategy_tag.hpp"
#include "matrix/matrix.hpp"
#include "framework/compute.hpp"

#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

namespace perflibs::linalg {

template<typename IntType, typename AType>
PERFLIBS_LINALG_INLINE
bool getrf_param_check(const IntType *m, const IntType *n,
                                  AType *a, const IntType *lda,
                                  IntType *ipiv, IntType *info) {

	*info = 0;

	if (*m < 0) {
		*info = -1;
	}
	else if (*n < 0) {
		*info = -2;
	}
	else if (*lda < max(1, *m)) {
		*info = -4;
	}
	if (*info != 0) {
		pl_linalg_int_t ninfo = -(*info);
		std::string_view fname;

		if constexpr (std::is_same_v<AType, double>) {
			fname = "DGETRF ";
		}
		else if constexpr (std::is_same_v<AType, float>) {
			fname = "SGETRF ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<float>>) {
			fname = "CGETRF ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<double>>) {
			fname = "ZGETRF ";
		}
		call_xerbla(fname, ninfo);
		return false;
	}
	return true;
}

template<bool Paramcheck, typename IntType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void getrf(const IntType *m, const IntType *n,
                      AType *a, const IntType *lda,
                      IntType *ipiv, IntType *info) {
	if constexpr (Paramcheck) {
		const bool res = getrf_param_check(m, n, a, lda, ipiv, info);

		if (!res) return;
	}

	if(*m == 0 || *n == 0) return;

	const kernel_inttype a_cntg      = *m;
	const kernel_inttype a_strd      = *n;
	const kernel_inttype a_cntg_step = 1;
	const kernel_inttype a_strd_step = *lda;
	const kernel_inttype min_mn      = min(*m, *n);

	auto pctx = spec::problem_context{
		factorization::lu_factorization{
			general_matrix{ matrix_base{ a, a_cntg, a_strd, a_cntg_step, a_strd_step } },
			std::span<IntType>(ipiv, min_mn), *info
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	};

	factorization::compute(pctx);
	// Update pivot to Fortran indexing
	for (kernel_inttype i = 0; i < min_mn; i++) {
		ipiv[i] += 1;
	}
}
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_FACTORIZATION_INTERFACES_GETRF_HPP

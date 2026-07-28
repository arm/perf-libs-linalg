/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_INTERFACES_GEQRF_HPP
#define PERFLIBS_LINALG_FACTORIZATION_INTERFACES_GEQRF_HPP

#include "packages/factorization/strategies.hpp"
#include "packages/factorization/problem_context_bases.hpp"

#include "spec/strategy_tag.hpp"
#include "matrix/matrix.hpp"
#include "framework/compute.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

namespace perflibs::linalg {

template<typename IntType, typename AType>
PERFLIBS_LINALG_INLINE bool geqrf_param_check(const IntType *m, const IntType *n, AType *a, const IntType *lda,
                                         AType *tau, AType *work, const IntType *lwork, IntType *info) {

	*info = 0;
	const bool lquery = (*lwork == -1);
	if (*m < 0) {
		*info = -1;
	}
	else if (*n < 0) {
		*info = -2;
	}
	else if (*lda < max(1, *m)) {
		*info = -4;
	}
	else if (!lquery && (*lwork <= 0 || (*m > 0 && *lwork < max(1, *n)))) {
		*info = -7;
	}

	if (*info != 0) {
		pl_linalg_int_t ninfo = -(*info);
		std::string_view fname;
		if constexpr (std::is_same_v<AType, double>) {
			fname = "DGEQRF ";
		}
		else if constexpr (std::is_same_v<AType, float>) {
			fname = "SGEQRF ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<float>>) {
			fname = "CGEQRF ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<double>>) {
			fname = "ZGEQRF ";
		}
		call_xerbla(fname, ninfo);
		return false;
	}
	return true;
}

template<bool Paramcheck, typename IntType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void geqrf(const IntType *m, const IntType *n, AType *a, const IntType *lda, AType *tau,
           AType *work, const IntType *lwork, IntType *info) {

	if constexpr (Paramcheck) {
		const bool res = geqrf_param_check(m, n, a, lda, tau, work, lwork, info);
		if (!res) return;
	}

	// Note that the value of nb set here is only used to compute lwork.
	// We use 192 to be allocate a workspace large enough to avoid internal
	// reallocation. The actual value of nb used for blocking is
	// obtained from tuning after a call to compute() function.
	constexpr IntType max_nb = 192;
	const IntType k          = min(*m, *n);
	const IntType max_lwork  = k == 0 ? 1 : *n * max_nb;
	work[0]                  = max_lwork;

	// Quick return
	if ((*lwork == -1) || k == 0)  return;

	const     kernel_inttype a_cntg      = *m;
	const     kernel_inttype a_strd      = *n;
	constexpr kernel_inttype a_cntg_step = 1;
	const     kernel_inttype a_strd_step = *lda;

	auto pctx = spec::problem_context{
		factorization::qr_factorization{
			general_matrix{ matrix_base{ a, a_cntg, a_strd, a_cntg_step, a_strd_step } },
			std::span<AType>(tau, k),
			std::span<AType>(work, *lwork)
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	};

	factorization::compute(pctx);

	//Set optimal value of lwork to comply with
	// the stantard
	const IntType optimal_lwork = *n < max_nb ? *n : *n * max_nb;
	work[0] = optimal_lwork;
}
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_FACTORIZATION_INTERFACES_GEQRF_HPP

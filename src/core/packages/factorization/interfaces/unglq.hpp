/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_INTERFACES_UNGLQ_HPP
#define PERFLIBS_LINALG_FACTORIZATION_INTERFACES_UNGLQ_HPP

#include "packages/factorization/strategies.hpp"
#include "packages/factorization/problem_context_bases.hpp"

#include "matrix/matrix.hpp"
#include "framework/compute.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

namespace perflibs::linalg {

template<typename IntType, typename AType>
PERFLIBS_LINALG_INLINE
bool unglq_param_check(const IntType* m, const IntType* n, const IntType* k,
                       AType* a, const IntType* lda, AType* tau,
                       AType* work, const IntType* lwork, IntType* info) {

	*info = 0;
	const bool lquery = (*lwork == -1);

	// Minimum required workspace
	IntType min_lwork = max(1, *m);

	if (*m < 0) {
		*info = -1;
	}
	else if (*n < *m) {
		*info = -2;
	}
	else if (*k < 0 || *k > *m) {
		*info = -3;
	}
	else if (*lda < max(1, *m)) {
		*info = -5;
	}
	else if (!lquery && *lwork < min_lwork) {
		*info = -8;
	}

	if (*info != 0) {
		pl_linalg_int_t ninfo = -(*info);
		const char* fname;
		if constexpr (std::is_same_v<AType, double>) {
			fname = "DORGLQ ";
		}
		else if constexpr (std::is_same_v<AType, float>) {
			fname = "SORGLQ ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<float>>) {
			fname = "CUNGLQ ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<double>>) {
			fname = "ZUNGLQ ";
		}
		call_xerbla(fname, ninfo);
		return false;
	}
	return true;
}

template<bool Paramcheck, typename IntType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void unglq(const IntType* m, const IntType* n, const IntType* k,
           AType* a, const IntType* lda, AType* tau,
           AType* work, const IntType* lwork, IntType* info) {

	if constexpr (Paramcheck) {
		const bool res = unglq_param_check(m, n, k, a, lda, tau, work, lwork, info);
		if (!res) return;
	}

	// Note that the value of nb set here is only used to compute lwork.
	// The actual value of nb used for blocking is
	// obtained from tuning after a call to compute() function.
	constexpr IntType max_nb = 128;

	// Calculate the optimal workspace size
	const IntType opt_lwork = max_nb * (*m);
	work[0] = opt_lwork;

	// Quick return if possible
	if (*lwork == -1 || *m == 0) return;

	// Matrix A dimensions
	const     kernel_inttype a_cntg      = *m;
	const     kernel_inttype a_strd      = *n;
	constexpr kernel_inttype a_cntg_step = 1;
	const     kernel_inttype a_strd_step = *lda;

	// Tau vector dimensions
	const     kernel_inttype tau_cntg      = *k;
	constexpr kernel_inttype tau_strd      = 1;
	constexpr kernel_inttype tau_cntg_step = 1;
	constexpr kernel_inttype tau_strd_step = 1;

	// Workspace dimensions
	const     kernel_inttype work_cntg      = *lwork;
	constexpr kernel_inttype work_strd      = 1;
	constexpr kernel_inttype work_cntg_step = 1;
	constexpr kernel_inttype work_strd_step = 1;

	auto pctx = spec::problem_context{
		factorization::generate_q_from_lq{
			general_matrix{ matrix_base{ a,    a_cntg,    a_strd,    a_cntg_step,    a_strd_step    } },
			general_matrix{ matrix_base{ tau,  tau_cntg,  tau_strd,  tau_cntg_step,  tau_strd_step  }, true },
			general_matrix{ matrix_base{ work, work_cntg, work_strd, work_cntg_step, work_strd_step } }
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	};

	factorization::compute(pctx);

	// Set optimal workspace size to comply with the standard
	work[0] = opt_lwork;
}
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_FACTORIZATION_INTERFACES_UNGLQ_HPP

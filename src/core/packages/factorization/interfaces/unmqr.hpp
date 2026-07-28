/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_INTERFACES_UNMQR_HPP
#define PERFLIBS_LINALG_FACTORIZATION_INTERFACES_UNMQR_HPP

#include "packages/factorization/strategies.hpp"
#include "packages/factorization/problem_context_bases.hpp"

#include "spec/strategy_tag.hpp"
#include "matrix/matrix.hpp"
#include "framework/compute.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

namespace perflibs::linalg {

template<typename IntType>
PERFLIBS_LINALG_INLINE
IntType unmqr_min_lwork(const bool left, IntType m, IntType n) {
	if (m > 0 && n > 0) {
		return left ? n : m;
	}
	return 1;
}

template<typename IntType, typename AType>
PERFLIBS_LINALG_INLINE
bool unmqr_param_check(const char* side, const char* trans, const IntType* m, const IntType* n,
					const IntType* k, AType* a, const IntType* lda, AType* tau,
					AType* c, const IntType* ldc, AType* work, const IntType* lwork, IntType* info) {

	*info = 0;
	const bool lquery = (*lwork == -1);
	const bool left   = option_matches(*side, 'L');
	const bool right  = option_matches(*side, 'R');
	const bool notran = option_matches(*trans, 'N');
	const bool tran   = (option_matches(*trans, 'C') ||
	                    (!std::is_same_v<AType, std::complex<float>> &&
	                    !std::is_same_v<AType, std::complex<double>> &&
	                    option_matches(*trans, 'T')));

	// Minimum required workspace
	IntType min_lwork = unmqr_min_lwork(left, *m, *n);;

	if (!left && !right) {
		*info = -1;
	}
	else if (!notran && !tran) {
		*info = -2;
	}
	else if (*m < 0) {
		*info = -3;
	}
	else if (*n < 0) {
		*info = -4;
	}
	else if (*k < 0 || (left && *k > *m) || (right && *k > *n)) {
		*info = -5;
	}
	else if ((*lda < max(1, left ? *m : *n))) {
		*info = -7;
	}
	else if (*ldc < max(1, *m)) {
		*info = -10;
	}
	else if (!lquery && *lwork < min_lwork) {
		*info = -12;
	}

	if (*info != 0) {
		pl_linalg_int_t ninfo = -(*info);
		std::string_view fname;
		if constexpr (std::is_same_v<AType, double>) {
			fname = "DORMQR ";
		}
		else if constexpr (std::is_same_v<AType, float>) {
			fname = "SORMQR ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<float>>) {
			fname = "CUNMQR ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<double>>) {
			fname = "ZUNMQR ";
		}
		call_xerbla(fname, ninfo);
		return false;
	}
	return true;
}

template<bool Paramcheck, typename IntType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void unmqr(const char* side, const char* trans, const IntType* m, const IntType* n,
		const IntType* k, AType* a, const IntType* lda, AType* tau,
		AType* c, const IntType* ldc, AType* work, const IntType* lwork, IntType* info) {

	if constexpr (Paramcheck) {
		const bool res = unmqr_param_check(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
		if (!res) return;
	}

	const bool left = option_matches(*side, 'L');

	// Note that the value of nb set here is only used to compute lwork.
	// The actual value of nb used for blocking is
	// obtained from tuning after a call to compute() function.
	constexpr IntType max_nb = 128;

	// Calculate the optimal workspace size
	IntType opt_lwork  = unmqr_min_lwork(left, *m, *n);
	if (*m > 0 && *n > 0) {
		opt_lwork = max_nb*(max_nb + max(*m, *n));
	}
	work[0] = opt_lwork;

	// Quick return if possible
	if (*lwork == -1 || *m == 0 || *n == 0 || *k == 0) return;

	// Matrix A dimensions
	const     kernel_inttype a_cntg      = left ? *m : *n;
	const     kernel_inttype a_strd      = *k;
	constexpr kernel_inttype a_cntg_step = 1;
	const     kernel_inttype a_strd_step = *lda;

	// Matrix C dimensions
	const     kernel_inttype c_cntg      = *m;
	const     kernel_inttype c_strd      = *n;
	constexpr kernel_inttype c_cntg_step = 1;
	const     kernel_inttype c_strd_step = *ldc;

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

	const     auto           q_trans        = c_to_trans(*trans);

	auto pctx = spec::problem_context{
		factorization::apply_q_from_qr{
			c_to_side(*side), q_trans,
			general_matrix{ matrix_base{ a,    a_cntg,    a_strd,    a_cntg_step,    a_strd_step    } },
			general_matrix{ matrix_base{ tau,  tau_cntg,  tau_strd,  tau_cntg_step,  tau_strd_step  }, is_conj(q_trans)},
			general_matrix{ matrix_base{ c,    c_cntg,    c_strd,    c_cntg_step,    c_strd_step    } },
			general_matrix{ matrix_base{ work, work_cntg, work_strd, work_cntg_step, work_strd_step } }
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	};

	factorization::compute(pctx);

	// Set optimal workspace size to comply with the standard
	work[0] = opt_lwork;
}
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_FACTORIZATION_INTERFACES_UNMQR_HPP

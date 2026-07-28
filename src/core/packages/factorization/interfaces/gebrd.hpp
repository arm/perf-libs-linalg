/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_INTERFACES_GEBRD_HPP
#define PERFLIBS_LINALG_FACTORIZATION_INTERFACES_GEBRD_HPP

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
bool gebrd_param_check(const IntType *m, const IntType *n, AType *a, const IntType *lda,
                       remove_complex_t<promote_t<AType>> *d, remove_complex_t<promote_t<AType>> *e,
                       AType *tauq, AType *taup, AType *work, const IntType *lwork, IntType *info) {

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
	else if (!lquery && *lwork < max(1, max(*m, *n))) {
		*info = -10;
	}

	if (*info != 0) {
		pl_linalg_int_t ninfo = -(*info);
		std::string_view fname;
		if constexpr (std::is_same_v<AType, double>) {
			fname = "DGEBRD ";
		}
		else if constexpr (std::is_same_v<AType, float>) {
			fname = "SGEBRD ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<float>>) {
			fname = "CGEBRD ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<double>>) {
			fname = "ZGEBRD ";
		}
		call_xerbla(fname, ninfo);
		return false;
	}
	return true;
}

template<bool Paramcheck, typename IntType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void gebrd(const IntType *m, const IntType *n, AType *a, const IntType *lda,
           remove_complex_t<promote_t<AType>> *d, remove_complex_t<promote_t<AType>> *e, AType *tauq,
           AType *taup, AType *work, const IntType *lwork, IntType *info) {

	if constexpr (Paramcheck) {
		const bool res = gebrd_param_check(m, n, a, lda, d, e, tauq, taup, work, lwork, info);
		if (!res) return;
	}

	// Note that the value of nb set here is only used to compute lwork.
	// We use 192 to be allocate a workspace large enough to avoid internal
	// reallocation. The actual value of nb used for blocking is
	// obtained from tuning after a call to compute() function.
	constexpr IntType max_nb    = 92;
	const     IntType minmn     = min(*m, *n);
	const     IntType opt_lwork =  minmn == 0 ? 1 : (*m + *n) * max_nb;
	                  *work     = opt_lwork;

	// Quick return if possible
	if (*m == 0 || *n == 0 || *lwork == -1) return;

	// Matrix A dimensions
	const     kernel_inttype a_cntg      = *m;
	const     kernel_inttype a_strd      = *n;
	constexpr kernel_inttype a_cntg_step = 1;
	const     kernel_inttype a_strd_step = *lda;

	// Vector d dimensions
	const     kernel_inttype d_cntg      = min(*m, *n);
	constexpr kernel_inttype d_strd      = 1;
	constexpr kernel_inttype d_cntg_step = 1;
	constexpr kernel_inttype d_strd_step = 1;

	// Vector e dimensions
	const     kernel_inttype e_cntg      = min(*m, *n) - 1;
	constexpr kernel_inttype e_strd      = 1;
	constexpr kernel_inttype e_cntg_step = 1;
	constexpr kernel_inttype e_strd_step = 1;

	// Vector tauq dimensions
	const     kernel_inttype tauq_cntg      = min(*m, *n);
	constexpr kernel_inttype tauq_strd      = 1;
	constexpr kernel_inttype tauq_cntg_step = 1;
	constexpr kernel_inttype tauq_strd_step = 1;

	// Vector taup dimensions
	const     kernel_inttype taup_cntg      = min(*m, *n);
	constexpr kernel_inttype taup_strd      = 1;
	constexpr kernel_inttype taup_cntg_step = 1;
	constexpr kernel_inttype taup_strd_step = 1;

	// Workspace dimensions
	const     kernel_inttype work_cntg      = *lwork;
	constexpr kernel_inttype work_strd      = 1;
	constexpr kernel_inttype work_cntg_step = 1;
	constexpr kernel_inttype work_strd_step = 1;

	auto pctx = spec::problem_context{
		factorization::bidiagonalization{
			general_matrix { matrix_base { a,    a_cntg,    a_strd,    a_cntg_step,    a_strd_step    } },
			general_matrix { matrix_base { d,    d_cntg,    d_strd,    d_cntg_step,    d_strd_step    } },
			general_matrix { matrix_base { e,    e_cntg,    e_strd,    e_cntg_step,    e_strd_step    } },
			general_matrix { matrix_base { tauq, tauq_cntg, tauq_strd, tauq_cntg_step, tauq_strd_step } },
			general_matrix { matrix_base { taup, taup_cntg, taup_strd, taup_cntg_step, taup_strd_step } },
			general_matrix { matrix_base { work, work_cntg, work_strd, work_cntg_step, work_strd_step } }
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};
	factorization::compute(pctx);

	// Set optimal workspace back to comply to the reference specification
	*work = opt_lwork;
}
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_FACTORIZATION_INTERFACES_GEBRD_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_INTERFACES_HETRD_HPP
#define PERFLIBS_LINALG_FACTORIZATION_INTERFACES_HETRD_HPP

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
bool hetrd_param_check(const char *uplo, const IntType *n,
                       AType *a, const IntType *lda,
                       remove_complex_t<promote_t<AType>> *diagonal,
                       remove_complex_t<promote_t<AType>> * off_diagonal,
                       AType *tau, AType *work, const IntType *lwork,
                       IntType *info) {

	*info = 0;
	const bool lquery = (*lwork == -1);
	const bool upper  = option_matches(*uplo, 'U');
	const bool lower  = option_matches(*uplo, 'L');

	if (!upper && !lower) {
		*info = -1;
	}
	else if (*n < 0) {
		*info = -2;
	}
	else if (*lda < max(1, *n)) {
		*info = -4;
	}
	else if (!lquery && (*lwork <= 0)) {
		*info = -9;
	}

	if (*info != 0) {
		pl_linalg_int_t ninfo = -(*info);
		std::string_view fname;
		if constexpr (std::is_same_v<AType, double>) {
			fname = "DSYTRD ";
		}
		else if constexpr (std::is_same_v<AType, float>) {
			fname = "SSYTRD ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<float>>) {
			fname = "CHETRD ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<double>>) {
			fname = "ZHETRD ";
		}
		call_xerbla(fname, ninfo);
		return false;
	}
	return true;
}

template<bool Paramcheck, typename IntType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void hetrd(const char *uplo, const IntType *n,
           AType *a, const IntType *lda,
           remove_complex_t<promote_t<AType>> *diagonal,
           remove_complex_t<promote_t<AType>> * off_diagonal,
           AType *tau, AType *work, const IntType *lwork,
           IntType *info) {

	if constexpr (Paramcheck) {
		const bool res = hetrd_param_check(uplo, n, a, lda, diagonal, off_diagonal, tau, work, lwork, info);
		if (!res) return;
	}

	// Note that the value of nb set here is only used to compute lwork.
	// We use 192 to be allocate a workspace large enough to avoid internal
	// reallocation. The actual value of nb used for blocking is
	// obtained from tuning after a call to compute() function.
	constexpr IntType max_nb = 192;
	const IntType max_lwork  = *n == 0 ? 1 : *n * max_nb;
	work[0]                  = max_lwork;

	// Quick return
	if ((*lwork == -1) || *n == 0)  return;

	const      kernel_inttype a_cntg                = *n;
	const      kernel_inttype a_strd                = *n;
	constexpr  kernel_inttype a_cntg_step           = 1;
	const      kernel_inttype a_strd_step           = *lda;
	const auto a_uplo                               = c_to_uplo(*uplo);

	const     kernel_inttype diagonal_cntg          = *n;
	constexpr kernel_inttype diagonal_strd          = 1;
	constexpr kernel_inttype diagonal_cntg_step     = 1;
	constexpr kernel_inttype diagonal_strd_step     = 1;

	const     kernel_inttype off_diagonal_cntg      = *n - 1;
	constexpr kernel_inttype off_diagonal_strd      = 1;
	constexpr kernel_inttype off_diagonal_cntg_step = 1;
	constexpr kernel_inttype off_diagonal_strd_step = 1;

	const     kernel_inttype tau_cntg               = *n - 1;
	constexpr kernel_inttype tau_strd               = 1;
	constexpr kernel_inttype tau_cntg_step          = 1;
	constexpr kernel_inttype tau_strd_step          = 1;

	const     kernel_inttype work_cntg              = *lwork;
	constexpr kernel_inttype work_strd              = 1;
	constexpr kernel_inttype work_cntg_step         = 1;
	constexpr kernel_inttype work_strd_step         = 1;

	using a_matrix_type = std::conditional_t<
		perflibs::is_complex_v<AType>,
		hermitian_matrix<matrix_base<AType>>,
		symmetric_matrix<matrix_base<AType>>
	>;

	auto pctx = spec::problem_context{
		factorization::tridiagonalization{
			a_matrix_type    { a_uplo, matrix_base { a,            a_cntg,             a_strd,            a_cntg_step,            a_strd_step            } },
			general_matrix   {         matrix_base { diagonal,     diagonal_cntg,      diagonal_strd,     diagonal_cntg_step,     diagonal_strd_step     } },
			general_matrix   {         matrix_base { off_diagonal, off_diagonal_cntg,  off_diagonal_strd, off_diagonal_cntg_step, off_diagonal_strd_step } },
			general_matrix   {         matrix_base { tau,          tau_cntg,           tau_strd,          tau_cntg_step,          tau_strd_step          } },
			general_matrix   {         matrix_base { work,         work_cntg,          work_strd,         work_cntg_step,         work_strd_step         } }
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	};
	factorization::compute(pctx);

	// Set optimal value of lwork to comply with
	// the stantard
	const IntType optimal_lwork = *n < max_nb ? *n : *n * max_nb;
	work[0] = optimal_lwork;
}
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_FACTORIZATION_INTERFACES_HETRD_HPP

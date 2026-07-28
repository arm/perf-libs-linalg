/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_INTERFACES_LATRD_HPP
#define PERFLIBS_LINALG_FACTORIZATION_INTERFACES_LATRD_HPP

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
bool latrd_param_check(const char* uplo, const IntType* n, const IntType* nb,
                       AType* a, const IntType* lda,
                       remove_complex_t<promote_t<AType>>* off_diagonal, AType* tau,
                       AType* w, const IntType* ldw) {
	IntType info = 0;

	bool upper = option_matches(*uplo, 'U');
	bool lower = option_matches(*uplo, 'L');

	if (!upper && !lower) {
		info = -1;
	}
	else if (*n < 0) {
		info = -2;
	}
	else if (*nb < 1 || *nb > *n) {
		info = -3;
	}
	else if (*lda < max(1, *n)) {
		info = -5;
	}
	else if (*ldw < max(1, *n)) {
		info = -9;
	}

	if (info != 0) {
		pl_linalg_int_t ninfo = -(info);
		std::string_view fname;

		if constexpr (std::is_same_v<AType, double>) {
			fname = "DLATRD ";
		}
		else if constexpr (std::is_same_v<AType, float>) {
			fname = "SLATRD ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<float>>) {
			fname = "CLATRD ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<double>>) {
			fname = "ZLATRD ";
		}
		call_xerbla(fname, ninfo);
		return false;
	}
	return true;
}

template<bool Paramcheck, typename IntType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void latrd(const char* uplo, const IntType* n, const IntType* nb,
           AType* a, const IntType* lda,
           remove_complex_t<promote_t<AType>>* off_diagonal, AType* tau,
           AType* w, const IntType* ldw) {

	if constexpr (Paramcheck) {
		const bool res = latrd_param_check(uplo, n, nb, a, lda, off_diagonal, tau, w, ldw);
		if (!res) return;
	}

	if (*n <= 0) return;

	const     kernel_inttype a_cntg                 = *n;
	const     kernel_inttype a_strd                 = *n;
	constexpr kernel_inttype a_cntg_step            = 1;
	const     kernel_inttype a_strd_step            = *lda;

	const     kernel_inttype w_cntg                 = *n;
	const     kernel_inttype w_strd                 = *nb;
	constexpr kernel_inttype w_cntg_step            = 1;
	const     kernel_inttype w_strd_step            = *ldw;

	const     kernel_inttype off_diagonal_cntg      = *n - 1;
	constexpr kernel_inttype off_diagonal_strd      = 1;
	constexpr kernel_inttype off_diagonal_cntg_step = 1;
	constexpr kernel_inttype off_diagonal_strd_step = 1;

	const     kernel_inttype tau_cntg               = *n - 1;
	constexpr kernel_inttype tau_strd               = 1;
	constexpr kernel_inttype tau_cntg_step          = 1;
	constexpr kernel_inttype tau_strd_step          = 1;

	const     auto           a_uplo                 = c_to_uplo(*uplo);

	using a_matrix_type = std::conditional_t<
		perflibs::is_complex_v<AType>,
		hermitian_matrix<matrix_base<AType>>,
		symmetric_matrix<matrix_base<AType>>
	>;

	auto pctx = spec::problem_context{
		factorization::tridiagonalization_block{
			a_matrix_type    { a_uplo, matrix_base { a,            a_cntg,            a_strd,            a_cntg_step,            a_strd_step            } },
			general_matrix   {         matrix_base { off_diagonal, off_diagonal_cntg, off_diagonal_strd, off_diagonal_cntg_step, off_diagonal_strd_step } },
			general_matrix   {         matrix_base { tau,          tau_cntg,          tau_strd,          tau_cntg_step,          tau_strd_step          } },
			general_matrix   {         matrix_base { w,            w_cntg,            w_strd,            w_cntg_step,            w_strd_step            } }
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	};
	factorization::compute(pctx);
}
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_FACTORIZATION_INTERFACES_LATRD_HPP

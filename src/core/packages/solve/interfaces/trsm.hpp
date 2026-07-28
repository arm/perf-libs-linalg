/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_TRSM_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_TRSM_HPP

#include "packages/solve/strategies.hpp"
#include "packages/solve/problem_context_bases.hpp"

#include "matrix/matrix.hpp"

#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

namespace perflibs::linalg {

template<typename AType, typename BType>
inline constexpr std::string_view trsm_name = "?TRSM ";

template<> inline constexpr std::string_view trsm_name<r32, r32> = "STRSM ";
template<> inline constexpr std::string_view trsm_name<r64, r64> = "DTRSM ";
template<> inline constexpr std::string_view trsm_name<c32, c32> = "CTRSM ";
template<> inline constexpr std::string_view trsm_name<c64, c64> = "ZTRSM ";

template<typename IntType, typename AType, typename BType>
PERFLIBS_LINALG_INLINE
bool trsm_param_check(
	const char *side, const char *uplo, const char *transa, const char *diag,
	const IntType *m, const IntType *n,
	const promote_t<AType, BType> *alpha,
	const AType *a, const IntType *lda,
	      BType *b, const IntType *ldb,
	std::string_view name) {

	const bool lside  = option_matches(*side, 'L');
	const bool rside  = option_matches(*side, 'R');

	const bool upper  = option_matches(*uplo, 'U');
	const bool lower  = option_matches(*uplo, 'L');

	const bool ttrans = option_matches(*transa, 'T');
	const bool ntrans = option_matches(*transa, 'N');
	const bool ctrans = option_matches(*transa, 'C');

	const bool udiag  = option_matches(*diag, 'U');
	const bool ndiag  = option_matches(*diag, 'N');

	const int_type nrowa = lside ? *m : *n;

	int_type info = 0;

	if (!lside && !rside) {
		info = 1;
	}
	if (!upper && !lower) {
		info = 2;
	}
	else if (!ttrans && !ntrans && !ctrans) {
		info = 3;
	}
	else if (!udiag && !ndiag) {
		info = 4;
	}
	else if (*m < 0) {
		info = 5;
	}
	else if (*n < 0) {
		info = 6;
	}
	else if (*lda < max(1, nrowa)) {
		info = 9;
	}
	else if (*ldb < max(1, *m)) {
		info = 11;
	}

	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}

	return true;
}

template<bool ParamCheck, typename IntType, typename AType, typename BType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void trsm(
	const char *side, const char *uplo, const char *transa, const char *diag,
	const IntType *m, const IntType *n,
	const promote_t<AType, BType> *alpha,
	const AType *a, const IntType *lda,
	      BType *b, const IntType *ldb) {

	if constexpr (ParamCheck) {
		const bool res = trsm_param_check(
			side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb,
				trsm_name<AType, BType>);

		if(! res) return;
	}

	const auto aside      = c_to_side(*side);
	const auto auplo      = c_to_uplo(*uplo);
	const auto atransa    = c_to_trans(*transa);
	const auto adiag      = c_to_diag(*diag);
	const bool is_lside   = is_left(aside);
	const bool is_trans_a = is_trans(atransa);
	const bool is_conj_a  = is_conj(atransa);

	if (is_lside) {
		const auto           auplo_mod   = is_trans_a ? auplo : lower_flip(auplo);
		const kernel_inttype b_cntg      = *m;
		const kernel_inttype b_strd      = *n;

		const kernel_inttype a_cntg_step = is_trans_a ?   1 : *lda;
		const kernel_inttype a_strd_step = is_trans_a ? *lda :   1;
		const kernel_inttype b_cntg_step = 1;
		const kernel_inttype b_strd_step = *ldb;

		spec::problem_context pctx {
			solve::solve {
				aside, atransa,
				triangular_matrix { auplo_mod, adiag, matrix_base { a, b_cntg, b_cntg, a_cntg_step, a_strd_step }, is_conj_a },
				general_matrix    {                   matrix_base { b, b_cntg, b_strd, b_cntg_step, b_strd_step }            },
				*alpha
			},
			ArchitectureSpec{ machine::get_system_unsafe() }
		};

		solve::compute(pctx);
	}
	else {
		const auto           auplo_mod   = is_trans_a ? lower_flip(auplo) : auplo;
		const kernel_inttype b_cntg      = *n;
		const kernel_inttype b_strd      = *m;

		const kernel_inttype a_cntg_step = is_trans_a ? *lda :   1;
		const kernel_inttype a_strd_step = is_trans_a ?   1 : *lda;
		const kernel_inttype b_cntg_step = *ldb;
		const kernel_inttype b_strd_step = 1;

		spec::problem_context pctx {
			solve::solve {
				aside, atransa,
				triangular_matrix { auplo_mod, adiag, matrix_base { a, b_cntg, b_cntg, a_cntg_step, a_strd_step }, is_conj_a },
				general_matrix    {                   matrix_base { b, b_cntg, b_strd, b_cntg_step, b_strd_step }            },
				*alpha
			},
			ArchitectureSpec{ machine::get_system_unsafe() }
		};

		solve::compute(pctx);
	}
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_TRSM_HPP

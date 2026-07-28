/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_TBSV_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_TBSV_HPP

#include "packages/solve/strategies.hpp"
#include "packages/solve/problem_context_bases.hpp"

#include "matrix/matrix.hpp"

#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

namespace perflibs::linalg {

template<typename AType, typename XType>
inline constexpr std::string_view tbsv_name = "?TBSV ";

template<> inline constexpr std::string_view tbsv_name<r32, r32> = "STBSV ";
template<> inline constexpr std::string_view tbsv_name<r64, r64> = "DTBSV ";
template<> inline constexpr std::string_view tbsv_name<c32, c32> = "CTBSV ";
template<> inline constexpr std::string_view tbsv_name<c64, c64> = "ZTBSV ";

template<typename IntType, typename AType, typename XType>
PERFLIBS_LINALG_INLINE
bool tbsv_param_check(
	const char *uplo, const char *trans, const char *diag,
	const IntType *n, const IntType *k,
	const AType *a, const IntType *lda,
	      XType *x, const IntType *incx,
	std::string_view name) {

	pl_linalg_int_t info = 0;
	if (! option_matches(*uplo, 'U', 'L')) {
		info = 1;
	}
	else if (! option_matches(*trans, 'N', 'T', 'C')) {
		info = 2;
	}
	else if (! option_matches(*diag, 'U', 'N')) {
		info = 3;
	}
	else if (*n < 0) {
		info = 4;
	}
	else if (*k < 0) {
		info = 5;
	}
	else if (*lda < *k + 1) {
		info = 7;
	}
	else if (*incx == 0) {
		info = 9;
	}

	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}

	return true;
}


template<bool ParamCheck, typename IntType, typename AType, typename XType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void tbsv(
	const char *uplo, const char *trans, const char *diag,
	const IntType *n, const IntType *k,
	const AType *a, const IntType *lda,
	      XType *x, const IntType *incx) {

	using scalar_type = promote_t<XType, AType>;

	if constexpr(ParamCheck) {
		const auto res = perflibs::linalg::tbsv_param_check(
			uplo, trans, diag, n, k, a, lda, x, incx,
				tbsv_name<AType, XType>);

		if(! res) return;
	}

	if (*n == 0)
		return;

	const     auto           aside       = PERFLIBS_LEFT;
	const     auto           atransa     = c_to_trans(*trans);
	const     bool           is_trans_a  = is_trans(atransa);
	const     auto           auplo       = c_to_uplo(*uplo);
	const     auto           auplo_mod   = is_trans_a ? lower_flip(auplo) : auplo;
	const     auto           adiag       = c_to_diag(*diag);

	const     kernel_inttype a_cntg      = *n;
	const     kernel_inttype c_cntg      = *n;
	constexpr kernel_inttype c_strd      = 1;
	const     kernel_inttype kl          = auplo_mod == PERFLIBS_LOWER ? *k : 0;
	const     kernel_inttype ku          = auplo_mod == PERFLIBS_LOWER ? 0 : *k;

	const     kernel_inttype a_cntg_step = is_trans_a ? *lda : 1;
	const     kernel_inttype a_strd_step = is_trans_a ? 1 : *lda;

	const     kernel_inttype b_cntg_step = *incx;
	constexpr kernel_inttype b_strd_step = 1;

	if(*incx < 0) x += (*incx * -1) * (*n - 1);

	spec::problem_context pctx {
		solve::solve {
			aside, atransa,
			triangular_matrix { auplo_mod, adiag, banded_matrix_base { a, c_cntg, a_cntg, a_cntg_step, a_strd_step, kl, ku }, is_conj(atransa) },
			general_matrix    {                   matrix_base        { x, a_cntg, c_strd, b_cntg_step, b_strd_step         }                   },
			one<scalar_type>
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	};

	solve::compute(pctx);
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_TBSV_HPP

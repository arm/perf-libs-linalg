/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_TRSV_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_TRSV_HPP

#include "packages/solve/strategies.hpp"
#include "packages/solve/problem_context_bases.hpp"

#include "matrix/matrix.hpp"

#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

namespace perflibs::linalg {

template<typename AType, typename XType>
inline constexpr std::string_view trsv_name = "?TRSV ";

template<> inline constexpr std::string_view trsv_name<r32, r32> = "STRSV ";
template<> inline constexpr std::string_view trsv_name<r64, r64> = "DTRSV ";
template<> inline constexpr std::string_view trsv_name<c32, c32> = "CTRSV ";
template<> inline constexpr std::string_view trsv_name<c64, c64> = "ZTRSV ";

template<typename IntType, typename AType, typename XType>
PERFLIBS_LINALG_INLINE
bool trsv_param_check(
	const char *uplo, const char *trans, const char *diag,
	const IntType *n,
	const AType *a, const IntType *lda,
	      XType *x, const IntType *incx,
	std::string_view name) {

	const bool upper  = option_matches(*uplo, 'U');
	const bool lower  = option_matches(*uplo, 'L');

	const bool ttrans = option_matches(*trans, 'T');
	const bool ntrans = option_matches(*trans, 'N');
	const bool ctrans = option_matches(*trans, 'C');

	const bool udiag  = option_matches(*diag, 'U');
	const bool ndiag  = option_matches(*diag, 'N');

	int_type info = 0;

	if (!upper && !lower) {
		info = 1;
	}
	else if (!ttrans && !ntrans && !ctrans) {
		info = 2;
	}
	else if (!udiag && !ndiag) {
		info = 3;
	}
	else if (*n < 0) {
		info = 4;
	}
	else if (*lda < max(1, *n)) {
		info = 6;
	}
	else if (*incx == 0) {
		info = 8;
	}

	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}

	return true;
}

template<bool ParamCheck, typename IntType, typename AType, typename XType, typename ArchitectureSpec>
void trsv(
	const char *uplo, const char *trans, const char *diag,
	const IntType *n,
	const AType *a, const IntType *lda,
	      XType *x, const IntType *incx) {

	using scalar_type = promote_t<AType, XType>;

	if constexpr (ParamCheck) {
		const bool res = trsv_param_check(
			uplo, trans, diag, n, a, lda, x, incx,
				trsv_name<AType, XType>);

		if(! res) return;
	}

	const     auto           atransa     = c_to_trans(*trans);
	const     bool           is_trans_a  = is_trans(atransa);
	const     auto           auplo       = c_to_uplo(*uplo);
	const     auto           auplo_mod   = is_trans_a ? auplo : lower_flip(auplo);
	const     auto           adiag       = c_to_diag(*diag);

	const     kernel_inttype b_cntg      = *n;
	constexpr kernel_inttype b_strd      = 1;

	const     auto           a_ptr       = a;
	const     kernel_inttype a_cntg_step = is_trans_a ? 1 : *lda;
	const     kernel_inttype a_strd_step = is_trans_a ? *lda : 1;

	/*
	* A negative incx simply means that we want to walk through the data
	* backwards, not that the data exist BEFORE the pointer address
	*/
	if(*incx < 0) x += (*incx * -1) * (*n - 1);

	const     auto           b_ptr       = x;
	const     kernel_inttype b_cntg_step = *incx;
	constexpr kernel_inttype b_strd_step = 0;

	spec::problem_context pctx {
		solve::solve {
			PERFLIBS_LEFT, atransa,
			triangular_matrix { auplo_mod, adiag, matrix_base { a_ptr, b_cntg, b_cntg, a_cntg_step, a_strd_step }, is_conj(atransa) },
			general_matrix    {                   matrix_base { b_ptr, b_cntg, b_strd, b_cntg_step, b_strd_step } },
			one<scalar_type>
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	};

	solve::compute(pctx);
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_TRSV_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_TRMM_OUT_OF_PLACE_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_TRMM_OUT_OF_PLACE_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"
#include "detect/system.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename AType, typename BType, typename CType>
inline constexpr std::string_view trmm_out_of_place_name = "?TRMM_OUT_OF_PLACE";

template<> inline constexpr std::string_view trmm_out_of_place_name<r32, r32, r32> = "STRMM_OUT_OF_PLACE";
template<> inline constexpr std::string_view trmm_out_of_place_name<r64, r64, r64> = "DTRMM_OUT_OF_PLACE";
template<> inline constexpr std::string_view trmm_out_of_place_name<c32, c32, c32> = "CTRMM_OUT_OF_PLACE";
template<> inline constexpr std::string_view trmm_out_of_place_name<c64, c64, c64> = "ZTRMM_OUT_OF_PLACE";

template<typename IntType, typename AType, typename BType, typename CType>
PERFLIBS_LINALG_INLINE
bool trmm_out_of_place_param_check(
	const char *side, const char *uplo,
	const char *transa, const char *transb,
	const char *diag,
	const IntType *m, const IntType *n,
	const promote_t<AType, BType, CType> *alpha,
	const AType *a, const IntType *lda,
	const BType *b, const IntType *ldb,
	const promote_t<AType, BType, CType> *beta,
	      CType *c, const IntType *ldc,
	std::string_view name) {

	const bool  lside = option_matches(*side, 'L');
	const bool  rside = option_matches(*side, 'R');

	const bool ttransa = option_matches(*transa, 'T');
	const bool ntransa = option_matches(*transa, 'N');
	const bool ctransa = option_matches(*transa, 'C');

	const bool ttransb = option_matches(*transb, 'T');
	const bool ntransb = option_matches(*transb, 'N');
	const bool ctransb = option_matches(*transb, 'C');

	const bool upper   = option_matches(*uplo, 'U');
	const bool lower   = option_matches(*uplo, 'L');

	const bool udiag   = option_matches(*diag, 'U');
	const bool ldiag   = option_matches(*diag, 'N');

	auto nrowa = lside ? *m : *n;
	auto ldb_bound = ntransb ? *m : *n;

	IntType info = 0;

	if (! lside && ! rside ) {
		info = 1;
	}
	else if (! upper && ! lower) {
		info = 2;
	}
	else if (! ntransa && ! ttransa  && ! ctransa) {
		info = 3;
	}
	else if (! ntransb && ! ttransb  && ! ctransb) {
		info = 4;
	}
	else if (! udiag && ! ldiag) {
		info = 5;
	}
	else if (*m < 0) {
		info = 6;
	}
	else if (*n < 0) {
		info = 7;
	}
	else if (*lda < max(1,nrowa)) {
		info = 10;
	}
	else if (*ldb < max(1,ldb_bound)) {
		info = 12;
	}
	else if (*ldc < max(1,*m)) {
		info = 15;
	}
	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}
	return true;
}

template<bool ParamCheck, typename IntType, typename AType, typename BType, typename CType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void trmm_out_of_place(
	const char *side, const char *uplo,
	const char *transa, const char *transb, const char *diag,
	const IntType *m, const IntType *n,
	const promote_t<AType, BType, CType> *alpha,
	const AType *a, const IntType *lda,
	const BType *b, const IntType *ldb,
	const promote_t<AType, BType, CType> *beta,
	      CType *c, const IntType *ldc) {

	using scalar_type = promote_t<AType, BType, CType>;

	if constexpr(ParamCheck) {
		const bool res = trmm_out_of_place_param_check(
			side, uplo, transa, transb, diag, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
				trmm_out_of_place_name<AType, BType, CType>);

		if(! res) return;
	}

	if(*m == 0 || *n == 0 || ( *alpha == zero<scalar_type> && *beta == one<scalar_type> ))
		return;


	if (option_matches(*side, 'L')) {

		const     kernel_inttype a_strd        = *m;
		const     kernel_inttype b_strd        = *n;
		const     kernel_inttype cntg          = *m;

		const     auto           a_trans       = c_to_trans(*transa);
		const     bool           is_a_trans    = is_trans(a_trans);
		const     auto           a_uplo        = is_a_trans ? c_to_uplo(*uplo) : lower_flip(c_to_uplo(*uplo));
		const     auto           a_ptr         = a;
		const     kernel_inttype a_cntg_stride = is_a_trans ? 1 : *lda;
		const     kernel_inttype a_strd_stride = is_a_trans ? *lda : 1;
		const     auto           a_is_conj     = is_conj(a_trans);
		const     auto           a_diag        = c_to_diag(*diag);

		const     auto           b_trans       = c_to_trans(*transb);
		const     bool           is_b_trans    = is_trans(b_trans);
		const     auto           b_ptr         = b;
		const     kernel_inttype b_cntg_stride = is_b_trans ? *ldb : 1;
		const     kernel_inttype b_strd_stride = is_b_trans ? 1 : *ldb;
		const     auto           b_is_conj     = is_conj(b_trans);

		const     auto           c_ptr         = c;
		constexpr kernel_inttype c_cntg_stride = 1;
		const     kernel_inttype c_strd_stride = *ldc;

		spec::problem_context pctx {
			matmul::matmul3 {
				triangular_matrix { a_uplo, a_diag, matrix_base { a_ptr, cntg,   a_strd, a_cntg_stride, a_strd_stride }, a_is_conj },
				general_matrix    {                 matrix_base { b_ptr, cntg,   b_strd, b_cntg_stride, b_strd_stride }, b_is_conj },
				general_matrix    {                 matrix_base { c_ptr, a_strd, b_strd, c_cntg_stride, c_strd_stride }            },
				*alpha, *beta
			},
			ArchitectureSpec { machine::get_system_unsafe() }
		};

		matmul::compute(pctx);
	}
	else {

		const     kernel_inttype a_strd        = *m;
		const     kernel_inttype b_strd        = *n;
		const     kernel_inttype cntg          = *n;

		const     auto           a_trans       = c_to_trans(*transb);
		const     bool           is_a_trans    = is_trans(a_trans);
		const     auto           a_is_conj     = is_conj(a_trans);
		const     auto           a_ptr         = b;

		const     kernel_inttype a_cntg_stride = is_a_trans ? 1 : *ldb;
		const     kernel_inttype a_strd_stride = is_a_trans ? *ldb : 1;

		const     auto           b_trans       = c_to_trans(*transa);
		const     bool           is_b_trans    = is_trans(b_trans);
		const     auto           b_is_conj     = is_conj(b_trans);
		const     auto           b_uplo        = is_b_trans ? lower_flip(c_to_uplo(*uplo)) : c_to_uplo(*uplo);
		const     auto           b_ptr         = a;
		const     kernel_inttype b_cntg_stride = is_b_trans ? *lda : 1;
		const     kernel_inttype b_strd_stride = is_b_trans ? 1 : *lda;
		const     auto           b_diag        = c_to_diag(*diag);

		const     auto           c_ptr         = c;
		constexpr kernel_inttype c_cntg_stride = 1;
		const     kernel_inttype c_strd_stride = *ldc;

		spec::problem_context pctx {
			matmul::matmul3 {
				general_matrix    {                 matrix_base { a_ptr, cntg,   a_strd, a_cntg_stride, a_strd_stride }, a_is_conj },
				triangular_matrix { b_uplo, b_diag, matrix_base { b_ptr, cntg,   b_strd, b_cntg_stride, b_strd_stride }, b_is_conj },
				general_matrix    {                 matrix_base { c_ptr, a_strd, b_strd, c_cntg_stride, c_strd_stride }            },
				*alpha, *beta
			},
			ArchitectureSpec { machine::get_system_unsafe() }
		};

		matmul::compute(pctx);
	}
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_TRMM_OUT_OF_PLACE_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_TRMM_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_TRMM_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include "detect/system.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename AType, typename BType>
inline constexpr std::string_view trmm_name = "?TRMM ";

template<> inline constexpr std::string_view trmm_name<r32, r32> = "STRMM ";
template<> inline constexpr std::string_view trmm_name<r64, r64> = "DTRMM ";
template<> inline constexpr std::string_view trmm_name<c32, c32> = "CTRMM ";
template<> inline constexpr std::string_view trmm_name<c64, c64> = "ZTRMM ";

template<typename IntType, typename AType, typename BType>
PERFLIBS_LINALG_INLINE
bool trmm_param_check(
	const char *side, const char *uplo,
	const char *transa, const char *diag,
	const IntType *m, const IntType *n,
	const promote_t<AType, BType> *alpha,
	const AType *a, const IntType *lda,
	      BType *b, const IntType *ldb,
	std::string_view name) {

	const bool  lside = option_matches(*side, 'L');
	const bool  rside = option_matches(*side, 'R');

	const bool ttransa = option_matches(*transa, 'T');
	const bool ntransa = option_matches(*transa, 'N');
	const bool ctransa = option_matches(*transa, 'C');

	const bool upper   = option_matches(*uplo, 'U');
	const bool lower   = option_matches(*uplo, 'L');

	const bool udiag   = option_matches(*diag, 'U');
	const bool ldiag   = option_matches(*diag, 'N');

	auto nrowa = lside ? *m : *n;

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
	else if (! udiag && ! ldiag) {
		info = 4;
	}
	else if (*m < 0) {
		info = 5;
	}
	else if (*n < 0) {
		info = 6;
	}
	else if (*lda < max(1,nrowa)) {
		info = 9;
	}
	else if (*ldb < max(1,*m)) {
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
void trmm(
	const char *side, const char *uplo,
	const char *transa, const char *diag,
	const IntType *m, const IntType *n,
	const promote_t<AType, BType> *alpha,
	const AType *a, const IntType *lda,
	      BType *b, const IntType *ldb) {

	using scalar_type = promote_t<AType, BType>;

	if constexpr(ParamCheck) {
		const bool res = trmm_param_check(
			side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb,
				trmm_name<AType, BType>);
		if(! res) return;
	}

	const auto aside                 = c_to_side(*side);
	const bool is_left               = aside == PERFLIBS_LEFT;
	const auto adiag                 = c_to_diag(*diag);
	const auto atransa               = c_to_trans(*transa);

	const bool is_trans_a            = is_trans(atransa) ^ (!is_left);

	const auto auplo                 = is_trans_a ? c_to_uplo(*uplo): lower_flip(c_to_uplo(*uplo));

	const kernel_inttype cntg        = is_left ? *m : *n; //a is square
	const kernel_inttype b_strd      = is_left ? *n : *m;

	const kernel_inttype a_cntg_step = is_trans_a ? 1 : *lda;
	const kernel_inttype a_strd_step = is_trans_a ? *lda  : 1;

	const kernel_inttype b_cntg_step = is_left ? 1 : *ldb;
	const kernel_inttype b_strd_step = is_left ? *ldb : 1;


	spec::problem_context pctx {
		matmul::matmul2 {
			triangular_matrix { auplo, adiag, matrix_base { a, cntg, cntg,   a_cntg_step, a_strd_step }, is_conj(atransa) },
			general_matrix    {               matrix_base { b, cntg, b_strd, b_cntg_step, b_strd_step }                   },
			*alpha, zero<scalar_type>
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx);
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_TRMM_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_TRMV_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_TRMV_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"
#include "detect/system.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename AType, typename XType>
inline constexpr std::string_view trmv_name = "?TRMV ";

template<> inline constexpr std::string_view trmv_name<r32, r32> = "STRMV ";
template<> inline constexpr std::string_view trmv_name<r64, r64> = "DTRMV ";
template<> inline constexpr std::string_view trmv_name<c32, c32> = "CTRMV ";
template<> inline constexpr std::string_view trmv_name<c64, c64> = "ZTRMV ";

template<typename IntType, typename AType, typename XType>
PERFLIBS_LINALG_INLINE
bool trmv_param_check(
	const char *uplo, const char *trans, const char *diag,
	const IntType *n,
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
	else if (*lda < max(1,*n)) {
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
PERFLIBS_LINALG_INLINE
void trmv(
	const char *uplo, const char *trans, const char *diag,
	const IntType *n,
    const AType *a, const IntType *lda,
	      XType *x, const IntType *incx) {

	using scalar_type = promote_t<AType, XType>;

	if constexpr(ParamCheck) {
		const bool res = trmv_param_check(
			uplo, trans, diag, n, a, lda, x, incx,
				trmv_name<AType, XType>);

		if(! res) return;
	}

	if (*n == 0) return;

	if (*incx < 0) x += (*incx * -1) * (*n - 1);

	const auto adiag                     = c_to_diag(*diag);
	const auto atransa                   = c_to_trans(*trans);
	const bool is_trans_a                = is_trans(atransa);

	const auto auplo                     = is_trans_a ? c_to_uplo(*uplo): lower_flip(c_to_uplo(*uplo));

	const     kernel_inttype cntg        = *n;
	constexpr kernel_inttype b_strd      = 1;

	const     kernel_inttype a_cntg_step = is_trans_a ? 1 : *lda;
	const     kernel_inttype a_strd_step = is_trans_a ? *lda : 1;

	const     kernel_inttype b_cntg_step = *incx;
	constexpr kernel_inttype b_strd_step = 0;

	spec::problem_context pctx {
		matmul::matmul2 {
			triangular_matrix { auplo, adiag, matrix_base { a, cntg, cntg,   a_cntg_step, a_strd_step }, is_conj(atransa) },
			general_matrix    {               matrix_base { x, cntg, b_strd, b_cntg_step, b_strd_step }                   },
			one<scalar_type>, zero<scalar_type>
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx);
}
} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_TRMV_HPP

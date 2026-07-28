/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_TPMV_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_TPMV_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename AType, typename XType>
inline constexpr std::string_view tpmv_name = "?TPMV ";

template<> inline constexpr std::string_view tpmv_name<r32, r32> = "STPMV ";
template<> inline constexpr std::string_view tpmv_name<r64, r64> = "DTPMV ";
template<> inline constexpr std::string_view tpmv_name<c32, c32> = "CTPMV ";
template<> inline constexpr std::string_view tpmv_name<c64, c64> = "ZTPMV ";

template<typename IntType, typename AType, typename XType>
PERFLIBS_LINALG_INLINE
bool tpmv_param_check(
	const char *uplo, const char *trans, const char *diag,
	const IntType *n,
	const AType *a,
	      XType *x, const IntType *incx,
	std::string_view name) {


	const bool is_upper = option_matches(*uplo, 'U');
	const bool is_lower = option_matches(*uplo, 'L');
	const bool is_not_trans = option_matches(*trans, 'N');
	const bool is_trans = option_matches(*trans, 'T');
	const bool is_conj_trans = option_matches(*trans, 'C');
	const bool is_unit = option_matches(*diag, 'U');
	const bool is_not_unit = option_matches(*diag, 'N');

	pl_linalg_int_t info = 0;
	if (!is_upper && !is_lower) {
		info = 1;
	}
	else if (!is_not_trans && !is_trans && !is_conj_trans) {
		info = 2;
	}
	else if (!is_unit && !is_not_unit) {
		info = 3;
	}
	else if (*n < 0) {
		info = 4;
	}
	else if (*incx == 0) {
		info = 7;
	}
	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}
	return true;
}

template<bool ParamCheck, typename IntType, typename AType, typename XType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void tpmv(
	const char *uplo, const char *trans, const char *diag,
	const IntType *n,
	const AType *a,
	      XType *x, const IntType *incx) {

	using scalar_type = promote_t<XType, AType>;

	if constexpr(ParamCheck) {
		const auto res = perflibs::linalg::tpmv_param_check(
			uplo, trans, diag, n, a, x, incx,
				tpmv_name<AType, XType>);

		if(! res) return;
	}

	if (*n == 0) {
		return;
	}

	const      auto           atransa       = c_to_trans(*trans);
	const      auto           auplo         = atransa == PERFLIBS_NOTRANS ? c_to_uplo(*uplo) : lower_flip(c_to_uplo(*uplo));
	const      auto           adiag         = c_to_diag(*diag);

	const      kernel_inttype a_strd        = *n;
	constexpr  kernel_inttype b_strd        = 1;
	const      kernel_inttype cntg          = *n;

	const      kernel_inttype b_cntg_stride = *incx;
	constexpr  kernel_inttype b_strd_stride = 1;

	if(*incx < 0) x += (*incx * -1) * (*n - 1);

	spec::problem_context pctx {
		matmul::matmul2 {
			triangular_matrix { auplo, adiag, packed_matrix_base { a, cntg  , a_strd, auplo, true, is_trans(atransa) }, is_conj(atransa) },
			general_matrix    {               matrix_base        { x, a_strd, b_strd, b_cntg_stride, b_strd_stride   }                   },
			one<scalar_type>, zero<scalar_type>
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx);
}

} // namespace perflibs::linalg
#endif //PERFLIBS_LINALG_BLAS_INTERFACES_TPMV_HPP

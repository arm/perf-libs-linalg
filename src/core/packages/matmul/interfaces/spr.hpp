/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_SPR_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_SPR_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename XType, typename AType>
inline constexpr std::string_view spr_name = "?SPR  ";

template<> inline constexpr std::string_view spr_name<r32, r32> = "SSPR  ";
template<> inline constexpr std::string_view spr_name<r64, r64> = "DSPR  ";
template<> inline constexpr std::string_view spr_name<c32, c32> = "CSPR  ";
template<> inline constexpr std::string_view spr_name<c64, c64> = "ZSPR  ";

template<typename IntType, typename XType, typename AType>
PERFLIBS_LINALG_INLINE
bool spr_param_check(
	const char *uplo, const IntType *n,
	const promote_t<XType, AType> *alpha,
	const XType *x, const IntType *incx,
	      AType *ap,
	std::string_view name) {

	const bool is_upper = option_matches(*uplo, 'U');
	const bool is_lower = option_matches(*uplo, 'L');

	pl_linalg_int_t info = 0;
	if (!is_upper && !is_lower) {
		info = 1;
	}
	else if (*n < 0) {
		info = 2;
	}
	else if (*incx == 0) {
		info = 5;
	}
	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}
	return true;
}

template<bool ParamCheck, typename IntType, typename XType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void spr(
	const char *uplo, const IntType *n,
	const promote_t<XType, AType> *alpha,
	const XType *x, const IntType *incx,
	      AType *ap) {

	using scalar_type = promote_t<XType, AType>;

	if constexpr(ParamCheck) {
		const auto res =  perflibs::linalg::spr_param_check(
			uplo, n, alpha, x, incx, ap,
				spr_name<XType, AType>);

		if(!res) return;
	}

	if (*n == 0 || *alpha == zero<scalar_type>) {
		return;
	}

	const     kernel_inttype a_strd        = *n;
	const     kernel_inttype b_strd        = *n;
	constexpr kernel_inttype cntg          = 1;

	constexpr kernel_inttype a_cntg_stride = 1;
	const     kernel_inttype a_strd_stride = *incx;

	constexpr kernel_inttype b_cntg_stride = 1;
	const     kernel_inttype b_strd_stride = *incx;

	const     perflibs_uplo     c_uplo        = c_to_uplo(*uplo);

	if (*incx < 0) x += (*incx * -1) * (*n - 1);

	spec::problem_context pctx {
		matmul::matmul3 {
			general_matrix   {         matrix_base        { x , cntg  , a_strd, a_cntg_stride, a_strd_stride } },
			general_matrix   {         matrix_base        { x , cntg  , b_strd, b_cntg_stride, b_strd_stride } },
			symmetric_matrix { c_uplo, packed_matrix_base { ap, a_strd, b_strd, c_uplo, false                } },
			*alpha, one<XType>
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx);
}

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_SPR_HPP

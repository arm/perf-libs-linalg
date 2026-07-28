/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_SPMV_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_SPMV_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename AType, typename XType, typename YType>
inline constexpr std::string_view hpmv_name = "?HPMV ";

template<> inline constexpr std::string_view hpmv_name<c32, c32, c32> = "CHPMV ";
template<> inline constexpr std::string_view hpmv_name<c64, c64, c64> = "ZHPMV ";

template<typename IntType, typename AType, typename XType, typename YType>
PERFLIBS_LINALG_INLINE
bool hpmv_param_check(	const char *uplo, const IntType *n,
	const promote_t<AType, XType, YType> *alpha,
	const AType *ap,
	const XType *x, const IntType *incx,
	const promote_t<AType, XType, YType> *beta,
	      YType *y, const IntType *incy,
	std::string_view name) {

	bool is_upper = option_matches(*uplo, 'U');
	bool is_lower = option_matches(*uplo, 'L');

	pl_linalg_int_t info = 0;
	if (!is_upper && !is_lower) {
		info = 1;
	}
	else if (*n < 0) {
		info = 2;
	}
	else if (*incx == 0) {
		info = 6;
	}
	else if (*incy == 0) {
		info = 9;
	}
	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}
	return true;
}

template<bool ParamCheck, typename IntType, typename AType, typename XType, typename YType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void hpmv(
	const char *uplo, const IntType *n,
	const promote_t<AType, XType, YType> *alpha,
	const AType *ap,
	const XType *x, const IntType *incx,
	const promote_t<AType, XType, YType> *beta,
	      YType *y, const IntType *incy) {

	using scalar_type = promote_t<XType, AType>;

	if (!perflibs::linalg::hpmv_param_check(uplo, n, alpha, ap, x, incx, beta, y, incy,
	    hpmv_name<AType, XType, YType>)) {
		return;
	}

	if ((*n == 0) || ((*alpha == zero<scalar_type>) && (*beta == one<scalar_type>))) {
		return;
	}

	const     perflibs_uplo     a_uplo        = lower_flip(c_to_uplo(*uplo));

	const     kernel_inttype a_strd        = *n;
	constexpr kernel_inttype b_strd        = 1;
	const     kernel_inttype cntg          = *n;

	const     kernel_inttype b_cntg_stride = *incx;
	constexpr kernel_inttype b_strd_stride = 1;

	const     kernel_inttype c_cntg_stride = *incy;
	constexpr kernel_inttype c_strd_stride = 1;

	if (*incx < 0) x += (*incx * -1) * (*n - 1);
	if (*incy < 0) y += (*incy * -1) * (*n - 1);

	spec::problem_context pctx {
		matmul::matmul3 {
			hermitian_matrix { a_uplo, packed_matrix_base { ap, cntg  , a_strd, lower_flip(a_uplo)                   } },
			general_matrix   {         matrix_base        {         x , cntg  , b_strd, b_cntg_stride, b_strd_stride } },
			general_matrix   {         matrix_base        {         y , a_strd, b_strd, c_cntg_stride, c_strd_stride } },
			*alpha, *beta
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx);
}

} // namespace perflibs::linalg {

#endif

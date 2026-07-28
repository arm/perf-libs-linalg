/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_HEMV_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_HEMV_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename AType, typename XType, typename YType>
inline constexpr std::string_view hemv_name = "?HEMV ";

template<> inline constexpr std::string_view hemv_name<c32, c32, c32> = "CHEMV ";
template<> inline constexpr std::string_view hemv_name<c64, c64, c64> = "ZHEMV ";

template<typename IntType, typename AType, typename XType, typename YType>
static inline bool hemv_param_check(
	const char *uplo, const IntType *n,
	const promote_t<AType, XType, YType> *alpha,
	const AType *a, const IntType *lda,
	const XType *x, const IntType *incx,
	const promote_t<AType, XType, YType> *beta,
	const YType *y, const IntType *incy,
	std::string_view name) {

	pl_linalg_int_t info = 0;
	if (! option_matches(*uplo, 'U', 'L')) {
		info = 1;
	}
	else if (*n < 0) {
		info = 2;
	}
	else if (*lda < max(1,*n)) {
		info = 5;
	}
	else if (*incx == 0) {
		info = 7;
	}
	else if (*incy == 0) {
		info = 10;
	}

	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}
	return true;
}

template<bool ParamCheck, typename IntType, typename AType, typename XType, typename YType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void hemv(
	const char *uplo, const IntType *n,
	const promote_t<AType, XType, YType> *alpha,
	const AType *a, const IntType *lda,
	const XType *x, const IntType *incx,
	const promote_t<AType, XType, YType> *beta,
	      YType *y, const IntType *incy) {

	if (!linalg::hemv_param_check(uplo, n, alpha, a, lda, x, incx, beta, y, incy,
	                            hemv_name<AType, XType, YType>)) {
		return;
	}

	if (*incx < 0) x += (*incx * -1) * (*n - 1);
	if (*incy < 0) y += (*incy * -1) * (*n - 1);

	const    kernel_inttype a_strd         = *n;
	const    kernel_inttype b_strd         = 1;
	const    kernel_inttype cntg           = *n;

	const     auto           a_uplo        = lower_flip(c_to_uplo(*uplo));
	const     auto           a_ptr         = a;
	const     kernel_inttype a_cntg_stride = *lda;
	constexpr kernel_inttype a_strd_stride = 1;

	const     auto           b_ptr         = x;
	const     kernel_inttype b_cntg_stride = *incx;
	const     kernel_inttype b_strd_stride = *n;

	const     auto           c_ptr         = y;
	const     kernel_inttype c_cntg_stride = *incy;
	const     kernel_inttype c_strd_stride = *n;



	spec::problem_context pctx {
		matmul::matmul3 {
			hermitian_matrix { a_uplo, matrix_base { a_ptr, cntg,   a_strd, a_cntg_stride, a_strd_stride } },
			general_matrix   {         matrix_base { b_ptr, cntg,   b_strd, b_cntg_stride, b_strd_stride } },
			general_matrix   {         matrix_base { c_ptr, a_strd, b_strd, c_cntg_stride, c_strd_stride } },
			*alpha, *beta
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx);
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_HEMV_HPP

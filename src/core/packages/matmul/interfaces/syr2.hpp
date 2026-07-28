/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_SYR2_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_SYR2_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename XType, typename YType, typename AType>
inline constexpr std::string_view syr2_name = "?SYR2 ";

template<> inline constexpr std::string_view syr2_name<r32, r32, r32> = "SSYR2 ";
template<> inline constexpr std::string_view syr2_name<r64, r64, r64> = "DSYR2 ";

template<typename IntType, typename XType, typename YType, typename AType>
PERFLIBS_LINALG_INLINE
bool syr2_param_check(
	const char *uplo, const IntType *n,
	const promote_t<XType, YType, AType> *alpha,
	const XType *x, const IntType *incx,
	const YType *y, const IntType *incy,
	const AType *a, const IntType *lda,
	std::string_view name) {

	pl_linalg_int_t nrowa, info, upper, lower;

	nrowa = *n;
	upper = option_matches(*uplo, 'U');
	lower = option_matches(*uplo, 'L');

	info = 0;
	if (!upper && !lower) {
		info = 1;
	}
	else if (*n < 0) {
		info = 2;
	}
	else if (*incx == 0) {
		info = 5;
	}
	else if (*incy == 0) {
		info = 7;
	}
	else if (*lda < max(1, nrowa)) {
		info = 9;
	}
	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}
	return true;
}

template<bool ParamCheck, typename IntType, typename XType, typename YType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void syr2(
	const char *uplo, const IntType *n,
	const promote_t<XType, YType, AType> *alpha,
	const XType *x, const IntType *incx,
	const YType *y, const IntType *incy,
	      AType *a, const IntType *lda) {

	using scalar_type = promote_t<XType, YType, AType>;

	if constexpr(ParamCheck) {
		const bool res = perflibs::linalg::syr2_param_check(uplo, n, alpha, x, incx, y, incy, a, lda,
			syr2_name<XType, YType, AType>);

		if (!res) return;
	}

	if (*n == 0 || *alpha == zero<scalar_type>) {
		return;
	}

	if (*incx < 0) x += (*incx * -1) * (*n - 1);

	if (*incy < 0) y += (*incy * -1) * (*n - 1);

	const     kernel_inttype a_strd        = *n;
	const     kernel_inttype b_strd        = *n;
	constexpr kernel_inttype cntg          = 1;

	const     auto           a_ptr         = x;
	constexpr kernel_inttype a_cntg_stride = 0;
	const     kernel_inttype a_strd_stride = *incx;

	const     auto           b_ptr         = y;
	constexpr kernel_inttype b_cntg_stride = 0;
	const     kernel_inttype b_strd_stride = *incy;

	const     auto           c_uplo        = c_to_uplo(*uplo);
	const     auto           c_ptr         = a;
	constexpr kernel_inttype c_cntg_stride = 1;
	const     kernel_inttype c_strd_stride = *lda;

	spec::problem_context pctx {
		matmul::rank_update_2k {
			general_matrix   {         matrix_base { a_ptr, cntg,   a_strd, a_cntg_stride, a_strd_stride } },
			general_matrix   {         matrix_base { b_ptr, cntg,   a_strd, b_cntg_stride, b_strd_stride } },
			symmetric_matrix { c_uplo, matrix_base { c_ptr, a_strd, b_strd, c_cntg_stride, c_strd_stride } },
			*alpha, one<scalar_type>
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx);
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_SYR2_HPP

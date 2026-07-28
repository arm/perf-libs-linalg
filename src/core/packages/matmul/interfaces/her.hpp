/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_HER_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_HER_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename XType, typename AType>
inline constexpr std::string_view her_name = "?HER  ";

template<> inline constexpr std::string_view her_name<c32, c32> = "CHER  ";
template<> inline constexpr std::string_view her_name<c64, c64> = "ZHER  ";

template<typename IntType, typename XType, typename AType>
PERFLIBS_LINALG_INLINE
bool her_param_check(
	const char *uplo, const IntType *n,
	const perflibs::remove_complex_t<promote_t<XType, AType>> *alpha,
	const XType *x, const IntType *incx,
	const AType *a, const IntType *lda,
	std::string_view name) {

	IntType nrowa = *n;
	IntType upper = option_matches(*uplo, 'U');
	IntType lower = option_matches(*uplo, 'L');

	IntType info = 0;
	if (!upper && !lower) {
		info = 1;
	}
	else if (*n < 0) {
		info = 2;
	}
	else if (*incx == 0) {
		info = 5;
	}
	else if (*lda < max(1, nrowa)) {
		info = 7;
	}
	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}
	return true;
}

template<bool ParamCheck, typename IntType, typename XType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void her(
	const char *uplo, const IntType *n,
	const perflibs::remove_complex_t<promote_t<XType, AType>> *alpha,
	const XType *x, const IntType *incx,
	      AType *a, const IntType *lda) {

	using promoted_type = promote_t<XType, AType>;
 	using scalar_type   = perflibs::remove_complex_t<promoted_type>;

	if constexpr(ParamCheck) {
		const bool res = perflibs::linalg::her_param_check(uplo, n, alpha, x, incx, a, lda,
			her_name<XType, AType>);

		if (!res) return;
	}

	if (*n == 0 || *alpha == zero<scalar_type>) {
		return;
	}

	if (*incx < 0) {
		x += (*incx * -1) * (*n - 1);
	}

	const     kernel_inttype a_strd          = *n;
	const     kernel_inttype b_strd          = *n;
	constexpr kernel_inttype cntg            = 1;

	const     auto           a_ptr          = x;
	constexpr kernel_inttype a_cntg_stride  = 0;
	const     kernel_inttype a_strd_stride  = *incx;

	const     auto           c_uplo        = c_to_uplo(*uplo);
	const     auto           c_ptr         = a;
	constexpr kernel_inttype c_cntg_stride = 1;
	const     kernel_inttype c_strd_stride = *lda;

	spec::problem_context pctx {
		matmul::matmul3 {
			general_matrix   {         matrix_base { a_ptr, cntg,   a_strd, a_cntg_stride, a_strd_stride }, false },
			general_matrix   {         matrix_base { a_ptr, cntg,   a_strd, a_cntg_stride, a_strd_stride }, true  },
			hermitian_matrix { c_uplo, matrix_base { c_ptr, a_strd, b_strd, c_cntg_stride, c_strd_stride }        },
			promoted_type{ *alpha },  one<promoted_type>
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx);
}
} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_HER_HPP

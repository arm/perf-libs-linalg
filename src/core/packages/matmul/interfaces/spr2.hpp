/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_SPR2_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_SPR2_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename XType, typename YType, typename AType>
inline constexpr std::string_view spr2_name = "?SPR2 ";

template<> inline constexpr std::string_view spr2_name<r32, r32, r32> = "SSPR2 ";
template<> inline constexpr std::string_view spr2_name<r64, r64, r64> = "DSPR2 ";

template<typename IntType, typename XType, typename YType, typename AType>
PERFLIBS_LINALG_INLINE
bool spr2_param_check(
	const char *uplo, const IntType *n,
	const promote_t<AType, XType, YType> *alpha,
	const XType *x, const IntType *incx,
	const YType *y, const IntType *incy,
	      AType *ap, std::string_view name) {

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
	else if (*incy == 0) {
		info = 7;
	}
	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}
	return true;
}

template<bool ParamCheck, typename IntType, typename XType, typename YType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void spr2(
	const char *uplo, const IntType *n,
	const promote_t<XType, YType, AType> *alpha,
	const XType *x, const IntType *incx,
	const YType *y, const IntType *incy,
	      AType *ap) {

	using scalar_type = promote_t<XType, YType, AType>;

	if constexpr(ParamCheck) {
		const auto res = perflibs::linalg::spr2_param_check(
			uplo, n, alpha, x, incx, y, incy, ap,
				spr2_name<XType, YType, AType>);

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
	const     kernel_inttype b_strd_stride = *incy;

	const     perflibs_uplo     c_uplo        = c_to_uplo(*uplo);

	if (*incx < 0) x += (*incx * -1) * (*n - 1);
	if (*incy < 0) y += (*incy * -1) * (*n - 1);

	spec::problem_context pctx {
		matmul::rank_update_2k {
			general_matrix   {         matrix_base        { x , cntg,   a_strd, a_cntg_stride, a_strd_stride } },
			general_matrix   {         matrix_base        { y , cntg,   b_strd, b_cntg_stride, b_strd_stride } },
			symmetric_matrix { c_uplo, packed_matrix_base { ap, a_strd, b_strd, c_uplo, false                } },
			*alpha, one<scalar_type>
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx);
}

} // namespace perflibs::linalg


#endif //PERFLIBS_LINALG_BLAS_INTERFACES_SPR2_HPP

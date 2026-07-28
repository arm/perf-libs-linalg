/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_GBMV_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_GBMV_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename AType, typename XType, typename YType>
inline constexpr std::string_view gbmv_name = "?GBMV ";

template<> inline constexpr std::string_view gbmv_name<r32, r32, r32> = "SGBMV ";
template<> inline constexpr std::string_view gbmv_name<r64, r64, r64> = "DGBMV ";
template<> inline constexpr std::string_view gbmv_name<c32, c32, c32> = "CGBMV ";
template<> inline constexpr std::string_view gbmv_name<c64, c64, c64> = "ZGBMV ";

template<typename IntType, typename AType, typename XType, typename YType>
PERFLIBS_LINALG_INLINE
bool gbmv_param_check(
	const char *trans,
	const IntType *m, const IntType *n,
	const IntType *kl, const IntType *ku,
    const promote_t<AType, XType, YType> *alpha,
	const AType *a, const IntType *lda,
    const XType *x, const IntType *incx,
	const promote_t<AType, XType, YType> *beta,
	      YType *y, const IntType *incy, std::string_view name) {

	const bool is_not_trans = option_matches(*trans, 'N');
	const bool is_trans = option_matches(*trans, 'T');
	const bool is_conj_trans = option_matches(*trans, 'C');

	pl_linalg_int_t info = 0;
	if (!is_not_trans && !is_trans && !is_conj_trans) {
		info = 1;
	}
	else if (*m < 0) {
		info = 2;
	}
	else if (*n < 0) {
		info = 3;
	}
	else if (*kl < 0) {
		info = 4;
	}
	else if (*ku < 0) {
		info = 5;
	}
	else if (*lda < *kl + *ku + 1) {
		info = 8;
	}
	else if (*incx == 0) {
		info = 10;
	}
	else if (*incy == 0) {
		info = 13;
	}
	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}
	return true;
}

template<bool ParamCheck, typename IntType, typename AType, typename XType, typename YType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void gbmv(
	const char *trans,
	const IntType *m, const IntType *n,
	const IntType *kl, const IntType *ku,
    const promote_t<AType, XType, YType> *alpha,
	const AType *a, const IntType *lda,
    const XType *x, const IntType *incx,
	const promote_t<AType, XType, YType> *beta,
	      YType *y, const IntType *incy) {

	if constexpr(ParamCheck) {
		auto res = perflibs::linalg::gbmv_param_check(
			trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
			gbmv_name<AType, XType, YType>);

		if(!res) return;
	}

	if (*m == 0 || *n == 0) {
		return;
	}

	const     auto          a_trans        = c_to_trans(*trans);
	const     bool          is_a_trans     = is_trans(a_trans);

	const     kernel_inttype a_strd        = is_a_trans ? *n : *m;
	constexpr kernel_inttype b_strd        = 1;
	const     kernel_inttype cntg          = is_a_trans ? *m : *n;

	const     kernel_inttype a_cntg_stride = is_a_trans ? *lda : 1;
	const     kernel_inttype a_strd_stride = is_a_trans ? 1 : *lda;

	const     kernel_inttype b_cntg_stride = *incx;
	constexpr kernel_inttype b_strd_stride = 1;

	const     kernel_inttype c_cntg_stride = *incy;
	constexpr kernel_inttype c_strd_stride = 1;

	const     kernel_inttype a_kl          = is_a_trans ? *ku : *kl;
	const     kernel_inttype a_ku          = is_a_trans ? *kl : *ku;

	if (*incx < 0) x += (*incx * -1) * (cntg   - 1);
	if (*incy < 0) y += (*incy * -1) * (a_strd - 1);

	spec::problem_context pctx {
		matmul::matmul3 {
			general_matrix { banded_matrix_base { a, cntg  , a_strd, a_cntg_stride, a_strd_stride, a_kl, a_ku }, is_conj(a_trans) },
			general_matrix { matrix_base        { x, cntg  , b_strd, b_cntg_stride, b_strd_stride             }                   },
			general_matrix { matrix_base        { y, a_strd, b_strd, c_cntg_stride, c_strd_stride             }                   },
			*alpha, *beta
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx);
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_GBMV_HPP

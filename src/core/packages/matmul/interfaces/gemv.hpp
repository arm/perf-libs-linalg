/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_GEMV_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_GEMV_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename AType, typename XType, typename YType>
inline constexpr std::string_view gemv_name = "?GEMV ";

template<> inline constexpr std::string_view gemv_name<bf16, bf16,  r32> = "SBGEMV";
template<> inline constexpr std::string_view gemv_name<bf16, bf16, bf16> = "BGEMV ";
template<> inline constexpr std::string_view gemv_name< r32,  r32,  r32> = "SGEMV ";
template<> inline constexpr std::string_view gemv_name< r64,  r64,  r64> = "DGEMV ";
template<> inline constexpr std::string_view gemv_name< c32,  c32,  c32> = "CGEMV ";
template<> inline constexpr std::string_view gemv_name< c64,  c64,  c64> = "ZGEMV ";

template<typename IntType, typename AType, typename XT, typename YT, typename ScalarType>
PERFLIBS_LINALG_INLINE
bool gemv_param_check(const char *const trans, const IntType *const m, const IntType *const n,
                       const ScalarType *const alpha, const AType *const a, const IntType *const lda,
                       const XT *const x, const IntType *const incx, const ScalarType*const beta,
                       YT *const y, const IntType *const incy, std::string_view name) {
	IntType info = 0;
	if (! option_matches(*trans, 'N', 'T', 'C')) {
		info = 1;
	}
	else if (*m < 0) {
		info = 2;
	}
	else if (*n < 0) {
		info = 3;
	}
	else if (*lda < max(1,*m)) {
		info = 6;
	}
	else if (*incx == 0) {
		info = 8;
	}
	else if (*incy == 0) {
		info = 11;
	}
	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}
	return true;
}

template<bool ParamCheck, typename IntType, typename AType, typename XType, typename YType, typename ArchitectureSpec>
void gemv(
	const char *trans,
	const IntType *m,
	const IntType *n,
	const promote_t<AType, XType, YType> *alpha,
	const AType *a, const IntType *lda,
	const XType *x, const IntType *incx,
	const promote_t<AType, XType, YType> *beta,
	      YType *y, const IntType *incy) {

	if constexpr(ParamCheck) {
		const bool succ = perflibs::linalg::gemv_param_check(trans, m, n, alpha, a, lda, x, incx, beta, y, incy,
			gemv_name<AType, XType, YType>);

		if(! succ) return;
	}

	// Note: this behavior differs from GEMM.
	if(*m == 0 || *n == 0)
		return;

	const     auto           a_trans      = c_to_trans(*trans);
	const     bool           is_a_trans   = is_trans(a_trans);

	const     kernel_inttype a_strd        = is_a_trans ? *n : *m;
	constexpr kernel_inttype b_strd        = 1;
	const     kernel_inttype cntg          = is_a_trans ? *m : *n;

	const     kernel_inttype a_cntg_stride = is_a_trans ? 1 : *lda;
	const     kernel_inttype a_strd_stride = is_a_trans ? *lda : 1;

	const     kernel_inttype b_cntg_stride = *incx;
	constexpr kernel_inttype b_strd_stride = 1;

	const     kernel_inttype c_cntg_stride = *incy;
	constexpr kernel_inttype c_strd_stride = 1;

	if(*incx < 0) x += (*incx * -1) * (cntg   - 1);
	if(*incy < 0) y += (*incy * -1) * (a_strd - 1);

	spec::problem_context pctx {
		matmul::matmul3 {
			general_matrix { matrix_base { a, cntg,   a_strd, a_cntg_stride, a_strd_stride }, is_conj(a_trans) },
			general_matrix { matrix_base { x, cntg,   b_strd, b_cntg_stride, b_strd_stride } },
			general_matrix { matrix_base { y, a_strd, b_strd, c_cntg_stride, c_strd_stride } },
			*alpha, *beta
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx);
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_GEMV_HPP

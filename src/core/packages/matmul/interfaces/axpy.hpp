/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_AXPY_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_AXPY_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

template<bool ParamCheck, typename IntType, typename XType, typename YType, typename ArchitectureSpec>
void axpy(const IntType *n,
	const promote_t<XType, YType> *alpha,
	const XType *x, const IntType *incx,
	      YType *y, const IntType *incy) {

	if(*n <= 0)
		return;

	using scalar_type = promote_t<XType, YType>;

	if (*incx < 0) x += (*incx * -1) * (*n - 1);
	if (*incy < 0) y += (*incy * -1) * (*n - 1);

	const     kernel_inttype a_strd        = *n;
	constexpr kernel_inttype b_strd        = 1;
	constexpr kernel_inttype cntg          = 1;

	constexpr kernel_inttype a_cntg_stride = 0;
	const     kernel_inttype a_strd_stride = *incx;

	const     auto           b             = &one<const XType>;
	constexpr kernel_inttype b_cntg_stride = 0;
	constexpr kernel_inttype b_strd_stride = 0;

	const     kernel_inttype c_cntg_stride = *incy;
	constexpr kernel_inttype c_strd_stride = 1;

	constexpr auto           beta          = one<scalar_type>;

	spec::problem_context pctx {
		matmul::matmul3 {
			general_matrix { matrix_base { x, cntg,   a_strd, a_cntg_stride, a_strd_stride } },
			general_matrix { matrix_base { b, cntg,   b_strd, b_cntg_stride, b_strd_stride } },
			general_matrix { matrix_base { y, a_strd, b_strd, c_cntg_stride, c_strd_stride } },
			*alpha, beta
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::matmul3_vector_scalar{}(pctx);
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_AXPY_HPP

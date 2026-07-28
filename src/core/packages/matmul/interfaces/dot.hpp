/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_DOT_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

template<bool ParamCheck, typename IntType, typename T1,  typename T2, typename T3, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
T3 dot(const IntType *n, const T1 *x, const IntType *incx, const T2 *y, const IntType *incy) {
	//because dot returns by value we create our own 1x1 C matrix here
	// i.e. a value, use pctx.c as its address, and return its value on completion
	auto c = zero<T3>;

	if(*n <= 0)
		return c;

	using scalar_type = promote_t<T2, T2, T3>;

	if (*incx < 0) x += (*incx * -1) * (*n - 1);
	if (*incy < 0) y += (*incy * -1) * (*n - 1);


	constexpr kernel_inttype a_strd        = 1;
	constexpr kernel_inttype b_strd        = 1;
	const     kernel_inttype cntg          = *n;

	const     kernel_inttype a_cntg_stride = *incx;
	constexpr kernel_inttype a_strd_stride = 0;

	const     kernel_inttype b_cntg_stride = *incy;
	constexpr kernel_inttype b_strd_stride = 0;

	constexpr kernel_inttype c_cntg_stride = 0;
	constexpr kernel_inttype c_strd_stride = 0;

	constexpr auto           alpha         = one<scalar_type>;
	constexpr auto           beta          = zero<scalar_type>;

	spec::problem_context pctx {
		matmul::matmul3 {
			general_matrix { matrix_base { x,  cntg,   a_strd, a_cntg_stride, a_strd_stride } },
			general_matrix { matrix_base { y,  cntg,   b_strd, b_cntg_stride, b_strd_stride } },
			general_matrix { matrix_base { &c, a_strd, b_strd, c_cntg_stride, c_strd_stride } },
			alpha, beta
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};


	matmul::matmul3_inner_product{}(pctx);

	return c;
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_DOT_HPP

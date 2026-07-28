/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_SCAL_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_SCAL_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

template<bool ParamCheck, typename IntType, typename AlphaType, typename XType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void scal(
	const IntType *n,
	const AlphaType *alpha,
	          XType *x, const IntType *incx) {

	if(*n <= 0 || *incx <= 0 || *alpha == one<AlphaType>)
		return;

	//if alpha is complex but X isn't, we want to "extend" alpha to complex
	const XType alpha_adj { *alpha };

	if (*incx < 0) x += (*incx * -1) * (*n - 1);

	const     kernel_inttype a_strd        = *n;
	constexpr kernel_inttype b_strd        = 1;
	constexpr kernel_inttype cntg          = 1;

	const     auto           a             = &one<XType>;
	constexpr kernel_inttype a_cntg_stride = 0;
	constexpr kernel_inttype a_strd_stride = one<IntType>;

	const     auto           b             = &one<XType>;
	constexpr kernel_inttype b_cntg_stride = 0;
	constexpr kernel_inttype b_strd_stride = 0;

	const     kernel_inttype c_cntg_stride = *incx;
	constexpr kernel_inttype c_strd_stride = 1;

	spec::problem_context pctx {
		matmul::matmul3 {
			general_matrix { matrix_base { a, cntg,   a_strd, a_cntg_stride, a_strd_stride } },
			general_matrix { matrix_base { b, cntg,   b_strd, b_cntg_stride, b_strd_stride } },
			general_matrix { matrix_base { x, a_strd, b_strd, c_cntg_stride, c_strd_stride } },
			zero<XType>, alpha_adj,
			zero_mode::scale
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::matmul3_vector_scalar{}(pctx);
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_SCAL_HPP

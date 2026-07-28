/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_GERBC_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_GERBC_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename XType, typename YType, typename AType>
inline constexpr std::string_view gerbc_name = "?GERBC ";

template<> inline constexpr std::string_view gerbc_name<c32, c32, c32> = "CGERBC ";
template<> inline constexpr std::string_view gerbc_name<c64, c64, c64> = "ZGERBC ";

template<typename IntType, typename T>
PERFLIBS_LINALG_INLINE
bool gerbc_param_check(const IntType *m, const IntType *n, const T *alpha, const T *x,
                            const IntType *incx, const T *y, const IntType *incy, const T *beta,
                            const T *a, const IntType *lda, std::string_view name) {
	IntType info, nrowa;

	nrowa = *m;

	info = 0;
	if (*m < 0)
		info = 1;
	else if (*n < 0)
		info = 2;
	else if (*incx == 0)
		info = 5;
	else if (*incy == 0)
		info = 7;
	else if (*lda < max(1, nrowa))
		info = 10;

	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}
	return true;
}

template <bool ParamCheck, typename IntType, typename XType, typename YType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void gerbc(const IntType *m, const IntType *n,
         const promote_t<XType, YType, AType> *alpha,
         const XType *x, const IntType *incx,
         const YType *y, const IntType *incy,
         const promote_t<XType, YType, AType> *beta,
               AType *a, const IntType *lda) {

	using scalar_type = promote_t<XType, YType, AType>;

	if constexpr(ParamCheck) {
		const bool res = perflibs::linalg::gerbc_param_check(m, n, alpha, x, incx, y, incy, beta, a, lda,
			gerbc_name<XType, YType, AType>);

		if(! res) return;
	}

	if(*m == 0 || *n == 0 || (*alpha == zero<scalar_type> && *beta == one<scalar_type>)) {
		return;
	}

	if (*incx < 0) x += (*incx * -1) * (*m - 1);
	if (*incy < 0) y += (*incy * -1) * (*n - 1);

	const     kernel_inttype a_strd        = *m;
	const     kernel_inttype b_strd        = *n;
	constexpr kernel_inttype cntg          = 1;

	constexpr kernel_inttype a_cntg_stride = 0;
	const     kernel_inttype a_strd_stride = *incx;

	constexpr kernel_inttype b_cntg_stride = 0;
	const     kernel_inttype b_strd_stride = *incy;

	constexpr kernel_inttype c_cntg_stride = 1;
	const     kernel_inttype c_strd_stride = *lda;

	spec::problem_context pctx {
		matmul::matmul3 {
			general_matrix { matrix_base { x, cntg,   a_strd, a_cntg_stride, a_strd_stride }, false },
			general_matrix { matrix_base { y, cntg,   b_strd, b_cntg_stride, b_strd_stride }, true  },
			general_matrix { matrix_base { a, a_strd, b_strd, c_cntg_stride, c_strd_stride }, false },
			*alpha, *beta
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx);
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_GERBC_HPP

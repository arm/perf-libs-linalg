/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_WAXPBY_KERNELS_HPP
#define PERFLIBS_LINALG_WAXPBY_KERNELS_HPP

#include "perflibs_complex.hpp"
#include "perflibs_util.hpp"
#include "framework/linalg_util.hpp"
#include "spec/problem_context.hpp"

namespace perflibs::linalg {

template<typename AType, typename BType=AType, typename CType=AType, typename DType = CType,
         typename ScalarType = promote_t<AType, BType, CType>>
using waxpby_kernel_t = void(kernel_inttype n, ScalarType alpha, const AType *x, ScalarType beta,
                             const CType *y, DType *w, const kernel_inttype incx, const kernel_inttype incy,
                             const kernel_inttype incw);

} //namespace perflibs::linalg

extern "C" {
perflibs::linalg::waxpby_kernel_t<float>          swaxpby_kernel;
perflibs::linalg::waxpby_kernel_t<double>         dwaxpby_kernel;
perflibs::linalg::waxpby_kernel_t<complex_float>  cwaxpby_kernel;
perflibs::linalg::waxpby_kernel_t<complex_double> zwaxpby_kernel;
perflibs::linalg::waxpby_kernel_t<float>          swaxpby_sve_kernel;
perflibs::linalg::waxpby_kernel_t<double>         dwaxpby_sve_kernel;
perflibs::linalg::waxpby_kernel_t<complex_float>  cwaxpby_sve_kernel;
perflibs::linalg::waxpby_kernel_t<complex_double> zwaxpby_sve_kernel;
perflibs::linalg::waxpby_kernel_t<complex_float>  cwaxpby_sve_kernel_fcmla;
perflibs::linalg::waxpby_kernel_t<complex_double> zwaxpby_sve_kernel_fcmla;
} //extern "C"

namespace perflibs::linalg {

template<typename T>
inline waxpby_kernel_t<T> *waxpby_neon_kernel;

template<> inline auto waxpby_neon_kernel<float>          = swaxpby_kernel;
template<> inline auto waxpby_neon_kernel<double>         = dwaxpby_kernel;
template<> inline auto waxpby_neon_kernel<complex_float>  = cwaxpby_kernel;
template<> inline auto waxpby_neon_kernel<complex_double> = zwaxpby_kernel;

template<typename T>
inline waxpby_kernel_t<T> *waxpby_sve_kernel;

template<> inline auto waxpby_sve_kernel<float>           = swaxpby_sve_kernel;
template<> inline auto waxpby_sve_kernel<double>          = dwaxpby_sve_kernel;
template<> inline auto waxpby_sve_kernel<complex_float>   = cwaxpby_sve_kernel;
template<> inline auto waxpby_sve_kernel<complex_double>  = zwaxpby_sve_kernel;

template<typename T>
inline waxpby_kernel_t<T> *waxpby_sve_kernel_fcmla = waxpby_sve_kernel<T>;

template<> inline auto waxpby_sve_kernel_fcmla<complex_float>   = cwaxpby_sve_kernel_fcmla;
template<> inline auto waxpby_sve_kernel_fcmla<complex_double>  = zwaxpby_sve_kernel_fcmla;


template<typename ProblemContext>
PERFLIBS_LINALG_INLINE
auto get_neon_waxpby_kernel(const ProblemContext& pctx) {
	return waxpby_neon_kernel<typename ProblemContext::d_matrix_type::value_type >;
}

template<typename ProblemContext>
PERFLIBS_LINALG_INLINE
auto get_sve_waxpby_kernel(const ProblemContext& pctx) {
	const bool is_contig = pctx.a.strd_step() == 1
	                    && pctx.c.cntg_step() == 1
	                    && pctx.d.cntg_step() == 1;

	using data_type = typename ProblemContext::a_matrix_type::value_type;

	if (is_contig) {
		return waxpby_sve_kernel_fcmla<data_type>;
	}

	return get_neon_waxpby_kernel(pctx);
}

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_WAXPBY_KERNELS_HPP

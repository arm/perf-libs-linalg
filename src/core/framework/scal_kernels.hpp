/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SCAL_KERNELS_HPP
#define PERFLIBS_LINALG_SCAL_KERNELS_HPP

#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

template<typename AType, typename BType, typename CType,
         typename ScalarType = perflibs::linalg::promote_t<AType, BType, CType>>
using scal_kernel_t = void (kernel_inttype n, ScalarType alpha, CType *x, kernel_inttype incx);

template<typename T1, typename T2>
using scal_kernel_2t = scal_kernel_t<T1, T1, T1, T2>;

} // namespace perflibs::linalg

extern "C" {

perflibs::linalg::scal_kernel_2t<float, float> sscal_sve_kernel;
perflibs::linalg::scal_kernel_2t<double, double> dscal_sve_kernel;
perflibs::linalg::scal_kernel_2t<std::complex<float>, std::complex<float>> cscal_sve_kernel_fcmla;
perflibs::linalg::scal_kernel_2t<std::complex<float>, std::complex<float>> cscal_sve_kernel;
perflibs::linalg::scal_kernel_2t<std::complex<float>, float> sscal_real_cplx_sve_kernel;
perflibs::linalg::scal_kernel_2t<std::complex<double>, std::complex<double>> zscal_sve_kernel_fcmla;
perflibs::linalg::scal_kernel_2t<std::complex<double>, std::complex<double>> zscal_sve_kernel;
perflibs::linalg::scal_kernel_2t<std::complex<double>, double> dscal_real_cplx_sve_kernel;
perflibs::linalg::scal_kernel_2t<float, float> sscal_kernel;
perflibs::linalg::scal_kernel_2t<double, double> dscal_kernel;
perflibs::linalg::scal_kernel_2t<std::complex<float>, std::complex<float>> cscal_kernel;
perflibs::linalg::scal_kernel_2t<std::complex<float>, float> sscal_real_cplx_kernel;
perflibs::linalg::scal_kernel_2t<std::complex<double>, std::complex<double>> zscal_kernel;
perflibs::linalg::scal_kernel_2t<std::complex<double>, double> dscal_real_cplx_kernel;

} //extern "C"


namespace perflibs::linalg {

template<typename T1, typename T2>
inline scal_kernel_2t<T1, T2> *scal_kernel;
template<>
inline auto scal_kernel<float, float> = sscal_kernel;
template<>
inline auto scal_kernel<double, double> = dscal_kernel;
template<>
inline auto scal_kernel<complex_float, complex_float> = cscal_kernel;
template<>
inline auto scal_kernel<complex_double, complex_double> = zscal_kernel;
template<>
inline auto scal_kernel<complex_float, float> = sscal_real_cplx_kernel;
template<>
inline auto scal_kernel<complex_double, double> = dscal_real_cplx_kernel;

template<typename T1, typename T2>
inline scal_kernel_2t<T1, T2> *scal_sve_kernel;
template<>
inline auto scal_sve_kernel<float, float> = sscal_sve_kernel;
template<>
inline auto scal_sve_kernel<double, double> = dscal_sve_kernel;
template<>
inline auto scal_sve_kernel<complex_float, complex_float> = cscal_sve_kernel;
template<>
inline auto scal_sve_kernel<complex_double, complex_double> = zscal_sve_kernel;
template<>
inline auto scal_sve_kernel<complex_float, float> = sscal_real_cplx_sve_kernel;
template<>
inline auto scal_sve_kernel<complex_double, double> = dscal_real_cplx_sve_kernel;

template<typename T1, typename T2>
inline scal_kernel_2t<T1, T2> *scal_sve_kernel_fcmla = scal_sve_kernel<T1, T2>;
template<>
inline auto scal_sve_kernel_fcmla<complex_float, complex_float> = cscal_sve_kernel_fcmla;
template<>
inline auto scal_sve_kernel_fcmla<complex_double, complex_double> = zscal_sve_kernel_fcmla;

template <typename T1, typename T2>
inline static
void scal_impl_inc0(kernel_inttype n, T2 alpha, T1 *x, kernel_inttype incx) {
	PERFLIBS_ASSERT(incx == 0, "INCX must be 0 to invoke this kernel");
	for (kernel_inttype i = 0; i < n; ++i) {
		x[0] *= alpha;
	}
}

template <typename T1, typename T2>
inline static
void scal_impl_fallback(kernel_inttype n, T2 alpha, T1 *x, kernel_inttype incx) {
	// x and y are assumed to have been pre-adjusted for negative increments.
	kernel_inttype ix = 0;
	for (kernel_inttype i = 0; i < n; ++i, ix += incx) {
		x[ix] = alpha * x[ix];
	}
}

template<typename ProblemContext>
PERFLIBS_LINALG_INLINE
auto get_neon_scal_kernel(const ProblemContext& pctx) {
	using T1 = typename ProblemContext::value_type_1;
	using T2 = typename ProblemContext::value_type_2;

	if (pctx.incx == 0) {
		return &scal_impl_inc0<T1, T2>;
	}
	return scal_kernel<T1, T2>;
}

template<typename ProblemContext>
PERFLIBS_LINALG_INLINE
auto get_sve_scal_kernel(const ProblemContext& pctx) {
	using T1 = typename ProblemContext::value_type_1;
	using T2 = typename ProblemContext::value_type_2;

	if (pctx.incx == 0) {
		return get_neon_scal_kernel(pctx);
	}

	if (pctx.incx == 1) {
		return scal_sve_kernel_fcmla<T1, T2>;
	}

	return get_neon_scal_kernel(pctx);
}

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_SCAL_KERNELS_HPP

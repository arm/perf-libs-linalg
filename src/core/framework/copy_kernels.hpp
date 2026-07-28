/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_COPY_KERNELS_HPP
#define PERFLIBS_LINALG_COPY_KERNELS_HPP

#include "perflibs_type_traits.hpp"
#include "framework/linalg_util.hpp"
#include "perflibs_assert.hpp"

namespace perflibs::linalg {

template<typename AType, typename BType=AType, typename CType=AType>
using copy_kernel_t = void(kernel_inttype n, const AType *x, kernel_inttype incx, CType *y,
                           kernel_inttype incy);

} //namespace perflibs::linalg

extern "C" {
//neon
perflibs::linalg::copy_kernel_t<float> scopy_kernel;
perflibs::linalg::copy_kernel_t<double> dcopy_kernel;
perflibs::linalg::copy_kernel_t<std::complex<double>> zcopy_kernel_with_inc;
//sve
perflibs::linalg::copy_kernel_t<float> scopy_sve_kernel;
perflibs::linalg::copy_kernel_t<double> dcopy_sve_kernel;
perflibs::linalg::copy_kernel_t<std::complex<double>> zcopy_sve_kernel_with_inc;
} //extern "C"

namespace perflibs::linalg {

inline static
void ccopy_kernel_generic(kernel_inttype n, const std::complex<float> *x, kernel_inttype incx,
                          std::complex<float> *y, kernel_inttype incy) {
	dcopy_kernel(n, reinterpret_cast<const double *>(x), incx, reinterpret_cast<double *>(y), incy);
}

inline static
void zcopy_kernel_generic(kernel_inttype n, const std::complex<double> *x, kernel_inttype incx,
                         std::complex<double> *y, kernel_inttype incy) {
	PERFLIBS_ASSERT(incx == 1 && incy == 1);
	dcopy_kernel(n * 2, reinterpret_cast<const double *>(x), incx, reinterpret_cast<double *>(y), incy);
}

inline static
void ccopy_sve_kernel_generic(kernel_inttype n, const std::complex<float> *x, kernel_inttype incx,
                              std::complex<float> *y, kernel_inttype incy) {
	dcopy_sve_kernel(n, reinterpret_cast<const double *>(x), incx, reinterpret_cast<double *>(y), incy);
}

inline static
void zcopy_sve_kernel_generic(kernel_inttype n, const std::complex<double> *x, kernel_inttype incx,
                              std::complex<double> *y, kernel_inttype incy) {
	PERFLIBS_ASSERT(incx == 1 && incy == 1);
	dcopy_sve_kernel(n * 2, reinterpret_cast<const double *>(x), incx, reinterpret_cast<double *>(y), incy);
}

template <typename T>
inline static
void copy_impl_incy0(kernel_inttype n, const T *x, kernel_inttype incx,
                     T *y, kernel_inttype incy) {
	PERFLIBS_ASSERT(incy == 0, "INCY must be 0 to invoke this kernel");
	auto ix = 0;
	y[0] = x[ix + incx * (n - 1)];
}

template <typename T>
inline static
void copy_impl_incx0(kernel_inttype n, const T *x, kernel_inttype incx,
                     T *y, kernel_inttype incy) {
	PERFLIBS_ASSERT(incx == 0, "INCX must be 0 to invoke this kernel");
	auto ix = 0;
	auto value = x[ix];
	kernel_inttype iy = 0;
	for (kernel_inttype i = 0; i < n; ++i, iy += incy) {
		y[iy] = value;
	}
}

template <typename T>
inline static
void copy_impl_fallback(kernel_inttype n, const T *x, kernel_inttype incx,
                        T *y, kernel_inttype incy) {
		// x and y are assumed to have been pre-adjusted for negative increments.
		kernel_inttype ix = 0;
		kernel_inttype iy = 0;
		for (kernel_inttype i = 0; i < n; ++i, ix += incx, iy += incy) {
			y[iy] = x[ix];
		}
}

template<typename T>
inline copy_kernel_t<T> *copy_kernel;
template<>
inline auto copy_kernel<float> = scopy_kernel;
template<>
inline auto copy_kernel<double> = dcopy_kernel;
template<>
inline auto copy_kernel<complex_float> = ccopy_kernel_generic;
template<>
inline auto copy_kernel<complex_double> = zcopy_kernel_generic;

template<typename T>
inline copy_kernel_t<T> *copy_sve_kernel;
template<>
inline auto copy_sve_kernel<float> = scopy_sve_kernel;
template<>
inline auto copy_sve_kernel<double> = dcopy_sve_kernel;
template<>
inline auto copy_sve_kernel<complex_float> = ccopy_sve_kernel_generic;
template<>
inline auto copy_sve_kernel<complex_double> = zcopy_sve_kernel_generic;

template<typename T>
PERFLIBS_LINALG_INLINE
copy_kernel_t<T> *get_neon_copy_kernel(kernel_inttype incx, kernel_inttype incy) {
	if (incy == 0) {
		return &copy_impl_incy0<T>;
	}
	else if (incx == 0) {
		return &copy_impl_incx0<T>;
	}
	else if constexpr (is_same_remove_cv_v<T, complex_double>) {
		if (!(incx == 1 && incy == 1)) {
			return &zcopy_kernel_with_inc; // a wrapper to dcopy_kernel
		}
	}
	return copy_kernel<T>;
}

template<typename T>
PERFLIBS_LINALG_INLINE
copy_kernel_t<T> *get_sve_copy_kernel(kernel_inttype incx, kernel_inttype incy) {
	if (incx == 0 || incy == 0) {
		return get_neon_copy_kernel<T>(incx, incy);
	}

	if (incx == 1 && incy == 1) {
		return copy_sve_kernel<T>;
	}

	return get_neon_copy_kernel<T>(incx, incy);
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_COPY_KERNELS_HPP

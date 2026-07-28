/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SWAP_KERNELS_HPP
#define PERFLIBS_LINALG_SWAP_KERNELS_HPP

#include "framework/linalg_util.hpp"

#include "perflibs_complex.hpp"

#include <complex>

namespace perflibs::linalg {

template<typename T>
using swap_kernel_t = void (kernel_inttype n, T *x, T *y, const kernel_inttype incx, const kernel_inttype incy);

namespace {

template <typename T>
PERFLIBS_LINALG_INLINE
void swap_fallback(kernel_inttype n, T *x, T *y, const kernel_inttype incx,
                  const kernel_inttype incy) {
	kernel_inttype ix = 0, iy = 0;
	for (kernel_inttype i = 0; i < n; i++) {
		T tmp = y[iy];
		y[iy] = x[ix];
		x[ix] = tmp;
		ix   += incx;
		iy   += incy;
	}
}

} // namespace
}

// Assembly kernels (only used for unit increments, otherwise fallback is used)
extern "C" {
perflibs::linalg::swap_kernel_t<float>                  sswap_kernel;
perflibs::linalg::swap_kernel_t<double>                 dswap_kernel;
perflibs::linalg::swap_kernel_t<std::complex<float>>    cswap_kernel;
perflibs::linalg::swap_kernel_t<std::complex<double>>   zswap_kernel;
} // extern "C"

#endif // PERFLIBS_LINALG_SWAP_KERNELS_HPP

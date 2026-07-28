/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_ROT_FALLBACK_KERNEL_HPP
#define PERFLIBS_LINALG_ROT_FALLBACK_KERNEL_HPP

#include "perflibs_complex.hpp"
#include "perflibs_util.hpp"
#include "linalg_util.hpp"

namespace perflibs::linalg {
namespace {

template<typename T1, typename T2>
inline void rot_kernel_fallback(kernel_inttype n, T1 *x, T1 *y, const kernel_inttype incx,
                                const kernel_inttype incy, const remove_complex_t<T1> c, const T2 s) {
	T2 sj = perflibs::conj(s);
	if (incx == 1 && incy == 1) {
		for (kernel_inttype i = 0; i < n; i++) {
			T1 tmp = c * x[i] + s * y[i];
			y[i] = c * y[i] - sj * x[i];
			x[i] = tmp;
		}
	}
	else {
		kernel_inttype ix=0, iy=0;
		for (kernel_inttype i=0 ; i<n ; i++) {
			T1 tmp = c*x[ix] + s*y[iy];
			y[iy] = c*y[iy] - sj*x[ix];
			x[ix] = tmp;
			ix+=incx;
			iy+=incy;
		}
	}
}

} // namespace
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_ROT_FALLBACK_KERNEL_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FRAMEWORK_L1_NORM_KERNELS_HPP
#define PERFLIBS_LINALG_FRAMEWORK_L1_NORM_KERNELS_HPP


#include "framework/linalg_util.hpp"
#include "perflibs_complex.hpp"
#include <complex>

namespace perflibs::linalg {

template<typename T>
using l1_norm_kernel_t = remove_complex_t<T> (kernel_inttype n, const T *x, const kernel_inttype incx);

namespace {

template<typename T>
PERFLIBS_LINALG_INLINE
remove_complex_t<T> l1_norm_fallback(kernel_inttype n, const T *x, const kernel_inttype incx) {

	constexpr kernel_inttype n_unroll = 6;
	remove_complex_t<T> x1 { 0 }, x2 { 0 }, x3 { 0 };
	remove_complex_t<T> x4 { 0 }, x5 { 0 }, x6 { 0 };

	kernel_inttype i = 0;
	const kernel_inttype remain_n = n % n_unroll;

	for (; i < remain_n; ++i) {
		x1 += perflibs::sum_abs(x[i * incx]);
	}
	if (n < n_unroll) {
		return x1;
	}

	for (; i < n; i += n_unroll) {
		x1 += perflibs::sum_abs(x[i * incx]);
		x2 += perflibs::sum_abs(x[(i + 1) * incx]);
		x3 += perflibs::sum_abs(x[(i + 2) * incx]);
		x4 += perflibs::sum_abs(x[(i + 3) * incx]);
		x5 += perflibs::sum_abs(x[(i + 4) * incx]);
		x6 += perflibs::sum_abs(x[(i + 5) * incx]);
	}

	return x1 + x2 + x3 + x4 + x5 + x6;
}
} // namespace
} // namespace perflibs::linalg

extern "C" {
perflibs::linalg::l1_norm_kernel_t<float>                sasum_kernel;
perflibs::linalg::l1_norm_kernel_t<double>               dasum_kernel;
perflibs::linalg::l1_norm_kernel_t<std::complex<float>>  scasum_kernel;
perflibs::linalg::l1_norm_kernel_t<std::complex<double>> dzasum_kernel;
} // extern "C"

#endif // PERFLIBS_LINALG_FRAMEWORK_L1_NORM_KERNELS_HPP

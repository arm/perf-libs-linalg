/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FRAMEWORK_L2_NORM_KERNELS_HPP
#define PERFLIBS_LINALG_FRAMEWORK_L2_NORM_KERNELS_HPP

#include "framework/linalg_util.hpp"

#include "perflibs_complex.hpp"

#include <algorithm>
#include <complex>
#include <cmath>

namespace perflibs::linalg {

template<typename T>
using l2_norm_kernel_t = remove_complex_t<T> (kernel_inttype n, const T *x, const kernel_inttype incx);

namespace {

template<typename T>
using real_type = remove_complex_t<T>;

template<typename T>
struct l2_norm_scale_constants;

template<>
struct l2_norm_scale_constants<float> {
	constexpr static float TR_SMALL  = 0x1p-63;
	constexpr static float TR_BIG    = 0x1p+52;
	constexpr static float SCALE_SMALL = 0x1p+75;
	constexpr static float SCALE_BIG   = 0x1p-76;
};

template<>
struct l2_norm_scale_constants<double> {
	constexpr static double TR_SMALL  = 0x1p-511;
	constexpr static double TR_BIG    = 0x1p+486;
	constexpr static double SCALE_SMALL = 0x1p+537;
	constexpr static double SCALE_BIG   = 0x1p-538;
};

template<typename T>
PERFLIBS_LINALG_INLINE
real_type<T> l2_norm_combine(real_type<T> ssq_sml, real_type<T> ssq_med, real_type<T> ssq_big) {
	using real_type_t      = real_type<T>;
	using scale_constants  = l2_norm_scale_constants<real_type_t>;
	constexpr real_type_t one  = 1;
	constexpr real_type_t zero = 0;

	real_type_t scale = one;
	real_type_t ssq;
	if (ssq_big > zero) {
		if (ssq_med > zero || !std::isfinite(ssq_med)) {
			ssq_big += (ssq_med * scale_constants::SCALE_BIG) * scale_constants::SCALE_BIG;
		}
		scale = one / scale_constants::SCALE_BIG;
		ssq   = ssq_big;
	}
	else if (ssq_sml > zero) {
		if (ssq_med > zero || !std::isfinite(ssq_med)) {
			ssq_med = std::sqrt(ssq_med);
			ssq_sml = std::sqrt(ssq_sml) / scale_constants::SCALE_SMALL;
			auto [y_min, y_max] = std::minmax(ssq_med, ssq_sml);
			ssq = (y_max * y_max) * (one + (y_min / y_max) * (y_min / y_max));
		}
		else {
			scale = one / scale_constants::SCALE_SMALL;
			ssq   = ssq_sml;
		}
	}
	else {
		ssq = ssq_med;
	}
	return scale * std::sqrt(ssq);
}

template<typename T>
struct l2_norm_accumulator {
	using real_type_t = real_type<T>;
	real_type_t ssq_sml {};
	real_type_t ssq_med {};
	real_type_t ssq_big {};
	PERFLIBS_LINALG_INLINE
  	void add(real_type_t x) { // only applicable for real positive values (norms)
		// Combine per-thread partial norms using safe scaling
		using scale_constants = l2_norm_scale_constants<real_type_t>;
		constexpr real_type_t zero = 0;
		if (x > scale_constants::TR_BIG) {
			auto temp = x * scale_constants::SCALE_BIG;
			this->ssq_big += temp * temp;
		}
		else if (x < scale_constants::TR_SMALL) {
			if (this->ssq_big == zero) {
				auto temp = x * scale_constants::SCALE_SMALL;
				this->ssq_sml += temp * temp;
			}
		}
		else {
			this->ssq_med += x * x;
		}

	}

	PERFLIBS_LINALG_INLINE
	void complex_add(T x) {
		add(std::abs(std::real(x)));
		if constexpr (is_complex_v<T>) {
			add(std::abs(std::imag(x)));
		}
	}

	PERFLIBS_LINALG_INLINE
	void merge(const l2_norm_accumulator<T>& acc) {
		this->ssq_sml += acc.ssq_sml;
		this->ssq_med += acc.ssq_med;
		this->ssq_big += acc.ssq_big;
	}

	PERFLIBS_LINALG_INLINE
	real_type_t finish() const {
		return l2_norm_combine<real_type_t>(this->ssq_sml, this->ssq_med, this->ssq_big);
	}
};

template <typename T>
PERFLIBS_LINALG_INLINE
real_type<T> l2_norm_fallback(kernel_inttype n, const T *x, const kernel_inttype incx) {
	constexpr kernel_inttype N_UNROLL = 8;
	l2_norm_accumulator<T> acc;
	kernel_inttype i = 0;
	const kernel_inttype remain_n = n % N_UNROLL;
	if (remain_n != 0) {
		for (; i < remain_n; ++i) {
			acc.complex_add(x[i * incx]);
		}
		if (n < N_UNROLL) {
			return acc.finish();
		}
	}
	for (; i < n; i+=N_UNROLL) {
		acc.complex_add(x[i * incx]);
		acc.complex_add(x[(i + 1) * incx]);
		acc.complex_add(x[(i + 2) * incx]);
		acc.complex_add(x[(i + 3) * incx]);
		acc.complex_add(x[(i + 4) * incx]);
		acc.complex_add(x[(i + 5) * incx]);
		acc.complex_add(x[(i + 6) * incx]);
		acc.complex_add(x[(i + 7) * incx]);
	}
	return acc.finish();
}
} // namespace
} // namespace perflibs::linalg


namespace perflibs {

template<typename T>
linalg::l2_norm_kernel_t<T> l2_norm_neon_kernel;

template<typename T>
linalg::l2_norm_kernel_t<T> l2_norm_sve_kernel;

} // end namespace perflibs

#endif // PERFLIBS_LINALG_FRAMEWORK_L2_NORM_KERNELS_HPP

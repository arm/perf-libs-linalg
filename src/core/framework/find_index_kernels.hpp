/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_FIND_INDEX_SPEC_HPP
#define PERFLIBS_FIND_INDEX_SPEC_HPP

#include "perflibs_type_traits.hpp"
#include "framework/linalg_util.hpp"

#include "perflibs_complex.hpp"

#include <complex>
#include <utility>
#include <cmath>

namespace perflibs::linalg {

// use decltype of the constructor of the pair to force template instantiation
template<typename T>
using find_index_kernel_t = decltype(perflibs::trivial_pair<kernel_inttype, remove_complex_t<T>>{})  (kernel_inttype n, const T *x, const kernel_inttype incx);

namespace {

template<typename T>
PERFLIBS_LINALG_INLINE
remove_complex_t<T> find_index_abs(const T& val) {
	if constexpr(is_complex_v<T>) {
		return std::abs(val.real())
		     + std::abs(val.imag());
	}
	else {
		return std::abs(val);
	}
}

/**
 * C++ implementation of iamax for non-unit increments, but it can also compute with
 * inc==1 - in practice this won't be the case as this function will only be
 * invoked with incx!=1. For incx==1 the assembly kernels will do the work.
 *
 * @return a pair of the index of the element with the max absolute value and that max absolute value (<index, max_abs>)
 */
template <typename T>
PERFLIBS_LINALG_INLINE
perflibs::trivial_pair<kernel_inttype, std::remove_cv_t<remove_complex_t<T>>>
iamax_fallback(const kernel_inttype n, const T *x, const kernel_inttype incx) {

	if (n <= 0 || incx <= 0) {
		return {0, 0};
	}

	using real_type = remove_complex_t<T>;
	real_type max = 0;
	kernel_inttype index = 1;

	for (int_type i = 0; i < n; ++i) {
		const real_type xi_abs = find_index_abs(x[i * incx]);

		if (xi_abs > max) {
			max = xi_abs;
			index = i+1;
		}
	}
	return {index, max};
}// iamax_fallback

/**
 * C++ implementation of iamin for non-unit increments, but it can also compute with
 * inc==1 - in practice this won't be the case as this function will only be
 * invoked with incx!=1. For incx==1 the assembly kernels will do the work.
 *
 * @return a pair of the index of the element with the min absolute value and that min absolute value (<index, min_abs>)
 */
template <typename T>
PERFLIBS_LINALG_INLINE
perflibs::trivial_pair<kernel_inttype, std::remove_cv_t<remove_complex_t<T>>>
iamin_fallback(const kernel_inttype n, const T *x, const kernel_inttype incx) {

	if (n <= 0 || incx <= 0) {
		return {0, 0};
	}

	using real_type = remove_complex_t<T>;
	real_type min = find_index_abs(x[0]);		// Possible as we know n>0 by now
	kernel_inttype index = 1;

	for (int_type i = 1; i < n; ++i) {
		const real_type xi_abs = find_index_abs(x[i * incx]);

		if (xi_abs < min) {
			min = xi_abs;
			index = i+1;
		}
	}
	return {index, min};
}// iamin_fallback


} // namespace <anon>
} // namespace perflibs::linalg

extern "C" {

perflibs::linalg::find_index_kernel_t<float> isamax_kernel;
perflibs::linalg::find_index_kernel_t<double> idamax_kernel;
perflibs::linalg::find_index_kernel_t<std::complex<float>> icamax_kernel;
perflibs::linalg::find_index_kernel_t<std::complex<double>> izamax_kernel;
perflibs::linalg::find_index_kernel_t<float> isamin_kernel;
perflibs::linalg::find_index_kernel_t<double> idamin_kernel;
perflibs::linalg::find_index_kernel_t<std::complex<float>> icamin_kernel;
perflibs::linalg::find_index_kernel_t<std::complex<double>> izamin_kernel;
}
#endif // PERFLIBS_IAMAX_SPEC_HPP

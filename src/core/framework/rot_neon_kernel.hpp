/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_ROT_NEON_KERNEL_HPP
#define PERFLIBS_LINALG_ROT_NEON_KERNEL_HPP

#include "perflibs_complex.hpp"
#include "perflibs_assert.hpp"
#include "perflibs_util.hpp"
#include "matrix/interleave_batch.hpp"

namespace perflibs::linalg {
namespace {

template<typename T>
PERFLIBS_LINALG_INLINE void rot_neon_kernel_complex(kernel_inttype n, T *x, T *y, const kernel_inttype incx,
                             const kernel_inttype incy, const remove_complex_t<T> c, const T s) {
	PERFLIBS_ASSERT(incx == 1 && incy == 1);

	// We de-interleave real & imaginary values, so
	// twice the number of elements per iteration
	constexpr int batch_size = 2 * 16 / sizeof(T);
	kernel_inttype i = 0;
	kernel_inttype remain_n = n % batch_size;

	// Scalar loop
	for (; i < remain_n; i++) {
		T tmp = c * x[i] + s * y[i];
		y[i] = c * y[i] - std::conj(s) * x[i];
		x[i] = tmp;
	}

	// Vectorized loop
	auto c_vec = simd::dup(c);
	auto s_vec_real = simd::dup(s.real());
	auto s_vec_imag = simd::dup(s.imag());
	for (; i < n; i += batch_size) {
		auto xi = reinterpret_cast<remove_complex_t<T> *>(&x[i]);
		auto yi = reinterpret_cast<remove_complex_t<T> *>(&y[i]);

		// LD2s
		auto x_vec = simd::load2(xi);
		auto y_vec = simd::load2(yi);

		// c * x
		auto mul_1_real = simd::mul(c_vec, x_vec.val[0]);
		auto mul_1_imag = simd::mul(c_vec, x_vec.val[1]);

		// c * y
		auto mul_2_real = simd::mul(c_vec, y_vec.val[0]);
		auto mul_2_imag = simd::mul(c_vec, y_vec.val[1]);

		// c * x + s * y
		mul_1_real = simd::mla(mul_1_real, s_vec_real, y_vec.val[0]);
		mul_1_imag = simd::mla(mul_1_imag, s_vec_imag, y_vec.val[0]);
		mul_1_real = simd::mls(mul_1_real, s_vec_imag, y_vec.val[1]);
		mul_1_imag = simd::mla(mul_1_imag, s_vec_real, y_vec.val[1]);

		// c * y - s_conj * x
		mul_2_real = simd::mls(mul_2_real, s_vec_real, x_vec.val[0]);
		mul_2_imag = simd::mla(mul_2_imag, s_vec_imag, x_vec.val[0]);
		mul_2_real = simd::mls(mul_2_real, s_vec_imag, x_vec.val[1]);
		mul_2_imag = simd::mls(mul_2_imag, s_vec_real, x_vec.val[1]);

		// ST2s
		simd::store2(xi, { mul_1_real, mul_1_imag });
		simd::store2(yi, { mul_2_real, mul_2_imag });
	}
}
} // namespace
} // end namespace perflibs::linalg

#endif // PERFLIBS_LINALG_ROT_NEON_KERNEL_HPP

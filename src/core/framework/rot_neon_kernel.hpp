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

	if (n <= 0)
		return;

	using real_type = remove_complex_t<T>;
	constexpr int batch_size = 16 / sizeof(T);
	// Avoid vector setup for a single complex float.
	if constexpr (batch_size == 2) {
		if (n == 1) {
			const T tmp = c * x[0] + s * y[0];
			y[0] = c * y[0] - std::conj(s) * x[0];
			x[0] = tmp;
			return;
		}
	}

	const auto c_vec = simd::dup(c);
	const auto s_vec_real = simd::dup(s.real());
	const auto s_vec_imag = simd::dup_pair(-s.imag(), s.imag());

	// Keep real/imaginary lanes interleaved. Swapping adjacent lanes and
	// multiplying by (-imag(s), imag(s)) supplies the imaginary products.
	auto rotate = [&](auto &x_vec, auto &y_vec) {
		auto new_x = simd::mul(c_vec, x_vec);
		auto new_y = simd::mul(c_vec, y_vec);
		new_x = simd::mla(new_x, s_vec_real, y_vec);
		new_y = simd::mls(new_y, s_vec_real, x_vec);
		new_x = simd::mla(new_x, s_vec_imag, simd::swap_pairs(y_vec));
		new_y = simd::mla(new_y, s_vec_imag, simd::swap_pairs(x_vec));
		x_vec = new_x;
		y_vec = new_y;
	};

	kernel_inttype i = 0;
	for (; n - i >= 2 * batch_size; i += 2 * batch_size) {
		auto xi = reinterpret_cast<real_type *>(x + i);
		auto yi = reinterpret_cast<real_type *>(y + i);
		auto x0 = simd::load1(xi);
		auto y0 = simd::load1(yi);
		auto x1 = simd::load1(xi + 2 * batch_size);
		auto y1 = simd::load1(yi + 2 * batch_size);
		rotate(x0, y0);
		rotate(x1, y1);
		simd::store1(xi, x0);
		simd::store1(yi, y0);
		simd::store1(xi + 2 * batch_size, x1);
		simd::store1(yi + 2 * batch_size, y1);
	}
	if (n - i >= batch_size) {
		auto xi = reinterpret_cast<real_type *>(x + i);
		auto yi = reinterpret_cast<real_type *>(y + i);
		auto x_vec = simd::load1(xi);
		auto y_vec = simd::load1(yi);
		rotate(x_vec, y_vec);
		simd::store1(xi, x_vec);
		simd::store1(yi, y_vec);
		i += batch_size;
	}
	if constexpr (batch_size == 2) {
		if (i < n) {
			const T tmp = c * x[i] + s * y[i];
			y[i] = c * y[i] - std::conj(s) * x[i];
			x[i] = tmp;
		}
	}
}
} // namespace
} // end namespace perflibs::linalg

#endif // PERFLIBS_LINALG_ROT_NEON_KERNEL_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_ROT_SVE_KERNEL_HPP
#define PERFLIBS_LINALG_ROT_SVE_KERNEL_HPP

#include "perflibs_assert.hpp"
#include "perflibs_complex.hpp"
#include "detect/simd.hpp"
#include "perflibs_util.hpp"
#include "framework/vector_length.hpp"
#include "matrix/interleave_batch.hpp"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

namespace perflibs::linalg {
namespace {

template<typename T>
struct vec_type;

#ifdef __ARM_FEATURE_SVE
template<>
struct vec_type<float> {
	using type = svfloat32_t;
	static inline svbool_t whilelt(const kernel_inttype i, const kernel_inttype n) {
		return svwhilelt_b32(i, n);
	}

	static inline type broadcast_complex(const complex_float x) {
		return svdupq_f32(x.real(), x.imag(), x.real(), x.imag());
	}

	static inline svbool_t ptrue() {
		return svptrue_b32();
	}
};

template<>
struct vec_type<double> {
	using type = svfloat64_t;
	static inline svbool_t whilelt(const kernel_inttype i, const kernel_inttype n) {
		return svwhilelt_b64(i, n);
	}

	static inline type broadcast_complex(const complex_double x) {
		return svdupq_f64(x.real(), x.imag());
	}

	static inline svbool_t ptrue() {
		return svptrue_b64();
	}
};
#endif

template<typename T>
void rot_kernel_sve_complex(kernel_inttype n, T *x, T *y, const kernel_inttype incx,
                            const kernel_inttype incy, const remove_complex_t<T> c, const T s) {
	if constexpr (simd::is_sve) {
		PERFLIBS_ASSERT(incx == 1 && incy == 1);
		using vec_t = vec_type<remove_complex_t<T>>;

		auto c_vec = simd::dup<typename vec_t::type>(c);
		auto s_vec = vec_t::broadcast_complex(s);

		kernel_inttype i = 0;

		const auto vec_size = vl<extensions::sve, remove_complex_t<T>, 1>();
		const auto ptrue = vec_t::ptrue();

		auto pg = vec_t::whilelt(i, 2 * n);

		while (svptest_first(ptrue, pg)) {
			auto xi = reinterpret_cast<remove_complex_t<T> *>(&(x[i / 2]));
			auto yi = reinterpret_cast<remove_complex_t<T> *>(&(y[i / 2]));

			auto x_vec = svld1(pg, xi);
			auto y_vec = svld1(pg, yi);

			// x[i] = c * x + s * y
			auto acc_x = svmul_x(pg, c_vec, x_vec);
			acc_x = svcmla_x(pg, acc_x, s_vec, y_vec, 0);
			acc_x = svcmla_x(pg, acc_x, s_vec, y_vec, 90);
			svst1(pg, xi, acc_x);

			// y[i] = c * y - s_conj * x
			auto acc_y = svmul_x(pg, c_vec, y_vec);
			acc_y = svcmla_x(pg, acc_y, s_vec, x_vec, 90);
			acc_y = svcmla_x(pg, acc_y, s_vec, x_vec, 180);
			svst1(pg, yi, acc_y);

			i += vec_size;
			pg = vec_t::whilelt(i, 2 * n);
		}
	}
}

} // namespace
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_ROT_SVE_KERNEL_HPP

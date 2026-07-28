/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "perflibs_complex.hpp"
#include "perflibs_assert.hpp"
#include "framework/l2_norm_kernels.hpp"
#include "framework/vector_length.hpp"
#include "matrix/interleave_batch.hpp"

#include <arm_sve.h>

namespace perflibs {

template<typename T>
using real_type = typename perflibs::remove_complex_t<T>;

template<typename T>
struct vec_type;

template<>
struct vec_type<float> {
	using type = svfloat32_t;

	static inline svbool_t whilelt(const kernel_inttype i, const kernel_inttype n) {
		return svwhilelt_b32(i, n);
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

	static inline svbool_t ptrue() {
		return svptrue_b64();
	}
};

template<typename T>
using vec_type_t  = typename vec_type<T>::type;

template<typename T>
inline void l2_norm_step_vec(kernel_inttype i, const T *x, vec_type_t<real_type<T>> &ssq_sml, vec_type_t<real_type<T>> &ssq_med,
                          vec_type_t<real_type<T>> &ssq_big, svbool_t pg) {
	using real_type_t = real_type<T>;
	using scale_constants = linalg::l2_norm_scale_constants<real_type_t>;

	vec_type_t<real_type_t> xis = svabs_x(pg, svld1(pg, reinterpret_cast<const real_type_t *>(&(x[i]))));

	// Compute predicates
	svbool_t p_big = svcmpgt(pg, xis, scale_constants::TR_BIG);
	svbool_t p_sml = svcmplt(pg, xis, scale_constants::TR_SMALL);
	svbool_t p_med = svnor_b_z(pg, p_big, p_sml);

	// Scale down large values
	auto scaled_big = svmul_x(p_big, xis, scale_constants::SCALE_BIG);
	ssq_big = svmla_m(p_big, ssq_big, scaled_big, scaled_big);

	// Scale up small values
	auto scaled_sml = svmul_x(p_sml, xis, scale_constants::SCALE_SMALL);
	ssq_sml = svmla_m(p_sml, ssq_sml, scaled_sml, scaled_sml);

	ssq_med = svmla_m(p_med, ssq_med, xis, xis);
}


template<typename T>
real_type<T> l2_norm_sve_kernel(kernel_inttype n, const T *x, const kernel_inttype incx) {
	PERFLIBS_ASSERT(incx == 1);
	using real_type_t = real_type<T>;
	using vec_t = vec_type<real_type_t>;

	constexpr real_type_t zero { 0.0 };

	auto ssq_sml_vec = linalg::simd::dup<vec_type_t<real_type_t>>(zero);
	auto ssq_med_vec = linalg::simd::dup<vec_type_t<real_type_t>>(zero);
	auto ssq_big_vec = linalg::simd::dup<vec_type_t<real_type_t>>(zero);

	constexpr bool is_complex = perflibs::is_complex_v<T>;
	const auto ptrue = vec_t::ptrue();
	const auto vec_size = linalg::vl<linalg::extensions::sve, real_type_t, 1>();

	kernel_inttype i = 0;
	kernel_inttype n_values = is_complex ? 2 * n : n;

	svbool_t pg = vec_t::whilelt(i, n_values);
	do {
		#pragma GCC unroll(4)
		for (int j = 0; j < 4; j++) {
			l2_norm_step_vec<T>(is_complex ? (i * incx) / 2 : (i * incx),
			                 x, ssq_sml_vec, ssq_med_vec, ssq_big_vec, pg);
			i += vec_size;
			pg = vec_t::whilelt(i, n_values);
		}
	} while (svptest_any(ptrue, pg));

	// Use ordered reduce to ensure sufficient accuracy for CSTEDC/CSTEGR
	perflibs::linalg::l2_norm_accumulator<real_type_t> acc;
	acc.ssq_sml = svadda(ptrue, real_type_t(0), ssq_sml_vec);
	acc.ssq_med = svadda(ptrue, real_type_t(0), ssq_med_vec);
	acc.ssq_big = svadda(ptrue, real_type_t(0), ssq_big_vec);

	return acc.finish();
};

template float l2_norm_sve_kernel<float>(kernel_inttype n, const float *x, const kernel_inttype incx);
template double l2_norm_sve_kernel<double>(kernel_inttype n, const double *x, const kernel_inttype incx);
template float l2_norm_sve_kernel<complex_float>(kernel_inttype n, const complex_float *x, const kernel_inttype incx);
template double l2_norm_sve_kernel<complex_double>(kernel_inttype n, const complex_double *x, const kernel_inttype incx);

} // end namespace perflibs

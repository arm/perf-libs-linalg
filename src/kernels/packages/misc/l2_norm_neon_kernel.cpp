/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "perflibs_complex.hpp"
#include "perflibs_assert.hpp"
#include "framework/l2_norm_kernels.hpp"
#include "matrix/interleave_batch.hpp"

#include <arm_neon.h>

namespace perflibs {

template<typename T>
using real_type = typename perflibs::remove_complex_t<T>;

template<typename T>
struct vec_type;

template<>
struct vec_type<float> {
	using type            = float32x4_t;
	using type_batch      = float32x4x2_t;
	using type_batch_mask = uint32x4x2_t;
};

template<>
struct vec_type<double> {
	using type            = float64x2_t;
	using type_batch      = float64x2x2_t;
	using type_batch_mask = uint64x2x2_t;
};

template<typename T>
using vec_type_t  = typename vec_type<T>::type;
template<typename T>
using vec_batch_t = typename vec_type<T>::type_batch;
template<typename T>
using vec_mask_t  = typename vec_type<T>::type_batch_mask;

template<typename T>
struct l2_norm_scale_constants_vec {
	vec_type_t<T> TR_SMALL;
	vec_type_t<T> TR_BIG;
	vec_type_t<T> SCALE_SMALL;
	vec_type_t<T> SCALE_BIG;
};

template<typename T>
inline void l2_norm_step_vec(kernel_inttype i, const T *x, vec_type_t<real_type<T>> &ssq_sml, vec_type_t<real_type<T>> &ssq_med, vec_type_t<real_type<T>> &ssq_big,
                          const l2_norm_scale_constants_vec<real_type<T>> &scale_constants) {
	using real_type_t = real_type<T>;
	vec_batch_t<real_type_t> xis_batch = linalg::simd::load_batch2(reinterpret_cast<const real_type_t*>(&x[i]));

	vec_mask_t<real_type_t> mask_big;
	vec_mask_t<real_type_t> mask_sml;
	vec_mask_t<real_type_t> mask_med;

	#pragma GCC unroll 2
	for (kernel_inttype i = 0; i < 2; i++) {
		// Compute masks
		xis_batch.val[i] = linalg::simd::abs(xis_batch.val[i]);
		mask_big.val[i]  = linalg::simd::cmgt(xis_batch.val[i], scale_constants.TR_BIG);
		mask_sml.val[i]  = linalg::simd::cmlt(xis_batch.val[i], scale_constants.TR_SMALL);
		mask_med.val[i]  = linalg::simd::orr(mask_big.val[i], mask_sml.val[i]);
		auto mask_zero   = linalg::simd::cmeqz(xis_batch.val[i]);
		mask_med.val[i]  = linalg::simd::bit_clear(mask_med.val[i], mask_zero);
	}

	// Combine medium masks
	auto mask_med_all = linalg::simd::orr(mask_med.val[0], mask_med.val[1]);

	// Skip scaling computation if there are only medium values
	if (__builtin_expect(!linalg::simd::is_zero(mask_med_all), 0)) {
		#pragma GCC unroll 2
		for (kernel_inttype i = 0; i < 2; i++) {
			// Scale down large values
			auto big_xs = linalg::simd::apply_mask(mask_big.val[i], xis_batch.val[i]);
			big_xs      = linalg::simd::mul(big_xs, scale_constants.SCALE_BIG);
			ssq_big     = linalg::simd::mla(ssq_big, big_xs, big_xs);

			// Scale up sml values
			auto sml_xs = linalg::simd::apply_mask(mask_sml.val[i], xis_batch.val[i]);
			sml_xs      = linalg::simd::mul(sml_xs, scale_constants.SCALE_SMALL);
			ssq_sml     = linalg::simd::mla(ssq_sml, sml_xs, sml_xs);
		}
	}

	#pragma GCC unroll 2
	for (kernel_inttype i = 0; i < 2; i++) {
		// Leave medium values unscaled
		auto med_xs = linalg::simd::bit_clear(xis_batch.val[i], mask_med.val[i]);
		ssq_med     = linalg::simd::mla(ssq_med, med_xs, med_xs);
	}
}

template<typename T>
real_type<T> l2_norm_neon_kernel(kernel_inttype n, const T *x, const kernel_inttype incx) {
	PERFLIBS_ASSERT(incx == 1);
	using real_type_t = real_type<T>;
	using vec_t = vec_type_t<real_type_t>;

	constexpr real_type_t zero { 0.0 };
	using scale_constants_scalar = linalg::l2_norm_scale_constants<real_type_t>;
	// Expand scale constants to vector type
	l2_norm_scale_constants_vec<real_type_t> scale_constants = {
		linalg::simd::dup(scale_constants_scalar::TR_SMALL),
		linalg::simd::dup(scale_constants_scalar::TR_BIG),
		linalg::simd::dup(scale_constants_scalar::SCALE_SMALL),
		linalg::simd::dup(scale_constants_scalar::SCALE_BIG)
	};

	perflibs::linalg::l2_norm_accumulator<T> acc;
	constexpr kernel_inttype vec_size = 16 / sizeof(T);
	constexpr kernel_inttype batch_size = 2 * vec_size;

	vec_t ssq_sml_vec = linalg::simd::dup(zero);
	vec_t ssq_med_vec = linalg::simd::dup(zero);
	vec_t ssq_big_vec = linalg::simd::dup(zero);

	kernel_inttype remain_n = n % batch_size;
	kernel_inttype i = 0;

	for (; i < remain_n; i++) {
		acc.complex_add(x[i * incx]);
	}

	#pragma GCC unroll 4
	for (; i < n; i += batch_size) {
		l2_norm_step_vec<T>(i * incx, x, ssq_sml_vec, ssq_med_vec, ssq_big_vec, scale_constants);
	}


	acc.ssq_sml += linalg::simd::sum(ssq_sml_vec);
	acc.ssq_med += linalg::simd::sum(ssq_med_vec);
	acc.ssq_big += linalg::simd::sum(ssq_big_vec);

	return acc.finish();
};

template float l2_norm_neon_kernel<float>(kernel_inttype n, const float *x, const kernel_inttype incx);
template double l2_norm_neon_kernel<double>(kernel_inttype n, const double *x, const kernel_inttype incx);
template float l2_norm_neon_kernel<complex_float>(kernel_inttype n, const complex_float *x, const kernel_inttype incx);
template double l2_norm_neon_kernel<complex_double>(kernel_inttype n, const complex_double *x, const kernel_inttype incx);

} // end namespace perflibs

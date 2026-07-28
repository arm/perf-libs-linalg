/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include <arm_neon.h>
#include <arm_bf16.h>

#include "perflibs_util.hpp"

extern "C" {

void sbgemv_t_bfdot_4x24_acle(
		kernel_inttype a_cntg, kernel_inttype a_strd,
		float alpha,
		const bfloat16_t *a, kernel_inttype lda,
		const bfloat16_t *x, kernel_inttype incx,
		float beta,
		      float      *y, kernel_inttype incy) {

	kernel_inttype j = 0u;

	constexpr auto a_strd_unroll0 = 4u;
	for(; (j + a_strd_unroll0) <= a_strd; j += a_strd_unroll0) {
		float32x4_t c00_f32x4 = vdupq_n_f32(0.0f);
		float32x4_t c10_f32x4 = vdupq_n_f32(0.0f);
		float32x4_t c20_f32x4 = vdupq_n_f32(0.0f);
		float32x4_t c30_f32x4 = vdupq_n_f32(0.0f);

		float32x4_t c01_f32x4 = vdupq_n_f32(0.0f);
		float32x4_t c11_f32x4 = vdupq_n_f32(0.0f);
		float32x4_t c21_f32x4 = vdupq_n_f32(0.0f);
		float32x4_t c31_f32x4 = vdupq_n_f32(0.0f);

		float32x4_t c02_f32x4 = vdupq_n_f32(0.0f);
		float32x4_t c12_f32x4 = vdupq_n_f32(0.0f);
		float32x4_t c22_f32x4 = vdupq_n_f32(0.0f);
		float32x4_t c32_f32x4 = vdupq_n_f32(0.0f);

		auto a_ptr_0 = a + lda * j;
		auto a_ptr_1 = a + lda * ( j + 1 );
		auto a_ptr_2 = a + lda * ( j + 2 );
		auto a_ptr_3 = a + lda * ( j + 3 );

		auto x_ptr = x;

		kernel_inttype i = 0u;

		constexpr auto cntg_urnoll0 = 8u * 3u;
		for(; (i + cntg_urnoll0) < a_cntg; i += cntg_urnoll0, a_ptr_0 += cntg_urnoll0, a_ptr_1 += cntg_urnoll0, a_ptr_2 += cntg_urnoll0, a_ptr_3 += cntg_urnoll0, x_ptr += cntg_urnoll0) {
			auto x0_bf16x8 = vld1q_bf16(x_ptr + 0 * 8);
			auto x1_bf16x8 = vld1q_bf16(x_ptr + 1 * 8);
			auto x2_bf16x8 = vld1q_bf16(x_ptr + 2 * 8);

			auto a00_bf16x8 = vld1q_bf16(a_ptr_0 + 0 * 8);
			auto a01_bf16x8 = vld1q_bf16(a_ptr_0 + 1 * 8);
			auto a02_bf16x8 = vld1q_bf16(a_ptr_0 + 2 * 8);

			auto a10_bf16x8 = vld1q_bf16(a_ptr_1 + 0 * 8);
			auto a11_bf16x8 = vld1q_bf16(a_ptr_1 + 1 * 8);
			auto a12_bf16x8 = vld1q_bf16(a_ptr_1 + 2 * 8);

			auto a20_bf16x8 = vld1q_bf16(a_ptr_2 + 0 * 8);
			auto a21_bf16x8 = vld1q_bf16(a_ptr_2 + 1 * 8);
			auto a22_bf16x8 = vld1q_bf16(a_ptr_2 + 2 * 8);

			auto a30_bf16x8 = vld1q_bf16(a_ptr_3 + 0 * 8);
			auto a31_bf16x8 = vld1q_bf16(a_ptr_3 + 1 * 8);
			auto a32_bf16x8 = vld1q_bf16(a_ptr_3 + 2 * 8);

			c00_f32x4 = vbfdotq_f32(c00_f32x4, x0_bf16x8, a00_bf16x8);
			c01_f32x4 = vbfdotq_f32(c01_f32x4, x1_bf16x8, a01_bf16x8);
			c02_f32x4 = vbfdotq_f32(c02_f32x4, x2_bf16x8, a02_bf16x8);

			c10_f32x4 = vbfdotq_f32(c10_f32x4, x0_bf16x8, a10_bf16x8);
			c11_f32x4 = vbfdotq_f32(c11_f32x4, x1_bf16x8, a11_bf16x8);
			c12_f32x4 = vbfdotq_f32(c12_f32x4, x2_bf16x8, a12_bf16x8);

			c20_f32x4 = vbfdotq_f32(c20_f32x4, x0_bf16x8, a20_bf16x8);
			c21_f32x4 = vbfdotq_f32(c21_f32x4, x1_bf16x8, a21_bf16x8);
			c22_f32x4 = vbfdotq_f32(c22_f32x4, x2_bf16x8, a22_bf16x8);

			c30_f32x4 = vbfdotq_f32(c30_f32x4, x0_bf16x8, a30_bf16x8);
			c31_f32x4 = vbfdotq_f32(c31_f32x4, x1_bf16x8, a31_bf16x8);
			c32_f32x4 = vbfdotq_f32(c32_f32x4, x2_bf16x8, a32_bf16x8);
		}


		constexpr auto cntg_urnoll1 = 8u;
		for(; (i + cntg_urnoll1) < a_cntg; i += cntg_urnoll1, a_ptr_0 += cntg_urnoll1, a_ptr_1 += cntg_urnoll1, a_ptr_2 += cntg_urnoll1, a_ptr_3 += cntg_urnoll1, x_ptr += cntg_urnoll1) {
			auto x0_bf16x8 = vld1q_bf16(x_ptr);

			auto a00_bf16x8 = vld1q_bf16(a_ptr_0);
			auto a10_bf16x8 = vld1q_bf16(a_ptr_1);
			auto a20_bf16x8 = vld1q_bf16(a_ptr_2);
			auto a30_bf16x8 = vld1q_bf16(a_ptr_3);

			c00_f32x4 = vbfdotq_f32(c00_f32x4, x0_bf16x8, a00_bf16x8);
			c10_f32x4 = vbfdotq_f32(c10_f32x4, x0_bf16x8, a10_bf16x8);
			c20_f32x4 = vbfdotq_f32(c20_f32x4, x0_bf16x8, a20_bf16x8);
			c30_f32x4 = vbfdotq_f32(c30_f32x4, x0_bf16x8, a30_bf16x8);
		}

		for(; i < a_cntg; ++i, ++a_ptr_0, ++a_ptr_1, ++a_ptr_2, ++a_ptr_3, ++x_ptr) {
			const auto x_tmp = *x_ptr;
			c00_f32x4[0] += x_tmp * *a_ptr_0;
			c10_f32x4[0] += x_tmp * *a_ptr_1;
			c20_f32x4[0] += x_tmp * *a_ptr_2;
			c30_f32x4[0] += x_tmp * *a_ptr_3;
		}

		auto c_out_f32x4 = vld1q_f32(y + j);

		c00_f32x4[0] = vaddvq_f32( vaddq_f32( vaddq_f32(c00_f32x4, c01_f32x4), c02_f32x4 ) );
		c00_f32x4[1] = vaddvq_f32( vaddq_f32( vaddq_f32(c10_f32x4, c11_f32x4), c12_f32x4 ) );
		c00_f32x4[2] = vaddvq_f32( vaddq_f32( vaddq_f32(c20_f32x4, c21_f32x4), c22_f32x4 ) );
		c00_f32x4[3] = vaddvq_f32( vaddq_f32( vaddq_f32(c30_f32x4, c31_f32x4), c32_f32x4 ) );

		c_out_f32x4 = vmulq_n_f32(c_out_f32x4, beta);
		c_out_f32x4 = vfmaq_n_f32(c_out_f32x4, c00_f32x4, alpha);

		vst1q_f32(y + j, c_out_f32x4);
	}

	constexpr auto a_strd_unroll1 = 1u;
	for(; (j + a_strd_unroll1) <= a_strd; j += a_strd_unroll1) {
		float32x4_t c00_f32x4 = vdupq_n_f32(0.0f);
		float32x4_t c01_f32x4 = vdupq_n_f32(0.0f);
		float32x4_t c02_f32x4 = vdupq_n_f32(0.0f);

		auto a_ptr_0 = a + lda * j;

		auto x_ptr = x;

		kernel_inttype i = 0u;

		constexpr auto cntg_urnoll0 = 8u * 3u;
		for(; (i + cntg_urnoll0) < a_cntg; i += cntg_urnoll0, a_ptr_0 += cntg_urnoll0, x_ptr += cntg_urnoll0) {
			auto x0_bf16x8 = vld1q_bf16(x_ptr + 0 * 8);
			auto x1_bf16x8 = vld1q_bf16(x_ptr + 1 * 8);
			auto x2_bf16x8 = vld1q_bf16(x_ptr + 2 * 8);

			auto a00_bf16x8 = vld1q_bf16(a_ptr_0 + 0 * 8);
			auto a01_bf16x8 = vld1q_bf16(a_ptr_0 + 1 * 8);
			auto a02_bf16x8 = vld1q_bf16(a_ptr_0 + 2 * 8);

			c00_f32x4 = vbfdotq_f32(c00_f32x4, x0_bf16x8, a00_bf16x8);
			c01_f32x4 = vbfdotq_f32(c01_f32x4, x1_bf16x8, a01_bf16x8);
			c02_f32x4 = vbfdotq_f32(c02_f32x4, x2_bf16x8, a02_bf16x8);
		}

		constexpr auto cntg_urnoll1 = 8u;
		for(; (i + cntg_urnoll1) < a_cntg; i += cntg_urnoll1, a_ptr_0 += cntg_urnoll1, x_ptr += cntg_urnoll1) {
			auto x0_bf16x8 = vld1q_bf16(x_ptr);
			auto a00_bf16x8 = vld1q_bf16(a_ptr_0);

			c00_f32x4 = vbfdotq_f32(c00_f32x4, x0_bf16x8, a00_bf16x8);
		}

		for(; i < a_cntg; ++i, ++a_ptr_0, ++x_ptr) {
			c00_f32x4[0] += *x_ptr * *a_ptr_0;
		}

		y[j] = alpha * vaddvq_f32( vaddq_f32( vaddq_f32(c00_f32x4, c01_f32x4), c02_f32x4 ) ) + beta * y[j];
	}
}

}

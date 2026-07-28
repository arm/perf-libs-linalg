/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "perflibs_util.hpp"
#include "perflibs_assert.hpp"

#include <arm_neon.h>
#include <arm_bf16.h>

#include "framework/dot_kernels.hpp"

namespace perflibs::linalg {
namespace {

template<typename ReturnType>
ReturnType dot_bfdot(kernel_inttype n, const bf16 *x, const  bf16 *y, const kernel_inttype incx, const kernel_inttype incy) {
	PERFLIBS_ASSERT(incx == 1, "x must be contiguous vector");
	PERFLIBS_ASSERT(incy == 1, "y must be contiguous vector");

	float32x4_t c0_f32x4 = vdupq_n_f32(0.0f);

	if(n >= 16) {
		float32x4_t c1_f32x4 = vdupq_n_f32(0.0f);
		float32x4_t c2_f32x4 = vdupq_n_f32(0.0f);

		if(n >= 32) {
			float32x4_t c3_f32x4 = vdupq_n_f32(0.0f);
			float32x4_t c4_f32x4 = vdupq_n_f32(0.0f);

			if(n >= 64) {
				float32x4_t c5_f32x4 = vdupq_n_f32(0.0f);
				float32x4_t c6_f32x4 = vdupq_n_f32(0.0f);
				float32x4_t c7_f32x4 = vdupq_n_f32(0.0f);

				auto x0_bf16x8 = vld1q_bf16(x + 0 * 8);
				auto x1_bf16x8 = vld1q_bf16(x + 1 * 8);
				auto x2_bf16x8 = vld1q_bf16(x + 2 * 8);
				auto x3_bf16x8 = vld1q_bf16(x + 3 * 8);
				auto x4_bf16x8 = vld1q_bf16(x + 4 * 8);
				auto x5_bf16x8 = vld1q_bf16(x + 5 * 8);
				auto x6_bf16x8 = vld1q_bf16(x + 6 * 8);
				auto x7_bf16x8 = vld1q_bf16(x + 7 * 8);

				auto y0_bf16x8 = vld1q_bf16(y + 0 * 8);
				auto y1_bf16x8 = vld1q_bf16(y + 1 * 8);
				auto y2_bf16x8 = vld1q_bf16(y + 2 * 8);
				auto y3_bf16x8 = vld1q_bf16(y + 3 * 8);
				auto y4_bf16x8 = vld1q_bf16(y + 4 * 8);
				auto y5_bf16x8 = vld1q_bf16(y + 5 * 8);
				auto y6_bf16x8 = vld1q_bf16(y + 6 * 8);
				auto y7_bf16x8 = vld1q_bf16(y + 7 * 8);

				n -= 64;
				x += 64;
				y += 64;

				for(; n >= 64; n -= 64, x += 64, y += 64) {
					c0_f32x4 = vbfdotq_f32(c0_f32x4, x0_bf16x8, y0_bf16x8);
					x0_bf16x8 = vld1q_bf16(x + 0 * 8);
					y0_bf16x8 = vld1q_bf16(y + 0 * 8);

					c1_f32x4 = vbfdotq_f32(c1_f32x4, x1_bf16x8, y1_bf16x8);
					x1_bf16x8 = vld1q_bf16(x + 1 * 8);
					y1_bf16x8 = vld1q_bf16(y + 1 * 8);

					c2_f32x4 = vbfdotq_f32(c2_f32x4, x2_bf16x8, y2_bf16x8);
					x2_bf16x8 = vld1q_bf16(x + 2 * 8);
					y2_bf16x8 = vld1q_bf16(y + 2 * 8);

					c3_f32x4 = vbfdotq_f32(c3_f32x4, x3_bf16x8, y3_bf16x8);
					x3_bf16x8 = vld1q_bf16(x + 3 * 8);
					y3_bf16x8 = vld1q_bf16(y + 3 * 8);

					c4_f32x4 = vbfdotq_f32(c4_f32x4, x4_bf16x8, y4_bf16x8);
					x4_bf16x8 = vld1q_bf16(x + 4 * 8);
					y4_bf16x8 = vld1q_bf16(y + 4 * 8);

					c5_f32x4 = vbfdotq_f32(c5_f32x4, x5_bf16x8, y5_bf16x8);
					x5_bf16x8 = vld1q_bf16(x + 5 * 8);
					y5_bf16x8 = vld1q_bf16(y + 5 * 8);

					c6_f32x4 = vbfdotq_f32(c6_f32x4, x6_bf16x8, y6_bf16x8);
					x6_bf16x8 = vld1q_bf16(x + 6 * 8);
					y6_bf16x8 = vld1q_bf16(y + 6 * 8);

					c7_f32x4 = vbfdotq_f32(c7_f32x4, x7_bf16x8, y7_bf16x8);
					x7_bf16x8 = vld1q_bf16(x + 7 * 8);
					y7_bf16x8 = vld1q_bf16(y + 7 * 8);

				}

				c0_f32x4 = vbfdotq_f32(c0_f32x4, x0_bf16x8, y0_bf16x8);
				c1_f32x4 = vbfdotq_f32(c1_f32x4, x1_bf16x8, y1_bf16x8);
				c2_f32x4 = vbfdotq_f32(c2_f32x4, x2_bf16x8, y2_bf16x8);
				c3_f32x4 = vbfdotq_f32(c3_f32x4, x3_bf16x8, y3_bf16x8);
				c4_f32x4 = vbfdotq_f32(c4_f32x4, x4_bf16x8, y4_bf16x8);
				c5_f32x4 = vbfdotq_f32(c5_f32x4, x5_bf16x8, y5_bf16x8);
				c7_f32x4 = vbfdotq_f32(c7_f32x4, x7_bf16x8, y7_bf16x8);
				c6_f32x4 = vbfdotq_f32(c6_f32x4, x6_bf16x8, y6_bf16x8);

				//reduce
				c0_f32x4 = vaddq_f32(c0_f32x4, c1_f32x4);
				c1_f32x4 = vaddq_f32(c2_f32x4, c3_f32x4);
				c2_f32x4 = vaddq_f32(c4_f32x4, c5_f32x4);
				c3_f32x4 = vaddq_f32(c6_f32x4, c7_f32x4);
			}

			if(n >= 32) {
				auto x0_bf16x8 = vld1q_bf16(x + 0 * 8);
				auto x1_bf16x8 = vld1q_bf16(x + 1 * 8);
				auto x2_bf16x8 = vld1q_bf16(x + 2 * 8);
				auto x3_bf16x8 = vld1q_bf16(x + 3 * 8);

				auto y0_bf16x8 = vld1q_bf16(y + 0 * 8);
				auto y1_bf16x8 = vld1q_bf16(y + 1 * 8);
				auto y2_bf16x8 = vld1q_bf16(y + 2 * 8);
				auto y3_bf16x8 = vld1q_bf16(y + 3 * 8);

				for(; n >= 32; n -= 32, x += 32, y += 32) {
					c0_f32x4 = vbfdotq_f32(c0_f32x4, x0_bf16x8, y0_bf16x8);
					x0_bf16x8 = vld1q_bf16(x + 0 * 8);
					y0_bf16x8 = vld1q_bf16(y + 0 * 8);

					c1_f32x4 = vbfdotq_f32(c1_f32x4, x1_bf16x8, y1_bf16x8);
					x1_bf16x8 = vld1q_bf16(x + 1 * 8);
					y1_bf16x8 = vld1q_bf16(y + 1 * 8);

					c2_f32x4 = vbfdotq_f32(c2_f32x4, x2_bf16x8, y2_bf16x8);
					x2_bf16x8 = vld1q_bf16(x + 2 * 8);
					y2_bf16x8 = vld1q_bf16(y + 2 * 8);

					c3_f32x4 = vbfdotq_f32(c3_f32x4, x3_bf16x8, y3_bf16x8);
					x3_bf16x8 = vld1q_bf16(x + 3 * 8);
					y3_bf16x8 = vld1q_bf16(y + 3 * 8);
				}
			}
			///reduce
			c0_f32x4 = vaddq_f32(c0_f32x4, c1_f32x4);
			c1_f32x4 = vaddq_f32(c2_f32x4, c3_f32x4);
		}
		if(n >= 16) {
			auto x0_bf16x8 = vld1q_bf16(x + 0 * 8);
			auto x1_bf16x8 = vld1q_bf16(x + 1 * 8);

			auto y0_bf16x8 = vld1q_bf16(y + 0 * 8);
			auto y1_bf16x8 = vld1q_bf16(y + 1 * 8);

			for(; n >= 16; n -= 16, x += 16, y += 16) {
				c0_f32x4 = vbfdotq_f32(c0_f32x4, x0_bf16x8, y0_bf16x8);
				x0_bf16x8 = vld1q_bf16(x + 0 * 8);
				y0_bf16x8 = vld1q_bf16(y + 0 * 8);

				c1_f32x4 = vbfdotq_f32(c1_f32x4, x1_bf16x8, y1_bf16x8);
				x1_bf16x8 = vld1q_bf16(x + 1 * 8);
				y1_bf16x8 = vld1q_bf16(y + 1 * 8);
			}
		}
		//reduce
		c0_f32x4 = vaddq_f32(c0_f32x4, c1_f32x4);
	}

	for(;n >= 8; n -= 8, x+=8, y+=8)  {
		auto x0_bf16x8 = vld1q_bf16(x);
		auto y0_bf16x8 = vld1q_bf16(y);

		c0_f32x4 = vbfdotq_f32(c0_f32x4, x0_bf16x8, y0_bf16x8);
	}

	float c = vaddvq_f32(c0_f32x4);

	for(; n > 0; --n, ++x, ++y) {
		c += static_cast<float>(*x) * static_cast<float>(*y);
	}

	return static_cast<ReturnType>(c);
}

} //namespace <anon>

extern "C" {

float sbdot_kernel(kernel_inttype n, const bf16 *x, const  bf16 *y, const kernel_inttype incx, const kernel_inttype incy) {
	if(incx == 1 && incy == 1) {
		return dot_bfdot<r32>(n, x, y, incx, incy);
	}

	return dotu_fallback<bf16, r32>(n, x, y, incx, incy);
}

bf16 bdot_kernel(kernel_inttype n, const bf16 *x, const  bf16 *y, const kernel_inttype incx, const kernel_inttype incy) {
	if(incx == 1 && incy == 1) {
		return dot_bfdot<bf16>(n, x, y, incx, incy);
	}

	return dotu_fallback<bf16, bf16>(n, x, y, incx, incy);
}


} //extern "C"

} // namespace perflibs::linalg

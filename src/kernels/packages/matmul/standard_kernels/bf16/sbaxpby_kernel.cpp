/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "framework/linalg_util.hpp"

#include "perflibs_util.hpp"
#include "perflibs_float.hpp"
#include "perflibs_assert.hpp"

#include <arm_bf16.h>
#include <arm_neon.h>

#include "detect/cpu_info.hpp" //features.neon_bf16

#include <cstddef>

extern"C" {

void sbaxpby_kernel_f32_fmla_x8_ldq(kernel_inttype n, float alpha, const __bf16 *x, float beta, float *y, kernel_inttype incx, kernel_inttype incy) {
	PERFLIBS_ASSERT(incx == 1, "incx must be 1");
	PERFLIBS_ASSERT(incy == 1, "incy must be 1");
	constexpr kernel_inttype unroll = 8;

	kernel_inttype i;
	for(i = 0; i <= ( n - unroll ); i += unroll, x += unroll, y += unroll) {
		bfloat16x8_t x0_bf16x8 = vld1q_bf16(x);
		uint16x8_t   x0_u16x8 = vreinterpretq_u16_bf16(x0_bf16x8);

		float32x4_t y0_fp32x4 = vld1q_f32(y);
		float32x4_t y1_fp32x4 = vld1q_f32(y + 4 );

		y0_fp32x4 = vmulq_n_f32(y0_fp32x4, beta);
		y1_fp32x4 = vmulq_n_f32(y1_fp32x4, beta);

		float32x4_t x0_fp32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(x0_u16x8), 16));
 		float32x4_t x1_fp32x4 = vreinterpretq_f32_u32(vshll_high_n_u16(x0_u16x8, 16));

		y0_fp32x4 = vfmaq_n_f32(y0_fp32x4, x0_fp32x4, alpha);
		y1_fp32x4 = vfmaq_n_f32(y1_fp32x4, x1_fp32x4, alpha);

		vst1q_f32(y,     y0_fp32x4);
		vst1q_f32(y + 4, y1_fp32x4);
	}

	bfloat16x4_t x0_bf16x4;
	float32x4_t  x0_fp32x4;
	float32x4_t  x1_fp32x4;
	float32x4_t  y0_fp32x4;
	float32x4_t  y1_fp32x4;

	const kernel_inttype rem = n - i;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
	switch(rem) {
	case 0:
		return;
	default:
		PERFLIBS_ASSERT(false, "invalid path");

	case 7:
		x0_bf16x4 = vset_lane_bf16(x[6], x0_bf16x4, 2);
		y1_fp32x4 = vsetq_lane_f32(y[6], y1_fp32x4, 2);
	case 6:
		x0_bf16x4 = vset_lane_bf16(x[5], x0_bf16x4, 1);
		y1_fp32x4 = vsetq_lane_f32(y[5], y1_fp32x4, 1);
	case 5:
		x0_bf16x4 = vset_lane_bf16(x[4], x0_bf16x4, 0);
		y1_fp32x4 = vsetq_lane_f32(y[4], y1_fp32x4, 0);

		x1_fp32x4 = vcvt_f32_bf16(x0_bf16x4);

	case 4:
		x0_bf16x4 = vset_lane_bf16(x[3], x0_bf16x4, 3);
		y0_fp32x4 = vsetq_lane_f32(y[3], y0_fp32x4, 3);
	case 3:
		x0_bf16x4 = vset_lane_bf16(x[2], x0_bf16x4, 2);
		y0_fp32x4 = vsetq_lane_f32(y[2], y0_fp32x4, 2);
	case 2:
		x0_bf16x4 = vset_lane_bf16(x[1], x0_bf16x4, 1);
		y0_fp32x4 = vsetq_lane_f32(y[1], y0_fp32x4, 1);
	case 1:
		x0_bf16x4 = vset_lane_bf16(x[0], x0_bf16x4, 0);
		y0_fp32x4 = vsetq_lane_f32(y[0], y0_fp32x4, 0);

	}
#pragma GCC diagnostic pop

	x0_fp32x4 = vcvt_f32_bf16(x0_bf16x4);

	y0_fp32x4 = vmulq_n_f32(y0_fp32x4, beta);
	y1_fp32x4 = vmulq_n_f32(y1_fp32x4, beta);

	y0_fp32x4 = vfmaq_n_f32(y0_fp32x4, x0_fp32x4, alpha);
	y1_fp32x4 = vfmaq_n_f32(y1_fp32x4, x1_fp32x4, alpha);

	switch(rem) {
	case 7: y[6] = y1_fp32x4[2];
	case 6: y[5] = y1_fp32x4[1];
	case 5: y[4] = y1_fp32x4[0];
	case 4: y[3] = y0_fp32x4[3];
	case 3: y[2] = y0_fp32x4[2];
	case 2: y[1] = y0_fp32x4[1];
	case 1: y[0] = y0_fp32x4[0];
	}
}

void sbaxpby_kernel_f32_fmla_x4_ldd(kernel_inttype n, float alpha, const __bf16 *x, float beta, float *y, kernel_inttype incx, kernel_inttype incy) {
	PERFLIBS_ASSERT(incx == 1, "incx must be 1");
	PERFLIBS_ASSERT(incy == 1, "incy must be 1");

	constexpr kernel_inttype unroll = 4;

	kernel_inttype i;
	for(i = 0; i <= ( n - unroll ); i += unroll, x += unroll, y += unroll) {
		bfloat16x4_t x0_bf16x4 = vld1_bf16(x);
		float32x4_t  y0_fp32x4 = vld1q_f32(y);

		float32x4_t  x0_fp32x4 = vcvt_f32_bf16(x0_bf16x4);

		y0_fp32x4 = vmulq_n_f32(y0_fp32x4, beta);
		y0_fp32x4 = vfmaq_n_f32(y0_fp32x4, x0_fp32x4, alpha);

		vst1q_f32(y, y0_fp32x4);
	}

	bfloat16x4_t x0_bf16x4;
	float32x4_t  x0_fp32x4;
	float32x4_t  y0_fp32x4;

	const kernel_inttype rem = n - i;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
	switch(rem) {
	case 0:
		return;
	default:
		PERFLIBS_ASSERT(false, "invalid path");

	case 3:
		x0_bf16x4 = vset_lane_bf16(x[2], x0_bf16x4, 2);
		y0_fp32x4 = vsetq_lane_f32(y[2], y0_fp32x4, 2);
	case 2:
		x0_bf16x4 = vset_lane_bf16(x[1], x0_bf16x4, 1);
		y0_fp32x4 = vsetq_lane_f32(y[1], y0_fp32x4, 1);
	case 1:
		x0_bf16x4 = vset_lane_bf16(x[0], x0_bf16x4, 0);
		y0_fp32x4 = vsetq_lane_f32(y[0], y0_fp32x4, 0);

	}
#pragma GCC diagnostic pop

	x0_fp32x4 = vcvt_f32_bf16(x0_bf16x4);
	y0_fp32x4 = vmulq_n_f32(y0_fp32x4, beta);
	y0_fp32x4 = vfmaq_n_f32(y0_fp32x4, x0_fp32x4, alpha);

	switch(rem) {
	case 3: y[2] = y0_fp32x4[2];
	case 2: y[1] = y0_fp32x4[1];
	case 1: y[0] = y0_fp32x4[0];
	}
}

void sbaxpby_kernel_f32_fmla_x8_ldd(kernel_inttype n, float alpha, const __bf16 *x, float beta, float *y, kernel_inttype incx, kernel_inttype incy) {
	PERFLIBS_ASSERT(incx == 1, "incx must be 1");
	PERFLIBS_ASSERT(incy == 1, "incy must be 1");

	constexpr kernel_inttype unroll = 8;

	kernel_inttype i;
	for(i = 0; i <= ( n - unroll ); i += unroll, x += unroll, y += unroll) {
		bfloat16x4_t x0_bf16x4 = vld1_bf16(x);
		float32x4_t  y0_fp32x4 = vld1q_f32(y);

		bfloat16x4_t x1_bf16x4 = vld1_bf16(x + 4);
		float32x4_t  y1_fp32x4 = vld1q_f32(y + 4);

		float32x4_t  x0_fp32x4 = vcvt_f32_bf16(x0_bf16x4);
		float32x4_t  x1_fp32x4 = vcvt_f32_bf16(x1_bf16x4);

		y0_fp32x4 = vmulq_n_f32(y0_fp32x4, beta);
		y1_fp32x4 = vmulq_n_f32(y1_fp32x4, beta);

		y0_fp32x4 = vfmaq_n_f32(y0_fp32x4, x0_fp32x4, alpha);
		y1_fp32x4 = vfmaq_n_f32(y1_fp32x4, x1_fp32x4, alpha);

		vst1q_f32(y,     y0_fp32x4);
		vst1q_f32(y + 4, y1_fp32x4);
	}


	bfloat16x4_t x0_bf16x4;
	float32x4_t  x0_fp32x4;
	float32x4_t  x1_fp32x4;
	float32x4_t  y0_fp32x4;
	float32x4_t  y1_fp32x4;

	const kernel_inttype rem = n - i;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
	switch(rem) {
	case 0:
		return;
	default:
		PERFLIBS_ASSERT(false, "invalid path");

	case 7:
		x0_bf16x4 = vset_lane_bf16(x[6], x0_bf16x4, 2);
		y1_fp32x4 = vsetq_lane_f32(y[6], y1_fp32x4, 2);
	case 6:
		x0_bf16x4 = vset_lane_bf16(x[5], x0_bf16x4, 1);
		y1_fp32x4 = vsetq_lane_f32(y[5], y1_fp32x4, 1);
	case 5:
		x0_bf16x4 = vset_lane_bf16(x[4], x0_bf16x4, 0);
		y1_fp32x4 = vsetq_lane_f32(y[4], y1_fp32x4, 0);

		x1_fp32x4 = vcvt_f32_bf16(x0_bf16x4);

	case 4:
		x0_bf16x4 = vset_lane_bf16(x[3], x0_bf16x4, 3);
		y0_fp32x4 = vsetq_lane_f32(y[3], y0_fp32x4, 3);
	case 3:
		x0_bf16x4 = vset_lane_bf16(x[2], x0_bf16x4, 2);
		y0_fp32x4 = vsetq_lane_f32(y[2], y0_fp32x4, 2);
	case 2:
		x0_bf16x4 = vset_lane_bf16(x[1], x0_bf16x4, 1);
		y0_fp32x4 = vsetq_lane_f32(y[1], y0_fp32x4, 1);
	case 1:
		x0_bf16x4 = vset_lane_bf16(x[0], x0_bf16x4, 0);
		y0_fp32x4 = vsetq_lane_f32(y[0], y0_fp32x4, 0);
	}
#pragma GCC diagnostic pop

	x0_fp32x4 = vcvt_f32_bf16(x0_bf16x4);

	y0_fp32x4 = vmulq_n_f32(y0_fp32x4, beta);
	y1_fp32x4 = vmulq_n_f32(y1_fp32x4, beta);

	y0_fp32x4 = vfmaq_n_f32(y0_fp32x4, x0_fp32x4, alpha);
	y1_fp32x4 = vfmaq_n_f32(y1_fp32x4, x1_fp32x4, alpha);

	switch(rem) {
	case 7: y[6] = y1_fp32x4[2];
	case 6: y[5] = y1_fp32x4[1];
	case 5: y[4] = y1_fp32x4[0];
	case 4: y[3] = y0_fp32x4[3];
	case 3: y[2] = y0_fp32x4[2];
	case 2: y[1] = y0_fp32x4[1];
	case 1: y[0] = y0_fp32x4[0];
	}
}



void sbaxpby_kernel_f32_fmla_x16_ldq(kernel_inttype n, float alpha, const __bf16 *x, float beta, float *y, kernel_inttype incx, kernel_inttype incy) {
	PERFLIBS_ASSERT(incx == 1, "incx must be 1");
	PERFLIBS_ASSERT(incy == 1, "incy must be 1");

	constexpr kernel_inttype unroll = 16;

	kernel_inttype i;
	for(i = 0; i <= ( n - unroll ); i += unroll, x += unroll, y += unroll) {

		bfloat16x8_t x0_bf16x8 = vld1q_bf16(x);
		bfloat16x8_t x1_bf16x8 = vld1q_bf16(x + 8);

		uint16x8_t   x0_u16x8 = vreinterpretq_u16_bf16(x0_bf16x8);
		uint16x8_t   x1_u16x8 = vreinterpretq_u16_bf16(x1_bf16x8);

		float32x4_t y0_fp32x4 = vld1q_f32(y);
		float32x4_t y1_fp32x4 = vld1q_f32(y + 4);

		float32x4_t y2_fp32x4 = vld1q_f32(y + 8);
		float32x4_t y3_fp32x4 = vld1q_f32(y + 12);

		y0_fp32x4 = vmulq_n_f32(y0_fp32x4, beta);
		y1_fp32x4 = vmulq_n_f32(y1_fp32x4, beta);
		y2_fp32x4 = vmulq_n_f32(y2_fp32x4, beta);
		y3_fp32x4 = vmulq_n_f32(y3_fp32x4, beta);

		float32x4_t x0_fp32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(x0_u16x8), 16));
 		float32x4_t x1_fp32x4 = vreinterpretq_f32_u32(vshll_high_n_u16(x0_u16x8, 16));

		float32x4_t x2_fp32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(x1_u16x8), 16));
 		float32x4_t x3_fp32x4 = vreinterpretq_f32_u32(vshll_high_n_u16(x1_u16x8, 16));


		y0_fp32x4 = vfmaq_n_f32(y0_fp32x4, x0_fp32x4, alpha);
		y1_fp32x4 = vfmaq_n_f32(y1_fp32x4, x1_fp32x4, alpha);

		y2_fp32x4 = vfmaq_n_f32(y2_fp32x4, x2_fp32x4, alpha);
		y3_fp32x4 = vfmaq_n_f32(y3_fp32x4, x3_fp32x4, alpha);

		vst1q_f32(y,      y0_fp32x4);
		vst1q_f32(y + 4,  y1_fp32x4);
		vst1q_f32(y + 8,  y2_fp32x4);
		vst1q_f32(y + 12, y3_fp32x4);
	}


	constexpr kernel_inttype unroll1 = 4;

	for(;i <= ( n - unroll1 ); i += unroll1, x += unroll1, y += unroll1) {
		bfloat16x4_t x0_bf16x4 = vld1_bf16(x);
		float32x4_t  y0_fp32x4 = vld1q_f32(y);

		float32x4_t  x0_fp32x4 = vcvt_f32_bf16(x0_bf16x4);

		y0_fp32x4 = vmulq_n_f32(y0_fp32x4, beta);
		y0_fp32x4 = vfmaq_n_f32(y0_fp32x4, x0_fp32x4, alpha);

		vst1q_f32(y,     y0_fp32x4);
	}

	bfloat16x4_t x0_bf16x4;
	float32x4_t  x0_fp32x4;
	float32x4_t  y0_fp32x4;

	const kernel_inttype rem = n - i;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
	switch(rem) {
	case 0:
		return;
	default:
		PERFLIBS_ASSERT(false, "invalid path");

	case 3:
		x0_bf16x4 = vset_lane_bf16(x[2], x0_bf16x4, 2);
		y0_fp32x4 = vsetq_lane_f32(y[2], y0_fp32x4, 2);
	case 2:
		x0_bf16x4 = vset_lane_bf16(x[1], x0_bf16x4, 1);
		y0_fp32x4 = vsetq_lane_f32(y[1], y0_fp32x4, 1);
	case 1:
		x0_bf16x4 = vset_lane_bf16(x[0], x0_bf16x4, 0);
		y0_fp32x4 = vsetq_lane_f32(y[0], y0_fp32x4, 0);

	}
#pragma GCC diagnostic pop

	x0_fp32x4 = vcvt_f32_bf16(x0_bf16x4);
	y0_fp32x4 = vmulq_n_f32(y0_fp32x4, beta);
	y0_fp32x4 = vfmaq_n_f32(y0_fp32x4, x0_fp32x4, alpha);

	switch(rem) {
	case 3: y[2] = y0_fp32x4[2];
	case 2: y[1] = y0_fp32x4[1];
	case 1: y[0] = y0_fp32x4[0];
	}
}

} //extern "C"

namespace perflibs::linalg {

template<bool BFloat16Extensions, typename YDataType>
void axpby_bf16_bf16_r32_fmla_x16_ldd_acle(kernel_inttype n, float alpha, const __bf16 *x, float beta, YDataType *y, kernel_inttype incx, kernel_inttype incy) {
	PERFLIBS_ASSERT(incx == 1, "incx must be 1");
	PERFLIBS_ASSERT(incy == 1, "incy must be 1");

	constexpr kernel_inttype unroll0 = 16;

	const auto st4 = [](auto *ptr, auto v) {
		if constexpr (std::is_same_v<decltype(ptr), bf16*>) {
			if constexpr(BFloat16Extensions) {
				return vst1_bf16(ptr, vcvt_bf16_f32(v));
			}
			else {
				uint16x4_t bf16 = vshrn_n_u32(vreinterpretq_u32_f32(v), 16);
				vst1_u16(reinterpret_cast<std::uint16_t *>(ptr), bf16);
			}
		}
		else {
			vst1q_f32(ptr, v);
		}
	}; //st4

	auto load_n_of_4 = [](std::size_t n, const auto *ptr) {
		if constexpr (std::is_same_v<decltype(ptr), const bf16*>) {
			uint16x8_t out = vmovq_n_u16(0);
			switch(n) {
				case 4: out = vld1q_lane_u16(reinterpret_cast<const std::uint16_t*>(ptr+3), out, 7);
				case 3: out = vld1q_lane_u16(reinterpret_cast<const std::uint16_t*>(ptr+2), out, 5);
				case 2: out = vld1q_lane_u16(reinterpret_cast<const std::uint16_t*>(ptr+1), out, 3);
				case 1: out = vld1q_lane_u16(reinterpret_cast<const std::uint16_t*>(ptr+0), out, 1);
				default:
					break;
			}
			return vreinterpretq_f32_u16(out);
		}
		else {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
			float32x4_t out;
			switch(n) {
				case 4: out = vld1q_lane_f32(ptr+3, out, 3);
				case 3: out = vld1q_lane_f32(ptr+2, out, 2);
				case 2: out = vld1q_lane_f32(ptr+1, out, 1);
				case 1: out = vld1q_lane_f32(ptr+0, out, 0);
				default:
					break;
			}
			return out;
#pragma GCC diagnostic pop
		}
	}; //load_n_of_4

	const auto ld4 = [](const auto *ptr) {
		if constexpr (std::is_same_v<decltype(ptr), const bf16*>) {
			// the line below is equivelant to vcvt_f32_bf16(vld1_bf16(ptr)); but without +bf16
			return vreinterpretq_f32_u32(
				vshll_n_u16(
					vld1_u16(reinterpret_cast<const std::uint16_t *>(ptr)),
				16)
			);
		}
		else {
			return vld1q_f32(ptr);
		}
	}; //ld4

	auto store_n_of_4 = [](std::size_t n, auto *ptr, auto out) {
		if constexpr (std::is_same_v<decltype(ptr), bf16*>) {
			if constexpr(BFloat16Extensions) {
				auto out_bf16x4 = vcvt_bf16_f32(out);
				switch(n) {
					case 4: vst1_lane_bf16(ptr+3, out_bf16x4, 3);
					case 3: vst1_lane_bf16(ptr+2, out_bf16x4, 2);
					case 2: vst1_lane_bf16(ptr+1, out_bf16x4, 1);
					case 1: vst1_lane_bf16(ptr+0, out_bf16x4, 0);
					default:
						break;
				}
			}
			else {
				uint16x8_t out_u16 = vreinterpretq_u16_f32(out);
				switch(n) {
					case 4: vst1q_lane_u16(reinterpret_cast<std::uint16_t*>(ptr+3), out_u16, 7);
					case 3: vst1q_lane_u16(reinterpret_cast<std::uint16_t*>(ptr+2), out_u16, 5);
					case 2: vst1q_lane_u16(reinterpret_cast<std::uint16_t*>(ptr+1), out_u16, 3);
					case 1: vst1q_lane_u16(reinterpret_cast<std::uint16_t*>(ptr+0), out_u16, 1);
					default:
						break;
				}
			}
		}
		else {
			switch(n) {
				case 4: vst1q_lane_f32(ptr+3, out, 3);
				case 3: vst1q_lane_f32(ptr+2, out, 2);
				case 2: vst1q_lane_f32(ptr+1, out, 1);
				case 1: vst1q_lane_f32(ptr+0, out, 0);
				default:
					break;
			}
		}
	}; //load_n_of_4

	kernel_inttype i = 0;
	for(;i <= ( n - unroll0 ); i += unroll0, x += unroll0, y += unroll0) {
		float32x4_t x0_fp32x4 = ld4(x);
		float32x4_t y0_fp32x4 = ld4(y);

		float32x4_t x1_fp32x4 = ld4(x + 4);
		float32x4_t y1_fp32x4 = ld4(y + 4);

		float32x4_t x2_fp32x4 = ld4(x + 8);
		float32x4_t y2_fp32x4 = ld4(y + 8);

		float32x4_t x3_fp32x4 = ld4(x + 12);
		float32x4_t y3_fp32x4 = ld4(y + 12);

		y0_fp32x4 = vmulq_n_f32(y0_fp32x4, beta);
		y1_fp32x4 = vmulq_n_f32(y1_fp32x4, beta);
		y2_fp32x4 = vmulq_n_f32(y2_fp32x4, beta);
		y3_fp32x4 = vmulq_n_f32(y3_fp32x4, beta);

		y0_fp32x4 = vfmaq_n_f32(y0_fp32x4, x0_fp32x4, alpha);
		y1_fp32x4 = vfmaq_n_f32(y1_fp32x4, x1_fp32x4, alpha);

		y2_fp32x4 = vfmaq_n_f32(y2_fp32x4, x2_fp32x4, alpha);
		y3_fp32x4 = vfmaq_n_f32(y3_fp32x4, x3_fp32x4, alpha);

		st4(y,      y0_fp32x4);
		st4(y + 4,  y1_fp32x4);

		st4(y + 8,  y2_fp32x4);
		st4(y + 12, y3_fp32x4);
	}


	constexpr kernel_inttype unroll1 = 4;

	for(;i <= ( n - unroll1 ); i += unroll1, x += unroll1, y += unroll1) {
		float32x4_t x0_fp32x4 = ld4(x);
		float32x4_t y0_fp32x4 = ld4(y);

		y0_fp32x4 = vmulq_n_f32(y0_fp32x4, beta);
		y0_fp32x4 = vfmaq_n_f32(y0_fp32x4, x0_fp32x4, alpha);

		st4(y, y0_fp32x4);
	}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
	if(const kernel_inttype rem = n - i; rem > 0) {
		auto x0_fp32x4 = load_n_of_4(rem, x);
		auto y0_fp32x4 = load_n_of_4(rem, y);

		y0_fp32x4 = vmulq_n_f32(y0_fp32x4, beta);
		y0_fp32x4 = vfmaq_n_f32(y0_fp32x4, x0_fp32x4, alpha);

		store_n_of_4(rem, y, y0_fp32x4);
	}
#pragma GCC diagnostic pop
}

extern "C" {

void sbaxpby_kernel(kernel_inttype n, r32 alpha, const bf16 *x, r32 beta, r32 *y, kernel_inttype incx, kernel_inttype incy) {
	axpby_bf16_bf16_r32_fmla_x16_ldd_acle<false>(n, alpha, x, beta, y, incx, incy);
}

void baxpby_kernel_nobf16exten(kernel_inttype n, bf16 alpha, const bf16 *x, bf16 beta, bf16 *y, kernel_inttype incx, kernel_inttype incy) {
	axpby_bf16_bf16_r32_fmla_x16_ldd_acle<false>(n, alpha, x, beta, y, incx, incy);
}

void baxpby_kernel_bf16exten(kernel_inttype n, bf16 alpha, const bf16 *x, bf16 beta, bf16 *y, kernel_inttype incx, kernel_inttype incy) {
	axpby_bf16_bf16_r32_fmla_x16_ldd_acle<true>(n, alpha, x, beta, y, incx, incy);
}

void baxpby_kernel(kernel_inttype n, bf16 alpha, const bf16 *x, bf16 beta, bf16 *y, kernel_inttype incx, kernel_inttype incy) {
	if(machine::cpu_info::get_cpu_features().neon_bf16) {
		baxpby_kernel_bf16exten(n, alpha, x, beta, y, incx, incy);
	}
	else {
		baxpby_kernel_nobf16exten(n, alpha, x, beta, y, incx, incy);
	}
}

} // extern "C"

} // namespace perflibs::linalg

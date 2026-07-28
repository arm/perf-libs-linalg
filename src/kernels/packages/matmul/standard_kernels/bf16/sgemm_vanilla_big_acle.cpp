/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "framework/linalg_util.hpp"

#include "perflibs_util.hpp"
#include "perflibs_float.hpp"

#include <arm_neon.h>
#include <algorithm>
#include <cstddef>

/*
 * ACLE equivalent of sgemm_vanilla_big assembly kernel.
 *
 * Expected packed layout (matching the assembly kernel):
 * - A is stored as contiguous K slices of 12 elements (k-major, 12-row panel)
 * - B is stored as contiguous K slices of 8 elements (k-major, 8-col panel)
 * - C is column-major with leading dimension ldc (in elements)
 */

namespace perflibs::linalg {

namespace {

constexpr std::size_t kMr = 12zu;
constexpr std::size_t kNr = 8zu;
constexpr std::size_t kVecRows = 4zu;
constexpr std::size_t kMrVecs = kMr / kVecRows;

inline __attribute__((always_inline))
auto load_4s(const r32 *p) {
	return vld1q_f32(p);
}

inline __attribute__((always_inline))
auto load_4s(const bf16 *p) {
	//code below is equivelant to return vcvt_f32_bf16( vld1_bf16(p) ); without +bf16
	return vreinterpretq_f32_u32(
		vshll_n_u16(vld1_u16(
			reinterpret_cast<const uint16_t*>(p)),
		16)
	);
}

template<bool BFloat16Extensions>
inline __attribute__((always_inline))
void store_4s(bf16 *ptr, auto v) {
	if constexpr(BFloat16Extensions) {
		vst1_bf16(ptr, vcvt_bf16_f32(v));
	}
	else {
		uint16x4_t bf16 = vshrn_n_u32(vreinterpretq_u32_f32(v), 16);
		vst1_u16(reinterpret_cast<std::uint16_t *>(ptr), bf16);
	}
}

template<bool BFloat16Extensions>
inline __attribute__((always_inline))
void store_4s(r32 *ptr, auto v) {
	vst1q_f32(ptr, v);
}

template<bool BFloat16Extensions>
inline __attribute__((always_inline))
auto store_n_of_4(std::size_t n, float32x4_t out, r32 *ptr)  {
	switch(n) {
		default:
			store_4s<BFloat16Extensions>(ptr, out);
			break;

		case 3: vst1q_lane_f32(ptr+2, out, 2);
		case 2: vst1q_lane_f32(ptr+1, out, 1);
		case 1: vst1q_lane_f32(ptr+0, out, 0);
		case 0:
			break;
	}
}

template<bool BFloat16Extensions>
inline __attribute__((always_inline))
auto store_n_of_4(std::size_t n, float32x4_t out, bf16 *ptr)  {
	if constexpr(BFloat16Extensions) {
		auto out_bf16x4 = vcvt_bf16_f32(out);
		switch(n) {
			default:
				vst1_bf16(ptr, out_bf16x4);
				break;

			case 3: vst1_lane_bf16(ptr+2, out_bf16x4, 2);
			case 2: vst1_lane_bf16(ptr+1, out_bf16x4, 1);
			case 1: vst1_lane_bf16(ptr+0, out_bf16x4, 0);
			case 0:
				break;
		}
	}
	else {
		uint16x8_t out_u16 = vreinterpretq_u16_f32(out);
		switch(n) {
			default:
				store_4s<BFloat16Extensions>(ptr, out);
				break;

			case 3: vst1q_lane_u16(reinterpret_cast<std::uint16_t*>(ptr+2), out_u16, 5);
			case 2: vst1q_lane_u16(reinterpret_cast<std::uint16_t*>(ptr+1), out_u16, 3);
			case 1: vst1q_lane_u16(reinterpret_cast<std::uint16_t*>(ptr+0), out_u16, 1);
			case 0:
				break;
		}
	}
}

inline __attribute__((always_inline))
auto load_n_of_4(std::size_t n, const r32 *ptr)  {
	float32x4_t out = vdupq_n_f32(0.0f);
	switch(n) {
		default:
			return load_4s(ptr);

		case 3: out = vld1q_lane_f32(ptr+2, out, 2);
		case 2: out = vld1q_lane_f32(ptr+1, out, 1);
		case 1: out = vld1q_lane_f32(ptr+0, out, 0);
		case 0:
			break;
	}
	return out;
}


inline __attribute__((always_inline))
auto load_n_of_4(std::size_t n, const bf16 *ptr)  {
	uint16x8_t out = vdupq_n_u16(0);
	switch(n) {
		default:
			return load_4s(ptr);

		case 3: out = vld1q_lane_u16(reinterpret_cast<const std::uint16_t*>(ptr+2), out, 5);
		case 2: out = vld1q_lane_u16(reinterpret_cast<const std::uint16_t*>(ptr+1), out, 3);
		case 1: out = vld1q_lane_u16(reinterpret_cast<const std::uint16_t*>(ptr+0), out, 1);
		case 0:
			break;
	}
	return vreinterpretq_f32_u16(out);
}

template<bool BFloat16Extensions>
inline void store_col(auto *c_col, std::size_t mblk, float32x4_t acc0, float32x4_t acc1, float32x4_t acc2, float alpha, float beta) {
	float32x4_t c0, c1, c2;
	auto *c1_col = c_col + std::min(4zu, mblk);
	auto *c2_col = c_col + std::min(8zu, mblk);

	if(beta == zero<>) {
		c0 = vdupq_n_f32(0.0f);
		c1 = vdupq_n_f32(0.0f);
		c2 = vdupq_n_f32(0.0f);
	}
	else {
		c0 = load_n_of_4(mblk,       c_col);
		c1 = load_n_of_4(max(4zu, mblk) - 4zu, c1_col);
		c2 = load_n_of_4(max(8zu, mblk) - 8zu, c2_col);

		if(beta != one<>) {
			c0 = vmulq_n_f32(c0, beta);
			c1 = vmulq_n_f32(c1, beta);
			c2 = vmulq_n_f32(c2, beta);
		}
	}

	if(alpha != one<>) {
		c0 = vfmaq_n_f32(c0, acc0, alpha);
		c1 = vfmaq_n_f32(c1, acc1, alpha);
		c2 = vfmaq_n_f32(c2, acc2, alpha);
	}
	else {
		c0 = vaddq_f32(c0, acc0);
		c1 = vaddq_f32(c1, acc1);
		c2 = vaddq_f32(c2, acc2);
	}

	store_n_of_4<BFloat16Extensions>(mblk,                 c0, c_col);
	store_n_of_4<BFloat16Extensions>(max(4zu, mblk) - 4zu, c1, c1_col);
	store_n_of_4<BFloat16Extensions>(max(8zu, mblk) - 8zu, c2, c2_col);
}

} //namespace <anon>


template<bool BFloat16Extensions>
void gemm_fp32_fmla_12x8_acle(const auto *a, const auto *b, auto *c,
		kernel_inttype k, kernel_inttype m, kernel_inttype n, kernel_inttype ldc,
		float alpha, float beta) {

	if (k <= 0 || m <= 0 || n <= 0) {
		return;
	}

	const auto ks = static_cast<std::size_t>(k);
	const auto ms = static_cast<std::size_t>(m);
	const auto ns = static_cast<std::size_t>(n);
	const auto ldcs = static_cast<std::size_t>(ldc);


	for (std::size_t b_row_idx = 0; b_row_idx < ns; b_row_idx += kNr) {
		const auto nblk = std::min(kNr, ns - b_row_idx);
		const auto *b_panel = b + b_row_idx * ks;

		for (std::size_t a_row_idx = 0; a_row_idx < ms; a_row_idx += kMr) {
			const auto mblk = std::min(kMr, ms - a_row_idx);
			const auto *a_panel = a + a_row_idx * ks;

			float32x4_t acc[kNr][kMrVecs];
			for (std::size_t col = 0; col < kNr; ++col) {
				acc[col][0] = vdupq_n_f32(0.0f);
				acc[col][1] = vdupq_n_f32(0.0f);
				acc[col][2] = vdupq_n_f32(0.0f);
			}

			for (std::size_t p = 0; p < ks; ++p) {
				const auto *a_k = a_panel + p * kMr;
				const auto *b_k = b_panel + p * kNr;

				const float32x4_t a0 = load_4s(a_k + 0);
				const float32x4_t a1 = load_4s(a_k + 4);
				const float32x4_t a2 = load_4s(a_k + 8);

				const float32x4_t b0 = load_4s(b_k + 0);
				const float32x4_t b1 = load_4s(b_k + 4);

				acc[0][0] = vfmaq_laneq_f32(acc[0][0], a0, b0, 0);
				acc[0][1] = vfmaq_laneq_f32(acc[0][1], a1, b0, 0);
				acc[0][2] = vfmaq_laneq_f32(acc[0][2], a2, b0, 0);

				acc[1][0] = vfmaq_laneq_f32(acc[1][0], a0, b0, 1);
				acc[1][1] = vfmaq_laneq_f32(acc[1][1], a1, b0, 1);
				acc[1][2] = vfmaq_laneq_f32(acc[1][2], a2, b0, 1);

				acc[2][0] = vfmaq_laneq_f32(acc[2][0], a0, b0, 2);
				acc[2][1] = vfmaq_laneq_f32(acc[2][1], a1, b0, 2);
				acc[2][2] = vfmaq_laneq_f32(acc[2][2], a2, b0, 2);

				acc[3][0] = vfmaq_laneq_f32(acc[3][0], a0, b0, 3);
				acc[3][1] = vfmaq_laneq_f32(acc[3][1], a1, b0, 3);
				acc[3][2] = vfmaq_laneq_f32(acc[3][2], a2, b0, 3);

				acc[4][0] = vfmaq_laneq_f32(acc[4][0], a0, b1, 0);
				acc[4][1] = vfmaq_laneq_f32(acc[4][1], a1, b1, 0);
				acc[4][2] = vfmaq_laneq_f32(acc[4][2], a2, b1, 0);

				acc[5][0] = vfmaq_laneq_f32(acc[5][0], a0, b1, 1);
				acc[5][1] = vfmaq_laneq_f32(acc[5][1], a1, b1, 1);
				acc[5][2] = vfmaq_laneq_f32(acc[5][2], a2, b1, 1);

				acc[6][0] = vfmaq_laneq_f32(acc[6][0], a0, b1, 2);
				acc[6][1] = vfmaq_laneq_f32(acc[6][1], a1, b1, 2);
				acc[6][2] = vfmaq_laneq_f32(acc[6][2], a2, b1, 2);

				acc[7][0] = vfmaq_laneq_f32(acc[7][0], a0, b1, 3);
				acc[7][1] = vfmaq_laneq_f32(acc[7][1], a1, b1, 3);
				acc[7][2] = vfmaq_laneq_f32(acc[7][2], a2, b1, 3);
			}

			for (std::size_t col = 0; col < nblk; ++col) {
				auto *c_col = c + (b_row_idx + col) * ldcs + a_row_idx;
				store_col<BFloat16Extensions>(c_col, mblk, acc[col][0], acc[col][1], acc[col][2], alpha, beta);
			}
		}
	}
}

extern "C" {
	void sgemm_vanilla_big_acle(const r32 *a, const r32 *b, r32 *c,
		kernel_inttype k, kernel_inttype m, kernel_inttype n, kernel_inttype ldc, r32 alpha, r32 beta) {
		return gemm_fp32_fmla_12x8_acle<false>(a, b, c, k, m, n, ldc, alpha, beta);
	}

	void sbgemm_fp32_fmla_acle(const bf16 *a, const bf16 *b, r32 *c,
		kernel_inttype k, kernel_inttype m, kernel_inttype n, kernel_inttype ldc, r32 alpha, r32 beta) {
		return gemm_fp32_fmla_12x8_acle<false>(a, b, c, k, m, n, ldc, alpha, beta);
	}

	void bgemm_fp32_fmla_acle_nobf16exten(const bf16 *a, const bf16 *b, bf16 *c,
		kernel_inttype k, kernel_inttype m, kernel_inttype n, kernel_inttype ldc, bf16 alpha, bf16 beta) {
		return gemm_fp32_fmla_12x8_acle<false>(a, b, c, k, m, n, ldc, alpha, beta);
	}

	void bgemm_fp32_fmla_acle_bf16exten(const bf16 *a, const bf16 *b, bf16 *c,
		kernel_inttype k, kernel_inttype m, kernel_inttype n, kernel_inttype ldc, bf16 alpha, bf16 beta) {
		return gemm_fp32_fmla_12x8_acle<true>(a, b, c, k, m, n, ldc, alpha, beta);
	}

} //extern "C"

} //namespace perflibs::linalg

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include <arm_neon.h>

#include <algorithm>
#include <cstddef>

namespace {

template <std::size_t Width>
inline void transpose_4x4(const float *src, std::size_t ld, float *dst) {
	const auto row0 = vld1q_f32(src);
	const auto row1 = vld1q_f32(src + ld);
	const auto row2 = vld1q_f32(src + 2 * ld);
	const auto row3 = vld1q_f32(src + 3 * ld);

	const auto even01 = vreinterpretq_u64_f32(vtrn1q_f32(row0, row1));
	const auto odd01 = vreinterpretq_u64_f32(vtrn2q_f32(row0, row1));
	const auto even23 = vreinterpretq_u64_f32(vtrn1q_f32(row2, row3));
	const auto odd23 = vreinterpretq_u64_f32(vtrn2q_f32(row2, row3));

	const auto col0 = vreinterpretq_f32_u64(vtrn1q_u64(even01, even23));
	const auto col1 = vreinterpretq_f32_u64(vtrn1q_u64(odd01, odd23));
	const auto col2 = vreinterpretq_f32_u64(vtrn2q_u64(even01, even23));
	const auto col3 = vreinterpretq_f32_u64(vtrn2q_u64(odd01, odd23));

	vst1q_f32(dst, col0);
	vst1q_f32(dst + Width, col1);
	vst1q_f32(dst + 2 * Width, col2);
	vst1q_f32(dst + 3 * Width, col3);
}

template <std::size_t Width>
inline void transpose_4x2(const float *src, std::size_t ld, float *dst) {
	const auto row0 = vld1_f32(src);
	const auto row1 = vld1_f32(src + ld);
	const auto row2 = vld1_f32(src + 2 * ld);
	const auto row3 = vld1_f32(src + 3 * ld);

	const auto pair01 = vtrn_f32(row0, row1);
	const auto pair23 = vtrn_f32(row2, row3);

	vst1q_f32(dst, vcombine_f32(pair01.val[0], pair23.val[0]));
	vst1q_f32(dst + Width, vcombine_f32(pair01.val[1], pair23.val[1]));
}

template <std::size_t Width, std::size_t Rows, std::size_t Cols>
inline void copy_partial_block(const float *src, std::size_t src_ld, float *dst, std::size_t k) {
	static_assert(Cols == 2 || Cols == 4);
	std::size_t row = 0;
	for (; row + 4 <= Rows; row += 4) {
		if constexpr (Cols == 4) {
			transpose_4x4<Width>(src + k + row * src_ld, src_ld, dst + k * Width + row);
		} else {
			transpose_4x2<Width>(src + k + row * src_ld, src_ld, dst + k * Width + row);
		}
	}
	for (; row < Rows; ++row) {
		for (std::size_t col = 0; col < Cols; ++col) {
			dst[(k + col) * Width + row] = src[k + col + row * src_ld];
		}
	}
}

template <std::size_t Width, std::size_t Rows>
inline void copy_rows(std::size_t cntg, std::size_t dst_cntg,
	const float *src, std::size_t src_ld, float *dst) {
	std::size_t k = 0;
	if constexpr (Rows >= 4) {
		for (; k + 4 <= cntg; k += 4) {
			copy_partial_block<Width, Rows, 4>(src, src_ld, dst, k);
		}
		if (k + 2 <= cntg) {
			copy_partial_block<Width, Rows, 2>(src, src_ld, dst, k);
			k += 2;
		}
	}
	for (; k < cntg; ++k) {
		for (std::size_t j = 0; j < Rows; ++j) {
			dst[k * Width + j] = src[k + j * src_ld];
		}
	}
	if constexpr (Rows == Width) {
		std::fill(dst + cntg * Width, dst + dst_cntg * Width, 0.0f);
	} else {
		for (std::size_t k = cntg; k < dst_cntg; ++k) {
			for (std::size_t j = 0; j < Rows; ++j) {
				dst[k * Width + j] = 0.0f;
			}
		}
	}
}

} // namespace

extern "C" void n_interleave_kernel_s12(
	std::size_t src_cntg, std::size_t src_strd, const float *__restrict src, std::size_t src_ld,
	std::size_t dst_cntg, std::size_t dst_strd, float *__restrict dst, std::size_t dst_ld) {
	const auto cntg = std::min(src_cntg, dst_cntg);
	const auto strd = std::min(src_strd, dst_strd);
	std::size_t row = 0;
	for (; row + 12 <= strd; row += 12, dst += dst_ld) {
		const auto panel = src + row * src_ld;
		copy_rows<12, 12>(cntg, dst_cntg, panel, src_ld, dst);
	}
	const auto remaining_rows = strd - row;
	if (remaining_rows == 0) {
		return;
	}
	const auto tail = cntg ? src + row * src_ld : src;
	switch (remaining_rows) {
	case 1: return copy_rows<12, 1>(cntg, dst_cntg, tail, src_ld, dst);
	case 2: return copy_rows<12, 2>(cntg, dst_cntg, tail, src_ld, dst);
	case 3: return copy_rows<12, 3>(cntg, dst_cntg, tail, src_ld, dst);
	case 4: return copy_rows<12, 4>(cntg, dst_cntg, tail, src_ld, dst);
	case 5: return copy_rows<12, 5>(cntg, dst_cntg, tail, src_ld, dst);
	case 6: return copy_rows<12, 6>(cntg, dst_cntg, tail, src_ld, dst);
	case 7: return copy_rows<12, 7>(cntg, dst_cntg, tail, src_ld, dst);
	case 8: return copy_rows<12, 8>(cntg, dst_cntg, tail, src_ld, dst);
	case 9: return copy_rows<12, 9>(cntg, dst_cntg, tail, src_ld, dst);
	case 10: return copy_rows<12, 10>(cntg, dst_cntg, tail, src_ld, dst);
	case 11: return copy_rows<12, 11>(cntg, dst_cntg, tail, src_ld, dst);
	default: return;
	}
}

extern "C" void n_interleave_kernel_s16(
	std::size_t src_cntg, std::size_t src_strd, const float *__restrict src, std::size_t src_ld,
	std::size_t dst_cntg, std::size_t dst_strd, float *__restrict dst, std::size_t dst_ld) {
	const auto cntg = std::min(src_cntg, dst_cntg);
	const auto strd = std::min(src_strd, dst_strd);
	if (dst_cntg == 0 || strd == 0) {
		return;
	}
	std::size_t row = 0;
	for (; row + 16 <= strd; row += 16, dst += dst_ld) {
		const auto panel = cntg ? src + row * src_ld : src;
		copy_rows<16, 16>(cntg, dst_cntg, panel, src_ld, dst);
	}
	const auto remaining_rows = strd - row;
	if (remaining_rows == 0) {
		return;
	}
	const auto tail = cntg ? src + row * src_ld : src;
	switch (remaining_rows) {
	case 1: return copy_rows<16, 1>(cntg, dst_cntg, tail, src_ld, dst);
	case 2: return copy_rows<16, 2>(cntg, dst_cntg, tail, src_ld, dst);
	case 3: return copy_rows<16, 3>(cntg, dst_cntg, tail, src_ld, dst);
	case 4: return copy_rows<16, 4>(cntg, dst_cntg, tail, src_ld, dst);
	case 5: return copy_rows<16, 5>(cntg, dst_cntg, tail, src_ld, dst);
	case 6: return copy_rows<16, 6>(cntg, dst_cntg, tail, src_ld, dst);
	case 7: return copy_rows<16, 7>(cntg, dst_cntg, tail, src_ld, dst);
	case 8: return copy_rows<16, 8>(cntg, dst_cntg, tail, src_ld, dst);
	case 9: return copy_rows<16, 9>(cntg, dst_cntg, tail, src_ld, dst);
	case 10: return copy_rows<16, 10>(cntg, dst_cntg, tail, src_ld, dst);
	case 11: return copy_rows<16, 11>(cntg, dst_cntg, tail, src_ld, dst);
	case 12: return copy_rows<16, 12>(cntg, dst_cntg, tail, src_ld, dst);
	case 13: return copy_rows<16, 13>(cntg, dst_cntg, tail, src_ld, dst);
	case 14: return copy_rows<16, 14>(cntg, dst_cntg, tail, src_ld, dst);
	case 15: return copy_rows<16, 15>(cntg, dst_cntg, tail, src_ld, dst);
	default: return;
	}
}

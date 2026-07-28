/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FRAMEWORK_TILE_INTERLEAVE_HPP
#define PERFLIBS_LINALG_FRAMEWORK_TILE_INTERLEAVE_HPP

#include "framework/linalg_util.hpp"

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace perflibs::linalg {

template<kernel_inttype CntgInterleave, kernel_inttype StrdInterleave, typename SrcDataType, typename DstDataType, typename SrcCntgStepType, typename SrcStrdStepType>
PERFLIBS_LINALG_INLINE
void tile_interleave_impl(
	std::size_t src_cntg, std::size_t src_strd, const SrcDataType *src, SrcCntgStepType src_cntg_step, SrcStrdStepType src_strd_step,
	std::size_t dst_cntg, std::size_t dst_strd,       DstDataType *dst,                                std::size_t     dst_strd_step) {


	constexpr kernel_inttype cntg_interleave_step = 1_ki;
	constexpr kernel_inttype strd_interleave_step = CntgInterleave;

	constexpr kernel_inttype cntg_tile_step = CntgInterleave * StrdInterleave;
	const     kernel_inttype strd_tile_step = dst_strd_step;


	auto full_cntg_tile_interleave = [&](auto& src, auto& dst, auto strd) {

		constexpr bool can_optimise = std::is_same_v<SrcCntgStepType, step_val_fixed<1>>
		                           && std::is_same_v<decltype(strd), step_val_fixed<StrdInterleave> >
		                           && (CntgInterleave * sizeof(DstDataType)) == sizeof(std::uint64_t)
		                           && (CntgInterleave * sizeof(SrcDataType)) == sizeof(std::uint64_t);

		if constexpr(can_optimise && StrdInterleave <= 12_ki) {
			if constexpr(StrdInterleave >=  1_ki) std::memcpy(dst + 0 * strd_interleave_step, src + 0 * src_strd_step, sizeof(std::uint64_t));
			if constexpr(StrdInterleave >=  2_ki) std::memcpy(dst + 1 * strd_interleave_step, src + 1 * src_strd_step, sizeof(std::uint64_t));
			if constexpr(StrdInterleave >=  3_ki) std::memcpy(dst + 2 * strd_interleave_step, src + 2 * src_strd_step, sizeof(std::uint64_t));
			if constexpr(StrdInterleave >=  4_ki) std::memcpy(dst + 3 * strd_interleave_step, src + 3 * src_strd_step, sizeof(std::uint64_t));
			if constexpr(StrdInterleave >=  5_ki) std::memcpy(dst + 4 * strd_interleave_step, src + 4 * src_strd_step, sizeof(std::uint64_t));
			if constexpr(StrdInterleave >=  6_ki) std::memcpy(dst + 5 * strd_interleave_step, src + 5 * src_strd_step, sizeof(std::uint64_t));
			if constexpr(StrdInterleave >=  7_ki) std::memcpy(dst + 6 * strd_interleave_step, src + 6 * src_strd_step, sizeof(std::uint64_t));
			if constexpr(StrdInterleave >=  8_ki) std::memcpy(dst + 7 * strd_interleave_step, src + 7 * src_strd_step, sizeof(std::uint64_t));
			if constexpr(StrdInterleave >=  9_ki) std::memcpy(dst + 8 * strd_interleave_step, src + 8 * src_strd_step, sizeof(std::uint64_t));
			if constexpr(StrdInterleave >= 10_ki) std::memcpy(dst + 9 * strd_interleave_step, src + 9 * src_strd_step, sizeof(std::uint64_t));
			if constexpr(StrdInterleave >= 11_ki) std::memcpy(dst + 10 * strd_interleave_step, src + 10 * src_strd_step, sizeof(std::uint64_t));
			if constexpr(StrdInterleave >= 12_ki) std::memcpy(dst + 11 * strd_interleave_step, src + 11 * src_strd_step, sizeof(std::uint64_t));

			return;
		}
		else if constexpr(CntgInterleave == 4_ki
		               && sizeof(SrcDataType) == sizeof(std::uint16_t)
		               && sizeof(DstDataType) == sizeof(std::uint16_t)
		               && std::is_same_v<SrcStrdStepType, step_val_fixed<1>>
		               && std::is_same_v<decltype(strd),  step_val_fixed<StrdInterleave> >) {

			if constexpr(StrdInterleave == 4_ki) {
				uint16x4_t src0 = vreinterpret_u16_u8(vld1_u8(reinterpret_cast<const std::uint8_t*>( src + 0 * src_cntg_step ) ) );
				uint16x4_t src1 = vreinterpret_u16_u8(vld1_u8(reinterpret_cast<const std::uint8_t*>( src + 1 * src_cntg_step ) ) );
				uint16x4_t src2 = vreinterpret_u16_u8(vld1_u8(reinterpret_cast<const std::uint8_t*>( src + 2 * src_cntg_step ) ) );
				uint16x4_t src3 = vreinterpret_u16_u8(vld1_u8(reinterpret_cast<const std::uint8_t*>( src + 3 * src_cntg_step ) ) );

				uint16x4x2_t t01 = vtrn_u16(src0, src1);
				uint16x4x2_t t23 = vtrn_u16(src2, src3);

				uint32x2x2_t r02 = vtrn_u32(vreinterpret_u32_u16(t01.val[0]), vreinterpret_u32_u16(t23.val[0]));
				uint32x2x2_t r13 = vtrn_u32(vreinterpret_u32_u16(t01.val[1]), vreinterpret_u32_u16(t23.val[1]));

				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 0 * strd_interleave_step ), vreinterpret_u8_u32(r02.val[0]));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 1 * strd_interleave_step ), vreinterpret_u8_u32(r13.val[0]));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 2 * strd_interleave_step ), vreinterpret_u8_u32(r02.val[1]));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 3 * strd_interleave_step ), vreinterpret_u8_u32(r13.val[1]));

				return;
			}
			else if constexpr(StrdInterleave == 6_ki) {
				uint16x4_t src0 = vreinterpret_u16_u8(vld1_u8(reinterpret_cast<const std::uint8_t*>( src + 0 * src_cntg_step ) ) );
				uint16x4_t src1 = vreinterpret_u16_u8(vld1_u8(reinterpret_cast<const std::uint8_t*>( src + 1 * src_cntg_step ) ) );
				uint16x4_t src2 = vreinterpret_u16_u8(vld1_u8(reinterpret_cast<const std::uint8_t*>( src + 2 * src_cntg_step ) ) );
				uint16x4_t src3 = vreinterpret_u16_u8(vld1_u8(reinterpret_cast<const std::uint8_t*>( src + 3 * src_cntg_step ) ) );

				uint16x4x2_t t01 = vtrn_u16(src0, src1);
				uint16x4x2_t t23 = vtrn_u16(src2, src3);

				uint32x2x2_t r02 = vtrn_u32(vreinterpret_u32_u16(t01.val[0]), vreinterpret_u32_u16(t23.val[0]));
				uint32x2x2_t r13 = vtrn_u32(vreinterpret_u32_u16(t01.val[1]), vreinterpret_u32_u16(t23.val[1]));

				uint8x8_t src02_tail_bytes = vdup_n_u8(0);
				src02_tail_bytes = vld1_lane_u8(reinterpret_cast<const std::uint8_t*>( src + 0 * src_cntg_step + 4 ) + 0, src02_tail_bytes, 0);
				src02_tail_bytes = vld1_lane_u8(reinterpret_cast<const std::uint8_t*>( src + 0 * src_cntg_step + 4 ) + 1, src02_tail_bytes, 1);
				src02_tail_bytes = vld1_lane_u8(reinterpret_cast<const std::uint8_t*>( src + 0 * src_cntg_step + 4 ) + 2, src02_tail_bytes, 2);
				src02_tail_bytes = vld1_lane_u8(reinterpret_cast<const std::uint8_t*>( src + 0 * src_cntg_step + 4 ) + 3, src02_tail_bytes, 3);
				src02_tail_bytes = vld1_lane_u8(reinterpret_cast<const std::uint8_t*>( src + 2 * src_cntg_step + 4 ) + 0, src02_tail_bytes, 4);
				src02_tail_bytes = vld1_lane_u8(reinterpret_cast<const std::uint8_t*>( src + 2 * src_cntg_step + 4 ) + 1, src02_tail_bytes, 5);
				src02_tail_bytes = vld1_lane_u8(reinterpret_cast<const std::uint8_t*>( src + 2 * src_cntg_step + 4 ) + 2, src02_tail_bytes, 6);
				src02_tail_bytes = vld1_lane_u8(reinterpret_cast<const std::uint8_t*>( src + 2 * src_cntg_step + 4 ) + 3, src02_tail_bytes, 7);
				uint32x2_t src02_tail = vreinterpret_u32_u8(src02_tail_bytes);

				uint8x8_t src13_tail_bytes = vdup_n_u8(0);
				src13_tail_bytes = vld1_lane_u8(reinterpret_cast<const std::uint8_t*>( src + 1 * src_cntg_step + 4 ) + 0, src13_tail_bytes, 0);
				src13_tail_bytes = vld1_lane_u8(reinterpret_cast<const std::uint8_t*>( src + 1 * src_cntg_step + 4 ) + 1, src13_tail_bytes, 1);
				src13_tail_bytes = vld1_lane_u8(reinterpret_cast<const std::uint8_t*>( src + 1 * src_cntg_step + 4 ) + 2, src13_tail_bytes, 2);
				src13_tail_bytes = vld1_lane_u8(reinterpret_cast<const std::uint8_t*>( src + 1 * src_cntg_step + 4 ) + 3, src13_tail_bytes, 3);
				src13_tail_bytes = vld1_lane_u8(reinterpret_cast<const std::uint8_t*>( src + 3 * src_cntg_step + 4 ) + 0, src13_tail_bytes, 4);
				src13_tail_bytes = vld1_lane_u8(reinterpret_cast<const std::uint8_t*>( src + 3 * src_cntg_step + 4 ) + 1, src13_tail_bytes, 5);
				src13_tail_bytes = vld1_lane_u8(reinterpret_cast<const std::uint8_t*>( src + 3 * src_cntg_step + 4 ) + 2, src13_tail_bytes, 6);
				src13_tail_bytes = vld1_lane_u8(reinterpret_cast<const std::uint8_t*>( src + 3 * src_cntg_step + 4 ) + 3, src13_tail_bytes, 7);
				uint32x2_t src13_tail = vreinterpret_u32_u8(src13_tail_bytes);

				uint16x4x2_t tail = vtrn_u16(vreinterpret_u16_u32(src02_tail), vreinterpret_u16_u32(src13_tail));

				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 0 * strd_interleave_step ), vreinterpret_u8_u32(r02.val[0]));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 1 * strd_interleave_step ), vreinterpret_u8_u32(r13.val[0]));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 2 * strd_interleave_step ), vreinterpret_u8_u32(r02.val[1]));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 3 * strd_interleave_step ), vreinterpret_u8_u32(r13.val[1]));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 4 * strd_interleave_step ), vreinterpret_u8_u32(vreinterpret_u32_u16(tail.val[0])));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 5 * strd_interleave_step ), vreinterpret_u8_u32(vreinterpret_u32_u16(tail.val[1])));

				return;
			}
			if constexpr(StrdInterleave == 8_ki) {
				uint16x8_t src0 = vreinterpretq_u16_u8(vld1q_u8(reinterpret_cast<const std::uint8_t*>( src + 0 * src_cntg_step ) ) );
				uint16x8_t src1 = vreinterpretq_u16_u8(vld1q_u8(reinterpret_cast<const std::uint8_t*>( src + 1 * src_cntg_step ) ) );
				uint16x8_t src2 = vreinterpretq_u16_u8(vld1q_u8(reinterpret_cast<const std::uint8_t*>( src + 2 * src_cntg_step ) ) );
				uint16x8_t src3 = vreinterpretq_u16_u8(vld1q_u8(reinterpret_cast<const std::uint8_t*>( src + 3 * src_cntg_step ) ) );

				uint16x8x2_t t01 = vtrnq_u16(src0, src1);
				uint16x8x2_t t23 = vtrnq_u16(src2, src3);

				uint32x4x2_t r02 = vtrnq_u32(vreinterpretq_u32_u16(t01.val[0]), vreinterpretq_u32_u16(t23.val[0]));
				uint32x4x2_t r13 = vtrnq_u32(vreinterpretq_u32_u16(t01.val[1]), vreinterpretq_u32_u16(t23.val[1]));

				uint64x2_t r0_u64x2 = vreinterpretq_u64_u32(r02.val[0]);
				uint64x2_t r2_u64x2 = vreinterpretq_u64_u32(r02.val[1]);
				uint64x2_t r1_u64x2 = vreinterpretq_u64_u32(r13.val[0]);
				uint64x2_t r3_u64x2 = vreinterpretq_u64_u32(r13.val[1]);

				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 0 * strd_interleave_step ), vget_low_u8(vreinterpretq_u8_u64(r0_u64x2)));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 1 * strd_interleave_step ), vget_low_u8(vreinterpretq_u8_u64(r1_u64x2)));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 2 * strd_interleave_step ), vget_low_u8(vreinterpretq_u8_u64(r2_u64x2)));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 3 * strd_interleave_step ), vget_low_u8(vreinterpretq_u8_u64(r3_u64x2)));

				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 4 * strd_interleave_step ), vget_high_u8(vreinterpretq_u8_u64(r0_u64x2)));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 5 * strd_interleave_step ), vget_high_u8(vreinterpretq_u8_u64(r1_u64x2)));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 6 * strd_interleave_step ), vget_high_u8(vreinterpretq_u8_u64(r2_u64x2)));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 7 * strd_interleave_step ), vget_high_u8(vreinterpretq_u8_u64(r3_u64x2)));

				return;
			}
			else if constexpr(StrdInterleave == 12_ki) {
				uint16x8_t src00 = vreinterpretq_u16_u8(vld1q_u8(reinterpret_cast<const std::uint8_t*>( src + 0 * src_cntg_step ) ) );
				uint16x8_t src01 = vreinterpretq_u16_u8(vld1q_u8(reinterpret_cast<const std::uint8_t*>( src + 1 * src_cntg_step ) ) );
				uint16x8_t src02 = vreinterpretq_u16_u8(vld1q_u8(reinterpret_cast<const std::uint8_t*>( src + 2 * src_cntg_step ) ) );
				uint16x8_t src03 = vreinterpretq_u16_u8(vld1q_u8(reinterpret_cast<const std::uint8_t*>( src + 3 * src_cntg_step ) ) );

				uint16x4_t src10 = vreinterpret_u16_u8(vld1_u8(reinterpret_cast<const std::uint8_t*>( src + 0 * src_cntg_step + 8 ) ) );
				uint16x4_t src11 = vreinterpret_u16_u8(vld1_u8(reinterpret_cast<const std::uint8_t*>( src + 1 * src_cntg_step + 8 ) ) );
				uint16x4_t src12 = vreinterpret_u16_u8(vld1_u8(reinterpret_cast<const std::uint8_t*>( src + 2 * src_cntg_step + 8 ) ) );
				uint16x4_t src13 = vreinterpret_u16_u8(vld1_u8(reinterpret_cast<const std::uint8_t*>( src + 3 * src_cntg_step + 8 ) ) );

				uint16x8x2_t t001 = vtrnq_u16(src00, src01);
				uint16x8x2_t t023 = vtrnq_u16(src02, src03);

				uint16x4x2_t t101 = vtrn_u16(src10, src11);
				uint16x4x2_t t123 = vtrn_u16(src12, src13);

				uint32x4x2_t r002 = vtrnq_u32(vreinterpretq_u32_u16(t001.val[0]), vreinterpretq_u32_u16(t023.val[0]));
				uint32x4x2_t r013 = vtrnq_u32(vreinterpretq_u32_u16(t001.val[1]), vreinterpretq_u32_u16(t023.val[1]));

				uint32x2x2_t r102 = vtrn_u32(vreinterpret_u32_u16(t101.val[0]), vreinterpret_u32_u16(t123.val[0]));
				uint32x2x2_t r113 = vtrn_u32(vreinterpret_u32_u16(t101.val[1]), vreinterpret_u32_u16(t123.val[1]));

				uint64x2_t r00_u64x2 = vreinterpretq_u64_u32(r002.val[0]);
				uint64x2_t r02_u64x2 = vreinterpretq_u64_u32(r002.val[1]);
				uint64x2_t r01_u64x2 = vreinterpretq_u64_u32(r013.val[0]);
				uint64x2_t r03_u64x2 = vreinterpretq_u64_u32(r013.val[1]);

				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 0 * strd_interleave_step ), vget_low_u8(vreinterpretq_u8_u64(r00_u64x2)));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 1 * strd_interleave_step ), vget_low_u8(vreinterpretq_u8_u64(r01_u64x2)));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 2 * strd_interleave_step ), vget_low_u8(vreinterpretq_u8_u64(r02_u64x2)));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 3 * strd_interleave_step ), vget_low_u8(vreinterpretq_u8_u64(r03_u64x2)));

				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 4 * strd_interleave_step ), vget_high_u8(vreinterpretq_u8_u64(r00_u64x2)));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 5 * strd_interleave_step ), vget_high_u8(vreinterpretq_u8_u64(r01_u64x2)));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 6 * strd_interleave_step ), vget_high_u8(vreinterpretq_u8_u64(r02_u64x2)));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 7 * strd_interleave_step ), vget_high_u8(vreinterpretq_u8_u64(r03_u64x2)));

				vst1_u8(reinterpret_cast<std::uint8_t*>( dst +  8 * strd_interleave_step ), vreinterpret_u8_u32(r102.val[0]));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst +  9 * strd_interleave_step ), vreinterpret_u8_u32(r113.val[0]));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 10 * strd_interleave_step ), vreinterpret_u8_u32(r102.val[1]));
				vst1_u8(reinterpret_cast<std::uint8_t*>( dst + 11 * strd_interleave_step ), vreinterpret_u8_u32(r113.val[1]));

				return;
			}
		}

		for(kernel_inttype si = 0_ki; si != strd; ++si) {
			for(kernel_inttype ci = 0_ki; ci != CntgInterleave; ++ci) {
				dst[ ci * cntg_interleave_step  + si * strd_interleave_step ] =
					src[ ci * src_cntg_step + si * src_strd_step ];
			}
		}
	};

	auto partial_cntg_tile_interleave = [&](auto& src, auto& dst, auto cntg_split, auto strd) {
		for(kernel_inttype si = 0_ki; si != strd; ++si) {
			for(kernel_inttype ci = 0_ki; ci != cntg_split; ++ci) {
				dst[ ci * cntg_interleave_step  + si * strd_interleave_step ] =
					src[ ci * src_cntg_step + si * src_strd_step ];
			}
			for(kernel_inttype ci = cntg_split; ci < CntgInterleave; ++ci) {
				dst[ ci * cntg_interleave_step  + si * strd_interleave_step ] = zero<>;
			}
		}
	};

	auto empty_cntg_tile_interleave = [&](auto& dst, auto strd) {
		for(kernel_inttype si = 0_ki; si != strd; ++si) {
			for(kernel_inttype ci = 0_ki; ci != CntgInterleave; ++ci) {
				dst[ ci * cntg_interleave_step  + si * strd_interleave_step ] = zero<>;
			}
		}
	};

	const auto cntg = min(src_cntg, dst_cntg);
	auto cntg_loops = [&](auto strd, auto src_row, auto dst_row) {
		kernel_inttype c = 0_ki, ct = 0_ki;
		for(; ( c + CntgInterleave ) <= cntg; c += CntgInterleave, ++ct) {
			auto dst_tile = dst_row + ct * cntg_tile_step;
			auto src_tile = src_row + c  * src_cntg_step;

			full_cntg_tile_interleave(src_tile, dst_tile, strd);
		}

		if(const auto remaining_cntg = cntg - c; remaining_cntg != 0_ki) {
			auto dst_tile = dst_row + ct * cntg_tile_step;
			auto src_tile = src_row + c  * src_cntg_step;

			partial_cntg_tile_interleave(src_tile, dst_tile, remaining_cntg, strd);

			c += CntgInterleave;
			++ct;
		}

		for(; c < dst_cntg; c += CntgInterleave, ++ct) {
			auto dst_tile = dst_row + ct * cntg_tile_step;

			empty_cntg_tile_interleave(dst_tile, strd);
		}
	};

	const auto strd = min(src_strd, dst_strd);
	constexpr step_val_fixed<StrdInterleave> strd_interleave;

	kernel_inttype s = 0_ki, st = 0_ki;
	for(; (s + StrdInterleave ) <= strd; s += StrdInterleave, ++st) {
		cntg_loops(strd_interleave, src + s * src_strd_step, dst + st * strd_tile_step);
	}

	if(s < strd) {
		const auto strd_chunk = min(StrdInterleave, strd - s);
		cntg_loops(strd_chunk, src + s * src_strd_step, dst + st * strd_tile_step);
	}
}

template<kernel_inttype CntgInterleave, kernel_inttype StrdInterleave, typename SrcDataType, typename DstDataType>
PERFLIBS_LINALG_INLINE
void tile_interleave(
	std::size_t src_cntg, std::size_t src_strd, const SrcDataType *src, std::size_t src_cntg_step, std::size_t src_strd_step,
	std::size_t dst_cntg, std::size_t dst_strd,       DstDataType *dst, std::size_t dst_strd_step,
	kernel_inttype src_submat_cntg, kernel_inttype src_submat_strd) {

	if(src_cntg_step == 1_ki) {
		tile_interleave_impl<CntgInterleave, StrdInterleave>(
			src_cntg, src_strd, src, step_val_fixed<1> {}, src_strd_step,
			dst_cntg, dst_strd, dst,                       dst_strd_step);
	}
	else if(src_strd_step == 1_ki) {
		tile_interleave_impl<CntgInterleave, StrdInterleave>(
			src_cntg, src_strd, src, src_cntg_step,        step_val_fixed<1> {},
			dst_cntg, dst_strd, dst,                       dst_strd_step);
	}
	else {
		tile_interleave_impl<CntgInterleave, StrdInterleave>(
			src_cntg, src_strd, src, src_cntg_step,        src_strd_step,
			dst_cntg, dst_strd, dst,                       dst_strd_step);
	}
}

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_FRAMEWORK_TILE_INTERLEAVE_HPP

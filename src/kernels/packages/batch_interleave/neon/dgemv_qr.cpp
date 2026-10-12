/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "packages/batch_interleave/factorization/kernels/qr_kernels.hpp"

#include "matrix/interleave_batch.hpp"

#include "perflibs_assert.hpp"

#include <array>

namespace perflibs::interleave_batch {

// a kernel specialized to calling DGEMV from with LARFL of QR (DGEQRFRR)
// alpha = 1, beta = 0 and b[0] = 1 (because b is a vector here)
void dgemv_qr_neon(
	kernel_inttype ninter,
	kernel_inttype m, kernel_inttype n,
	const double *a, kernel_inttype a_cntg_step, kernel_inttype a_strd_step,
	const double *b, kernel_inttype b_cntg_step,
	      double *c, kernel_inttype c_cntg_step) {

	switch(ninter) {
	default:
	{
		PERFLIBS_ASSERT(ninter <= 64, "ninter out of range");

		std::array<float64x2_t, 32> tmp;

		for(kernel_inttype i = 0; i < m; ++i) {
			auto a_row = a + i * a_strd_step;
			auto c_inter = c + i * c_cntg_step;

			// the first entry of B is always one
			for(kernel_inttype inter = 0; inter < ninter; inter+=2) {
				tmp[inter/2] = linalg::simd::load<float64x2_t>(a_row + inter);
			}

			for(kernel_inttype l = 1; l < n; ++l) {
				auto a_inter = a_row + l * a_cntg_step;
				auto b_inter = b + l * b_cntg_step;

				for(kernel_inttype inter = 0; inter < ninter; inter+=2) {
					auto a0 = linalg::simd::load<float64x2_t>(a_inter + inter);
					auto b0 = linalg::simd::load<float64x2_t>(b_inter + inter);

					linalg::simd::mla(tmp[inter/2], a0, b0);
				}
			}

			for(kernel_inttype inter = 0; inter != ninter; inter+=2) {
				linalg::simd::store(c_inter + inter, tmp[inter/2]);
			}
		}
		break;
	}
	case 4:
	{
		std::array<float64x2_t, 2> tmp;

		for(kernel_inttype i = 0; i < m; ++i) {
			auto a_row = a + i * a_strd_step;
			auto c_inter     = c + i * c_cntg_step;

			tmp[0] = linalg::simd::load<float64x2_t>(a_row);
			tmp[1] = linalg::simd::load<float64x2_t>(a_row + 2);

			for(kernel_inttype l = 1; l < n; ++l) {
				auto a_inter = a_row + l * a_cntg_step;
				auto b_inter = b + l * b_cntg_step;

				auto a0 = linalg::simd::load<float64x2_t>(a_inter);
				auto b0 = linalg::simd::load<float64x2_t>(b_inter);

				auto a1 = linalg::simd::load<float64x2_t>(a_inter + 2);
				auto b1 = linalg::simd::load<float64x2_t>(b_inter + 2);

				linalg::simd::mla(tmp[0], a0, b0);
				linalg::simd::mla(tmp[1], a1, b1);
			}

			linalg::simd::store(c_inter, tmp[0]);
			linalg::simd::store(c_inter + 2, tmp[1]);
		}
		break;
	}
	case 8:
	{
		std::array<float64x2_t, 4> tmp;

		for(kernel_inttype i = 0; i < m; ++i) {
			auto a_row = a + i * a_strd_step;
			auto c_inter = c + i * c_cntg_step;

			tmp[0] = linalg::simd::load<float64x2_t>(a_row);
			tmp[1] = linalg::simd::load<float64x2_t>(a_row + 2);
			tmp[2] = linalg::simd::load<float64x2_t>(a_row + 4);
			tmp[3] = linalg::simd::load<float64x2_t>(a_row + 6);

			for(kernel_inttype l = 1; l < n; ++l) {
				auto a_inter = a_row + l * a_cntg_step;
				auto b_inter = b + l * b_cntg_step;

				auto a0 = linalg::simd::load<float64x2_t>(a_inter);
				auto b0 = linalg::simd::load<float64x2_t>(b_inter);

				auto a1 = linalg::simd::load<float64x2_t>(a_inter + 2);
				auto b1 = linalg::simd::load<float64x2_t>(b_inter + 2);

				auto a2 = linalg::simd::load<float64x2_t>(a_inter + 4);
				auto b2 = linalg::simd::load<float64x2_t>(b_inter + 4);

				auto a3 = linalg::simd::load<float64x2_t>(a_inter + 6);
				auto b3 = linalg::simd::load<float64x2_t>(b_inter + 6);

				linalg::simd::mla(tmp[0], a0, b0);
				linalg::simd::mla(tmp[1], a1, b1);
				linalg::simd::mla(tmp[2], a2, b2);
				linalg::simd::mla(tmp[3], a3, b3);
			}

			linalg::simd::store(c_inter, tmp[0]);
			linalg::simd::store(c_inter + 2, tmp[1]);
			linalg::simd::store(c_inter + 4, tmp[2]);
			linalg::simd::store(c_inter + 6, tmp[3]);
		}
		break;
	}
	case 16:
	{
		std::array<float64x2_t, 8> tmp;

		for(kernel_inttype i = 0; i < m; ++i) {
			auto a_row = a + i * a_strd_step;
			auto c_inter = c + i * c_cntg_step;

			tmp[0] = linalg::simd::load<float64x2_t>(a_row);
			tmp[1] = linalg::simd::load<float64x2_t>(a_row + 2);
			tmp[2] = linalg::simd::load<float64x2_t>(a_row + 4);
			tmp[3] = linalg::simd::load<float64x2_t>(a_row + 6);
			tmp[4] = linalg::simd::load<float64x2_t>(a_row + 8);
			tmp[5] = linalg::simd::load<float64x2_t>(a_row + 10);
			tmp[6] = linalg::simd::load<float64x2_t>(a_row + 12);
			tmp[7] = linalg::simd::load<float64x2_t>(a_row + 14);

			for(kernel_inttype l = 1; l < n; ++l) {
				auto a_inter = a_row + l * a_cntg_step;
				auto b_inter = b + l * b_cntg_step;

				auto a0 = linalg::simd::load<float64x2_t>(a_inter);
				auto b0 = linalg::simd::load<float64x2_t>(b_inter);

				auto a1 = linalg::simd::load<float64x2_t>(a_inter + 2);
				auto b1 = linalg::simd::load<float64x2_t>(b_inter + 2);

				auto a2 = linalg::simd::load<float64x2_t>(a_inter + 4);
				auto b2 = linalg::simd::load<float64x2_t>(b_inter + 4);

				auto a3 = linalg::simd::load<float64x2_t>(a_inter + 6);
				auto b3 = linalg::simd::load<float64x2_t>(b_inter + 6);

				linalg::simd::mla(tmp[0], a0, b0);
				a0 = linalg::simd::load<float64x2_t>(a_inter + 8);
				b0 = linalg::simd::load<float64x2_t>(b_inter + 8);
				linalg::simd::mla(tmp[1], a1, b1);
				a1 = linalg::simd::load<float64x2_t>(a_inter + 10);
				b1 = linalg::simd::load<float64x2_t>(b_inter + 10);
				linalg::simd::mla(tmp[2], a2, b2);
				a2 = linalg::simd::load<float64x2_t>(a_inter + 12);
				b2 = linalg::simd::load<float64x2_t>(b_inter + 12);
				linalg::simd::mla(tmp[3], a3, b3);
				a3 = linalg::simd::load<float64x2_t>(a_inter + 14);
				b3 = linalg::simd::load<float64x2_t>(b_inter + 14);

				linalg::simd::mla(tmp[4], a0, b0);
				linalg::simd::mla(tmp[5], a1, b1);
				linalg::simd::mla(tmp[6], a2, b2);
				linalg::simd::mla(tmp[7], a3, b3);
			}

			linalg::simd::store(c_inter, tmp[0]);
			linalg::simd::store(c_inter + 2, tmp[1]);
			linalg::simd::store(c_inter + 4, tmp[2]);
			linalg::simd::store(c_inter + 6, tmp[3]);
			linalg::simd::store(c_inter + 8, tmp[4]);
			linalg::simd::store(c_inter + 10, tmp[5]);
			linalg::simd::store(c_inter + 12, tmp[6]);
			linalg::simd::store(c_inter + 14, tmp[7]);
		}
	}
	}
}
} //namespace perflibs::interleave_batch

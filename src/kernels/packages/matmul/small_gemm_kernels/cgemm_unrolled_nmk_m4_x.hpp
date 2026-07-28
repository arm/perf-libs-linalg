/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#define P99_PROTECT(...) __VA_ARGS__

#if defined(A_TRANSPOSE) || defined(A_CONJUGATE)
#error "Transpose/conjugate mode for matrix A not supported for this implementation"
#endif

#include "cgemm_unrolled_address_calculation.hpp"

static void impl(const int_type m, const int_type n, const int_type k,
		const std::complex<float> alpha_, const std::complex<float> * restrict A, const size_t lda,
		const std::complex<float> * restrict B, const size_t ldb, const std::complex<float> beta,
		std::complex<float> * restrict C, const size_t ldc) {

	constexpr int_type unrollm_ = unrollm / 4;

	float32x2_t alpha { alpha_.real(), alpha_.imag() };
#ifndef BETA_ONE
	float32x4_t alpha_beta { alpha_.real(), alpha_.imag(), beta.real(), beta.imag() };
#endif

	const bool beta_zero =   beta.real() == float(0.0)   &&   beta.imag() == float(0.0);

	for (int_type in = 0; in < n; in+=unrolln) {
		for (int_type im=0; im < m; im+=4*unrollm_) {
			float32x4_t cr0_re[unrollm_*unrolln];
			float32x4_t cr0_im[unrollm_*unrolln];

			// setup accumulators and load cr
			for (int_type iun = 0; iun < unrolln; iun++) {
				for (int_type ium = 0; ium < unrollm_; ium++) {
					cr0_re[iun*unrollm_ + ium] = cr0_im[iun*unrollm_ + ium] = vdupq_n_f32(0.f);
				}
			}

			for (int_type ik = 0; ik < k; ik+=unrollk) {
				float32x4_t ar_re[unrollk*unrollm_];
				float32x4_t ar_im[unrollk*unrollm_];
				float32x2_t br[unrollk*unrolln];

				// load matrix A
				for (int_type ium_ = 0; ium_ < unrollm_; ium_++) {
					for (int_type iuk = 0; iuk < unrollk; iuk++) {
						auto ium = ium_ * 4;
						auto a_ptr = reinterpret_cast<const float32_t*>(&A[A_ADDRESS_CALCULATION]);
						float32x4x2_t ar = vld2q_f32(a_ptr);
						ar_re[iuk*unrollm_ + ium_] = ar.val[0];
						ar_im[iuk*unrollm_ + ium_] = ar.val[1];
					}
				}

				// load matrix B
				for (int_type iun = 0; iun < unrolln; iun++) {
					for (int_type iuk = 0; iuk < unrollk; iuk++) {
						auto b_ptr = reinterpret_cast<const float32_t*>(&B[B_ADDRESS_CALCULATION]);
						br[iuk*unrolln + iun] = vld1_f32(b_ptr);
					}
				}

				// multiply and accumulate into cr0_*
				for (int_type ium = 0; ium < unrollm_; ium++) {
					for (int_type iun = 0; iun < unrolln; iun++) {
						for (int_type iuk = 0; iuk < unrollk; iuk++) {
#ifdef B_CONJUGATE
							auto cr1_re = vfmaq_lane_f32(cr0_re[iun*unrollm_ + ium], ar_re[iuk*unrollm_ + ium], br[iuk*unrolln + iun], 0);
							auto cr1_im = vfmsq_lane_f32(cr0_im[iun*unrollm_ + ium], ar_re[iuk*unrollm_ + ium], br[iuk*unrolln + iun], 1);
							auto cr2_re = vfmaq_lane_f32(cr1_re, ar_im[iuk*unrollm_ + ium], br[iuk*unrolln + iun], 1);
							auto cr2_im = vfmaq_lane_f32(cr1_im, ar_im[iuk*unrollm_ + ium], br[iuk*unrolln + iun], 0);
#else
							auto cr1_re = vfmaq_lane_f32(cr0_re[iun*unrollm_ + ium], ar_re[iuk*unrollm_ + ium], br[iuk*unrolln + iun], 0);
							auto cr1_im = vfmaq_lane_f32(cr0_im[iun*unrollm_ + ium], ar_re[iuk*unrollm_ + ium], br[iuk*unrolln + iun], 1);
							auto cr2_re = vfmsq_lane_f32(cr1_re, ar_im[iuk*unrollm_ + ium], br[iuk*unrolln + iun], 1);
							auto cr2_im = vfmaq_lane_f32(cr1_im, ar_im[iuk*unrollm_ + ium], br[iuk*unrolln + iun], 0);
#endif
							cr0_re[iun*unrollm_ + ium] = cr2_re;
							cr0_im[iun*unrollm_ + ium] = cr2_im;
						}
					}
				}
			}

			if (beta_zero) {
				for (int_type iun = 0; iun < unrolln; iun++) {
					for (int_type ium = 0; ium < unrollm_; ium++) {

						auto c_ptr = reinterpret_cast<float*>(&C[ldc*(in + iun) + im + 4*ium]);
						auto cr_re = vdupq_n_f32(0.f);
						auto cr_im = vdupq_n_f32(0.f);

						auto cr0_idx = iun*unrollm_ + ium;
						auto cr1_re = vfmaq_lane_f32(cr_re, cr0_re[cr0_idx], alpha, 0);
						auto cr1_im = vfmaq_lane_f32(cr_im, cr0_re[cr0_idx], alpha, 1);
						auto crr_re = vfmsq_lane_f32(cr1_re, cr0_im[cr0_idx], alpha, 1);
						auto crr_im = vfmaq_lane_f32(cr1_im, cr0_im[cr0_idx], alpha, 0);

						vst2q_f32(c_ptr, P99_PROTECT({ crr_re, crr_im }));
					}
				}
			}
			else {
				for (int_type iun = 0; iun < unrolln; iun++) {
					for (int_type ium = 0; ium < unrollm_; ium++) {

						auto c_ptr = reinterpret_cast<float*>(&C[ldc*(in + iun) + im + 4*ium]);
						auto cr = vld2q_f32(c_ptr);
						auto cr_re = cr.val[0];
						auto cr_im = cr.val[1];

						auto cr0_idx = iun*unrollm_ + ium;
#ifdef BETA_ONE
						auto cr1_re = vfmaq_lane_f32(cr_re, cr0_re[cr0_idx], alpha, 0);
						auto cr1_im = vfmaq_lane_f32(cr_im, cr0_re[cr0_idx], alpha, 1);
						auto crr_re = vfmsq_lane_f32(cr1_re, cr0_im[cr0_idx], alpha, 1);
						auto crr_im = vfmaq_lane_f32(cr1_im, cr0_im[cr0_idx], alpha, 0);
#else
						auto cr1_re = vmulq_laneq_f32(cr0_re[cr0_idx], alpha_beta, 0);
						auto cr1_im = vmulq_laneq_f32(cr0_re[cr0_idx], alpha_beta, 1);
						auto cr2_re = vfmsq_laneq_f32(cr1_re, cr0_im[cr0_idx], alpha_beta, 1);
						auto cr2_im = vfmaq_laneq_f32(cr1_im, cr0_im[cr0_idx], alpha_beta, 0);

						auto cr3_re = vfmaq_laneq_f32(cr2_re, cr_re, alpha_beta, 2);
						auto cr3_im = vfmaq_laneq_f32(cr2_im, cr_re, alpha_beta, 3);
						auto crr_re = vfmsq_laneq_f32(cr3_re, cr_im, alpha_beta, 3);
						auto crr_im = vfmaq_laneq_f32(cr3_im, cr_im, alpha_beta, 2);
#endif
						vst2q_f32(c_ptr, P99_PROTECT({ crr_re, crr_im }));
					}
				}
			}
		}
	}
}

#undef A_ADDRESS_CALCULATION
#undef B_ADDRESS_CALCULATION
#undef A_TRANSPOSE
#undef B_TRANSPOSE
#undef A_CONJUGATE
#undef B_CONJUGATE
#undef BETA_ONE

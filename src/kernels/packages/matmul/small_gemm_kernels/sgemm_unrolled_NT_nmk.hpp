/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef SGEMM_UNROLLED_NT_NMK_HPP
#define SGEMM_UNROLLED_NT_NMK_HPP

#include "perflibs_util.hpp"
#include <arm_neon.h>
#include <cstddef>

namespace perflibs { namespace gemm {

template<int unrolln, int unrollm, int unrollk>
void unrolled_kernel_NT_nmk(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const float alpha,
							const float * restrict A, const size_t lda,
							const float * restrict B, const size_t ldb, const float beta,
							float * restrict C, const size_t ldc) {

	float32x4_t ar[unrollk*unrollm];
	float32x2_t brs[unrollk*unrolln] = {0};
	float32x4_t cr[unrollm*unrolln];

	float32x2_t alpha_rs = {0};
	alpha_rs = vld1_lane_f32(&alpha, alpha_rs, 0);
	float32x2_t beta_rs = {0};
	beta_rs = vld1_lane_f32(&beta, beta_rs, 0);

	for (int_type in = 0; in < n - (unrolln - 1); in+=unrolln) {

		for (int_type im = 0; im < m - (4*unrollm - 1); im+=4*unrollm) {

			float32_t *c_ptr = &C[ldc*in + im];

			float32x4_t cr0[unrollm*unrolln] = {0};

			for (int_type ik = 0; ik < k - (unrollk  - 1); ik+=unrollk) {

				const float32_t *b_ptr = &B[ldb*ik + in];

				for (int_type iuk = 0; iuk < unrollk; iuk++) {
					for (int_type iun = 0; iun < unrolln; iun++) {
						brs[unrollk*iun + iuk] = vld1_lane_f32(b_ptr + ldb*iuk + iun, brs[unrollk*iun + iuk], 0);
					}
				}

				const float32_t *a_ptr = &A[lda*ik + im];

				for (int_type ium = 0; ium < unrollm; ium++) {
					for (int_type iuk = 0; iuk < unrollk; iuk++) {
						ar[unrollm*iuk + ium] = vld1q_f32(a_ptr + lda*iuk + 4*ium);
					}
				}


				for (int_type iun = 0; iun < unrolln; iun++) {
					for (int_type ium = 0; ium < unrollm; ium++) {
						for (int_type iuk = 0; iuk < unrollk; iuk++) {
							cr0[unrollm*iun + ium] = vfmaq_lane_f32(cr0[iun*unrollm + ium], ar[iuk*unrollm + ium], brs[unrollk*iun + iuk], 0);
						}
					}
				}

			}

			if (beta == float(0.0)) {
				for (int iun = 0; iun < unrolln; iun++) {
					for (int ium = 0; ium < unrollm; ium++) {
						cr0[unrollm*iun + ium] = vmulq_lane_f32(cr0[unrollm*iun + ium], alpha_rs, 0);

						vst1q_f32(c_ptr + ldc*iun + 4*ium, cr0[iun*unrollm + ium]);
					}
				}
			}
			else {
				for (int iun = 0; iun < unrolln; iun++) {
					for (int ium = 0; ium < unrollm; ium++) {
						cr[unrollm*iun + ium] = vld1q_f32(c_ptr + ldc*iun + 4*ium);

						cr0[unrollm*iun + ium] = vmulq_lane_f32(cr0[unrollm*iun + ium], alpha_rs, 0);

						cr[unrollm*iun + ium] = vfmaq_lane_f32(cr0[unrollm*iun + ium], cr[unrollm*iun + ium], beta_rs, 0);

						vst1q_f32(c_ptr + ldc*iun + 4*ium, cr[iun*unrollm + ium]);
					}
				}
			}

		}

		for (int_type im = m - (m%4); im < m; im++) {

			float32_t *c_ptr = &C[ldc*in + im];

			float32x2_t crs0[unrolln] = {0};
			float32x2_t crs[unrolln] = {0};
			float32x2_t ars[unrollk] = {0};

			for (int_type ik = 0; ik < k - (unrollk  - 1); ik+=unrollk) {

				const float32_t *b_ptr = &B[ldb*ik + in];

				for (int_type iun = 0; iun < unrolln; iun++) {
					for (int_type iuk = 0; iuk < unrollk; iuk++) {
						brs[unrollk*iun + iuk] = vld1_lane_f32(b_ptr + ldb*iuk + iun, brs[unrollk*iun + iuk], 0);
					}
				}

				const float32_t *a_ptr = &A[lda*ik + im];

				for (int_type iuk = 0; iuk < unrollk; iuk++) {
					ars[iuk] = vld1_lane_f32(a_ptr + lda*iuk, ars[iuk], 0);
				}

				for (int_type iun = 0; iun < unrolln; iun++) {
					for (int_type iuk = 0; iuk < unrollk; iuk++) {
						crs0[iun] = vfma_f32(crs0[iun], ars[iuk], brs[unrollk*iun + iuk]);
					}
				}

			}

			if (beta == float(0.0)) {
				for (int iun =0; iun < unrolln; iun++) {
					crs0[iun] = vmul_f32(crs0[iun], alpha_rs);

					vst1_lane_f32(c_ptr + ldc*iun, crs0[iun], 0);
				}
			}
			else {
				for (int iun =0; iun < unrolln; iun++) {
					crs[iun] = vld1_lane_f32(c_ptr + ldc*iun, crs[iun], 0);

					crs0[iun] = vmul_f32(crs0[iun], alpha_rs);

					crs[iun] = vfma_f32(crs0[iun], crs[iun], beta_rs);

					vst1_lane_f32(c_ptr + ldc*iun, crs[iun], 0);
				}
			}

		}

	}

}


}}

#endif

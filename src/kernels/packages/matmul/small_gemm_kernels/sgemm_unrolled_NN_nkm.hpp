/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef SGEMM_UNROLLED_NN_NKM_HPP
#define SGEMM_UNROLLED_NN_NKM_HPP

#include "perflibs_util.hpp"
#include <arm_neon.h>
#include <cstddef>

namespace perflibs { namespace gemm {

template<int unrolln, int unrollm, int unrollk>
void unrolled_kernel_NN_nkm(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const float alpha,
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

		for (int_type ik = 0; ik < k - (unrollk - 1); ik+=unrollk) {

			const float32_t *b_ptr = &B[in*ldb + ik];

			for (int iun =0; iun < unrolln; iun++) {
				for (int iuk =0; iuk < unrollk; iuk++) {
					brs[unrollk*iun + iuk] = vld1_lane_f32(&b_ptr[ldb*iun + iuk], brs[unrollk*iun + iuk], 0);
					brs[unrollk*iun + iuk] = vmul_lane_f32(brs[unrollk*iun + iuk], alpha_rs, 0);
				}
			}

			const float32_t *a_ptr = &A[ik*lda];
			float32_t *c_ptr = &C[in*ldc];

			for (int_type im = 0; im < m - (4*unrollm - 1); im+=4*unrollm) {

				if (ik > 0) { // if this is not the first iteration, simply load C
					for (int iun =0; iun < unrolln; iun++) {
						for (int ium =0; ium < unrollm; ium++) {
							cr[unrollm*iun + ium] = vld1q_f32(c_ptr + ldc*iun + im + 4*ium);
						}
					}
				}
				else if (beta == float(0.0)) { // otherwise if it is the first iter and beta is zero, set C to zero (as per reference DGEMM)
					for (int iun =0; iun < unrolln; iun++) {
						for (int ium =0; ium < unrollm; ium++) {
							cr[unrollm*iun + ium] = vdupq_n_f32(0.0);
						}
					}
				}
				else { // otherwise if it is the first iteration and beta is non-zero load C and multiply by beta
					for (int iun =0; iun < unrolln; iun++) {
						for (int ium =0; ium < unrollm; ium++) {
							cr[unrollm*iun + ium] = vld1q_f32(c_ptr + ldc*iun + im + 4*ium);
							cr[unrollm*iun + ium] = vmulq_lane_f32(cr[unrollm*iun + ium], beta_rs, 0);
						}
					}
				}

				for (int ium =0; ium < unrollm; ium++) {
					for (int iuk =0; iuk < unrollk; iuk++) {
						ar[iuk*unrollm + ium] = vld1q_f32(a_ptr + lda*iuk + im + 4*ium);
						for (int iun = 0; iun < unrolln; iun++) {
							cr[iun*unrollm + ium] = vfmaq_lane_f32(cr[iun*unrollm + ium], ar[iuk*unrollm + ium], brs[unrollk*iun + iuk], 0);
						}
					}
				}

				for (int iun =0; iun < unrolln; iun++) {
					for (int ium =0; ium < unrollm; ium++) {
						vst1q_f32(c_ptr + ldc*iun + im + 4*ium, cr[iun*unrollm + ium]);
					}
				}

			}

			for (int_type im = m - (m%4); im < m; im++) {

				float32x2_t crs[unrolln] = {0};
				float32x2_t ars[unrollk] = {0};

				if (ik > 0) {
					for (int iun =0; iun < unrolln; iun++) {
						crs[iun] = vld1_lane_f32(c_ptr + ldc*iun + im, crs[iun], 0);
					}
				}
				else if (beta == float(0.0)) {
					for (int iun =0; iun < unrolln; iun++) {
						crs[iun] = vdup_n_f32(0.0);
					}
				}
				else {
					for (int iun =0; iun < unrolln; iun++) {
						crs[iun] = vld1_lane_f32(c_ptr + ldc*iun + im, crs[iun], 0);
						crs[iun] = vmul_lane_f32(crs[iun], beta_rs, 0);
					}
				}

				for (int iuk =0; iuk < unrollk; iuk++) {
					ars[iuk] = vld1_lane_f32(a_ptr + lda*iuk + im, ars[iuk], 0);
					for (int iun = 0; iun < unrolln; iun++) {
						crs[iun] = vfma_f32(crs[iun], ars[iuk], brs[unrollk*iun + iuk]);
					}
				}

				for (int iun =0; iun < unrolln; iun++) {
					vst1_lane_f32(c_ptr + ldc*iun + im, crs[iun], 0);
				}

			}

		}

	}

}

}}

#endif

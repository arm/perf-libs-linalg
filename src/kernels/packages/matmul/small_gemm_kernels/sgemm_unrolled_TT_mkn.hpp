/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef SGEMM_UNROLLED_TT_MKN_HPP
#define SGEMM_UNROLLED_TT_MKN_HPP

#include "perflibs_util.hpp"
#include <arm_neon.h>
#include <cstddef>

namespace perflibs { namespace gemm {

/*
 Note: Note that the names of n and m and A and B are swapped compared with the interface: m as passed in becomes n in here,
 and vice-versa, and A as passed in is B in here, vice-versa.
 This is because this is the same routine as the N,N case, but with the matrices swapped, and we write the transpose of the
 C that is computed in the vector registers.
*/
template<int unrollm, int unrolln, int unrollk>
void unrolled_kernel_TT_mkn(const kernel_inttype n, const kernel_inttype m, const kernel_inttype k, const float alpha,
							const float * restrict B, const size_t ldb,
							const float * restrict A, const size_t lda, const float beta,
							float * restrict C, const size_t ldc) {

	float32x4_t ar[unrollk*unrollm];
	float32x2_t brs[unrollk*unrolln] = {0};
	float32x4_t cr[unrollm*unrolln] = {0};

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
			float32_t *c_ptr = &C[in];

			for (int_type im = 0; im < m - (4*unrollm - 1); im+=4*unrollm) {

				if (ik > 0) {
					for (int iun =0; iun < unrolln; iun++) {
						for (int ium =0; ium < unrollm; ium++) {
							cr[unrolln*ium + iun] = vld1q_lane_f32(c_ptr + ldc*(im + 4*ium) + iun, cr[unrolln*ium + iun], 0);
							cr[unrolln*ium + iun] = vld1q_lane_f32(c_ptr + ldc*(im + 4*ium + 1) + iun, cr[unrolln*ium + iun], 1);
							cr[unrolln*ium + iun] = vld1q_lane_f32(c_ptr + ldc*(im + 4*ium + 2) + iun, cr[unrolln*ium + iun], 2);
							cr[unrolln*ium + iun] = vld1q_lane_f32(c_ptr + ldc*(im + 4*ium + 3) + iun, cr[unrolln*ium + iun], 3);
						}
					}
				}
				else if (beta == float(0.0)) {
					for (int iun =0; iun < unrolln; iun++) {
						for (int ium =0; ium < unrollm; ium++) {
							cr[unrolln*ium + iun] = vdupq_n_f32(0.0);
						}
					}
				}
				else {
					for (int iun =0; iun < unrolln; iun++) {
						for (int ium =0; ium < unrollm; ium++) {
							cr[unrolln*ium + iun] = vld1q_lane_f32(c_ptr + ldc*(im + 4*ium) + iun, cr[unrolln*ium + iun], 0);
							cr[unrolln*ium + iun] = vld1q_lane_f32(c_ptr + ldc*(im + 4*ium + 1) + iun, cr[unrolln*ium + iun], 1);
							cr[unrolln*ium + iun] = vld1q_lane_f32(c_ptr + ldc*(im + 4*ium + 2) + iun, cr[unrolln*ium + iun], 2);
							cr[unrolln*ium + iun] = vld1q_lane_f32(c_ptr + ldc*(im + 4*ium + 3) + iun, cr[unrolln*ium + iun], 3);
							cr[unrolln*ium + iun] = vmulq_lane_f32(cr[unrolln*ium + iun], beta_rs, 0);
						}
					}
				}

				for (int ium =0; ium < unrollm; ium++) {
					for (int iuk =0; iuk < unrollk; iuk++) {
						ar[iuk*unrollm + ium] = vld1q_f32(a_ptr + lda*iuk + im + 4*ium);
						for (int iun = 0; iun < unrolln; iun++) {
							cr[ium*unrolln + iun] = vfmaq_lane_f32(cr[ium*unrolln + iun], ar[iuk*unrollm + ium], brs[unrollk*iun + iuk], 0);
						}
					}
				}

				for (int iun =0; iun < unrolln; iun++) {
					for (int ium =0; ium < unrollm; ium++) {
						vst1q_lane_f32(c_ptr + ldc*(4*ium +im) + iun, cr[unrolln*ium + iun], 0);
						vst1q_lane_f32(c_ptr + ldc*(4*ium + im + 1) + iun, cr[unrolln*ium + iun], 1);
						vst1q_lane_f32(c_ptr + ldc*(4*ium + im + 2) + iun, cr[unrolln*ium + iun], 2);
						vst1q_lane_f32(c_ptr + ldc*(4*ium + im + 3) + iun, cr[unrolln*ium + iun], 3);
					}
				}

			}

			for (int_type im = m - (m%4); im < m; im++) {

				float32x2_t crs[unrolln] = {0};
				float32x2_t ars[unrollk] = {0};

				if (ik > 0) {
					for (int iun =0; iun < unrolln; iun++) {
						crs[iun] = vld1_lane_f32(c_ptr + ldc*im + iun, crs[iun], 0);
					}
				}
				else if (beta == float(0.0)) {
					for (int iun =0; iun < unrolln; iun++) {
						crs[iun] = vdup_n_f32(0.0);
					}
				}
				else {
					for (int iun =0; iun < unrolln; iun++) {
						crs[iun] = vld1_lane_f32(c_ptr + ldc*im + iun, crs[iun], 0);
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
					vst1_lane_f32(c_ptr + iun + ldc*im, crs[iun], 0);
				}
			}

		}

	}

}

}}
#endif

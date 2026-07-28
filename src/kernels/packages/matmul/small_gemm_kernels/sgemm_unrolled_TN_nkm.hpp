/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef SGEMM_UNROLLED_TN_NKM_HPP
#define SGEMM_UNROLLED_TN_NKM_HPP

#include "perflibs_util.hpp"
#include <arm_neon.h>
#include <cstddef>

namespace perflibs { namespace gemm {

template<int unrolln, int unrollm, int unrollk>
void unrolled_kernel_TN_nkm(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const float alpha,
							const float * restrict A, const size_t lda,
							const float * restrict B, const size_t ldb, float beta,
							float * restrict C, const size_t ldc) {

	float32x4_t ar[unrollk*unrollm] = {0};
	float32x2_t brs[unrollk*unrolln] = {0};
	float32x4_t cr[unrollm*unrolln];

	float32x2_t alpha_rs = {0};
	alpha_rs = vld1_lane_f32(&alpha, alpha_rs, 0);
	float32x2_t beta_rs = {0};
	beta_rs = vld1_lane_f32(&beta, beta_rs, 0);

	for (int in=0; in<n - (unrolln - 1); in+=unrolln) {

		for (int ik=0; ik<k - (unrollk - 1); ik+=unrollk) {

			for (int iun=0; iun<unrolln; iun++) {
				for (int iuk=0; iuk<unrollk; iuk++) {
					brs[iun*unrollk + iuk] = vld1_lane_f32(&B[(in+iun)*ldb + ik + iuk], brs[iun*unrollk + iuk], 0);
					brs[unrollk*iun + iuk] = vmul_lane_f32(brs[iun*unrollk + iuk], alpha_rs, 0);
				}
			}

			for (int im=0; im<m - (4*unrollm - 1); im+=4*unrollm) {

				float32_t *c_ptr = &C[in*ldc + im];

				if (ik > 0) {
					for (int iun=0; iun<unrolln; iun++) {
						for (int ium=0; ium<unrollm; ium++) {
							cr[iun*unrollm + ium] = vld1q_f32(c_ptr + iun*ldc + 4*ium);
						}
					}
				}
				else if (beta == float(0.0)) {
					for (int iun=0; iun<unrolln; iun++) {
						for (int ium=0; ium<unrollm; ium++) {
							cr[iun*unrollm + ium] = vdupq_n_f32(0.0);
						}
					}
				}
				else {
					for (int iun=0; iun<unrolln; iun++) {
						for (int ium=0; ium<unrollm; ium++) {
							cr[iun*unrollm + ium] = vld1q_f32(c_ptr + iun*ldc + 4*ium);
							cr[iun*unrollm + ium] = vmulq_lane_f32(cr[iun*unrollm + ium], beta_rs, 0);
						}
					}
				}

				for (int ium=0; ium<unrollm; ium++) {
					for (int iuk=0; iuk<unrollk; iuk++) {
						ar[ium*unrollk + iuk] = vld1q_lane_f32(&A[(im + 4*ium)*lda + ik + iuk], ar[ium*unrollk + iuk], 0);
						ar[ium*unrollk + iuk] = vld1q_lane_f32(&A[(im + 1 + 4*ium)*lda + ik + iuk], ar[ium*unrollk + iuk], 1);
						ar[ium*unrollk + iuk] = vld1q_lane_f32(&A[(im + 2 + 4*ium)*lda + ik + iuk], ar[ium*unrollk + iuk], 2);
						ar[ium*unrollk + iuk] = vld1q_lane_f32(&A[(im + 3 + 4*ium)*lda + ik + iuk], ar[ium*unrollk + iuk], 3);
					}

					for (int iun=0; iun<unrolln; iun++) {
						for (int iuk=0; iuk<unrollk; iuk++) {
							cr[iun*unrollm + ium] = vfmaq_lane_f32(cr[iun*unrollm + ium], ar[ium*unrollk + iuk], brs[iun*unrollk + iuk], 0);
						}
					}
				}

				for (int iun=0; iun<unrolln; iun++) {
					for (int ium=0; ium<unrollm; ium++) {
						vst1q_f32(c_ptr + iun*ldc + 4*ium, cr[iun*unrollm + ium]);
					}
				}

			}

			for (int_type im = m - (m%4); im < m; im++) {

				float32_t *c_ptr = &C[in*ldc + im];
				float32x2_t crs0[unrolln] = {0};
				float32x2_t ars[unrollk] = {0};

				if (ik > 0) {
					for (int iun=0; iun<unrolln; iun++) {
						crs0[iun] = vld1_lane_f32(c_ptr + iun*ldc, crs0[iun], 0);
					}
				}
				else if (beta == float(0.0)) {
					for (int iun=0; iun<unrolln; iun++) {
						crs0[iun] = vdup_n_f32(0.0);
					}
				}
				else {
					for (int iun=0; iun<unrolln; iun++) {
						crs0[iun] = vld1_lane_f32(c_ptr + iun*ldc, crs0[iun], 0);
						crs0[iun] = vmul_f32(crs0[iun], beta_rs);
					}
				}

				for (int iuk=0; iuk<unrollk; iuk++) {
					ars[iuk] = vld1_lane_f32(&A[im*lda + ik + iuk], ars[iuk], 0);
				}

				for (int iun=0; iun<unrolln; iun++) {
					for (int iuk=0; iuk<unrollk; iuk++) {
						brs[iun*unrollk + iuk] = vld1_lane_f32(&B[(in+iun)*ldb + ik + iuk], brs[iun*unrollk + iuk], 0);
						brs[unrollk*iun + iuk] = vmul_lane_f32(brs[iun*unrollk + iuk], alpha_rs, 0);
						crs0[iun] = vfma_f32(crs0[iun], ars[iuk], brs[iun*unrollk + iuk]);
					}
				}

				for (int iun=0; iun<unrolln; iun++) {
					vst1_lane_f32(c_ptr + iun*ldc, crs0[iun], 0);
				}

			}

		}


	}

}

}}

#endif

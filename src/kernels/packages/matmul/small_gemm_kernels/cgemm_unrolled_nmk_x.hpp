/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */


#include "cgemm_unrolled_address_calculation.hpp"

static void impl(const int_type m, const int_type n, const int_type k,
		const std::complex<float> alpha, const std::complex<float> * restrict A, const size_t lda,
		const std::complex<float> * restrict B, const size_t ldb, const std::complex<float> beta,
		std::complex<float> * restrict C, const size_t ldc) {

	float alpha_re = alpha.real();
	float alpha_im = alpha.imag();
	float beta_re = beta.real();
	float beta_im = beta.imag();

	const bool beta_zero =   beta_re == float(0.0)   &&   beta_im == float(0.0);

	for (int_type in = 0; in < n; in+=unrolln) {
		for (int_type im=0; im < m; im+=unrollm) {
			float cr0_re[unrollm*unrolln];
			float cr0_im[unrollm*unrolln];

			// setup accumulators and load cr
			for (int_type iun = 0; iun < unrolln; iun++) {
				for (int_type ium = 0; ium < unrollm; ium++) {
					cr0_re[iun*unrollm + ium] = cr0_im[iun*unrollm + ium] = 0.f;
				}
			}

			for (int_type ik = 0; ik < k; ik+=unrollk) {
				float ar_re[unrollk*unrollm];
				float ar_im[unrollk*unrollm];
				float br_re[unrollk*unrolln];
				float br_im[unrollk*unrolln];

				// load matrix A
				for (int_type ium = 0; ium < unrollm; ium++) {
					for (int_type iuk = 0; iuk < unrollk; iuk++) {
						auto a_ptr = reinterpret_cast<const float*>(&A[A_ADDRESS_CALCULATION]);
						ar_re[iuk*unrollm + ium] = a_ptr[0];
						ar_im[iuk*unrollm + ium] = a_ptr[1];
					}
				}

				// load matrix B
				for (int_type iun = 0; iun < unrolln; iun++) {
					for (int_type iuk = 0; iuk < unrollk; iuk++) {
						auto b_ptr = reinterpret_cast<const float*>(&B[B_ADDRESS_CALCULATION]);
						br_re[iuk*unrolln + iun] = b_ptr[0];
						br_im[iuk*unrolln + iun] = b_ptr[1];
					}
				}

				// multiply and accumulate into cr0_*
				for (int_type ium = 0; ium < unrollm; ium++) {
					for (int_type iun = 0; iun < unrolln; iun++) {
						for (int_type iuk = 0; iuk < unrollk; iuk++) {
#if defined(A_CONJUGATE) && defined(B_CONJUGATE)
							auto cr1_re = cr0_re[iun*unrollm + ium] + ar_re[iuk*unrollm + ium] * br_re[iuk*unrolln + iun];
							auto cr1_im = cr0_im[iun*unrollm + ium] - ar_re[iuk*unrollm + ium] * br_im[iuk*unrolln + iun];
							auto cr2_re = cr1_re - ar_im[iuk*unrollm + ium] * br_im[iuk*unrolln + iun];
							auto cr2_im = cr1_im - ar_im[iuk*unrollm + ium] * br_re[iuk*unrolln + iun];
#elif defined(A_CONJUGATE)
							auto cr1_re = cr0_re[iun*unrollm + ium] + ar_re[iuk*unrollm + ium] * br_re[iuk*unrolln + iun];
							auto cr1_im = cr0_im[iun*unrollm + ium] + ar_re[iuk*unrollm + ium] * br_im[iuk*unrolln + iun];
							auto cr2_re = cr1_re + ar_im[iuk*unrollm + ium] * br_im[iuk*unrolln + iun];
							auto cr2_im = cr1_im - ar_im[iuk*unrollm + ium] * br_re[iuk*unrolln + iun];
#elif defined(B_CONJUGATE)
							auto cr1_re = cr0_re[iun*unrollm + ium] + ar_re[iuk*unrollm + ium] * br_re[iuk*unrolln + iun];
							auto cr1_im = cr0_im[iun*unrollm + ium] - ar_re[iuk*unrollm + ium] * br_im[iuk*unrolln + iun];
							auto cr2_re = cr1_re + ar_im[iuk*unrollm + ium] * br_im[iuk*unrolln + iun];
							auto cr2_im = cr1_im + ar_im[iuk*unrollm + ium] * br_re[iuk*unrolln + iun];
#else
							auto cr1_re = cr0_re[iun*unrollm + ium] + ar_re[iuk*unrollm + ium] * br_re[iuk*unrolln + iun];
							auto cr1_im = cr0_im[iun*unrollm + ium] + ar_re[iuk*unrollm + ium] * br_im[iuk*unrolln + iun];
							auto cr2_re = cr1_re - ar_im[iuk*unrollm + ium] * br_im[iuk*unrolln + iun];
							auto cr2_im = cr1_im + ar_im[iuk*unrollm + ium] * br_re[iuk*unrolln + iun];
#endif
							cr0_re[iun*unrollm + ium] = cr2_re;
							cr0_im[iun*unrollm + ium] = cr2_im;
						}
					}
				}
			}

			if (beta_zero) {
				for (int_type iun = 0; iun < unrolln; iun++) {
					for (int_type ium = 0; ium < unrollm; ium++) {
						auto c_ptr = reinterpret_cast<float*>(&C[ldc*(in + iun) + im + ium]);

						auto cr_idx = iun*unrollm + ium;
						auto cr1_re = cr0_re[cr_idx] * alpha_re;
						auto cr1_im = cr0_re[cr_idx] * alpha_im;
						auto cr2_re = cr1_re - cr0_im[cr_idx] * alpha_im;
						auto cr2_im = cr1_im + cr0_im[cr_idx] * alpha_re;

						c_ptr[0] = cr2_re;
						c_ptr[1] = cr2_im;
					}
				}
			}
			else {
				for (int_type iun = 0; iun < unrolln; iun++) {
					for (int_type ium = 0; ium < unrollm; ium++) {
						auto c_ptr = reinterpret_cast<float*>(&C[ldc*(in + iun) + im + ium]);
						auto cr_re = c_ptr[0];
						auto cr_im = c_ptr[1];

						auto cr_idx = iun*unrollm + ium;
						auto cr1_re = cr0_re[cr_idx] * alpha_re;
						auto cr1_im = cr0_re[cr_idx] * alpha_im;
						auto cr2_re = cr1_re - cr0_im[cr_idx] * alpha_im;
						auto cr2_im = cr1_im + cr0_im[cr_idx] * alpha_re;

#ifdef BETA_ONE
						auto crr_re = cr2_re + cr_re;
						auto crr_im = cr2_im + cr_im;
#else
						auto cr3_re = cr2_re + cr_re * beta_re;
						auto cr3_im = cr2_im + cr_re * beta_im;
						auto crr_re = cr3_re - cr_im * beta_im;
						auto crr_im = cr3_im + cr_im * beta_re;
#endif
						c_ptr[0] = crr_re;
						c_ptr[1] = crr_im;
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

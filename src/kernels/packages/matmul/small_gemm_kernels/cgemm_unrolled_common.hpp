/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include <algorithm>
#include <array>
#include <arm_neon.h>
#include <complex>
#include <cstdio>

#include "perflibs_util.hpp"

#define P99_PROTECT(...) __VA_ARGS__

namespace perflibs { namespace gemm {

constexpr int max_unroll = 4;
constexpr int max_unroll_cubed = max_unroll*max_unroll*max_unroll;

template<char transa, char transb, bool beta_one, int unrolln, int unrollm, int unrollk, typename = void>
struct cgemm_unrolled_impl {
	static void impl(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k,
				const std::complex<float> alpha, const std::complex<float> * restrict A, const size_t lda,
				const std::complex<float> * restrict B, const size_t ldb, const std::complex<float> beta,
				std::complex<float> * restrict C, const size_t ldc);
};

template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'N', 'N', true, unrolln, unrollm, unrollk, typename std::enable_if<(unrollm % 4 == 0)>::type> {
#define BETA_ONE
#include "cgemm_unrolled_nmk_m4_x.hpp"
};


template<int unrolln, int unrollm_, int unrollk>
struct cgemm_unrolled_impl<'N', 'N', true, unrolln, unrollm_, unrollk, typename std::enable_if<(unrollm_ % 4 == 2)>::type> {
	static void impl(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k,
				const std::complex<float> alpha_, const std::complex<float> * restrict A, const size_t lda,
				const std::complex<float> * restrict B, const size_t ldb, const std::complex<float> beta,
				std::complex<float> * restrict C, const size_t ldc) {

		constexpr kernel_inttype unrollm = unrollm_ / 2;

		float32x2_t alpha { alpha_.real(), alpha_.imag() };

		for (kernel_inttype in = 0; in < n; in+=unrolln) {
			kernel_inttype im=0;
			for (; im < m; im+=2*unrollm) {
				float32x2_t cr0_re[unrollm*unrolln];
				float32x2_t cr0_im[unrollm*unrolln];
				float32x2_t cr_re[unrollm*unrolln];
				float32x2_t cr_im[unrollm*unrolln];

				// setup accumulators and load cr
				for (kernel_inttype iun = 0; iun < unrolln; iun++) {
					for (kernel_inttype ium = 0; ium < unrollm; ium++) {
						cr0_re[iun*unrollm + ium] = cr0_im[iun*unrollm + ium] = vdup_n_f32(0.f);

						auto c_ptr = reinterpret_cast<float*>(&C[ldc*(in + iun) + im + 2*ium]);
						float32x2x2_t cr = vld2_f32(c_ptr);
						cr_re[iun*unrollm + ium] = cr.val[0];
						cr_im[iun*unrollm + ium] = cr.val[1];
					}
				}

				for (kernel_inttype ik = 0; ik < k; ik+=unrollk) {
					float32x2_t ar_re[unrollk*unrollm];
					float32x2_t ar_im[unrollk*unrollm];
					float32x2_t br[unrollk*unrolln];

					// load matrix A
					for (kernel_inttype ium = 0; ium < unrollm; ium++) {
						for (kernel_inttype iuk = 0; iuk < unrollk; iuk++) {
							auto a_ptr = reinterpret_cast<const float32_t*>(&A[lda*(ik + iuk) + im + 2*ium]);
							float32x2x2_t ar = vld2_f32(a_ptr);
							ar_re[iuk*unrollm + ium] = ar.val[0];
							ar_im[iuk*unrollm + ium] = ar.val[1];
						}
					}

					// load matrix B
					for (kernel_inttype iun = 0; iun < unrolln; iun++) {
						for (kernel_inttype iuk = 0; iuk < unrollk; iuk++) {
							auto b_ptr = reinterpret_cast<const float32_t*>(&B[ldb*(in + iun) + ik + iuk]);
							br[iuk*unrolln + iun] = vld1_f32(b_ptr);
						}
					}

					// multiply and accumulate into cr0_*
					for (kernel_inttype ium = 0; ium < unrollm; ium++) {
						for (kernel_inttype iun = 0; iun < unrolln; iun++) {
							for (kernel_inttype iuk = 0; iuk < unrollk; iuk++) {
								auto cr1_re = vfma_lane_f32(cr0_re[iun*unrollm + ium], ar_re[iuk*unrollm + ium], br[iuk*unrolln + iun], 0);
								auto cr1_im = vfma_lane_f32(cr0_im[iun*unrollm + ium], ar_re[iuk*unrollm + ium], br[iuk*unrolln + iun], 1);
								auto cr2_re = vfms_lane_f32(cr1_re, ar_im[iuk*unrollm + ium], br[iuk*unrolln + iun], 1);
								auto cr2_im = vfma_lane_f32(cr1_im, ar_im[iuk*unrollm + ium], br[iuk*unrolln + iun], 0);
								cr0_re[iun*unrollm + ium] = cr2_re;
								cr0_im[iun*unrollm + ium] = cr2_im;
							}
						}
					}
				}

				for (kernel_inttype iun = 0; iun < unrolln; iun++) {
					for (kernel_inttype ium = 0; ium < unrollm; ium++) {
						auto cr_idx = iun*unrollm + ium;
						auto cr1_re = vfma_lane_f32(cr_re[cr_idx], cr0_re[cr_idx], alpha, 0);
						auto cr1_im = vfma_lane_f32(cr_im[cr_idx], cr0_re[cr_idx], alpha, 1);
						auto cr2_re = vfms_lane_f32(cr1_re, cr0_im[cr_idx], alpha, 1);
						auto cr2_im = vfma_lane_f32(cr1_im, cr0_im[cr_idx], alpha, 0);

						auto c_ptr = reinterpret_cast<float*>(&C[ldc*(in + iun) + im + 2*ium]);
						vst2_f32(c_ptr, P99_PROTECT({ cr2_re, cr2_im }));
					}
				}
			}
		}
	}
};

template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'N', 'N', true, unrolln, unrollm, unrollk, typename std::enable_if<(unrollm % 2 == 1)>::type> {
#define BETA_ONE
#include "cgemm_unrolled_nmk_x.hpp"
};

template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'N', 'N', false, unrolln, unrollm, unrollk, typename std::enable_if<(unrollm % 4 == 0)>::type> {
#include "cgemm_unrolled_nmk_m4_x.hpp"
};


template<int unrolln, int unrollm_, int unrollk>
struct cgemm_unrolled_impl<'N', 'N', false, unrolln, unrollm_, unrollk, typename std::enable_if<(unrollm_ % 4 == 2)>::type> {
	static void impl(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k,
				const std::complex<float> alpha, const std::complex<float> * restrict A, const size_t lda,
				const std::complex<float> * restrict B, const size_t ldb, const std::complex<float> beta,
				std::complex<float> * restrict C, const size_t ldc) {

		constexpr kernel_inttype unrollm = unrollm_ / 2;

		float32x4_t alpha_beta { alpha.real(), alpha.imag(), beta.real(), beta.imag() };
		float32x2_t alphar = { alpha.real(), alpha.imag() };

		const bool beta_zero =   beta.real() == float(0.0)   &&   beta.imag() == float(0.0);

		for (kernel_inttype in = 0; in < n; in+=unrolln) {
			kernel_inttype im=0;
			for (; im < m; im+=2*unrollm) {
				float32x2_t cr0_re[unrollm*unrolln];
				float32x2_t cr0_im[unrollm*unrolln];

				// setup accumulators and load cr
				for (kernel_inttype iun = 0; iun < unrolln; iun++) {
					for (kernel_inttype ium = 0; ium < unrollm; ium++) {
						cr0_re[iun*unrollm + ium] = cr0_im[iun*unrollm + ium] = vdup_n_f32(0.f);
					}
				}

				for (kernel_inttype ik = 0; ik < k; ik+=unrollk) {
					float32x2_t ar_re[unrollk*unrollm];
					float32x2_t ar_im[unrollk*unrollm];
					float32x2_t br[unrollk*unrolln];

					// load matrix A
					for (kernel_inttype ium = 0; ium < unrollm; ium++) {
						for (kernel_inttype iuk = 0; iuk < unrollk; iuk++) {
							auto a_ptr = reinterpret_cast<const float32_t*>(&A[lda*(ik + iuk) + im + 2*ium]);
							float32x2x2_t ar = vld2_f32(a_ptr);
							ar_re[iuk*unrollm + ium] = ar.val[0];
							ar_im[iuk*unrollm + ium] = ar.val[1];
						}
					}

					// load matrix B
					for (kernel_inttype iun = 0; iun < unrolln; iun++) {
						for (kernel_inttype iuk = 0; iuk < unrollk; iuk++) {
							auto b_ptr = reinterpret_cast<const float32_t*>(&B[ldb*(in + iun) + ik + iuk]);
							br[iuk*unrolln + iun] = vld1_f32(b_ptr);
						}
					}

					// multiply and accumulate into cr0_*
					for (kernel_inttype ium = 0; ium < unrollm; ium++) {
						for (kernel_inttype iun = 0; iun < unrolln; iun++) {
							for (kernel_inttype iuk = 0; iuk < unrollk; iuk++) {
								auto cr1_re = vfma_lane_f32(cr0_re[iun*unrollm + ium], ar_re[iuk*unrollm + ium], br[iuk*unrolln + iun], 0);
								auto cr1_im = vfma_lane_f32(cr0_im[iun*unrollm + ium], ar_re[iuk*unrollm + ium], br[iuk*unrolln + iun], 1);
								auto cr2_re = vfms_lane_f32(cr1_re, ar_im[iuk*unrollm + ium], br[iuk*unrolln + iun], 1);
								auto cr2_im = vfma_lane_f32(cr1_im, ar_im[iuk*unrollm + ium], br[iuk*unrolln + iun], 0);
								cr0_re[iun*unrollm + ium] = cr2_re;
								cr0_im[iun*unrollm + ium] = cr2_im;
							}
						}
					}
				}

				if (beta_zero) {
					for (kernel_inttype iun = 0; iun < unrolln; iun++) {
						for (kernel_inttype ium = 0; ium < unrollm; ium++) {
							auto cr_re = vdup_n_f32(0.f);
							auto cr_im = vdup_n_f32(0.f);

							auto cr_idx = iun*unrollm + ium;

							auto cr1_re = vfma_lane_f32(cr_re, cr0_re[cr_idx], alphar, 0);
							auto cr1_im = vfma_lane_f32(cr_im, cr0_re[cr_idx], alphar, 1);
							auto crr_re = vfms_lane_f32(cr1_re, cr0_im[cr_idx], alphar, 1);
							auto crr_im = vfma_lane_f32(cr1_im, cr0_im[cr_idx], alphar, 0);

							auto c_ptr = reinterpret_cast<float*>(&C[ldc*(in + iun) + im + 2*ium]);
							vst2_f32(c_ptr, P99_PROTECT({ crr_re, crr_im }));
						}
					}
				}
				else {
					for (kernel_inttype iun = 0; iun < unrolln; iun++) {
						for (kernel_inttype ium = 0; ium < unrollm; ium++) {
							auto c_ptr = reinterpret_cast<float*>(&C[ldc*(in + iun) + im + 2*ium]);
							auto cr = vld2_f32(c_ptr);
							auto cr_re = cr.val[0];
							auto cr_im = cr.val[1];

							auto cr_idx = iun*unrollm + ium;
							auto cr1_re = vmul_laneq_f32(cr0_re[cr_idx], alpha_beta, 0);
							auto cr1_im = vmul_laneq_f32(cr0_re[cr_idx], alpha_beta, 1);
							auto cr2_re = vfms_laneq_f32(cr1_re, cr0_im[cr_idx], alpha_beta, 1);
							auto cr2_im = vfma_laneq_f32(cr1_im, cr0_im[cr_idx], alpha_beta, 0);

							auto cr3_re = vfma_laneq_f32(cr2_re, cr_re, alpha_beta, 2);
							auto cr3_im = vfma_laneq_f32(cr2_im, cr_re, alpha_beta, 3);
							auto crr_re = vfms_laneq_f32(cr3_re, cr_im, alpha_beta, 3);
							auto crr_im = vfma_laneq_f32(cr3_im, cr_im, alpha_beta, 2);

							vst2_f32(c_ptr, P99_PROTECT({ crr_re, crr_im }));
						}
					}
				}
			}
		}
	}
};

template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'N', 'N', false, unrolln, unrollm, unrollk, typename std::enable_if<(unrollm % 2 == 1)>::type> {
#include "cgemm_unrolled_nmk_x.hpp"
};

template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'N', 'T', false, unrolln, unrollm, unrollk, typename std::enable_if<(unrollm % 4 != 0)>::type> {
#define B_TRANSPOSE
#include "cgemm_unrolled_nmk_x.hpp"
};

template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'N', 'T', true, unrolln, unrollm, unrollk, typename std::enable_if<(unrollm % 4 != 0)>::type> {
#define B_TRANSPOSE
#define BETA_ONE
#include "cgemm_unrolled_nmk_x.hpp"
};

template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'N', 'T', false, unrolln, unrollm, unrollk, typename std::enable_if<(unrollm % 4 == 0)>::type> {
#define B_TRANSPOSE
#include "cgemm_unrolled_nmk_m4_x.hpp"
};

template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'N', 'T', true, unrolln, unrollm, unrollk, typename std::enable_if<(unrollm % 4 == 0)>::type> {
#define B_TRANSPOSE
#define BETA_ONE
#include "cgemm_unrolled_nmk_m4_x.hpp"
};

template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'N', 'C', false, unrolln, unrollm, unrollk, typename std::enable_if<(unrollm % 4 != 0)>::type> {
#define B_CONJUGATE
#include "cgemm_unrolled_nmk_x.hpp"
};

template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'N', 'C', true, unrolln, unrollm, unrollk, typename std::enable_if<(unrollm % 4 != 0)>::type> {
#define B_CONJUGATE
#define BETA_ONE
#include "cgemm_unrolled_nmk_x.hpp"
};

template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'N', 'C', false, unrolln, unrollm, unrollk, typename std::enable_if<(unrollm % 4 == 0)>::type> {
#define B_CONJUGATE
#include "cgemm_unrolled_nmk_m4_x.hpp"
};

template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'N', 'C', true, unrolln, unrollm, unrollk, typename std::enable_if<(unrollm % 4 == 0)>::type> {
#define B_CONJUGATE
#define BETA_ONE
#include "cgemm_unrolled_nmk_m4_x.hpp"
};

template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'T', 'N', false, unrolln, unrollm, unrollk> {
#define A_TRANSPOSE
#include "cgemm_unrolled_nmk_x.hpp"
};


template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'T', 'N', true, unrolln, unrollm, unrollk> {
#define A_TRANSPOSE
#define BETA_ONE
#include "cgemm_unrolled_nmk_x.hpp"
};


template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'T', 'T', false, unrolln, unrollm, unrollk> {
#define A_TRANSPOSE
#define B_TRANSPOSE
#include "cgemm_unrolled_mnk_x.hpp"
};

template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'T', 'T', true, unrolln, unrollm, unrollk> {
#define A_TRANSPOSE
#define B_TRANSPOSE
#define BETA_ONE
#include "cgemm_unrolled_mnk_x.hpp"
};


template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'T', 'C', false, unrolln, unrollm, unrollk> {
#define A_TRANSPOSE
#define B_CONJUGATE
#include "cgemm_unrolled_nmk_x.hpp"
};

template< int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'T', 'C', true, unrolln, unrollm, unrollk> {
#define A_TRANSPOSE
#define B_CONJUGATE
#define BETA_ONE
#include "cgemm_unrolled_nmk_x.hpp"
};


template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'C', 'N', false, unrolln, unrollm, unrollk> {
#define A_CONJUGATE
#include "cgemm_unrolled_nmk_x.hpp"
};

template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'C', 'N', true, unrolln, unrollm, unrollk> {
#define A_CONJUGATE
#define BETA_ONE
#include "cgemm_unrolled_nmk_x.hpp"
};


template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'C', 'T', false, unrolln, unrollm, unrollk> {
#define A_CONJUGATE
#define B_TRANSPOSE
#include "cgemm_unrolled_nmk_x.hpp"
};

template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'C', 'T', true, unrolln, unrollm, unrollk> {
#define A_CONJUGATE
#define B_TRANSPOSE
#define BETA_ONE
#include "cgemm_unrolled_nmk_x.hpp"
};


template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'C', 'C', false, unrolln, unrollm, unrollk> {
#define A_CONJUGATE
#define B_CONJUGATE
#include "cgemm_unrolled_nmk_x.hpp"
};

template<int unrolln, int unrollm, int unrollk>
struct cgemm_unrolled_impl<'C', 'C', true, unrolln, unrollm, unrollk> {
#define A_CONJUGATE
#define B_CONJUGATE
#define BETA_ONE
#include "cgemm_unrolled_nmk_x.hpp"
};

template<char transa, char transb, int unrolln, int unrollm, int unrollk>
static inline void cgemm_unrolled_kernel(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k,
			const std::complex<float> alpha, const std::complex<float> * restrict A, const size_t lda,
			const std::complex<float> * restrict B, const size_t ldb, const std::complex<float> beta,
			std::complex<float> * restrict C, const size_t ldc) {
	if (beta == 1.f) {
		cgemm_unrolled_impl<transa, transb, true, unrolln, unrollm, unrollk>::impl(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	} else {
		cgemm_unrolled_impl<transa, transb, false, unrolln, unrollm, unrollk>::impl(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}
}

using fptr = decltype(&cgemm_unrolled_kernel<'N', 'N', 1, 1, 1>);
using fptr_array_t = std::array<fptr, max_unroll_cubed>;

template<char transa, char transb, std::size_t I=max_unroll_cubed>
struct populate {
	static fptr_array_t impl() {
		auto array = populate<transa, transb, I-1>::impl();
		constexpr int unrolln = (I-1)/(max_unroll*max_unroll) + 1;
		constexpr int unrollm = (((I-1)/max_unroll)%max_unroll) + 1;
		constexpr int unrollk = (I-1)%max_unroll + 1;
		array[I-1]= &cgemm_unrolled_kernel<transa, transb, unrolln, unrollm, unrollk>;
		return array;
	}
};

template<char transa, char transb>
struct populate<transa, transb, 0> {
	static fptr_array_t impl() { return {}; }
};

template<char transa, char transb, typename...Args>
void run_unrolled_impl(kernel_inttype m, kernel_inttype n, kernel_inttype k, Args&&...args) {
	int iunrollk = 1;
	int iunrollm = 1;
	int iunrolln = 1;

	for (int i = max_unroll; i > 0; i--) {
		if (k % i == 0) {
			iunrollk = i;
			break;
		}
	}
	for (int i = max_unroll; i > 0; i--) {
		if (n % i == 0) {
			iunrolln = i;
			break;
		}
	}
	for (int i = max_unroll; i > 0; i--) {
		if (m % i == 0) {
			iunrollm = i;
			break;
		}
	}

	auto lookup_array = populate<transa, transb>::impl();
	int lt_index = max_unroll*max_unroll*(iunrolln-1) + max_unroll*(iunrollm-1) + (iunrollk-1);
	lookup_array[lt_index](m, n, k, std::forward<Args>(args)...);
}
}}

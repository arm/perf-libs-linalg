/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "packages/batch_interleave/matmul/kernels/gemm_kernels.hpp"
#include "matrix/interleave_batch.hpp"
#include <arm_sve.h>
#include <cstdint>

typedef int64_t kernel_inttype;

namespace perflibs::interleave_batch {

template<bool BetaIsZero>
static PERFLIBS_LINALG_INLINE
svfloat64_t load_and_scale_c(const svbool_t pg, const double *c, const svfloat64_t beta) {
	if constexpr (BetaIsZero) {
		return svdup_f64(0.0);
	}
	else {
		return svmul_f64_x(pg, svld1_f64(pg, c), beta);
	}
}

template<bool BetaIsZero>
static void dgemm_sve_vls8_impl(kernel_inttype ninter, kernel_inttype nbatch, kernel_inttype m, kernel_inttype n,
                                kernel_inttype k, double alpha, const double *A, kernel_inttype bstrd_A,
                                kernel_inttype istrd_A, kernel_inttype jstrd_A, const double *B, kernel_inttype bstrd_B,
                                kernel_inttype istrd_B, kernel_inttype jstrd_B, double beta, double *C,
                                kernel_inttype bstrd_C, kernel_inttype istrd_C, kernel_inttype jstrd_C) {

	auto vec_width = svcntd();
	svfloat64_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
	svfloat64_t alpha_v = svdup_f64(alpha);
	svfloat64_t beta_v = svdup_f64(beta);
	const svbool_t pg = svptrue_b64();

	for (kernel_inttype bi = 0; bi < nbatch; bi++) {
		auto Ab = A + bi * bstrd_A;
		auto Bb = B + bi * bstrd_B;
		auto Cb = C + bi * bstrd_C;

		for (kernel_inttype j = 0; j < n; j++) {
			auto Bbj = Bb + j * jstrd_B;
			auto Cbj = Cb + j * jstrd_C;

			for (kernel_inttype i = 0; i < m; i++) {
				auto Abi = Ab + i * jstrd_A; // kernels work on transposed A
				auto Cbij = Cbj + i * istrd_C;

				tmp0 = svdup_f64(0.0);
				tmp1 = svdup_f64(0.0);
				tmp2 = svdup_f64(0.0);
				tmp3 = svdup_f64(0.0);
				tmp4 = svdup_f64(0.0);
				tmp5 = svdup_f64(0.0);
				tmp6 = svdup_f64(0.0);
				tmp7 = svdup_f64(0.0);

				for (kernel_inttype ki = 0; ki < k; ki++) {
					auto Abij = Abi + ki * istrd_A;
					auto Bbij = Bbj + ki * istrd_B;

					auto Av0 = svld1_f64(pg, Abij);
					auto Bv0 = svld1_f64(pg, Bbij);

					auto Av1 = svld1_f64(pg, Abij + vec_width);
					auto Bv1 = svld1_f64(pg, Bbij + vec_width);

					auto Av2 = svld1_f64(pg, Abij + vec_width * 2);
					auto Bv2 = svld1_f64(pg, Bbij + vec_width * 2);

					auto Av3 = svld1_f64(pg, Abij + vec_width * 3);
					auto Bv3 = svld1_f64(pg, Bbij + vec_width * 3);

					auto Av4 = svld1_f64(pg, Abij + vec_width * 4);
					auto Bv4 = svld1_f64(pg, Bbij + vec_width * 4);

					auto Av5 = svld1_f64(pg, Abij + vec_width * 5);
					auto Bv5 = svld1_f64(pg, Bbij + vec_width * 5);

					auto Av6 = svld1_f64(pg, Abij + vec_width * 6);
					auto Bv6 = svld1_f64(pg, Bbij + vec_width * 6);

					auto Av7 = svld1_f64(pg, Abij + vec_width * 7);
					auto Bv7 = svld1_f64(pg, Bbij + vec_width * 7);

					tmp0 = svmla_f64_x(pg, tmp0, Av0, Bv0);  // Accumulate into a temporary
					tmp1 = svmla_f64_x(pg, tmp1, Av1, Bv1);
					tmp2 = svmla_f64_x(pg, tmp2, Av2, Bv2);
					tmp3 = svmla_f64_x(pg, tmp3, Av3, Bv3);
					tmp4 = svmla_f64_x(pg, tmp4, Av4, Bv4);
					tmp5 = svmla_f64_x(pg, tmp5, Av5, Bv5);
					tmp6 = svmla_f64_x(pg, tmp6, Av6, Bv6);
					tmp7 = svmla_f64_x(pg, tmp7, Av7, Bv7);

				} // k

				// Load C, multiply by beta, accumulate alpha*tmp into C and store

				auto Cv0 = load_and_scale_c<BetaIsZero>(pg, Cbij, beta_v);

				auto Cv1 = load_and_scale_c<BetaIsZero>(pg, Cbij + vec_width, beta_v);

				auto Cv2 = load_and_scale_c<BetaIsZero>(pg, Cbij + vec_width * 2, beta_v);

				auto Cv3 = load_and_scale_c<BetaIsZero>(pg, Cbij + vec_width * 3, beta_v);

				auto Cv4 = load_and_scale_c<BetaIsZero>(pg, Cbij + vec_width * 4, beta_v);

				auto Cv5 = load_and_scale_c<BetaIsZero>(pg, Cbij + vec_width * 5, beta_v);

				auto Cv6 = load_and_scale_c<BetaIsZero>(pg, Cbij + vec_width * 6, beta_v);

				auto Cv7 = load_and_scale_c<BetaIsZero>(pg, Cbij + vec_width * 7, beta_v);

				tmp0 = svmla_f64_x(pg, Cv0, tmp0, alpha_v);
				svst1_f64(pg, Cbij, tmp0);
				tmp1 = svmla_f64_x(pg, Cv1, tmp1, alpha_v);
				svst1_f64(pg, Cbij + vec_width, tmp1);
				tmp2 = svmla_f64_x(pg, Cv2, tmp2, alpha_v);
				svst1_f64(pg, Cbij + vec_width * 2, tmp2);
				tmp3 = svmla_f64_x(pg, Cv3, tmp3, alpha_v);
				svst1_f64(pg, Cbij + vec_width * 3, tmp3);
				tmp4 = svmla_f64_x(pg, Cv4, tmp4, alpha_v);
				svst1_f64(pg, Cbij + vec_width * 4, tmp4);
				tmp5 = svmla_f64_x(pg, Cv5, tmp5, alpha_v);
				svst1_f64(pg, Cbij + vec_width * 5, tmp5);
				tmp6 = svmla_f64_x(pg, Cv6, tmp6, alpha_v);
				svst1_f64(pg, Cbij + vec_width * 6, tmp6);
				tmp7 = svmla_f64_x(pg, Cv7, tmp7, alpha_v);
				svst1_f64(pg, Cbij + vec_width * 7, tmp7);
			} // m

		} // n

	} // nbatch
}

template<bool BetaIsZero>
static void dgemm_sve_vla_impl(kernel_inttype ninter, kernel_inttype nbatch, kernel_inttype m, kernel_inttype n,
                               kernel_inttype k, double alpha, const double *A, kernel_inttype bstrd_A,
                               kernel_inttype istrd_A, kernel_inttype jstrd_A, const double *B, kernel_inttype bstrd_B,
                               kernel_inttype istrd_B, kernel_inttype jstrd_B, double beta, double *C,
                               kernel_inttype bstrd_C, kernel_inttype istrd_C, kernel_inttype jstrd_C) {

	svfloat64_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

	auto vec_width = svcntd();
	svfloat64_t alpha_v = svdup_f64(alpha);
	svfloat64_t beta_v = svdup_f64(beta);

	for (kernel_inttype bi = 0; bi < nbatch; bi++) {
		auto Ab = A + bi * bstrd_A;
		auto Bb = B + bi * bstrd_B;
		auto Cb = C + bi * bstrd_C;

		for (kernel_inttype j = 0; j < n; j++) {
			auto Bbj = Bb + j * jstrd_B;
			auto Cbj = Cb + j * jstrd_C;

			for (kernel_inttype i = 0; i < m; i++) {
				auto Abi = Ab + i * jstrd_A; // kernels work on transposed A
				auto Cbij = Cbj + i * istrd_C;

				tmp0 = svdup_f64(0.0);
				tmp1 = svdup_f64(0.0);
				tmp2 = svdup_f64(0.0);
				tmp3 = svdup_f64(0.0);
				tmp4 = svdup_f64(0.0);
				tmp5 = svdup_f64(0.0);
				tmp6 = svdup_f64(0.0);
				tmp7 = svdup_f64(0.0);

				for (kernel_inttype ki = 0; ki < k; ki++) {
					auto Abij = Abi + ki * istrd_A;
					auto Bbij = Bbj + ki * istrd_B;

					kernel_inttype inter = 0;
					svbool_t pg = svwhilelt_b64((kernel_inttype) 0, ninter);
					do {
						auto Av0 = svld1_f64(pg, Abij + inter);
						auto Bv0 = svld1_f64(pg, Bbij + inter);
						tmp0 = svmla_f64_x(pg, tmp0, Av0, Bv0); // Accumulate into a temporary

						inter += vec_width;
						pg = svwhilelt_b64(inter, ninter);

						if (!svptest_any(svptrue_b64(), pg)) {
							break;
						}

						auto Av1 = svld1_f64(pg, Abij + inter);
						auto Bv1 = svld1_f64(pg, Bbij + inter);
						tmp1 = svmla_f64_x(pg, tmp1, Av1, Bv1);

						inter += vec_width;
						pg = svwhilelt_b64(inter, ninter);

						if (!svptest_any(svptrue_b64(), pg)) {
							break;
						}

						auto Av2 = svld1_f64(pg, Abij + inter);
						auto Bv2 = svld1_f64(pg, Bbij + inter);
						tmp2 = svmla_f64_x(pg, tmp2, Av2, Bv2);

						inter += vec_width;
						pg = svwhilelt_b64(inter, ninter);

						if (!svptest_any(svptrue_b64(), pg)) {
							break;
						}

						auto Av3 = svld1_f64(pg, Abij + inter);
						auto Bv3 = svld1_f64(pg, Bbij + inter);
						tmp3 = svmla_f64_x(pg, tmp3, Av3, Bv3);

						inter += vec_width;
						pg = svwhilelt_b64(inter, ninter);

						if (!svptest_any(svptrue_b64(), pg)) {
							break;
						}

						auto Av4 = svld1_f64(pg, Abij + inter);
						auto Bv4 = svld1_f64(pg, Bbij + inter);
						tmp4 = svmla_f64_x(pg, tmp4, Av4, Bv4);

						inter += vec_width;
						pg = svwhilelt_b64(inter, ninter);

						if (!svptest_any(svptrue_b64(), pg)) {
							break;
						}

						auto Av5 = svld1_f64(pg, Abij + inter);
						auto Bv5 = svld1_f64(pg, Bbij + inter);
						tmp5 = svmla_f64_x(pg, tmp5, Av5, Bv5);

						inter += vec_width;
						pg = svwhilelt_b64(inter, ninter);

						if (!svptest_any(svptrue_b64(), pg)) {
							break;
						}

						auto Av6 = svld1_f64(pg, Abij + inter);
						auto Bv6 = svld1_f64(pg, Bbij + inter);
						tmp6 = svmla_f64_x(pg, tmp6, Av6, Bv6);

						inter += vec_width;
						pg = svwhilelt_b64(inter, ninter);

						if (!svptest_any(svptrue_b64(), pg)) {
							break;
						}

						auto Av7 = svld1_f64(pg, Abij + inter);
						auto Bv7 = svld1_f64(pg, Bbij + inter);
						tmp7 = svmla_f64_x(pg, tmp7, Av7, Bv7);

					} while (0); // ninter - contig dimension, so must be inner loop

				} // k

				// Load C, multiply by beta, accumulate alpha*tmp into C and store

				kernel_inttype inter = 0;
				svbool_t pg = svwhilelt_b64((kernel_inttype) 0, ninter);
				do {
					auto Cv0 = load_and_scale_c<BetaIsZero>(pg, Cbij + inter, beta_v);

					tmp0 = svmla_f64_x(pg, Cv0, tmp0, alpha_v);
					svst1_f64(pg, Cbij + inter, tmp0);

					inter += vec_width;
					pg = svwhilelt_b64(inter, ninter);

					if (!svptest_any(svptrue_b64(), pg)) {
						break;
					}

					auto Cv1 = load_and_scale_c<BetaIsZero>(pg, Cbij + inter, beta_v);

					tmp1 = svmla_f64_x(pg, Cv1, tmp1, alpha_v);
					svst1_f64(pg, Cbij + inter, tmp1);

					inter += vec_width;
					pg = svwhilelt_b64(inter, ninter);

					if (!svptest_any(svptrue_b64(), pg)) {
						break;
					}

					auto Cv2 = load_and_scale_c<BetaIsZero>(pg, Cbij + inter, beta_v);

					tmp2 = svmla_f64_x(pg, Cv2, tmp2, alpha_v);
					svst1_f64(pg, Cbij + inter, tmp2);

					inter += vec_width;
					pg = svwhilelt_b64(inter, ninter);

					if (!svptest_any(svptrue_b64(), pg)) {
						break;
					}

					auto Cv3 = load_and_scale_c<BetaIsZero>(pg, Cbij + inter, beta_v);

					tmp3 = svmla_f64_x(pg, Cv3, tmp3, alpha_v);
					svst1_f64(pg, Cbij + inter, tmp3);

					inter += vec_width;
					pg = svwhilelt_b64(inter, ninter);

					if (!svptest_any(svptrue_b64(), pg)) {
						break;
					}

					auto Cv4 = load_and_scale_c<BetaIsZero>(pg, Cbij + inter, beta_v);

					tmp4 = svmla_f64_x(pg, Cv4, tmp4, alpha_v);
					svst1_f64(pg, Cbij + inter, tmp4);

					inter += vec_width;
					pg = svwhilelt_b64(inter, ninter);

					if (!svptest_any(svptrue_b64(), pg)) {
						break;
					}

					auto Cv5 = load_and_scale_c<BetaIsZero>(pg, Cbij + inter, beta_v);

					tmp5 = svmla_f64_x(pg, Cv5, tmp5, alpha_v);
					svst1_f64(pg, Cbij + inter, tmp5);

					inter += vec_width;
					pg = svwhilelt_b64(inter, ninter);

					if (!svptest_any(svptrue_b64(), pg)) {
						break;
					}

					auto Cv6 = load_and_scale_c<BetaIsZero>(pg, Cbij + inter, beta_v);

					tmp6 = svmla_f64_x(pg, Cv6, tmp6, alpha_v);
					svst1_f64(pg, Cbij + inter, tmp6);

					inter += vec_width;
					pg = svwhilelt_b64(inter, ninter);

					if (!svptest_any(svptrue_b64(), pg)) {
						break;
					}

					auto Cv7 = load_and_scale_c<BetaIsZero>(pg, Cbij + inter, beta_v);

					tmp7 = svmla_f64_x(pg, Cv7, tmp7, alpha_v);
					svst1_f64(pg, Cbij + inter, tmp7);

				} while (0);

			} // m

		} // n

	} // nbatch
}

// Select the beta specialization once, before entering the matrix loops.
void dgemm_sve_vls8(kernel_inttype ninter, kernel_inttype nbatch, kernel_inttype m, kernel_inttype n,
                    kernel_inttype k, double alpha, const double *A, kernel_inttype bstrd_A,
                    kernel_inttype istrd_A, kernel_inttype jstrd_A, const double *B, kernel_inttype bstrd_B,
                    kernel_inttype istrd_B, kernel_inttype jstrd_B, double beta, double *C,
                    kernel_inttype bstrd_C, kernel_inttype istrd_C, kernel_inttype jstrd_C) {
	if (beta == 0.0) {
		dgemm_sve_vls8_impl<true>(ninter, nbatch, m, n, k, alpha, A, bstrd_A, istrd_A, jstrd_A,
		                          B, bstrd_B, istrd_B, jstrd_B, beta, C, bstrd_C, istrd_C, jstrd_C);
	}
	else {
		dgemm_sve_vls8_impl<false>(ninter, nbatch, m, n, k, alpha, A, bstrd_A, istrd_A, jstrd_A,
		                           B, bstrd_B, istrd_B, jstrd_B, beta, C, bstrd_C, istrd_C, jstrd_C);
	}
}

void dgemm_sve_vla(kernel_inttype ninter, kernel_inttype nbatch, kernel_inttype m, kernel_inttype n,
                   kernel_inttype k, double alpha, const double *A, kernel_inttype bstrd_A,
                   kernel_inttype istrd_A, kernel_inttype jstrd_A, const double *B, kernel_inttype bstrd_B,
                   kernel_inttype istrd_B, kernel_inttype jstrd_B, double beta, double *C,
                   kernel_inttype bstrd_C, kernel_inttype istrd_C, kernel_inttype jstrd_C) {
	if (beta == 0.0) {
		dgemm_sve_vla_impl<true>(ninter, nbatch, m, n, k, alpha, A, bstrd_A, istrd_A, jstrd_A,
		                         B, bstrd_B, istrd_B, jstrd_B, beta, C, bstrd_C, istrd_C, jstrd_C);
	}
	else {
		dgemm_sve_vla_impl<false>(ninter, nbatch, m, n, k, alpha, A, bstrd_A, istrd_A, jstrd_A,
		                          B, bstrd_B, istrd_B, jstrd_B, beta, C, bstrd_C, istrd_C, jstrd_C);
	}
}

} // namespace perflibs::interleave_batch

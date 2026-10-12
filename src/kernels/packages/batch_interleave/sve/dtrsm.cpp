/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "packages/batch_interleave/solve/kernels/trsm_kernels.hpp"
#include "matrix/interleave_batch.hpp"

namespace perflibs::interleave_batch {

inline void dtrsm_sve_vla(kernel_inttype ninter, const double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A,
                          double *B_p, kernel_inttype istrd_B, kernel_inttype jstrd_B, kernel_inttype m, kernel_inttype n,
                          bool nounit, double alpha) {
	auto vec_width = svcntd();
	const svbool_t ptrue = svptrue_b64();
	svbool_t pg0, pg1, pg2, pg3, pg4, pg5, pg6, pg7;
	kernel_inttype inter;
	inter = 0;
	pg0 = svwhilelt_b64(inter, ninter);
	inter += svcntd();
	pg1 = svwhilelt_b64(inter, ninter);
	inter += svcntd();
	pg2 = svwhilelt_b64(inter, ninter);
	inter += svcntd();
	pg3 = svwhilelt_b64(inter, ninter);
	inter += svcntd();
	pg4 = svwhilelt_b64(inter, ninter);
	inter += svcntd();
	pg5 = svwhilelt_b64(inter, ninter);
	inter += svcntd();
	pg6 = svwhilelt_b64(inter, ninter);
	inter += svcntd();
	pg7 = svwhilelt_b64(inter, ninter);
	const double one = 1.0;

	// Initialize all bar the first to avoid compiler errors
	// with gcc10, 11 when using -Wuninitialized -Werror
	svfloat64_t a_ii0;
	svfloat64_t a_ii1 = svdup_f64(0);
	svfloat64_t a_ii2 = svdup_f64(0);
	svfloat64_t a_ii3 = svdup_f64(0);
	svfloat64_t a_ii4 = svdup_f64(0);
	svfloat64_t a_ii5 = svdup_f64(0);
	svfloat64_t a_ii6 = svdup_f64(0);
	svfloat64_t a_ii7 = svdup_f64(0);
	for (kernel_inttype i = m - 1_ki; i >= 0_ki; i--) {
		auto a_ii = A_p + i*istrd_A + i*jstrd_A;
		if (nounit) {
			do {
				a_ii0 = svld1_vnum(pg0, a_ii, 0);
				a_ii0 = svdivr_x(pg0, a_ii0, one);

				if (!svptest_any(ptrue, pg1)) {
					break;
				}

				a_ii1 = svld1_vnum(pg1, a_ii, 1);
				a_ii1 = svdivr_x(pg1, a_ii1, one);

				if (!svptest_any(ptrue, pg2)) {
					break;
				}

				a_ii2 = svld1_vnum(pg2, a_ii, 2);
				a_ii2 = svdivr_x(pg2, a_ii2, one);

				if (!svptest_any(ptrue, pg3)) {
					break;
				}

				a_ii3 = svld1_vnum(pg3, a_ii, 3);
				a_ii3 = svdivr_x(pg3, a_ii3, one);

				if (!svptest_any(ptrue, pg4)) {
					break;
				}

				a_ii4 = svld1_vnum(pg4, a_ii, 4);
				a_ii4 = svdivr_x(pg4, a_ii4, one);

				if (!svptest_any(ptrue, pg5)) {
					break;
				}

				a_ii5 = svld1_vnum(pg5, a_ii, 5);
				a_ii5 = svdivr_x(pg5, a_ii5, one);

				if (!svptest_any(ptrue, pg6)) {
					break;
				}

				a_ii6 = svld1_vnum(pg6, a_ii, 6);
				a_ii6 = svdivr_x(pg6, a_ii6, one);

				if (!svptest_any(ptrue, pg7)) {
					break;
				}

				a_ii7 = svld1_vnum(pg7, a_ii, 7);
				a_ii7 = svdivr_x(pg7, a_ii7, one);
			} while (0);
		}

		// Initialize all bar the first to avoid compiler errors
		// with gcc10, 11 when using -Wuninitialized -Werror
		svfloat64_t b_ij0;
		svfloat64_t b_ij1 = svdup_f64(0);
		svfloat64_t b_ij2 = svdup_f64(0);
		svfloat64_t b_ij3 = svdup_f64(0);
		svfloat64_t b_ij4 = svdup_f64(0);
		svfloat64_t b_ij5 = svdup_f64(0);
		svfloat64_t b_ij6 = svdup_f64(0);
		svfloat64_t b_ij7 = svdup_f64(0);
		for (kernel_inttype j = 0_ki; j<n; j++) {
			auto b_ij = B_p + i*istrd_B + j*jstrd_B;
			// B(i, j) *= alpha
			do {
				b_ij0 = svld1_vnum(pg0, b_ij, 0);
				b_ij0 = svmul_x(pg0, b_ij0, alpha);

				if (!svptest_any(ptrue, pg1)) {
					break;
				}

				b_ij1 = svld1_vnum(pg1, b_ij, 1);
				b_ij1 = svmul_x(pg1, b_ij1, alpha);

				if (!svptest_any(ptrue, pg2)) {
					break;
				}

				b_ij2 = svld1_vnum(pg2, b_ij, 2);
				b_ij2 = svmul_x(pg2, b_ij2, alpha);

				if (!svptest_any(ptrue, pg3)) {
					break;
				}

				b_ij3 = svld1_vnum(pg3, b_ij, 3);
				b_ij3 = svmul_x(pg3, b_ij3, alpha);

				if (!svptest_any(ptrue, pg4)) {
					break;
				}

				b_ij4 = svld1_vnum(pg4, b_ij, 4);
				b_ij4 = svmul_x(pg4, b_ij4, alpha);

				if (!svptest_any(ptrue, pg5)) {
					break;
				}

				b_ij5 = svld1_vnum(pg5, b_ij, 5);
				b_ij5 = svmul_x(pg5, b_ij5, alpha);

				if (!svptest_any(ptrue, pg6)) {
					break;
				}

				b_ij6 = svld1_vnum(pg6, b_ij, 6);
				b_ij6 = svmul_x(pg6, b_ij6, alpha);

				if (!svptest_any(ptrue, pg7)) {
					break;
				}

				b_ij7 = svld1_vnum(pg7, b_ij, 7);
				b_ij7 = svmul_x(pg7, b_ij7, alpha);
			} while (0);

			for (kernel_inttype k = i + 1_ki; k<m; k++) {
				svfloat64_t a_ik0, a_ik1, a_ik2, a_ik3, a_ik4, a_ik5, a_ik6, a_ik7;
				auto a_ik = A_p + i*istrd_A + k*jstrd_A;
				svfloat64_t b_kj0, b_kj1, b_kj2, b_kj3, b_kj4, b_kj5, b_kj6, b_kj7;
				auto b_kj = B_p + k*istrd_B + j*jstrd_B;
				// B(i, j) = B(i, j) - A(i, k)*B(k, j)
				do {
					a_ik0 = svld1_vnum(pg0, a_ik, 0);
					b_kj0 = svld1_vnum(pg0, b_kj, 0);
					b_ij0 = svmls_x(pg0, b_ij0, a_ik0, b_kj0);

					if (!svptest_any(ptrue, pg1)) {
						break;
					}

					a_ik1 = svld1_vnum(pg1, a_ik, 1);
					b_kj1 = svld1_vnum(pg1, b_kj, 1);
					b_ij1 = svmls_x(pg1, b_ij1, a_ik1, b_kj1);

					if (!svptest_any(ptrue, pg2)) {
						break;
					}

					a_ik2 = svld1_vnum(pg2, a_ik, 2);
					b_kj2 = svld1_vnum(pg2, b_kj, 2);
					b_ij2 = svmls_x(pg2, b_ij2, a_ik2, b_kj2);

					if (!svptest_any(ptrue, pg3)) {
						break;
					}

					a_ik3 = svld1_vnum(pg3, a_ik, 3);
					b_kj3 = svld1_vnum(pg3, b_kj, 3);
					b_ij3 = svmls_x(pg3, b_ij3, a_ik3, b_kj3);

					if (!svptest_any(ptrue, pg4)) {
						break;
					}

					a_ik4 = svld1_vnum(pg4, a_ik, 4);
					b_kj4 = svld1_vnum(pg4, b_kj, 4);
					b_ij4 = svmls_x(pg4, b_ij4, a_ik4, b_kj4);

					if (!svptest_any(ptrue, pg5)) {
						break;
					}

					a_ik5 = svld1_vnum(pg5, a_ik, 5);
					b_kj5 = svld1_vnum(pg5, b_kj, 5);
					b_ij5 = svmls_x(pg5, b_ij5, a_ik5, b_kj5);

					if (!svptest_any(ptrue, pg6)) {
						break;
					}

					a_ik6 = svld1_vnum(pg6, a_ik, 6);
					b_kj6 = svld1_vnum(pg6, b_kj, 6);
					b_ij6 = svmls_x(pg6, b_ij6, a_ik6, b_kj6);

					if (!svptest_any(ptrue, pg7)) {
						break;
					}

					a_ik7 = svld1_vnum(pg7, a_ik, 7);
					b_kj7 = svld1_vnum(pg7, b_kj, 7);
					b_ij7 = svmls_x(pg7, b_ij7, a_ik7, b_kj7);
				} while (0);
			}

			if (nounit) {
				// B(i, j) /= A(i, i)
				do {
					b_ij0 = svmul_x(pg0, b_ij0, a_ii0);
					svst1_vnum(pg0, b_ij, 0, b_ij0);

					if (!svptest_any(ptrue, pg1)) {
						break;
					}

					b_ij1 = svmul_x(pg1, b_ij1, a_ii1);
					svst1_vnum(pg1, b_ij, 1, b_ij1);

					if (!svptest_any(ptrue, pg2)) {
						break;
					}

					b_ij2 = svmul_x(pg2, b_ij2, a_ii2);
					svst1_vnum(pg2, b_ij, 2, b_ij2);

					if (!svptest_any(ptrue, pg3)) {
						break;
					}

					b_ij3 = svmul_x(pg3, b_ij3, a_ii3);
					svst1_vnum(pg3, b_ij, 3, b_ij3);

					if (!svptest_any(ptrue, pg4)) {
						break;
					}

					b_ij4 = svmul_x(pg4, b_ij4, a_ii4);
					svst1_vnum(pg4, b_ij, 4, b_ij4);

					if (!svptest_any(ptrue, pg5)) {
						break;
					}

					b_ij5 = svmul_x(pg5, b_ij5, a_ii5);
					svst1_vnum(pg5, b_ij, 5, b_ij5);

					if (!svptest_any(ptrue, pg6)) {
						break;
					}

					b_ij6 = svmul_x(pg6, b_ij6, a_ii6);
					svst1_vnum(pg6, b_ij, 6, b_ij6);

					if (!svptest_any(ptrue, pg7)) {
						break;
					}

					b_ij7 = svmul_x(pg7, b_ij7, a_ii7);
					svst1_vnum(pg7, b_ij, 7, b_ij7);
				} while (0);
			}
			else {
				do {
					svst1_vnum(pg0, b_ij, 0, b_ij0);

					if (!svptest_any(ptrue, pg1)) {
						break;
					}

					svst1_vnum(pg1, b_ij, 1, b_ij1);

					if (!svptest_any(ptrue, pg2)) {
						break;
					}

					svst1_vnum(pg2, b_ij, 2, b_ij2);

					if (!svptest_any(ptrue, pg3)) {
						break;
					}

					svst1_vnum(pg3, b_ij, 3, b_ij3);

					if (!svptest_any(ptrue, pg4)) {
						break;
					}

					svst1_vnum(pg4, b_ij, 4, b_ij4);

					if (!svptest_any(ptrue, pg5)) {
						break;
					}

					svst1_vnum(pg5, b_ij, 5, b_ij5);

					if (!svptest_any(ptrue, pg6)) {
						break;
					}

					svst1_vnum(pg6, b_ij, 6, b_ij6);

					if (!svptest_any(ptrue, pg7)) {
						break;
					}

					svst1_vnum(pg7, b_ij, 7, b_ij7);
				} while (0);
			}
		}
	}
}

inline void dtrsm_sve_vls8(kernel_inttype ninter, const double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A,
                          double *B_p, kernel_inttype istrd_B, kernel_inttype jstrd_B, kernel_inttype m, kernel_inttype n,
                          bool nounit, double alpha) {
	const svbool_t ptrue = svptrue_b64();
	const double one = 1.0;

	for (kernel_inttype i = m - 1_ki; i >= 0_ki; i--) {
		svfloat64_t a_ii0, a_ii1, a_ii2, a_ii3, a_ii4, a_ii5, a_ii6, a_ii7;
		auto a_ii = A_p + i*istrd_A + i*jstrd_A;
		if (nounit) {
			a_ii0 = svld1_vnum(ptrue, a_ii, 0);
			a_ii0 = svdivr_x(ptrue, a_ii0, one);
			a_ii1 = svld1_vnum(ptrue, a_ii, 1);
			a_ii1 = svdivr_x(ptrue, a_ii1, one);
			a_ii2 = svld1_vnum(ptrue, a_ii, 2);
			a_ii2 = svdivr_x(ptrue, a_ii2, one);
			a_ii3 = svld1_vnum(ptrue, a_ii, 3);
			a_ii3 = svdivr_x(ptrue, a_ii3, one);
			a_ii4 = svld1_vnum(ptrue, a_ii, 4);
			a_ii4 = svdivr_x(ptrue, a_ii4, one);
			a_ii5 = svld1_vnum(ptrue, a_ii, 5);
			a_ii5 = svdivr_x(ptrue, a_ii5, one);
			a_ii6 = svld1_vnum(ptrue, a_ii, 6);
			a_ii6 = svdivr_x(ptrue, a_ii6, one);
			a_ii7 = svld1_vnum(ptrue, a_ii, 7);
			a_ii7 = svdivr_x(ptrue, a_ii7, one);
		}

		for (kernel_inttype j = 0_ki; j<n; j++) {
			svfloat64_t b_ij0, b_ij1, b_ij2, b_ij3, b_ij4, b_ij5, b_ij6, b_ij7;
			auto b_ij = B_p + i*istrd_B + j*jstrd_B;
			// B(i, j) *= alpha
			b_ij0 = svld1_vnum(ptrue, b_ij, 0);
			b_ij0 = svmul_x(ptrue, b_ij0, alpha);
			b_ij1 = svld1_vnum(ptrue, b_ij, 1);
			b_ij1 = svmul_x(ptrue, b_ij1, alpha);
			b_ij2 = svld1_vnum(ptrue, b_ij, 2);
			b_ij2 = svmul_x(ptrue, b_ij2, alpha);
			b_ij3 = svld1_vnum(ptrue, b_ij, 3);
			b_ij3 = svmul_x(ptrue, b_ij3, alpha);
			b_ij4 = svld1_vnum(ptrue, b_ij, 4);
			b_ij4 = svmul_x(ptrue, b_ij4, alpha);
			b_ij5 = svld1_vnum(ptrue, b_ij, 5);
			b_ij5 = svmul_x(ptrue, b_ij5, alpha);
			b_ij6 = svld1_vnum(ptrue, b_ij, 6);
			b_ij6 = svmul_x(ptrue, b_ij6, alpha);
			b_ij7 = svld1_vnum(ptrue, b_ij, 7);
			b_ij7 = svmul_x(ptrue, b_ij7, alpha);

			for (kernel_inttype k = i + 1_ki; k<m; k++) {
				svfloat64_t a_ik0, a_ik1, a_ik2, a_ik3, a_ik4, a_ik5, a_ik6, a_ik7;
				auto a_ik = A_p + i*istrd_A + k*jstrd_A;
				svfloat64_t b_kj0, b_kj1, b_kj2, b_kj3, b_kj4, b_kj5, b_kj6, b_kj7;
				auto b_kj = B_p + k*istrd_B + j*jstrd_B;
				// B(i, j) = B(i, j) - A(i, k)*B(k, j)
				a_ik0 = svld1_vnum(ptrue, a_ik, 0);
				b_kj0 = svld1_vnum(ptrue, b_kj, 0);
				b_ij0 = svmls_x(ptrue, b_ij0, a_ik0, b_kj0);
				a_ik1 = svld1_vnum(ptrue, a_ik, 1);
				b_kj1 = svld1_vnum(ptrue, b_kj, 1);
				b_ij1 = svmls_x(ptrue, b_ij1, a_ik1, b_kj1);
				a_ik2 = svld1_vnum(ptrue, a_ik, 2);
				b_kj2 = svld1_vnum(ptrue, b_kj, 2);
				b_ij2 = svmls_x(ptrue, b_ij2, a_ik2, b_kj2);
				a_ik3 = svld1_vnum(ptrue, a_ik, 3);
				b_kj3 = svld1_vnum(ptrue, b_kj, 3);
				b_ij3 = svmls_x(ptrue, b_ij3, a_ik3, b_kj3);
				a_ik4 = svld1_vnum(ptrue, a_ik, 4);
				b_kj4 = svld1_vnum(ptrue, b_kj, 4);
				b_ij4 = svmls_x(ptrue, b_ij4, a_ik4, b_kj4);
				a_ik5 = svld1_vnum(ptrue, a_ik, 5);
				b_kj5 = svld1_vnum(ptrue, b_kj, 5);
				b_ij5 = svmls_x(ptrue, b_ij5, a_ik5, b_kj5);
				a_ik6 = svld1_vnum(ptrue, a_ik, 6);
				b_kj6 = svld1_vnum(ptrue, b_kj, 6);
				b_ij6 = svmls_x(ptrue, b_ij6, a_ik6, b_kj6);
				a_ik7 = svld1_vnum(ptrue, a_ik, 7);
				b_kj7 = svld1_vnum(ptrue, b_kj, 7);
				b_ij7 = svmls_x(ptrue, b_ij7, a_ik7, b_kj7);
			}

			if (nounit) {
				// B(i, j) /= A(i, i)
				b_ij0 = svmul_x(ptrue, b_ij0, a_ii0);
				b_ij1 = svmul_x(ptrue, b_ij1, a_ii1);
				b_ij2 = svmul_x(ptrue, b_ij2, a_ii2);
				b_ij3 = svmul_x(ptrue, b_ij3, a_ii3);
				b_ij4 = svmul_x(ptrue, b_ij4, a_ii4);
				b_ij5 = svmul_x(ptrue, b_ij5, a_ii5);
				b_ij6 = svmul_x(ptrue, b_ij6, a_ii6);
				b_ij7 = svmul_x(ptrue, b_ij7, a_ii7);
			}
			svst1_vnum(ptrue, b_ij, 0, b_ij0);
			svst1_vnum(ptrue, b_ij, 1, b_ij1);
			svst1_vnum(ptrue, b_ij, 2, b_ij2);
			svst1_vnum(ptrue, b_ij, 3, b_ij3);
			svst1_vnum(ptrue, b_ij, 4, b_ij4);
			svst1_vnum(ptrue, b_ij, 5, b_ij5);
			svst1_vnum(ptrue, b_ij, 6, b_ij6);
			svst1_vnum(ptrue, b_ij, 7, b_ij7);
		}
	}
}

void dtrsm_sve_kernel(kernel_inttype ninter, const double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A,
                      double *B_p, kernel_inttype istrd_B, kernel_inttype jstrd_B, kernel_inttype m, kernel_inttype n,
                      bool nounit, double alpha) {
	// The input parameters assume an LNU (left, notrans, upper) case.
	// For other cases, the problem is converted to an LNU problem by simple
	// transformations of the input parameters before calling this routine.
	auto vec_width = svcntd();
	kernel_inttype ni = 0;
	kernel_inttype ninter_remain = ninter;
	kernel_inttype vec8 = vec_width * 8;

	for (; ni < ninter - vec8 + 1; ni += vec8) {
		perflibs::interleave_batch::dtrsm_sve_vls8(vec8, A_p + ni, istrd_A, jstrd_A, B_p + ni,
		                                         istrd_B, jstrd_B, m, n, nounit, alpha);
		ninter_remain -= vec8;
	}
	if (ninter_remain > 0) {
		perflibs::interleave_batch::dtrsm_sve_vla(ninter_remain, A_p + ni, istrd_A, jstrd_A, B_p + ni,
		                                         istrd_B, jstrd_B, m, n, nounit, alpha);
	}
}

} // namespace perflibs::interleave_batch

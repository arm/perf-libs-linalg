/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include <arm_sve.h>
#include "packages/batch_interleave/matmul/kernels/gemm_kernels.hpp"

namespace perflibs::interleave_batch {
inline void dger_sve_vla(kernel_inttype ninter, kernel_inttype m, kernel_inttype n, double alpha,
                         const double *x_p, kernel_inttype istrd_x,
                         const double *y_p, kernel_inttype istrd_y,
                         double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A) {
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
	// Initialize all bar the first to avoid compiler errors
	// with gcc10, 11 when using -Wuninitialized -Werror
	svfloat64_t x_i0;
	svfloat64_t x_i1 = svdup_f64(0);
	svfloat64_t x_i2 = svdup_f64(0);
	svfloat64_t x_i3 = svdup_f64(0);
	svfloat64_t x_i4 = svdup_f64(0);
	svfloat64_t x_i5 = svdup_f64(0);
	svfloat64_t x_i6 = svdup_f64(0);
	svfloat64_t x_i7 = svdup_f64(0);
	for (kernel_inttype i = 0; i != m; i ++) {
		auto x_i = x_p + i*istrd_x;
		// x(i) *= alpha
		do {
			x_i0 = svld1_vnum(pg0, x_i, 0);
			x_i0 = svmul_x(pg0, x_i0, alpha);

			if (!svptest_any(ptrue, pg1)) {
				break;
			}

			x_i1 = svld1_vnum(pg1, x_i, 1);
			x_i1 = svmul_x(pg1, x_i1, alpha);

			if (!svptest_any(ptrue, pg2)) {
				break;
			}

			x_i2 = svld1_vnum(pg2, x_i, 2);
			x_i2 = svmul_x(pg2, x_i2, alpha);

			if (!svptest_any(ptrue, pg3)) {
				break;
			}

			x_i3 = svld1_vnum(pg3, x_i, 3);
			x_i3 = svmul_x(pg3, x_i3, alpha);

			if (!svptest_any(ptrue, pg4)) {
				break;
			}

			x_i4 = svld1_vnum(pg4, x_i, 4);
			x_i4 = svmul_x(pg4, x_i4, alpha);

			if (!svptest_any(ptrue, pg5)) {
				break;
			}

			x_i5 = svld1_vnum(pg5, x_i, 5);
			x_i5 = svmul_x(pg5, x_i5, alpha);

			if (!svptest_any(ptrue, pg6)) {
				break;
			}

			x_i6 = svld1_vnum(pg6, x_i, 6);
			x_i6 = svmul_x(pg6, x_i6, alpha);

			if (!svptest_any(ptrue, pg7)) {
				break;
			}

			x_i7 = svld1_vnum(pg7, x_i, 7);
			x_i7 = svmul_x(pg7, x_i7, alpha);
		} while (0);

		for (kernel_inttype j = 0; j < n; j++) {
			svfloat64_t y_j0, y_j1, y_j2, y_j3, y_j4, y_j5, y_j6, y_j7;
			auto y_j = y_p + j*istrd_y;
			svfloat64_t A_ij0, A_ij1, A_ij2, A_ij3, A_ij4, A_ij5, A_ij6, A_ij7;
			auto A_ij = A_p + i*istrd_A + j*jstrd_A;
			// A(i, j) += (x(i)*alpha) * y(j)
			do {
				y_j0 = svld1_vnum(pg0, y_j, 0);
				A_ij0 = svld1_vnum(pg0, A_ij, 0);
				A_ij0 = svmla_x(pg0, A_ij0, x_i0, y_j0);
				svst1_vnum(pg0, A_ij, 0, A_ij0);

				if (!svptest_any(ptrue, pg1)) {
					break;
				}

				y_j1 = svld1_vnum(pg1, y_j, 1);
				A_ij1 = svld1_vnum(pg1, A_ij, 1);
				A_ij1 = svmla_x(pg1, A_ij1, x_i1, y_j1);
				svst1_vnum(pg1, A_ij, 1, A_ij1);

				if (!svptest_any(ptrue, pg2)) {
					break;
				}

				y_j2 = svld1_vnum(pg2, y_j, 2);
				A_ij2 = svld1_vnum(pg2, A_ij, 2);
				A_ij2 = svmla_x(pg2, A_ij2, x_i2, y_j2);
				svst1_vnum(pg2, A_ij, 2, A_ij2);

				if (!svptest_any(ptrue, pg3)) {
					break;
				}

				y_j3 = svld1_vnum(pg3, y_j, 3);
				A_ij3 = svld1_vnum(pg3, A_ij, 3);
				A_ij3 = svmla_x(pg3, A_ij3, x_i3, y_j3);
				svst1_vnum(pg3, A_ij, 3, A_ij3);

				if (!svptest_any(ptrue, pg4)) {
					break;
				}

				y_j4 = svld1_vnum(pg4, y_j, 4);
				A_ij4 = svld1_vnum(pg4, A_ij, 4);
				A_ij4 = svmla_x(pg4, A_ij4, x_i4, y_j4);
				svst1_vnum(pg4, A_ij, 4, A_ij4);

				if (!svptest_any(ptrue, pg5)) {
					break;
				}

				y_j5 = svld1_vnum(pg5, y_j, 5);
				A_ij5 = svld1_vnum(pg5, A_ij, 5);
				A_ij5 = svmla_x(pg5, A_ij5, x_i5, y_j5);
				svst1_vnum(pg5, A_ij, 5, A_ij5);

				if (!svptest_any(ptrue, pg6)) {
					break;
				}

				y_j6 = svld1_vnum(pg6, y_j, 6);
				A_ij6 = svld1_vnum(pg6, A_ij, 6);
				A_ij6 = svmla_x(pg6, A_ij6, x_i6, y_j6);
				svst1_vnum(pg6, A_ij, 6, A_ij6);

				if (!svptest_any(ptrue, pg7)) {
					break;
				}

				y_j7 = svld1_vnum(pg7, y_j, 7);
				A_ij7 = svld1_vnum(pg7, A_ij, 7);
				A_ij7 = svmla_x(pg7, A_ij7, x_i7, y_j7);
				svst1_vnum(pg7, A_ij, 7, A_ij7);
			} while (0);
		}
	}
}

inline void dger_sve_vls8(kernel_inttype ninter, kernel_inttype m, kernel_inttype n, double alpha,
                          const double *x_p, kernel_inttype istrd_x,
                          const double *y_p, kernel_inttype istrd_y,
                          double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A) {
	const svbool_t ptrue = svptrue_b64();
	for (kernel_inttype i = 0; i != m; i ++) {
		svfloat64_t x_i0, x_i1, x_i2, x_i3, x_i4, x_i5, x_i6, x_i7;
		auto x_i = x_p + i*istrd_x;
		// x(i) *= alpha
		x_i0 = svld1_vnum(ptrue, x_i, 0);
		x_i0 = svmul_x(ptrue, x_i0, alpha);
		x_i1 = svld1_vnum(ptrue, x_i, 1);
		x_i1 = svmul_x(ptrue, x_i1, alpha);
		x_i2 = svld1_vnum(ptrue, x_i, 2);
		x_i2 = svmul_x(ptrue, x_i2, alpha);
		x_i3 = svld1_vnum(ptrue, x_i, 3);
		x_i3 = svmul_x(ptrue, x_i3, alpha);
		x_i4 = svld1_vnum(ptrue, x_i, 4);
		x_i4 = svmul_x(ptrue, x_i4, alpha);
		x_i5 = svld1_vnum(ptrue, x_i, 5);
		x_i5 = svmul_x(ptrue, x_i5, alpha);
		x_i6 = svld1_vnum(ptrue, x_i, 6);
		x_i6 = svmul_x(ptrue, x_i6, alpha);
		x_i7 = svld1_vnum(ptrue, x_i, 7);
		x_i7 = svmul_x(ptrue, x_i7, alpha);

		for (kernel_inttype j = 0; j < n; j++) {
			svfloat64_t y_j0, y_j1, y_j2, y_j3, y_j4, y_j5, y_j6, y_j7;
			auto y_j = y_p + j*istrd_y;
			svfloat64_t A_ij0, A_ij1, A_ij2, A_ij3, A_ij4, A_ij5, A_ij6, A_ij7;
			auto A_ij = A_p + i*istrd_A + j*jstrd_A;
			// A(i, j) += (x(i)*alpha) * y(j)
			y_j0 = svld1_vnum(ptrue, y_j, 0);
			A_ij0 = svld1_vnum(ptrue, A_ij, 0);
			A_ij0 = svmla_x(ptrue, A_ij0, x_i0, y_j0);
			svst1_vnum(ptrue, A_ij, 0, A_ij0);
			y_j1 = svld1_vnum(ptrue, y_j, 1);
			A_ij1 = svld1_vnum(ptrue, A_ij, 1);
			A_ij1 = svmla_x(ptrue, A_ij1, x_i1, y_j1);
			svst1_vnum(ptrue, A_ij, 1, A_ij1);
			y_j2 = svld1_vnum(ptrue, y_j, 2);
			A_ij2 = svld1_vnum(ptrue, A_ij, 2);
			A_ij2 = svmla_x(ptrue, A_ij2, x_i2, y_j2);
			svst1_vnum(ptrue, A_ij, 2, A_ij2);
			y_j3 = svld1_vnum(ptrue, y_j, 3);
			A_ij3 = svld1_vnum(ptrue, A_ij, 3);
			A_ij3 = svmla_x(ptrue, A_ij3, x_i3, y_j3);
			svst1_vnum(ptrue, A_ij, 3, A_ij3);
			y_j4 = svld1_vnum(ptrue, y_j, 4);
			A_ij4 = svld1_vnum(ptrue, A_ij, 4);
			A_ij4 = svmla_x(ptrue, A_ij4, x_i4, y_j4);
			svst1_vnum(ptrue, A_ij, 4, A_ij4);
			y_j5 = svld1_vnum(ptrue, y_j, 5);
			A_ij5 = svld1_vnum(ptrue, A_ij, 5);
			A_ij5 = svmla_x(ptrue, A_ij5, x_i5, y_j5);
			svst1_vnum(ptrue, A_ij, 5, A_ij5);
			y_j6 = svld1_vnum(ptrue, y_j, 6);
			A_ij6 = svld1_vnum(ptrue, A_ij, 6);
			A_ij6 = svmla_x(ptrue, A_ij6, x_i6, y_j6);
			svst1_vnum(ptrue, A_ij, 6, A_ij6);
			y_j7 = svld1_vnum(ptrue, y_j, 7);
			A_ij7 = svld1_vnum(ptrue, A_ij, 7);
			A_ij7 = svmla_x(ptrue, A_ij7, x_i7, y_j7);
			svst1_vnum(ptrue, A_ij, 7, A_ij7);
		}
	}
}


void ger_sve(kernel_inttype ninter, kernel_inttype m, kernel_inttype n, double alpha,
             const double *x_p, kernel_inttype istrd_x,
             const double *y_p, kernel_inttype istrd_y,
             double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A) {
	auto vec_width = svcntd();
	kernel_inttype ni = 0;
	kernel_inttype ninter_remain = ninter;
	kernel_inttype vec8 = vec_width * 8;

	/*
	 * We want to avoid a store followed closely by a load that would
	 * overlap if predication were ignored. Otherwise, it is faster
	 * for the inner loop to step through A with a smaller stride.
	 */
	bool trans = (std::abs(istrd_A) < std::abs(jstrd_A));
	bool bad_ninter = ((ninter < vec8) && (ninter % vec_width != 0));

	if ((trans && !bad_ninter) || (!trans && bad_ninter)) {
		for (; ni < ninter - vec8 + 1; ni += vec8) {
			perflibs::interleave_batch::dger_sve_vls8(vec8, n, m, alpha, y_p + ni,
			                                       istrd_y, x_p + ni, istrd_x,
			                                       A_p + ni, jstrd_A, istrd_A);
			ninter_remain -= vec8;
		}
		if (ninter_remain > 0) {
			perflibs::interleave_batch::dger_sve_vla(ninter_remain, n, m, alpha, y_p + ni,
			                                      istrd_y, x_p + ni, istrd_x,
			                                      A_p + ni, jstrd_A, istrd_A);
		}
	}
	else {
		for (; ni < ninter - vec8 + 1; ni += vec8) {
			perflibs::interleave_batch::dger_sve_vls8(vec8, m, n, alpha, x_p + ni,
			                                       istrd_x, y_p + ni, istrd_y,
			                                       A_p + ni, istrd_A, jstrd_A);
			ninter_remain -= vec8;
		}
		if (ninter_remain > 0) {
			perflibs::interleave_batch::dger_sve_vla(ninter_remain, m, n, alpha, x_p + ni,
			                                      istrd_x, y_p + ni, istrd_y,
			                                      A_p + ni, istrd_A, jstrd_A);
		}
	}
}
} // namespace perflibs::interleave_batch

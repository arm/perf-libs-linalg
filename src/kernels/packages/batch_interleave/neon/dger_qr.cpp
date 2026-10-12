/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "packages/batch_interleave/factorization/kernels/qr_kernels.hpp"
#include "matrix/interleave_batch.hpp"

namespace perflibs::interleave_batch {
inline void dger_qr_neon_vl16(kernel_inttype ninter, kernel_inttype m, kernel_inttype n, const double *alpha,
                           const double *x_p, kernel_inttype istrd_x,
                           const double *y_p, kernel_inttype istrd_y,
                           double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A) {

	float64x2_t alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7;
	alpha0 = linalg::simd::load<float64x2_t>(alpha + 0);
	alpha1 = linalg::simd::load<float64x2_t>(alpha + 2);
	alpha2 = linalg::simd::load<float64x2_t>(alpha + 4);
	alpha3 = linalg::simd::load<float64x2_t>(alpha + 6);
	alpha4 = linalg::simd::load<float64x2_t>(alpha + 8);
	alpha5 = linalg::simd::load<float64x2_t>(alpha + 10);
	alpha6 = linalg::simd::load<float64x2_t>(alpha + 12);
	alpha7 = linalg::simd::load<float64x2_t>(alpha + 14);

	// The first entry of x is always 1
	for (kernel_inttype j = 0; j < n; j++) {
		float64x2_t y_j0, y_j1, y_j2, y_j3, y_j4, y_j5, y_j6, y_j7;
		auto y_j = y_p + j*istrd_y;
		float64x2_t A_0j0, A_0j1, A_0j2, A_0j3, A_0j4, A_0j5, A_0j6, A_0j7;
		auto A_0j = A_p + 0*istrd_A + j*jstrd_A;
		// A(1, j) += alpha * y(j)
		y_j0 = linalg::simd::load<float64x2_t>(y_j + 0);
		A_0j0 = linalg::simd::load<float64x2_t>(A_0j + 0);
		A_0j0 = linalg::simd::mls(A_0j0, alpha0, y_j0);
		linalg::simd::store(A_0j + 0, A_0j0);
		y_j1 = linalg::simd::load<float64x2_t>(y_j + 2);
		A_0j1 = linalg::simd::load<float64x2_t>(A_0j + 2);
		A_0j1 = linalg::simd::mls(A_0j1, alpha1, y_j1);
		linalg::simd::store(A_0j + 2, A_0j1);
		y_j2 = linalg::simd::load<float64x2_t>(y_j + 4);
		A_0j2 = linalg::simd::load<float64x2_t>(A_0j + 4);
		A_0j2 = linalg::simd::mls(A_0j2, alpha2, y_j2);
		linalg::simd::store(A_0j + 4, A_0j2);
		y_j3 = linalg::simd::load<float64x2_t>(y_j + 6);
		A_0j3 = linalg::simd::load<float64x2_t>(A_0j + 6);
		A_0j3 = linalg::simd::mls(A_0j3, alpha3, y_j3);
		linalg::simd::store(A_0j + 6, A_0j3);
		y_j4 = linalg::simd::load<float64x2_t>(y_j + 8);
		A_0j4 = linalg::simd::load<float64x2_t>(A_0j + 8);
		A_0j4 = linalg::simd::mls(A_0j4, alpha4, y_j4);
		linalg::simd::store(A_0j + 8, A_0j4);
		y_j5 = linalg::simd::load<float64x2_t>(y_j + 10);
		A_0j5 = linalg::simd::load<float64x2_t>(A_0j + 10);
		A_0j5 = linalg::simd::mls(A_0j5, alpha5, y_j5);
		linalg::simd::store(A_0j + 10, A_0j5);
		y_j6 = linalg::simd::load<float64x2_t>(y_j + 12);
		A_0j6 = linalg::simd::load<float64x2_t>(A_0j + 12);
		A_0j6 = linalg::simd::mls(A_0j6, alpha6, y_j6);
		linalg::simd::store(A_0j + 12, A_0j6);
		y_j7 = linalg::simd::load<float64x2_t>(y_j + 14);
		A_0j7 = linalg::simd::load<float64x2_t>(A_0j + 14);
		A_0j7 = linalg::simd::mls(A_0j7, alpha7, y_j7);
		linalg::simd::store(A_0j + 14, A_0j7);
	}

	for (kernel_inttype i = 1; i < m; i++) {
		float64x2_t x_i0, x_i1, x_i2, x_i3, x_i4, x_i5, x_i6, x_i7;
		auto x_i = x_p + i*istrd_x;
		// x(i) *= alpha
		x_i0 = linalg::simd::load<float64x2_t>(x_i + 0);
		x_i0 = linalg::simd::mul(x_i0, alpha0);
		x_i1 = linalg::simd::load<float64x2_t>(x_i + 2);
		x_i1 = linalg::simd::mul(x_i1, alpha1);
		x_i2 = linalg::simd::load<float64x2_t>(x_i + 4);
		x_i2 = linalg::simd::mul(x_i2, alpha2);
		x_i3 = linalg::simd::load<float64x2_t>(x_i + 6);
		x_i3 = linalg::simd::mul(x_i3, alpha3);
		x_i4 = linalg::simd::load<float64x2_t>(x_i + 8);
		x_i4 = linalg::simd::mul(x_i4, alpha4);
		x_i5 = linalg::simd::load<float64x2_t>(x_i + 10);
		x_i5 = linalg::simd::mul(x_i5, alpha5);
		x_i6 = linalg::simd::load<float64x2_t>(x_i + 12);
		x_i6 = linalg::simd::mul(x_i6, alpha6);
		x_i7 = linalg::simd::load<float64x2_t>(x_i + 14);
		x_i7 = linalg::simd::mul(x_i7, alpha7);

		for (kernel_inttype j = 0; j < n; j++) {
			float64x2_t y_j0, y_j1, y_j2, y_j3, y_j4, y_j5, y_j6, y_j7;
			auto y_j = y_p + j*istrd_y;
			float64x2_t A_ij0, A_ij1, A_ij2, A_ij3, A_ij4, A_ij5, A_ij6, A_ij7;
			auto A_ij = A_p + i*istrd_A + j*jstrd_A;
			// A(i, j) += (x(i)*alpha) * y(j)
			y_j0 = linalg::simd::load<float64x2_t>(y_j + 0);
			A_ij0 = linalg::simd::load<float64x2_t>(A_ij + 0);
			A_ij0 = linalg::simd::mls(A_ij0, x_i0, y_j0);
			linalg::simd::store(A_ij + 0, A_ij0);
			y_j1 = linalg::simd::load<float64x2_t>(y_j + 2);
			A_ij1 = linalg::simd::load<float64x2_t>(A_ij + 2);
			A_ij1 = linalg::simd::mls(A_ij1, x_i1, y_j1);
			linalg::simd::store(A_ij + 2, A_ij1);
			y_j2 = linalg::simd::load<float64x2_t>(y_j + 4);
			A_ij2 = linalg::simd::load<float64x2_t>(A_ij + 4);
			A_ij2 = linalg::simd::mls(A_ij2, x_i2, y_j2);
			linalg::simd::store(A_ij + 4, A_ij2);
			y_j3 = linalg::simd::load<float64x2_t>(y_j + 6);
			A_ij3 = linalg::simd::load<float64x2_t>(A_ij + 6);
			A_ij3 = linalg::simd::mls(A_ij3, x_i3, y_j3);
			linalg::simd::store(A_ij + 6, A_ij3);
			y_j4 = linalg::simd::load<float64x2_t>(y_j + 8);
			A_ij4 = linalg::simd::load<float64x2_t>(A_ij + 8);
			A_ij4 = linalg::simd::mls(A_ij4, x_i4, y_j4);
			linalg::simd::store(A_ij + 8, A_ij4);
			y_j5 = linalg::simd::load<float64x2_t>(y_j + 10);
			A_ij5 = linalg::simd::load<float64x2_t>(A_ij + 10);
			A_ij5 = linalg::simd::mls(A_ij5, x_i5, y_j5);
			linalg::simd::store(A_ij + 10, A_ij5);
			y_j6 = linalg::simd::load<float64x2_t>(y_j + 12);
			A_ij6 = linalg::simd::load<float64x2_t>(A_ij + 12);
			A_ij6 = linalg::simd::mls(A_ij6, x_i6, y_j6);
			linalg::simd::store(A_ij + 12, A_ij6);
			y_j7 = linalg::simd::load<float64x2_t>(y_j + 14);
			A_ij7 = linalg::simd::load<float64x2_t>(A_ij + 14);
			A_ij7 = linalg::simd::mls(A_ij7, x_i7, y_j7);
			linalg::simd::store(A_ij + 14, A_ij7);
		}
	}
}

inline void dger_qr_neon_vl8(kernel_inttype ninter, kernel_inttype m, kernel_inttype n, const double *alpha,
                           const double *x_p, kernel_inttype istrd_x,
                           const double *y_p, kernel_inttype istrd_y,
                           double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A) {

	float64x2_t alpha0, alpha1, alpha2, alpha3;
	alpha0 = linalg::simd::load<float64x2_t>(alpha + 0);
	alpha1 = linalg::simd::load<float64x2_t>(alpha + 2);
	alpha2 = linalg::simd::load<float64x2_t>(alpha + 4);
	alpha3 = linalg::simd::load<float64x2_t>(alpha + 6);

	// The first entry of x is always 1
	for (kernel_inttype j = 0; j < n; j++) {
		float64x2_t y_j0, y_j1, y_j2, y_j3;
		auto y_j = y_p + j*istrd_y;
		float64x2_t A_0j0, A_0j1, A_0j2, A_0j3;
		auto A_0j = A_p + 0*istrd_A + j*jstrd_A;
		// A(1, j) += alpha * y(j)
		y_j0 = linalg::simd::load<float64x2_t>(y_j + 0);
		A_0j0 = linalg::simd::load<float64x2_t>(A_0j + 0);
		A_0j0 = linalg::simd::mls(A_0j0, alpha0, y_j0);
		linalg::simd::store(A_0j + 0, A_0j0);
		y_j1 = linalg::simd::load<float64x2_t>(y_j + 2);
		A_0j1 = linalg::simd::load<float64x2_t>(A_0j + 2);
		A_0j1 = linalg::simd::mls(A_0j1, alpha1, y_j1);
		linalg::simd::store(A_0j + 2, A_0j1);
		y_j2 = linalg::simd::load<float64x2_t>(y_j + 4);
		A_0j2 = linalg::simd::load<float64x2_t>(A_0j + 4);
		A_0j2 = linalg::simd::mls(A_0j2, alpha2, y_j2);
		linalg::simd::store(A_0j + 4, A_0j2);
		y_j3 = linalg::simd::load<float64x2_t>(y_j + 6);
		A_0j3 = linalg::simd::load<float64x2_t>(A_0j + 6);
		A_0j3 = linalg::simd::mls(A_0j3, alpha3, y_j3);
		linalg::simd::store(A_0j + 6, A_0j3);
	}

	for (kernel_inttype i = 1; i < m; i++) {
		float64x2_t x_i0, x_i1, x_i2, x_i3;
		auto x_i = x_p + i*istrd_x;
		// x(i) *= alpha
		x_i0 = linalg::simd::load<float64x2_t>(x_i + 0);
		x_i0 = linalg::simd::mul(x_i0, alpha0);
		x_i1 = linalg::simd::load<float64x2_t>(x_i + 2);
		x_i1 = linalg::simd::mul(x_i1, alpha1);
		x_i2 = linalg::simd::load<float64x2_t>(x_i + 4);
		x_i2 = linalg::simd::mul(x_i2, alpha2);
		x_i3 = linalg::simd::load<float64x2_t>(x_i + 6);
		x_i3 = linalg::simd::mul(x_i3, alpha3);

		for (kernel_inttype j = 0; j < n; j++) {
			float64x2_t y_j0, y_j1, y_j2, y_j3;
			auto y_j = y_p + j*istrd_y;
			float64x2_t A_ij0, A_ij1, A_ij2, A_ij3;
			auto A_ij = A_p + i*istrd_A + j*jstrd_A;
			// A(i, j) += (x(i)*alpha) * y(j)
			y_j0 = linalg::simd::load<float64x2_t>(y_j + 0);
			A_ij0 = linalg::simd::load<float64x2_t>(A_ij + 0);
			A_ij0 = linalg::simd::mls(A_ij0, x_i0, y_j0);
			linalg::simd::store(A_ij + 0, A_ij0);
			y_j1 = linalg::simd::load<float64x2_t>(y_j + 2);
			A_ij1 = linalg::simd::load<float64x2_t>(A_ij + 2);
			A_ij1 = linalg::simd::mls(A_ij1, x_i1, y_j1);
			linalg::simd::store(A_ij + 2, A_ij1);
			y_j2 = linalg::simd::load<float64x2_t>(y_j + 4);
			A_ij2 = linalg::simd::load<float64x2_t>(A_ij + 4);
			A_ij2 = linalg::simd::mls(A_ij2, x_i2, y_j2);
			linalg::simd::store(A_ij + 4, A_ij2);
			y_j3 = linalg::simd::load<float64x2_t>(y_j + 6);
			A_ij3 = linalg::simd::load<float64x2_t>(A_ij + 6);
			A_ij3 = linalg::simd::mls(A_ij3, x_i3, y_j3);
			linalg::simd::store(A_ij + 6, A_ij3);
		}
	}
}

inline void dger_qr_neon_vl4(kernel_inttype ninter, kernel_inttype m, kernel_inttype n, const double *alpha,
                           const double *x_p, kernel_inttype istrd_x,
                           const double *y_p, kernel_inttype istrd_y,
                           double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A) {

	float64x2_t alpha0, alpha1;
	alpha0 = linalg::simd::load<float64x2_t>(alpha + 0);
	alpha1 = linalg::simd::load<float64x2_t>(alpha + 2);

	// The first entry of x is always 1
	for (kernel_inttype j = 0; j < n; j++) {
		float64x2_t y_j0, y_j1;
		auto y_j = y_p + j*istrd_y;
		float64x2_t A_0j0, A_0j1;
		auto A_0j = A_p + 0*istrd_A + j*jstrd_A;
		// A(0, j) += alpha * y(j)
		y_j0 = linalg::simd::load<float64x2_t>(y_j + 0);
		A_0j0 = linalg::simd::load<float64x2_t>(A_0j + 0);
		A_0j0 = linalg::simd::mls(A_0j0, alpha0, y_j0);
		linalg::simd::store(A_0j + 0, A_0j0);
		y_j1 = linalg::simd::load<float64x2_t>(y_j + 2);
		A_0j1 = linalg::simd::load<float64x2_t>(A_0j + 2);
		A_0j1 = linalg::simd::mls(A_0j1, alpha1, y_j1);
		linalg::simd::store(A_0j + 2, A_0j1);
	}

	for (kernel_inttype i = 1; i < m; i++) {
		float64x2_t x_i0, x_i1;
		auto x_i = x_p + i*istrd_x;
		// x(i) *= alpha
		x_i0 = linalg::simd::load<float64x2_t>(x_i + 0);
		x_i0 = linalg::simd::mul(x_i0, alpha0);
		x_i1 = linalg::simd::load<float64x2_t>(x_i + 2);
		x_i1 = linalg::simd::mul(x_i1, alpha1);

		for (kernel_inttype j = 0; j < n; j++) {
			float64x2_t y_j0, y_j1;
			auto y_j = y_p + j*istrd_y;
			float64x2_t A_ij0, A_ij1;
			auto A_ij = A_p + i*istrd_A + j*jstrd_A;
			// A(i, j) += (x(i)*alpha) * y(j)
			y_j0 = linalg::simd::load<float64x2_t>(y_j + 0);
			A_ij0 = linalg::simd::load<float64x2_t>(A_ij + 0);
			A_ij0 = linalg::simd::mls(A_ij0, x_i0, y_j0);
			linalg::simd::store(A_ij + 0, A_ij0);
			y_j1 = linalg::simd::load<float64x2_t>(y_j + 2);
			A_ij1 = linalg::simd::load<float64x2_t>(A_ij + 2);
			A_ij1 = linalg::simd::mls(A_ij1, x_i1, y_j1);
			linalg::simd::store(A_ij + 2, A_ij1);
		}
	}
}

inline void dger_qr_neon_vl2(kernel_inttype ninter, kernel_inttype m, kernel_inttype n, const double *alpha,
                           const double *x_p, kernel_inttype istrd_x,
                           const double *y_p, kernel_inttype istrd_y,
                           double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A) {

	float64x2_t alpha0;
	alpha0 = linalg::simd::load<float64x2_t>(alpha + 0);

	// The first entry of x is always 1
	for (kernel_inttype j = 0; j < n; j++) {
		float64x2_t y_j0;
		auto y_j = y_p + j*istrd_y;
		float64x2_t A_0j0;
		auto A_0j = A_p + 0*istrd_A + j*jstrd_A;
		// A(1, j) += alpha * y(j)
		y_j0 = linalg::simd::load<float64x2_t>(y_j + 0);
		A_0j0 = linalg::simd::load<float64x2_t>(A_0j + 0);
		A_0j0 = linalg::simd::mls(A_0j0, alpha0, y_j0);
		linalg::simd::store(A_0j + 0, A_0j0);
	}

	for (kernel_inttype i = 1; i < m; i++) {
		float64x2_t x_i0;
		auto x_i = x_p + i*istrd_x;
		// x(i) *= alpha
		x_i0 = linalg::simd::load<float64x2_t>(x_i + 0);
		x_i0 = linalg::simd::mul(x_i0, alpha0);

		for (kernel_inttype j = 0; j < n; j++) {
			float64x2_t y_j0;
			auto y_j = y_p + j*istrd_y;
			float64x2_t A_ij0;
			auto A_ij = A_p + i*istrd_A + j*jstrd_A;
			// A(i, j) += (x(i)*alpha) * y(j)
			y_j0 = linalg::simd::load<float64x2_t>(y_j + 0);
			A_ij0 = linalg::simd::load<float64x2_t>(A_ij + 0);
			A_ij0 = linalg::simd::mls(A_ij0, x_i0, y_j0);
			linalg::simd::store(A_ij + 0, A_ij0);
		}
	}
}

inline void dger_qr_neon_vl1(kernel_inttype ninter, kernel_inttype m, kernel_inttype n, const double *alpha,
                           const double *x_p, kernel_inttype istrd_x,
                           const double *y_p, kernel_inttype istrd_y,
                           double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A) {

	double alpha0;
	alpha0 = *alpha;

	// The first entry of x is always 1
	for (kernel_inttype j = 0; j < n; j++) {
		double y_j0;
		auto y_j = y_p + j*istrd_y;
		double A_0j0;
		auto A_0j = A_p + j*jstrd_A;
		// A(1, j) += alpha * y(j)
		y_j0 = *y_j;
		A_0j0 = *A_0j;
		A_0j0 -= alpha0 * y_j0;
		*A_0j = A_0j0;
	}

	for (kernel_inttype i = 1; i < m; i++) {
		double x_i0;
		auto x_i = x_p + i*istrd_x;
		// x(i) *= alpha
		x_i0 = *x_i;
		x_i0 *= alpha0;

		for (kernel_inttype j = 0; j < n; j++) {
			double y_j0;
			auto y_j = y_p + j*istrd_y;
			double A_ij0;
			auto A_ij = A_p + i*istrd_A + j*jstrd_A;
			// A(i, j) += (x(i)*alpha) * y(j)
			y_j0 = *y_j;
			A_ij0 = *A_ij;
			A_ij0 -= x_i0 * y_j0;
			*A_ij = A_ij0;
		}
	}
}

void ger_qr_neon(kernel_inttype ninter, kernel_inttype m, kernel_inttype n, const double *alpha,
               const double *x_p, kernel_inttype istrd_x,
               const double *y_p, kernel_inttype istrd_y,
               double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A) {

	kernel_inttype ni;

	for (ni = 0; ni <= ninter - 16; ni += 16) {
		perflibs::interleave_batch::dger_qr_neon_vl16(16, m, n, alpha + ni, x_p + ni, istrd_x, y_p + ni, istrd_y,
                                                        A_p + ni, istrd_A, jstrd_A);
	}
	if (ni <= ninter - 8) {
		perflibs::interleave_batch::dger_qr_neon_vl8(8, m, n, alpha + ni, x_p + ni, istrd_x, y_p + ni, istrd_y,
                                                       A_p + ni, istrd_A, jstrd_A);
		ni += 8;
	}
	if (ni <= ninter - 4) {
		perflibs::interleave_batch::dger_qr_neon_vl4(4, m, n, alpha + ni, x_p + ni, istrd_x, y_p + ni, istrd_y,
                                                       A_p + ni, istrd_A, jstrd_A);
		ni += 4;
	}
	if (ni <= ninter - 2) {
		perflibs::interleave_batch::dger_qr_neon_vl2(2, m, n, alpha + ni, x_p + ni, istrd_x, y_p + ni, istrd_y,
                                                       A_p + ni, istrd_A, jstrd_A);
		ni += 2;
	}
	if (ni == ninter - 1) {
		perflibs::interleave_batch::dger_qr_neon_vl1(1, m, n, alpha + ni, x_p + ni, istrd_x, y_p + ni, istrd_y,
                                                       A_p + ni, istrd_A, jstrd_A);
	}
}
} // namespace perflibs::interleave_batch

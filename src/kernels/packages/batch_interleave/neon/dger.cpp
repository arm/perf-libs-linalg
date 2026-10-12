/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "packages/batch_interleave/matmul/kernels/gemm_kernels.hpp"
#include "matrix/interleave_batch.hpp"

namespace perflibs::interleave_batch {
inline void dger_neon_vl16(kernel_inttype ninter, kernel_inttype m, kernel_inttype n, double alpha,
                           const double *x_p, kernel_inttype istrd_x,
                           const double *y_p, kernel_inttype istrd_y,
                           double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A) {
	float64x2_t alphax2 = {alpha, alpha};

	kernel_inttype i = 0;
	if (m % 2 == 1) {
		float64x2_t x_00, x_01, x_02, x_03, x_04, x_05, x_06, x_07;
		auto x_0 = x_p + 0*istrd_x;
		// x(0) *= alpha
		x_00 = linalg::simd::load<float64x2_t>(x_0 + 0);
		x_00 = linalg::simd::mul(x_00, alphax2);
		x_01 = linalg::simd::load<float64x2_t>(x_0 + 2);
		x_01 = linalg::simd::mul(x_01, alphax2);
		x_02 = linalg::simd::load<float64x2_t>(x_0 + 4);
		x_02 = linalg::simd::mul(x_02, alphax2);
		x_03 = linalg::simd::load<float64x2_t>(x_0 + 6);
		x_03 = linalg::simd::mul(x_03, alphax2);
		x_04 = linalg::simd::load<float64x2_t>(x_0 + 8);
		x_04 = linalg::simd::mul(x_04, alphax2);
		x_05 = linalg::simd::load<float64x2_t>(x_0 + 10);
		x_05 = linalg::simd::mul(x_05, alphax2);
		x_06 = linalg::simd::load<float64x2_t>(x_0 + 12);
		x_06 = linalg::simd::mul(x_06, alphax2);
		x_07 = linalg::simd::load<float64x2_t>(x_0 + 14);
		x_07 = linalg::simd::mul(x_07, alphax2);

		for (kernel_inttype j = 0; j < n; j++) {
			float64x2_t y_j0, y_j1, y_j2, y_j3, y_j4, y_j5, y_j6, y_j7;
			auto y_j = y_p + j*istrd_y;
			float64x2_t a_0j0, a_0j1, a_0j2, a_0j3, a_0j4, a_0j5, a_0j6, a_0j7;
			auto a_0j = A_p + 0*istrd_A + j*jstrd_A;
			// A(0, j) += (x(0)*alpha) * y(j)
			y_j0 = linalg::simd::load<float64x2_t>(y_j + 0);
			a_0j0 = linalg::simd::load<float64x2_t>(a_0j + 0);
			a_0j0 = linalg::simd::mla(a_0j0, x_00, y_j0);
			linalg::simd::store(a_0j + 0, a_0j0);
			y_j1 = linalg::simd::load<float64x2_t>(y_j + 2);
			a_0j1 = linalg::simd::load<float64x2_t>(a_0j + 2);
			a_0j1 = linalg::simd::mla(a_0j1, x_01, y_j1);
			linalg::simd::store(a_0j + 2, a_0j1);
			y_j2 = linalg::simd::load<float64x2_t>(y_j + 4);
			a_0j2 = linalg::simd::load<float64x2_t>(a_0j + 4);
			a_0j2 = linalg::simd::mla(a_0j2, x_02, y_j2);
			linalg::simd::store(a_0j + 4, a_0j2);
			y_j3 = linalg::simd::load<float64x2_t>(y_j + 6);
			a_0j3 = linalg::simd::load<float64x2_t>(a_0j + 6);
			a_0j3 = linalg::simd::mla(a_0j3, x_03, y_j3);
			linalg::simd::store(a_0j + 6, a_0j3);
			y_j4 = linalg::simd::load<float64x2_t>(y_j + 8);
			a_0j4 = linalg::simd::load<float64x2_t>(a_0j + 8);
			a_0j4 = linalg::simd::mla(a_0j4, x_04, y_j4);
			linalg::simd::store(a_0j + 8, a_0j4);
			y_j5 = linalg::simd::load<float64x2_t>(y_j + 10);
			a_0j5 = linalg::simd::load<float64x2_t>(a_0j + 10);
			a_0j5 = linalg::simd::mla(a_0j5, x_05, y_j5);
			linalg::simd::store(a_0j + 10, a_0j5);
			y_j6 = linalg::simd::load<float64x2_t>(y_j + 12);
			a_0j6 = linalg::simd::load<float64x2_t>(a_0j + 12);
			a_0j6 = linalg::simd::mla(a_0j6, x_06, y_j6);
			linalg::simd::store(a_0j + 12, a_0j6);
			y_j7 = linalg::simd::load<float64x2_t>(y_j + 14);
			a_0j7 = linalg::simd::load<float64x2_t>(a_0j + 14);
			a_0j7 = linalg::simd::mla(a_0j7, x_07, y_j7);
			linalg::simd::store(a_0j + 14, a_0j7);
		}
		i++;
	}

	for (; i != m; i += 2) {
		kernel_inttype h = i + 1;
		float64x2_t x_i0, x_i1, x_i2, x_i3, x_i4, x_i5, x_i6, x_i7;
		auto x_i = x_p + i*istrd_x;
		float64x2_t x_h0, x_h1, x_h2, x_h3, x_h4, x_h5, x_h6, x_h7;
		auto x_h = x_p + h*istrd_x;
		// x(i) *= alpha
		// x(i + 1) *= alpha
		x_i0 = linalg::simd::load<float64x2_t>(x_i + 0);
		x_h0 = linalg::simd::load<float64x2_t>(x_h + 0);
		x_i0 = linalg::simd::mul(x_i0, alphax2);
		x_h0 = linalg::simd::mul(x_h0, alphax2);
		x_i1 = linalg::simd::load<float64x2_t>(x_i + 2);
		x_h1 = linalg::simd::load<float64x2_t>(x_h + 2);
		x_i1 = linalg::simd::mul(x_i1, alphax2);
		x_h1 = linalg::simd::mul(x_h1, alphax2);
		x_i2 = linalg::simd::load<float64x2_t>(x_i + 4);
		x_h2 = linalg::simd::load<float64x2_t>(x_h + 4);
		x_i2 = linalg::simd::mul(x_i2, alphax2);
		x_h2 = linalg::simd::mul(x_h2, alphax2);
		x_i3 = linalg::simd::load<float64x2_t>(x_i + 6);
		x_h3 = linalg::simd::load<float64x2_t>(x_h + 6);
		x_i3 = linalg::simd::mul(x_i3, alphax2);
		x_h3 = linalg::simd::mul(x_h3, alphax2);
		x_i4 = linalg::simd::load<float64x2_t>(x_i + 8);
		x_h4 = linalg::simd::load<float64x2_t>(x_h + 8);
		x_i4 = linalg::simd::mul(x_i4, alphax2);
		x_h4 = linalg::simd::mul(x_h4, alphax2);
		x_i5 = linalg::simd::load<float64x2_t>(x_i + 10);
		x_h5 = linalg::simd::load<float64x2_t>(x_h + 10);
		x_i5 = linalg::simd::mul(x_i5, alphax2);
		x_h5 = linalg::simd::mul(x_h5, alphax2);
		x_i6 = linalg::simd::load<float64x2_t>(x_i + 12);
		x_h6 = linalg::simd::load<float64x2_t>(x_h + 12);
		x_i6 = linalg::simd::mul(x_i6, alphax2);
		x_h6 = linalg::simd::mul(x_h6, alphax2);
		x_i7 = linalg::simd::load<float64x2_t>(x_i + 14);
		x_h7 = linalg::simd::load<float64x2_t>(x_h + 14);
		x_i7 = linalg::simd::mul(x_i7, alphax2);
		x_h7 = linalg::simd::mul(x_h7, alphax2);

		for (kernel_inttype j = 0; j < n; j++) {
			float64x2_t y_j0, y_j1, y_j2, y_j3, y_j4, y_j5, y_j6, y_j7;
			auto y_j = y_p + j*istrd_y;
			float64x2_t a_ij0, a_ij1, a_ij2, a_ij3, a_ij4, a_ij5, a_ij6, a_ij7;
			auto a_ij = A_p + i*istrd_A + j*jstrd_A;
			float64x2_t a_hj0, a_hj1, a_hj2, a_hj3, a_hj4, a_hj5, a_hj6, a_hj7;
			auto a_hj = A_p + h*istrd_A + j*jstrd_A;
			// A(i, j) += (x(i)*alpha) * y(j)
			// A(i + 1, j) += (x(i + 1)*alpha) * y(j)
			y_j0 = linalg::simd::load<float64x2_t>(y_j + 0);
			a_ij0 = linalg::simd::load<float64x2_t>(a_ij + 0);
			a_hj0 = linalg::simd::load<float64x2_t>(a_hj + 0);
			a_ij0 = linalg::simd::mla(a_ij0, x_i0, y_j0);
			a_hj0 = linalg::simd::mla(a_hj0, x_h0, y_j0);
			linalg::simd::store(a_ij + 0, a_ij0);
			linalg::simd::store(a_hj + 0, a_hj0);
			y_j1 = linalg::simd::load<float64x2_t>(y_j + 2);
			a_ij1 = linalg::simd::load<float64x2_t>(a_ij + 2);
			a_hj1 = linalg::simd::load<float64x2_t>(a_hj + 2);
			a_ij1 = linalg::simd::mla(a_ij1, x_i1, y_j1);
			a_hj1 = linalg::simd::mla(a_hj1, x_h1, y_j1);
			linalg::simd::store(a_ij + 2, a_ij1);
			linalg::simd::store(a_hj + 2, a_hj1);
			y_j2 = linalg::simd::load<float64x2_t>(y_j + 4);
			a_ij2 = linalg::simd::load<float64x2_t>(a_ij + 4);
			a_hj2 = linalg::simd::load<float64x2_t>(a_hj + 4);
			a_ij2 = linalg::simd::mla(a_ij2, x_i2, y_j2);
			a_hj2 = linalg::simd::mla(a_hj2, x_h2, y_j2);
			linalg::simd::store(a_ij + 4, a_ij2);
			linalg::simd::store(a_hj + 4, a_hj2);
			y_j3 = linalg::simd::load<float64x2_t>(y_j + 6);
			a_ij3 = linalg::simd::load<float64x2_t>(a_ij + 6);
			a_hj3 = linalg::simd::load<float64x2_t>(a_hj + 6);
			a_ij3 = linalg::simd::mla(a_ij3, x_i3, y_j3);
			a_hj3 = linalg::simd::mla(a_hj3, x_h3, y_j3);
			linalg::simd::store(a_ij + 6, a_ij3);
			linalg::simd::store(a_hj + 6, a_hj3);
			y_j4 = linalg::simd::load<float64x2_t>(y_j + 8);
			a_ij4 = linalg::simd::load<float64x2_t>(a_ij + 8);
			a_hj4 = linalg::simd::load<float64x2_t>(a_hj + 8);
			a_ij4 = linalg::simd::mla(a_ij4, x_i4, y_j4);
			a_hj4 = linalg::simd::mla(a_hj4, x_h4, y_j4);
			linalg::simd::store(a_ij + 8, a_ij4);
			linalg::simd::store(a_hj + 8, a_hj4);
			y_j5 = linalg::simd::load<float64x2_t>(y_j + 10);
			a_ij5 = linalg::simd::load<float64x2_t>(a_ij + 10);
			a_hj5 = linalg::simd::load<float64x2_t>(a_hj + 10);
			a_ij5 = linalg::simd::mla(a_ij5, x_i5, y_j5);
			a_hj5 = linalg::simd::mla(a_hj5, x_h5, y_j5);
			linalg::simd::store(a_ij + 10, a_ij5);
			linalg::simd::store(a_hj + 10, a_hj5);
			y_j6 = linalg::simd::load<float64x2_t>(y_j + 12);
			a_ij6 = linalg::simd::load<float64x2_t>(a_ij + 12);
			a_hj6 = linalg::simd::load<float64x2_t>(a_hj + 12);
			a_ij6 = linalg::simd::mla(a_ij6, x_i6, y_j6);
			a_hj6 = linalg::simd::mla(a_hj6, x_h6, y_j6);
			linalg::simd::store(a_ij + 12, a_ij6);
			linalg::simd::store(a_hj + 12, a_hj6);
			y_j7 = linalg::simd::load<float64x2_t>(y_j + 14);
			a_ij7 = linalg::simd::load<float64x2_t>(a_ij + 14);
			a_hj7 = linalg::simd::load<float64x2_t>(a_hj + 14);
			a_ij7 = linalg::simd::mla(a_ij7, x_i7, y_j7);
			a_hj7 = linalg::simd::mla(a_hj7, x_h7, y_j7);
			linalg::simd::store(a_ij + 14, a_ij7);
			linalg::simd::store(a_hj + 14, a_hj7);
		}
	}
}

inline void dger_neon_vl8(kernel_inttype ninter, kernel_inttype m, kernel_inttype n, double alpha,
                           const double *x_p, kernel_inttype istrd_x,
                           const double *y_p, kernel_inttype istrd_y,
                           double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A) {
	float64x2_t alphax2 = {alpha, alpha};

	kernel_inttype i = 0;
	if (m % 2 == 1) {
		float64x2_t x_00, x_01, x_02, x_03;
		auto x_0 = x_p + 0*istrd_x;
		// x(0) *= alpha
		x_00 = linalg::simd::load<float64x2_t>(x_0 + 0);
		x_00 = linalg::simd::mul(x_00, alphax2);
		x_01 = linalg::simd::load<float64x2_t>(x_0 + 2);
		x_01 = linalg::simd::mul(x_01, alphax2);
		x_02 = linalg::simd::load<float64x2_t>(x_0 + 4);
		x_02 = linalg::simd::mul(x_02, alphax2);
		x_03 = linalg::simd::load<float64x2_t>(x_0 + 6);
		x_03 = linalg::simd::mul(x_03, alphax2);

		for (kernel_inttype j = 0; j < n; j++) {
			float64x2_t y_j0, y_j1, y_j2, y_j3;
			auto y_j = y_p + j*istrd_y;
			float64x2_t a_0j0, a_0j1, a_0j2, a_0j3;
			auto a_0j = A_p + 0*istrd_A + j*jstrd_A;
			// A(0, j) += (x(0)*alpha) * y(j)
			y_j0 = linalg::simd::load<float64x2_t>(y_j + 0);
			a_0j0 = linalg::simd::load<float64x2_t>(a_0j + 0);
			a_0j0 = linalg::simd::mla(a_0j0, x_00, y_j0);
			linalg::simd::store(a_0j + 0, a_0j0);
			y_j1 = linalg::simd::load<float64x2_t>(y_j + 2);
			a_0j1 = linalg::simd::load<float64x2_t>(a_0j + 2);
			a_0j1 = linalg::simd::mla(a_0j1, x_01, y_j1);
			linalg::simd::store(a_0j + 2, a_0j1);
			y_j2 = linalg::simd::load<float64x2_t>(y_j + 4);
			a_0j2 = linalg::simd::load<float64x2_t>(a_0j + 4);
			a_0j2 = linalg::simd::mla(a_0j2, x_02, y_j2);
			linalg::simd::store(a_0j + 4, a_0j2);
			y_j3 = linalg::simd::load<float64x2_t>(y_j + 6);
			a_0j3 = linalg::simd::load<float64x2_t>(a_0j + 6);
			a_0j3 = linalg::simd::mla(a_0j3, x_03, y_j3);
			linalg::simd::store(a_0j + 6, a_0j3);
		}
		i++;
	}

	for (; i != m; i += 2) {
		kernel_inttype h = i + 1;
		float64x2_t x_i0, x_i1, x_i2, x_i3;
		auto x_i = x_p + i*istrd_x;
		float64x2_t x_h0, x_h1, x_h2, x_h3;
		auto x_h = x_p + h*istrd_x;
		// x(i) *= alpha
		// x(i + 1) *= alpha
		x_i0 = linalg::simd::load<float64x2_t>(x_i + 0);
		x_h0 = linalg::simd::load<float64x2_t>(x_h + 0);
		x_i0 = linalg::simd::mul(x_i0, alphax2);
		x_h0 = linalg::simd::mul(x_h0, alphax2);
		x_i1 = linalg::simd::load<float64x2_t>(x_i + 2);
		x_h1 = linalg::simd::load<float64x2_t>(x_h + 2);
		x_i1 = linalg::simd::mul(x_i1, alphax2);
		x_h1 = linalg::simd::mul(x_h1, alphax2);
		x_i2 = linalg::simd::load<float64x2_t>(x_i + 4);
		x_h2 = linalg::simd::load<float64x2_t>(x_h + 4);
		x_i2 = linalg::simd::mul(x_i2, alphax2);
		x_h2 = linalg::simd::mul(x_h2, alphax2);
		x_i3 = linalg::simd::load<float64x2_t>(x_i + 6);
		x_h3 = linalg::simd::load<float64x2_t>(x_h + 6);
		x_i3 = linalg::simd::mul(x_i3, alphax2);
		x_h3 = linalg::simd::mul(x_h3, alphax2);

		for (kernel_inttype j = 0; j < n; j++) {
			float64x2_t y_j0, y_j1, y_j2, y_j3;
			auto y_j = y_p + j*istrd_y;
			float64x2_t a_ij0, a_ij1, a_ij2, a_ij3;
			auto a_ij = A_p + i*istrd_A + j*jstrd_A;
			float64x2_t a_hj0, a_hj1, a_hj2, a_hj3;
			auto a_hj = A_p + h*istrd_A + j*jstrd_A;
			// A(i, j) += (x(i)*alpha) * y(j)
			// A(i + 1, j) += (x(i + 1)*alpha) * y(j)
			y_j0 = linalg::simd::load<float64x2_t>(y_j + 0);
			a_ij0 = linalg::simd::load<float64x2_t>(a_ij + 0);
			a_hj0 = linalg::simd::load<float64x2_t>(a_hj + 0);
			a_ij0 = linalg::simd::mla(a_ij0, x_i0, y_j0);
			a_hj0 = linalg::simd::mla(a_hj0, x_h0, y_j0);
			linalg::simd::store(a_ij + 0, a_ij0);
			linalg::simd::store(a_hj + 0, a_hj0);
			y_j1 = linalg::simd::load<float64x2_t>(y_j + 2);
			a_ij1 = linalg::simd::load<float64x2_t>(a_ij + 2);
			a_hj1 = linalg::simd::load<float64x2_t>(a_hj + 2);
			a_ij1 = linalg::simd::mla(a_ij1, x_i1, y_j1);
			a_hj1 = linalg::simd::mla(a_hj1, x_h1, y_j1);
			linalg::simd::store(a_ij + 2, a_ij1);
			linalg::simd::store(a_hj + 2, a_hj1);
			y_j2 = linalg::simd::load<float64x2_t>(y_j + 4);
			a_ij2 = linalg::simd::load<float64x2_t>(a_ij + 4);
			a_hj2 = linalg::simd::load<float64x2_t>(a_hj + 4);
			a_ij2 = linalg::simd::mla(a_ij2, x_i2, y_j2);
			a_hj2 = linalg::simd::mla(a_hj2, x_h2, y_j2);
			linalg::simd::store(a_ij + 4, a_ij2);
			linalg::simd::store(a_hj + 4, a_hj2);
			y_j3 = linalg::simd::load<float64x2_t>(y_j + 6);
			a_ij3 = linalg::simd::load<float64x2_t>(a_ij + 6);
			a_hj3 = linalg::simd::load<float64x2_t>(a_hj + 6);
			a_ij3 = linalg::simd::mla(a_ij3, x_i3, y_j3);
			a_hj3 = linalg::simd::mla(a_hj3, x_h3, y_j3);
			linalg::simd::store(a_ij + 6, a_ij3);
			linalg::simd::store(a_hj + 6, a_hj3);
		}
	}
}

inline void dger_neon_vl4(kernel_inttype ninter, kernel_inttype m, kernel_inttype n, double alpha,
                           const double *x_p, kernel_inttype istrd_x,
                           const double *y_p, kernel_inttype istrd_y,
                           double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A) {
	float64x2_t alphax2 = {alpha, alpha};

	kernel_inttype i = 0;
	if (m % 2 == 1) {
		float64x2_t x_00, x_01;
		auto x_0 = x_p + 0*istrd_x;
		// x(0) *= alpha
		x_00 = linalg::simd::load<float64x2_t>(x_0 + 0);
		x_00 = linalg::simd::mul(x_00, alphax2);
		x_01 = linalg::simd::load<float64x2_t>(x_0 + 2);
		x_01 = linalg::simd::mul(x_01, alphax2);

		for (kernel_inttype j = 0; j < n; j++) {
			float64x2_t y_j0, y_j1;
			auto y_j = y_p + j*istrd_y;
			float64x2_t a_0j0, a_0j1;
			auto a_0j = A_p + 0*istrd_A + j*jstrd_A;
			// A(0, j) += (x(0)*alpha) * y(j)
			y_j0 = linalg::simd::load<float64x2_t>(y_j + 0);
			a_0j0 = linalg::simd::load<float64x2_t>(a_0j + 0);
			a_0j0 = linalg::simd::mla(a_0j0, x_00, y_j0);
			linalg::simd::store(a_0j + 0, a_0j0);
			y_j1 = linalg::simd::load<float64x2_t>(y_j + 2);
			a_0j1 = linalg::simd::load<float64x2_t>(a_0j + 2);
			a_0j1 = linalg::simd::mla(a_0j1, x_01, y_j1);
			linalg::simd::store(a_0j + 2, a_0j1);
		}
		i++;
	}

	for (; i != m; i += 2) {
		kernel_inttype h = i + 1;
		float64x2_t x_i0, x_i1;
		auto x_i = x_p + i*istrd_x;
		float64x2_t x_h0, x_h1;
		auto x_h = x_p + h*istrd_x;
		// x(i) *= alpha
		// x(i + 1) *= alpha
		x_i0 = linalg::simd::load<float64x2_t>(x_i + 0);
		x_h0 = linalg::simd::load<float64x2_t>(x_h + 0);
		x_i0 = linalg::simd::mul(x_i0, alphax2);
		x_h0 = linalg::simd::mul(x_h0, alphax2);
		x_i1 = linalg::simd::load<float64x2_t>(x_i + 2);
		x_h1 = linalg::simd::load<float64x2_t>(x_h + 2);
		x_i1 = linalg::simd::mul(x_i1, alphax2);
		x_h1 = linalg::simd::mul(x_h1, alphax2);

		for (kernel_inttype j = 0; j < n; j++) {
			float64x2_t y_j0, y_j1;
			auto y_j = y_p + j*istrd_y;
			float64x2_t a_ij0, a_ij1;
			auto a_ij = A_p + i*istrd_A + j*jstrd_A;
			float64x2_t a_hj0, a_hj1;
			auto a_hj = A_p + h*istrd_A + j*jstrd_A;
			// A(i, j) += (x(i)*alpha) * y(j)
			// A(i + 1, j) += (x(i + 1)*alpha) * y(j)
			y_j0 = linalg::simd::load<float64x2_t>(y_j + 0);
			a_ij0 = linalg::simd::load<float64x2_t>(a_ij + 0);
			a_hj0 = linalg::simd::load<float64x2_t>(a_hj + 0);
			a_ij0 = linalg::simd::mla(a_ij0, x_i0, y_j0);
			a_hj0 = linalg::simd::mla(a_hj0, x_h0, y_j0);
			linalg::simd::store(a_ij + 0, a_ij0);
			linalg::simd::store(a_hj + 0, a_hj0);
			y_j1 = linalg::simd::load<float64x2_t>(y_j + 2);
			a_ij1 = linalg::simd::load<float64x2_t>(a_ij + 2);
			a_hj1 = linalg::simd::load<float64x2_t>(a_hj + 2);
			a_ij1 = linalg::simd::mla(a_ij1, x_i1, y_j1);
			a_hj1 = linalg::simd::mla(a_hj1, x_h1, y_j1);
			linalg::simd::store(a_ij + 2, a_ij1);
			linalg::simd::store(a_hj + 2, a_hj1);
		}
	}
}

inline void dger_neon_vl2(kernel_inttype ninter, kernel_inttype m, kernel_inttype n, double alpha,
                           const double *x_p, kernel_inttype istrd_x,
                           const double *y_p, kernel_inttype istrd_y,
                           double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A) {
	float64x2_t alphax2 = {alpha, alpha};

	kernel_inttype i = 0;
	if (m % 2 == 1) {
		float64x2_t x_00;
		auto x_0 = x_p + 0*istrd_x;
		// x(0) *= alpha
		x_00 = linalg::simd::load<float64x2_t>(x_0 + 0);
		x_00 = linalg::simd::mul(x_00, alphax2);

		for (kernel_inttype j = 0; j < n; j++) {
			float64x2_t y_j0;
			auto y_j = y_p + j*istrd_y;
			float64x2_t a_0j0;
			auto a_0j = A_p + 0*istrd_A + j*jstrd_A;
			// A(0, j) += (x(0)*alpha) * y(j)
			y_j0 = linalg::simd::load<float64x2_t>(y_j + 0);
			a_0j0 = linalg::simd::load<float64x2_t>(a_0j + 0);
			a_0j0 = linalg::simd::mla(a_0j0, x_00, y_j0);
			linalg::simd::store(a_0j + 0, a_0j0);
		}
		i++;
	}

	for (; i != m; i += 2) {
		kernel_inttype h = i + 1;
		float64x2_t x_i0;
		auto x_i = x_p + i*istrd_x;
		float64x2_t x_h0;
		auto x_h = x_p + h*istrd_x;
		// x(i) *= alpha
		// x(i + 1) *= alpha
		x_i0 = linalg::simd::load<float64x2_t>(x_i + 0);
		x_h0 = linalg::simd::load<float64x2_t>(x_h + 0);
		x_i0 = linalg::simd::mul(x_i0, alphax2);
		x_h0 = linalg::simd::mul(x_h0, alphax2);

		for (kernel_inttype j = 0; j < n; j++) {
			float64x2_t y_j0;
			auto y_j = y_p + j*istrd_y;
			float64x2_t a_ij0;
			auto a_ij = A_p + i*istrd_A + j*jstrd_A;
			float64x2_t a_hj0;
			auto a_hj = A_p + h*istrd_A + j*jstrd_A;
			// A(i, j) += (x(i)*alpha) * y(j)
			// A(i + 1, j) += (x(i + 1)*alpha) * y(j)
			y_j0 = linalg::simd::load<float64x2_t>(y_j + 0);
			a_ij0 = linalg::simd::load<float64x2_t>(a_ij + 0);
			a_hj0 = linalg::simd::load<float64x2_t>(a_hj + 0);
			a_ij0 = linalg::simd::mla(a_ij0, x_i0, y_j0);
			a_hj0 = linalg::simd::mla(a_hj0, x_h0, y_j0);
			linalg::simd::store(a_ij + 0, a_ij0);
			linalg::simd::store(a_hj + 0, a_hj0);
		}
	}
}

inline void dger_neon_vl1(kernel_inttype ninter, kernel_inttype m, kernel_inttype n, double alpha,
                           const double *x_p, kernel_inttype istrd_x,
                           const double *y_p, kernel_inttype istrd_y,
                           double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A) {

	kernel_inttype i = 0;
	if (m % 2 == 1) {
		double x_00;
		auto x_0 = x_p + 0*istrd_x;
		// x(0) *= alpha
		x_00 = *x_0;
		x_00 *= alpha;

		for (kernel_inttype j = 0; j < n; j++) {
			double y_j0;
			auto y_j = y_p + j*istrd_y;
			double a_0j0;
			auto a_0j = A_p + 0*istrd_A + j*jstrd_A;
			// A(0, j) += (x(0)*alpha) * y(j)
			y_j0 = *y_j;
			a_0j0 = *a_0j;
			a_0j0 += x_00 * y_j0;
			*a_0j = a_0j0;
		}
		i++;
	}

	for (; i != m; i += 2) {
		kernel_inttype h = i + 1;
		double x_i0, x_h0;
		auto x_i = x_p + i*istrd_x;
		auto x_h = x_p + h*istrd_x;
		// x(i) *= alpha
		// x(i + 1) *= alpha
		x_i0 = *x_i;
		x_h0 = *x_h;
		x_i0 *= alpha;
		x_h0 *= alpha;

		for (kernel_inttype j = 0; j < n; j++) {
			double y_j0;
			auto y_j = y_p + j*istrd_y;
			double a_ij0, a_hj0;
			auto a_ij = A_p + i*istrd_A + j*jstrd_A;
			auto a_hj = A_p + h*istrd_A + j*jstrd_A;
			// A(i, j) += (x(i)*alpha) * y(j)
			// A(i + 1, j) += (x(i + 1)*alpha) * y(j)
			y_j0 = *y_j;
			a_ij0 = *a_ij;
			a_hj0 = *a_hj;
			a_ij0 += x_i0 * y_j0;
			a_hj0 += x_h0 * y_j0;
			*a_ij = a_ij0;
			*a_hj = a_hj0;
		}
	}
}

void ger_neon(kernel_inttype ninter, kernel_inttype m, kernel_inttype n, double alpha,
               const double *x_p, kernel_inttype istrd_x,
               const double *y_p, kernel_inttype istrd_y,
               double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A) {

	kernel_inttype ni;

	for (ni = 0; ni <= ninter - 16; ni += 16) {
		perflibs::interleave_batch::dger_neon_vl16(16, m, n, alpha, x_p + ni, istrd_x, y_p + ni, istrd_y,
		                                        A_p + ni, istrd_A, jstrd_A);
	}
	if (ni <= ninter - 8) {
		perflibs::interleave_batch::dger_neon_vl8(8, m, n, alpha, x_p + ni, istrd_x, y_p + ni, istrd_y,
		                                       A_p + ni, istrd_A, jstrd_A);
		ni += 8;
	}
	if (ni <= ninter - 4) {
		perflibs::interleave_batch::dger_neon_vl4(4, m, n, alpha, x_p + ni, istrd_x, y_p + ni, istrd_y,
		                                       A_p + ni, istrd_A, jstrd_A);
		ni += 4;
	}
	if (ni <= ninter - 2) {
		perflibs::interleave_batch::dger_neon_vl2(2, m, n, alpha, x_p + ni, istrd_x, y_p + ni, istrd_y,
		                                       A_p + ni, istrd_A, jstrd_A);
		ni += 2;
	}
	if (ni == ninter - 1) {
		perflibs::interleave_batch::dger_neon_vl1(1, m, n, alpha, x_p + ni, istrd_x, y_p + ni, istrd_y,
		                                       A_p + ni, istrd_A, jstrd_A);
	}
}
} // namespace perflibs::interleave_batch

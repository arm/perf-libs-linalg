/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "packages/batch_interleave/solve/kernels/trsm_kernels.hpp"
#include "matrix/interleave_batch.hpp"

namespace perflibs::interleave_batch {
inline void dtrsm_neon_vl16(kernel_inttype ninter, const double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A,
                           double *B_p, kernel_inttype istrd_B, kernel_inttype jstrd_B, kernel_inttype m, kernel_inttype n,
                           bool nounit, double alpha) {
	const float64x2_t onex2 = {1.0, 1.0};
	float64x2_t alphax2 = {alpha, alpha};

	for (kernel_inttype i = m - 1; i >= 0; i--) {
		float64x2_t a_ii0, a_ii1, a_ii2, a_ii3, a_ii4, a_ii5, a_ii6, a_ii7;
		auto a_ii = A_p + i*istrd_A + i*jstrd_A;
		if (nounit) {
			a_ii0 = linalg::simd::load<float64x2_t>(a_ii + 0);
			a_ii0 = linalg::simd::div(onex2, a_ii0);
			a_ii1 = linalg::simd::load<float64x2_t>(a_ii + 2);
			a_ii1 = linalg::simd::div(onex2, a_ii1);
			a_ii2 = linalg::simd::load<float64x2_t>(a_ii + 4);
			a_ii2 = linalg::simd::div(onex2, a_ii2);
			a_ii3 = linalg::simd::load<float64x2_t>(a_ii + 6);
			a_ii3 = linalg::simd::div(onex2, a_ii3);
			a_ii4 = linalg::simd::load<float64x2_t>(a_ii + 8);
			a_ii4 = linalg::simd::div(onex2, a_ii4);
			a_ii5 = linalg::simd::load<float64x2_t>(a_ii + 10);
			a_ii5 = linalg::simd::div(onex2, a_ii5);
			a_ii6 = linalg::simd::load<float64x2_t>(a_ii + 12);
			a_ii6 = linalg::simd::div(onex2, a_ii6);
			a_ii7 = linalg::simd::load<float64x2_t>(a_ii + 14);
			a_ii7 = linalg::simd::div(onex2, a_ii7);
		}

		for (kernel_inttype j = 0; j<n; j++) {
			float64x2_t b_ij0, b_ij1, b_ij2, b_ij3, b_ij4, b_ij5, b_ij6, b_ij7;
			auto b_ij = B_p + i*istrd_B + j*jstrd_B;
			// B(i, j) *= alpha
			b_ij0 = linalg::simd::load<float64x2_t>(b_ij + 0);
			b_ij0 = linalg::simd::mul(b_ij0, alphax2);
			b_ij1 = linalg::simd::load<float64x2_t>(b_ij + 2);
			b_ij1 = linalg::simd::mul(b_ij1, alphax2);
			b_ij2 = linalg::simd::load<float64x2_t>(b_ij + 4);
			b_ij2 = linalg::simd::mul(b_ij2, alphax2);
			b_ij3 = linalg::simd::load<float64x2_t>(b_ij + 6);
			b_ij3 = linalg::simd::mul(b_ij3, alphax2);
			b_ij4 = linalg::simd::load<float64x2_t>(b_ij + 8);
			b_ij4 = linalg::simd::mul(b_ij4, alphax2);
			b_ij5 = linalg::simd::load<float64x2_t>(b_ij + 10);
			b_ij5 = linalg::simd::mul(b_ij5, alphax2);
			b_ij6 = linalg::simd::load<float64x2_t>(b_ij + 12);
			b_ij6 = linalg::simd::mul(b_ij6, alphax2);
			b_ij7 = linalg::simd::load<float64x2_t>(b_ij + 14);
			b_ij7 = linalg::simd::mul(b_ij7, alphax2);

			for (kernel_inttype k = i + 1; k<m; k++) {
				float64x2_t a_ik0, a_ik1, a_ik2, a_ik3, a_ik4, a_ik5, a_ik6, a_ik7;
				auto a_ik = A_p + i*istrd_A + k*jstrd_A;
				float64x2_t b_kj0, b_kj1, b_kj2, b_kj3, b_kj4, b_kj5, b_kj6, b_kj7;
				auto b_kj = B_p + k*istrd_B + j*jstrd_B;
				// B(i, j) = B(i, j) - A(i, k)*B(k, j)
				a_ik0 = linalg::simd::load<float64x2_t>(a_ik + 0);
				b_kj0 = linalg::simd::load<float64x2_t>(b_kj + 0);
				b_ij0 = linalg::simd::mls(b_ij0, a_ik0, b_kj0);
				a_ik1 = linalg::simd::load<float64x2_t>(a_ik + 2);
				b_kj1 = linalg::simd::load<float64x2_t>(b_kj + 2);
				b_ij1 = linalg::simd::mls(b_ij1, a_ik1, b_kj1);
				a_ik2 = linalg::simd::load<float64x2_t>(a_ik + 4);
				b_kj2 = linalg::simd::load<float64x2_t>(b_kj + 4);
				b_ij2 = linalg::simd::mls(b_ij2, a_ik2, b_kj2);
				a_ik3 = linalg::simd::load<float64x2_t>(a_ik + 6);
				b_kj3 = linalg::simd::load<float64x2_t>(b_kj + 6);
				b_ij3 = linalg::simd::mls(b_ij3, a_ik3, b_kj3);
				a_ik4 = linalg::simd::load<float64x2_t>(a_ik + 8);
				b_kj4 = linalg::simd::load<float64x2_t>(b_kj + 8);
				b_ij4 = linalg::simd::mls(b_ij4, a_ik4, b_kj4);
				a_ik5 = linalg::simd::load<float64x2_t>(a_ik + 10);
				b_kj5 = linalg::simd::load<float64x2_t>(b_kj + 10);
				b_ij5 = linalg::simd::mls(b_ij5, a_ik5, b_kj5);
				a_ik6 = linalg::simd::load<float64x2_t>(a_ik + 12);
				b_kj6 = linalg::simd::load<float64x2_t>(b_kj + 12);
				b_ij6 = linalg::simd::mls(b_ij6, a_ik6, b_kj6);
				a_ik7 = linalg::simd::load<float64x2_t>(a_ik + 14);
				b_kj7 = linalg::simd::load<float64x2_t>(b_kj + 14);
				b_ij7 = linalg::simd::mls(b_ij7, a_ik7, b_kj7);
			}

			if (nounit) {
				// B(i, j) /= A(i, i)
				b_ij0 = linalg::simd::mul(b_ij0, a_ii0);
				b_ij1 = linalg::simd::mul(b_ij1, a_ii1);
				b_ij2 = linalg::simd::mul(b_ij2, a_ii2);
				b_ij3 = linalg::simd::mul(b_ij3, a_ii3);
				b_ij4 = linalg::simd::mul(b_ij4, a_ii4);
				b_ij5 = linalg::simd::mul(b_ij5, a_ii5);
				b_ij6 = linalg::simd::mul(b_ij6, a_ii6);
				b_ij7 = linalg::simd::mul(b_ij7, a_ii7);
			}
			linalg::simd::store(b_ij + 0, b_ij0);
			linalg::simd::store(b_ij + 2, b_ij1);
			linalg::simd::store(b_ij + 4, b_ij2);
			linalg::simd::store(b_ij + 6, b_ij3);
			linalg::simd::store(b_ij + 8, b_ij4);
			linalg::simd::store(b_ij + 10, b_ij5);
			linalg::simd::store(b_ij + 12, b_ij6);
			linalg::simd::store(b_ij + 14, b_ij7);
		}
	}
}

inline void dtrsm_neon_vl8(kernel_inttype ninter, const double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A,
                           double *B_p, kernel_inttype istrd_B, kernel_inttype jstrd_B, kernel_inttype m, kernel_inttype n,
                           bool nounit, double alpha) {
	const float64x2_t onex2 = {1.0, 1.0};
	float64x2_t alphax2 = {alpha, alpha};

	for (kernel_inttype i = m - 1; i >= 0; i--) {
		float64x2_t a_ii0, a_ii1, a_ii2, a_ii3;
		auto a_ii = A_p + i*istrd_A + i*jstrd_A;
		if (nounit) {
			a_ii0 = linalg::simd::load<float64x2_t>(a_ii + 0);
			a_ii0 = linalg::simd::div(onex2, a_ii0);
			a_ii1 = linalg::simd::load<float64x2_t>(a_ii + 2);
			a_ii1 = linalg::simd::div(onex2, a_ii1);
			a_ii2 = linalg::simd::load<float64x2_t>(a_ii + 4);
			a_ii2 = linalg::simd::div(onex2, a_ii2);
			a_ii3 = linalg::simd::load<float64x2_t>(a_ii + 6);
			a_ii3 = linalg::simd::div(onex2, a_ii3);
		}

		for (kernel_inttype j = 0; j<n; j++) {
			float64x2_t b_ij0, b_ij1, b_ij2, b_ij3;
			auto b_ij = B_p + i*istrd_B + j*jstrd_B;
			// B(i, j) *= alpha
			b_ij0 = linalg::simd::load<float64x2_t>(b_ij + 0);
			b_ij0 = linalg::simd::mul(b_ij0, alphax2);
			b_ij1 = linalg::simd::load<float64x2_t>(b_ij + 2);
			b_ij1 = linalg::simd::mul(b_ij1, alphax2);
			b_ij2 = linalg::simd::load<float64x2_t>(b_ij + 4);
			b_ij2 = linalg::simd::mul(b_ij2, alphax2);
			b_ij3 = linalg::simd::load<float64x2_t>(b_ij + 6);
			b_ij3 = linalg::simd::mul(b_ij3, alphax2);

			for (kernel_inttype k = i + 1; k<m; k++) {
				float64x2_t a_ik0, a_ik1, a_ik2, a_ik3;
				auto a_ik = A_p + i*istrd_A + k*jstrd_A;
				float64x2_t b_kj0, b_kj1, b_kj2, b_kj3;
				auto b_kj = B_p + k*istrd_B + j*jstrd_B;
				// B(i, j) = B(i, j) - A(i, k)*B(k, j)
				a_ik0 = linalg::simd::load<float64x2_t>(a_ik + 0);
				b_kj0 = linalg::simd::load<float64x2_t>(b_kj + 0);
				b_ij0 = linalg::simd::mls(b_ij0, a_ik0, b_kj0);
				a_ik1 = linalg::simd::load<float64x2_t>(a_ik + 2);
				b_kj1 = linalg::simd::load<float64x2_t>(b_kj + 2);
				b_ij1 = linalg::simd::mls(b_ij1, a_ik1, b_kj1);
				a_ik2 = linalg::simd::load<float64x2_t>(a_ik + 4);
				b_kj2 = linalg::simd::load<float64x2_t>(b_kj + 4);
				b_ij2 = linalg::simd::mls(b_ij2, a_ik2, b_kj2);
				a_ik3 = linalg::simd::load<float64x2_t>(a_ik + 6);
				b_kj3 = linalg::simd::load<float64x2_t>(b_kj + 6);
				b_ij3 = linalg::simd::mls(b_ij3, a_ik3, b_kj3);
			}

			if (nounit) {
				// B(i, j) /= A(i, i)
				b_ij0 = linalg::simd::mul(b_ij0, a_ii0);
				b_ij1 = linalg::simd::mul(b_ij1, a_ii1);
				b_ij2 = linalg::simd::mul(b_ij2, a_ii2);
				b_ij3 = linalg::simd::mul(b_ij3, a_ii3);
			}
			linalg::simd::store(b_ij + 0, b_ij0);
			linalg::simd::store(b_ij + 2, b_ij1);
			linalg::simd::store(b_ij + 4, b_ij2);
			linalg::simd::store(b_ij + 6, b_ij3);
		}
	}
}

inline void dtrsm_neon_vl4(kernel_inttype ninter, const double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A,
                           double *B_p, kernel_inttype istrd_B, kernel_inttype jstrd_B, kernel_inttype m, kernel_inttype n,
                           bool nounit, double alpha) {
	const float64x2_t onex2 = {1.0, 1.0};
	float64x2_t alphax2 = {alpha, alpha};

	for (kernel_inttype i = m - 1; i >= 0; i--) {
		float64x2_t a_ii0, a_ii1;
		auto a_ii = A_p + i*istrd_A + i*jstrd_A;
		if (nounit) {
			a_ii0 = linalg::simd::load<float64x2_t>(a_ii + 0);
			a_ii0 = linalg::simd::div(onex2, a_ii0);
			a_ii1 = linalg::simd::load<float64x2_t>(a_ii + 2);
			a_ii1 = linalg::simd::div(onex2, a_ii1);
		}

		for (kernel_inttype j = 0; j<n; j++) {
			float64x2_t b_ij0, b_ij1;
			auto b_ij = B_p + i*istrd_B + j*jstrd_B;
			// B(i, j) *= alpha
			b_ij0 = linalg::simd::load<float64x2_t>(b_ij + 0);
			b_ij0 = linalg::simd::mul(b_ij0, alphax2);
			b_ij1 = linalg::simd::load<float64x2_t>(b_ij + 2);
			b_ij1 = linalg::simd::mul(b_ij1, alphax2);

			for (kernel_inttype k = i + 1; k<m; k++) {
				float64x2_t a_ik0, a_ik1;
				auto a_ik = A_p + i*istrd_A + k*jstrd_A;
				float64x2_t b_kj0, b_kj1;
				auto b_kj = B_p + k*istrd_B + j*jstrd_B;
				// B(i, j) = B(i, j) - A(i, k)*B(k, j)
				a_ik0 = linalg::simd::load<float64x2_t>(a_ik + 0);
				b_kj0 = linalg::simd::load<float64x2_t>(b_kj + 0);
				b_ij0 = linalg::simd::mls(b_ij0, a_ik0, b_kj0);
				a_ik1 = linalg::simd::load<float64x2_t>(a_ik + 2);
				b_kj1 = linalg::simd::load<float64x2_t>(b_kj + 2);
				b_ij1 = linalg::simd::mls(b_ij1, a_ik1, b_kj1);
			}

			if (nounit) {
				// B(i, j) /= A(i, i)
				b_ij0 = linalg::simd::mul(b_ij0, a_ii0);
				b_ij1 = linalg::simd::mul(b_ij1, a_ii1);
			}
			linalg::simd::store(b_ij + 0, b_ij0);
			linalg::simd::store(b_ij + 2, b_ij1);
		}
	}
}

inline void dtrsm_neon_vl2(kernel_inttype ninter, const double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A,
                           double *B_p, kernel_inttype istrd_B, kernel_inttype jstrd_B, kernel_inttype m, kernel_inttype n,
                           bool nounit, double alpha) {
	const float64x2_t onex2 = {1.0, 1.0};
	float64x2_t alphax2 = {alpha, alpha};

	for (kernel_inttype i = m - 1; i >= 0; i--) {
		float64x2_t a_ii0;
		auto a_ii = A_p + i*istrd_A + i*jstrd_A;
		if (nounit) {
			a_ii0 = linalg::simd::load<float64x2_t>(a_ii + 0);
			a_ii0 = linalg::simd::div(onex2, a_ii0);
		}

		for (kernel_inttype j = 0; j<n; j++) {
			float64x2_t b_ij0;
			auto b_ij = B_p + i*istrd_B + j*jstrd_B;
			// B(i, j) *= alpha
			b_ij0 = linalg::simd::load<float64x2_t>(b_ij + 0);
			b_ij0 = linalg::simd::mul(b_ij0, alphax2);

			for (kernel_inttype k = i + 1; k<m; k++) {
				float64x2_t a_ik0;
				auto a_ik = A_p + i*istrd_A + k*jstrd_A;
				float64x2_t b_kj0;
				auto b_kj = B_p + k*istrd_B + j*jstrd_B;
				// B(i, j) = B(i, j) - A(i, k)*B(k, j)
				a_ik0 = linalg::simd::load<float64x2_t>(a_ik + 0);
				b_kj0 = linalg::simd::load<float64x2_t>(b_kj + 0);
				b_ij0 = linalg::simd::mls(b_ij0, a_ik0, b_kj0);
			}

			if (nounit) {
				// B(i, j) /= A(i, i)
				b_ij0 = linalg::simd::mul(b_ij0, a_ii0);
			}
			linalg::simd::store(b_ij + 0, b_ij0);
		}
	}
}

inline void dtrsm_neon_vl1(kernel_inttype ninter, const double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A,
                           double *B_p, kernel_inttype istrd_B, kernel_inttype jstrd_B, kernel_inttype m, kernel_inttype n,
                           bool nounit, double alpha) {

	const double one = 1.0;

	for (kernel_inttype i = m - 1; i >= 0; i--) {
		double a_ii0;
		auto a_ii = A_p + i*istrd_A + i*jstrd_A;

		if (nounit) {
			a_ii0 = *a_ii;
			a_ii0 = one / a_ii0;
		}

		for (kernel_inttype j = 0; j<n; j++) {
			double b_ij0;
			auto b_ij = B_p + i*istrd_B + j*jstrd_B;
			// B(i, j) *= alpha
			b_ij0 = *b_ij;
			b_ij0 *= alpha;

			for (kernel_inttype k = i + 1; k<m; k++) {
				double a_ik0;
				auto a_ik = A_p + i*istrd_A + k*jstrd_A;
				double b_kj0;
				auto b_kj = B_p + k*istrd_B + j*jstrd_B;
				// B(i, j) = B(i, j) - A(i, k)*B(k, j)
				a_ik0 = *a_ik;
				b_kj0 = *b_kj;
				b_ij0 -= a_ik0 * b_kj0;
			}

			if (nounit) {
				// B(i, j) /= A(i, i)
				b_ij0 *= a_ii0;
				*b_ij = b_ij0;
			}
			else {
				*b_ij = b_ij0;
			}
		}
	}
}

void dtrsm_neon_kernel(kernel_inttype ninter, const double *A_p, kernel_inttype istrd_A, kernel_inttype jstrd_A,
                      double *B_p, kernel_inttype istrd_B, kernel_inttype jstrd_B, kernel_inttype m, kernel_inttype n,
                      bool nounit, double alpha) {
	// The input parameters assume an LNU (left, notrans, upper) case.
	// For other cases, the problem is converted to an LNU problem by simple
	// transformations of the input parameters before calling this routine.
	kernel_inttype ni;

	for (ni = 0; ni <= ninter - 16; ni += 16) {
		perflibs::interleave_batch::dtrsm_neon_vl16(16, A_p + ni, istrd_A, jstrd_A, B_p + ni,
		                                         istrd_B, jstrd_B, m, n, nounit, alpha);
	}
	if (ni <= ninter - 8) {
		perflibs::interleave_batch::dtrsm_neon_vl8(8, A_p + ni, istrd_A, jstrd_A, B_p + ni,
		                                        istrd_B, jstrd_B, m, n, nounit, alpha);
		ni += 8;
	}
	if (ni <= ninter - 4) {
		perflibs::interleave_batch::dtrsm_neon_vl4(4, A_p + ni, istrd_A, jstrd_A, B_p + ni,
		                                        istrd_B, jstrd_B, m, n, nounit, alpha);
		ni += 4;
	}
	if (ni <= ninter - 2) {
		perflibs::interleave_batch::dtrsm_neon_vl2(2, A_p + ni, istrd_A, jstrd_A, B_p + ni,
		                                        istrd_B, jstrd_B, m, n, nounit, alpha);
		ni += 2;
	}
	if (ni == ninter - 1) {
		perflibs::interleave_batch::dtrsm_neon_vl1(1, A_p + ni, istrd_A, jstrd_A, B_p + ni,
		                                        istrd_B, jstrd_B, m, n, nounit, alpha);
	}
}

} // namespace perflibs::interleave_batch

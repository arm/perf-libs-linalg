/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "packages/batch_interleave/matmul/kernels/gemm_kernels.hpp"
#include "matrix/interleave_batch.hpp"
#include "perflibs_assert.hpp"

#include <array>

namespace perflibs::interleave_batch {

namespace {

template<bool BetaIsZero, typename ValueType, typename BetaType>
PERFLIBS_LINALG_INLINE
ValueType load_and_scale_c(const double *c, const ValueType zero, const BetaType beta) {
	if constexpr (BetaIsZero) {
		return zero;
	} else {
		return linalg::simd::mul(linalg::simd::load<ValueType>(c), beta);
	}
}

template<bool BetaIsZero>
void dgemm_neon_impl(
	kernel_inttype ninter, kernel_inttype nbatch,
	kernel_inttype m, kernel_inttype n, kernel_inttype k,
	double alpha,
	const double *a, kernel_inttype a_bat_step, kernel_inttype a_cntg_step, kernel_inttype a_strd_step,
	const double *b, kernel_inttype b_bat_step, kernel_inttype b_cntg_step, kernel_inttype b_strd_step,
	double beta,
	      double *c, kernel_inttype c_bat_step, kernel_inttype c_cntg_step, kernel_inttype c_strd_step) {

	const float64x2_t alpha0 { alpha, alpha };
	const float64x2_t beta0  { beta, beta };
	const float64x2_t zero0  { 0.0, 0.0 };


	switch(ninter) {
	default:
		PERFLIBS_ASSERT(false, "invalid interleave count");
		break;
	case 1:
	{
		for(kernel_inttype batch = 0; batch != nbatch; ++batch) {
			auto a_batch = a + batch * a_bat_step;
			auto b_batch = b + batch * b_bat_step;
			auto c_batch = c + batch * c_bat_step;

			for(kernel_inttype j = 0; j != n; ++j) {
				auto b_batch_row = b_batch + j * b_strd_step;

				for(kernel_inttype i = 0; i != m; ++i) {
					auto a_batch_row = a_batch + i * a_strd_step;
					auto c_inter     = c_batch + i * c_cntg_step + j * c_strd_step;

					double tmp = 0.0;

					for(kernel_inttype l = 0; l != k; ++l) {
						auto a_inter = a_batch_row + l * a_cntg_step;
						auto b_inter = b_batch_row + l * b_cntg_step;

						auto a0 = linalg::simd::load<double>(a_inter);
						auto b0 = linalg::simd::load<double>(b_inter);

						linalg::simd::mla(tmp, a0, b0);
					}

					auto c0 = load_and_scale_c<BetaIsZero>(c_inter, 0.0, beta);
					linalg::simd::mla(c0, tmp, alpha);
					linalg::simd::store(c_inter, c0);
				}
			}
		}
		break;
	}
	case 2:
	{
		for(kernel_inttype batch = 0; batch != nbatch; ++batch) {
			auto a_batch = a + batch * a_bat_step;
			auto b_batch = b + batch * b_bat_step;
			auto c_batch = c + batch * c_bat_step;

			for(kernel_inttype j = 0; j != n; ++j) {
				auto b_batch_row = b_batch + j * b_strd_step;

				for(kernel_inttype i = 0; i != m; ++i) {
					auto a_batch_row = a_batch + i * a_strd_step;
					auto c_inter     = c_batch + i * c_cntg_step + j * c_strd_step;

					float64x2_t tmp = zero0;

					for(kernel_inttype l = 0; l != k; ++l) {
						auto a_inter = a_batch_row + l * a_cntg_step;
						auto b_inter = b_batch_row + l * b_cntg_step;

						auto a0 = linalg::simd::load<float64x2_t>(a_inter);
						auto b0 = linalg::simd::load<float64x2_t>(b_inter);

						linalg::simd::mla(tmp, a0, b0);
					}

					auto c0 = load_and_scale_c<BetaIsZero>(c_inter, zero0, beta0);
					linalg::simd::mla(c0, tmp, alpha0);
					linalg::simd::store(c_inter, c0);
				}
			}
		}
		break;
	}
	case 4:
	{
		std::array<float64x2_t, 2> tmp;
		for(kernel_inttype batch = 0; batch != nbatch; ++batch) {
			auto a_batch = a + batch * a_bat_step;
			auto b_batch = b + batch * b_bat_step;
			auto c_batch = c + batch * c_bat_step;

			for(kernel_inttype j = 0; j != n; ++j) {
				auto b_batch_row = b_batch + j * b_strd_step;

				for(kernel_inttype i = 0; i != m; ++i) {
					auto a_batch_row = a_batch + i * a_strd_step;
					auto c_inter     = c_batch + i * c_cntg_step + j * c_strd_step;

					//zero temp
					auto c0 = load_and_scale_c<BetaIsZero>(c_inter, zero0, beta0);
					tmp[0] = zero0;
					auto c1 = load_and_scale_c<BetaIsZero>(c_inter + 2, zero0, beta0);
					tmp[1] = zero0;

					for(kernel_inttype l = 0; l != k; ++l) {
						auto a_inter = a_batch_row + l * a_cntg_step;
						auto b_inter = b_batch_row + l * b_cntg_step;

						auto a0 = linalg::simd::load<float64x2_t>(a_inter);
						auto b0 = linalg::simd::load<float64x2_t>(b_inter);

						auto a1 = linalg::simd::load<float64x2_t>(a_inter + 2);
						auto b1 = linalg::simd::load<float64x2_t>(b_inter + 2);

						linalg::simd::mla(tmp[0], a0, b0);
						linalg::simd::mla(tmp[1], a1, b1);
					}

					linalg::simd::mla(c0, tmp[0], alpha0);
					linalg::simd::store(c_inter, c0);
					linalg::simd::mla(c1, tmp[1], alpha0);
					linalg::simd::store(c_inter + 2, c1);
				}
			}
		}
		break;
	}
	case 8:
	{
		std::array<float64x2_t, 4> tmp;
		for(kernel_inttype batch = 0; batch != nbatch; ++batch) {
			auto a_batch = a + batch * a_bat_step;
			auto b_batch = b + batch * b_bat_step;
			auto c_batch = c + batch * c_bat_step;

			for(kernel_inttype j = 0; j != n; ++j) {
				auto b_batch_row = b_batch + j * b_strd_step;

				for(kernel_inttype i = 0; i != m; ++i) {
					auto a_batch_row = a_batch + i * a_strd_step;
					auto c_inter     = c_batch + i * c_cntg_step + j * c_strd_step;

					//zero temp
					auto c0 = load_and_scale_c<BetaIsZero>(c_inter, zero0, beta0);
					tmp[0] = zero0;
					auto c1 = load_and_scale_c<BetaIsZero>(c_inter + 2, zero0, beta0);
					tmp[1] = zero0;
					auto c2 = load_and_scale_c<BetaIsZero>(c_inter + 4, zero0, beta0);
					tmp[2] = zero0;
					auto c3 = load_and_scale_c<BetaIsZero>(c_inter + 6, zero0, beta0);
					tmp[3] = zero0;

					for(kernel_inttype l = 0; l != k; ++l) {
						auto a_inter = a_batch_row + l * a_cntg_step;
						auto b_inter = b_batch_row + l * b_cntg_step;

						auto a0 = linalg::simd::load<float64x2_t>(a_inter);
						auto b0 = linalg::simd::load<float64x2_t>(b_inter);

						auto a1 = linalg::simd::load<float64x2_t>(a_inter + 2);
						auto b1 = linalg::simd::load<float64x2_t>(b_inter + 2);

						auto a2 = linalg::simd::load<float64x2_t>(a_inter + 4);
						auto b2 = linalg::simd::load<float64x2_t>(b_inter + 4);

						auto a3 = linalg::simd::load<float64x2_t>(a_inter + 6);
						auto b3 = linalg::simd::load<float64x2_t>(b_inter + 6);

						linalg::simd::mla(tmp[0], a0, b0);
						linalg::simd::mla(tmp[1], a1, b1);
						linalg::simd::mla(tmp[2], a2, b2);
						linalg::simd::mla(tmp[3], a3, b3);
					}

					linalg::simd::mla(c0, tmp[0], alpha0);
					linalg::simd::store(c_inter, c0);
					linalg::simd::mla(c1, tmp[1], alpha0);
					linalg::simd::store(c_inter + 2, c1);
					linalg::simd::mla(c2, tmp[2], alpha0);
					linalg::simd::store(c_inter + 4, c2);
					linalg::simd::mla(c3, tmp[3], alpha0);
					linalg::simd::store(c_inter + 6, c3);
				}
			}
		}
		break;
	}
	case 16:
	{
		std::array<float64x2_t, 8> tmp;
		for(kernel_inttype batch = 0; batch != nbatch; ++batch) {
			auto a_batch = a + batch * a_bat_step;
			auto b_batch = b + batch * b_bat_step;
			auto c_batch = c + batch * c_bat_step;

			for(kernel_inttype j = 0; j != n; ++j) {
				auto b_batch_row = b_batch + j * b_strd_step;

				for(kernel_inttype i = 0; i != m; ++i) {
					auto a_batch_row = a_batch + i * a_strd_step;
					auto c_inter     = c_batch + i * c_cntg_step + j * c_strd_step;

					//zero temp
					tmp[0] = zero0;
					tmp[1] = zero0;
					tmp[2] = zero0;
					tmp[3] = zero0;
					tmp[4] = zero0;
					tmp[5] = zero0;
					tmp[6] = zero0;
					tmp[7] = zero0;

					for(kernel_inttype l = 0; l != k; ++l) {
						auto a_inter = a_batch_row + l * a_cntg_step;
						auto b_inter = b_batch_row + l * b_cntg_step;

						auto a0 = linalg::simd::load<float64x2_t>(a_inter);
						auto b0 = linalg::simd::load<float64x2_t>(b_inter);

						auto a1 = linalg::simd::load<float64x2_t>(a_inter + 2);
						auto b1 = linalg::simd::load<float64x2_t>(b_inter + 2);


						auto a2 = linalg::simd::load<float64x2_t>(a_inter + 4);
						auto b2 = linalg::simd::load<float64x2_t>(b_inter + 4);

						auto a3 = linalg::simd::load<float64x2_t>(a_inter + 6);
						auto b3 = linalg::simd::load<float64x2_t>(b_inter + 6);

						linalg::simd::mla(tmp[0], a0, b0);
						a0 = linalg::simd::load<float64x2_t>(a_inter + 8);
						b0 = linalg::simd::load<float64x2_t>(b_inter + 8);
						linalg::simd::mla(tmp[1], a1, b1);
						a1 = linalg::simd::load<float64x2_t>(a_inter + 10);
						b1 = linalg::simd::load<float64x2_t>(b_inter + 10);
						linalg::simd::mla(tmp[2], a2, b2);
						a2 = linalg::simd::load<float64x2_t>(a_inter + 12);
						b2 = linalg::simd::load<float64x2_t>(b_inter + 12);
						linalg::simd::mla(tmp[3], a3, b3);
						a3 = linalg::simd::load<float64x2_t>(a_inter + 14);
						b3 = linalg::simd::load<float64x2_t>(b_inter + 14);

						linalg::simd::mla(tmp[4], a0, b0);
						linalg::simd::mla(tmp[5], a1, b1);
						linalg::simd::mla(tmp[6], a2, b2);
						linalg::simd::mla(tmp[7], a3, b3);
					}
					auto c0 = load_and_scale_c<BetaIsZero>(c_inter, zero0, beta0);
					auto c1 = load_and_scale_c<BetaIsZero>(c_inter + 2, zero0, beta0);
					auto c2 = load_and_scale_c<BetaIsZero>(c_inter + 4, zero0, beta0);
					auto c3 = load_and_scale_c<BetaIsZero>(c_inter + 6, zero0, beta0);

					linalg::simd::mla(c0, tmp[0], alpha0);
					linalg::simd::mla(c1, tmp[1], alpha0);
					linalg::simd::mla(c2, tmp[2], alpha0);
					linalg::simd::mla(c3, tmp[3], alpha0);

					linalg::simd::store(c_inter, c0);
					linalg::simd::store(c_inter + 2, c1);
					linalg::simd::store(c_inter + 4, c2);
					linalg::simd::store(c_inter + 6, c3);

					c0 = load_and_scale_c<BetaIsZero>(c_inter + 8, zero0, beta0);
					c1 = load_and_scale_c<BetaIsZero>(c_inter + 10, zero0, beta0);
					c2 = load_and_scale_c<BetaIsZero>(c_inter + 12, zero0, beta0);
					c3 = load_and_scale_c<BetaIsZero>(c_inter + 14, zero0, beta0);

					linalg::simd::mla(c0, tmp[4], alpha0);
					linalg::simd::mla(c1, tmp[5], alpha0);
					linalg::simd::mla(c2, tmp[6], alpha0);
					linalg::simd::mla(c3, tmp[7], alpha0);

					linalg::simd::store(c_inter + 8, c0);
					linalg::simd::store(c_inter + 10, c1);
					linalg::simd::store(c_inter + 12, c2);
					linalg::simd::store(c_inter + 14, c3);
				}
			}
		}}
	}
}

} // namespace

void dgemm_neon_beta_zero(
	kernel_inttype ninter, kernel_inttype nbatch, kernel_inttype m, kernel_inttype n,
	kernel_inttype k, double alpha, const double *a, kernel_inttype a_bat_step,
	kernel_inttype a_cntg_step, kernel_inttype a_strd_step, const double *b,
	kernel_inttype b_bat_step, kernel_inttype b_cntg_step, kernel_inttype b_strd_step,
	double beta, double *c, kernel_inttype c_bat_step, kernel_inttype c_cntg_step,
	kernel_inttype c_strd_step) {
	dgemm_neon_impl<true>(ninter, nbatch, m, n, k, alpha, a, a_bat_step, a_cntg_step, a_strd_step,
	                      b, b_bat_step, b_cntg_step, b_strd_step, beta, c, c_bat_step, c_cntg_step,
	                      c_strd_step);
}

void dgemm_neon_beta_nonzero(
	kernel_inttype ninter, kernel_inttype nbatch, kernel_inttype m, kernel_inttype n,
	kernel_inttype k, double alpha, const double *a, kernel_inttype a_bat_step,
	kernel_inttype a_cntg_step, kernel_inttype a_strd_step, const double *b,
	kernel_inttype b_bat_step, kernel_inttype b_cntg_step, kernel_inttype b_strd_step,
	double beta, double *c, kernel_inttype c_bat_step, kernel_inttype c_cntg_step,
	kernel_inttype c_strd_step) {
	dgemm_neon_impl<false>(ninter, nbatch, m, n, k, alpha, a, a_bat_step, a_cntg_step, a_strd_step,
	                       b, b_bat_step, b_cntg_step, b_strd_step, beta, c, c_bat_step, c_cntg_step,
	                       c_strd_step);
}

} //namespace perflibs::interleave_batch

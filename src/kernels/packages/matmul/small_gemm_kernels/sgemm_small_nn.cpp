/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "perflibs_assert.hpp"
#include "detect/omp.hpp"
#include "perflibs_util.hpp"
#include "perflibs_numeric_utils.hpp"
#include "gemm_small_framework.hpp"
#include "gemm_small_tx2.hpp"
#include "sgemm_unrolled.hpp"

#include <cstdio>
#include <cstdlib>
#include <algorithm>

static const kernel_inttype M2_MIN = 40;
static const kernel_inttype N2_MIN = 16;

extern "C" void sgemm_small_kernel_nn_tx2(kernel_inttype m, kernel_inttype n, kernel_inttype k_, float alpha,
                                      const float *a, kernel_inttype lda, const float *b, kernel_inttype ldb,
                                      float beta, float *c, kernel_inttype ldc);

/** Single-threaded small sgemm implementation, responsible for blocking
 *  and calling into the sgemm_small_kernel_nn_tx2 assembly kernel.
 */
static void sgemm_small_nn_mp1_tx2(
		perflibs::perflibs_trans transa, perflibs::perflibs_trans transb, kernel_inttype m, kernel_inttype n, kernel_inttype k_,
		float alpha, const float *a, kernel_inttype lda,
		const float *b, kernel_inttype ldb, float beta,
		float *c, kernel_inttype ldc) {

	PERFLIBS_ASSERT(transa == perflibs::PERFLIBS_NOTRANS && transb == perflibs::PERFLIBS_NOTRANS);

	kernel_inttype kblk_base = std::max<kernel_inttype>(iround(static_cast<kernel_inttype>(exp(9.33 + -0.80*log(lda))), 4), 8);
	kernel_inttype kextend_max = kblk_base/2;

	for (kernel_inttype k=0; k<k_; ) {
		auto kblk = k_-k > kblk_base && k_-k <= kblk_base + kextend_max ? k_-k : kblk_base;
		auto a2 = &a[k*lda];
		auto b2 = &b[k];
		auto k2 = std::min(kblk, k_-k);
		auto beta2 = k == 0 ? beta : 1.f;
		sgemm_small_kernel_nn_tx2(m, n, k2, alpha, a2, lda, b2, ldb, beta2, c, ldc);
		k += kblk;
	}
}

/** Multi-threaded small sgemm implementation, divides up the total matrix
 *  according to the m-divide split and n-divide split provided, then calls
 *  into the single-threaded sgemm_small_nn_mp1_tx2 implementation.
 */
static void sgemm_small_nn_mp_tx2(
		perflibs::perflibs_trans transa, perflibs::perflibs_trans transb, kernel_inttype m, kernel_inttype n, kernel_inttype k,
		float alpha, const float *a, kernel_inttype lda,
		const float *b, kernel_inttype ldb, float beta,
		float *c, kernel_inttype ldc,
		kernel_inttype mdiv, kernel_inttype ndiv) {

	// figure out what thread this is, and hence what block it is doing.
	auto id = perflibs::linalg::omp::get_thread_num();
	auto n_id = id % ndiv;
	auto m_id = id / ndiv;

	auto n2 = std::max(iround(n, ndiv)/ndiv, N2_MIN);
	auto j2 = n_id * n2;
	n2 = std::min(n-j2, n2);

	auto m2 = std::max(iround(m, mdiv)/mdiv, M2_MIN);
	auto i2 = iround(m_id * m2, 4);
	auto i2_next = iround((m_id+1) * m2, 4);
	m2 = std::min(m-i2, i2_next-i2);

	auto a2 = &a[transa == perflibs::PERFLIBS_NOTRANS ? i2 : i2*lda];
	auto b2 = &b[transb == perflibs::PERFLIBS_NOTRANS ? j2*ldb : j2];
	auto c2 = &c[j2*ldc + i2];

	sgemm_small_nn_mp1_tx2(transa, transb, m2, n2, k, alpha, a2, lda, b2, ldb, beta, c2, ldc);
}

/** This function figures out a "plan" of how to divide up the matrix into
 *  blocks of work to do, returning a pair of "m-divisions" and "n-division"
 *  respectively. The total number of threads to be used is the product of
 *  the two, and may be less than that provided as a parameter if the problem
 *  is too small.
 */
static std::pair<kernel_inttype, kernel_inttype> sgemm_small_nn_mp_plan_tx2(
		kernel_inttype m, kernel_inttype n, kernel_inttype k, kernel_inttype nthreads) {

	kernel_inttype mdiv = 1, ndiv = 1;

	while (nthreads > 1) {
		// find a factor to split on, based on the factorisation of the number of
		// threads available. This is suboptimal when using a prime number of
		// threads, but that case is not particularly common.
		kernel_inttype factor = 0;
		factor = nthreads % 2 == 0 ? 2 : 0;
		for (kernel_inttype i=3; factor == 0 && i*i<=nthreads; i+=2) {
			if (nthreads % i == 0) {
				factor = i;
			}
		}
		factor = factor != 0 ? factor : nthreads;
		PERFLIBS_ASSERT(nthreads % factor == 0);

		// find a dimension to split on, accounting for minimum block sizes
		auto m2 = iround(m, mdiv)/mdiv;
		auto n2 = iround(n, ndiv)/ndiv;
		bool split_on_m = m2 > M2_MIN && m/mdiv > n/ndiv;
		bool split_on_n = n2 > N2_MIN && n/ndiv >= m/mdiv;

		if (split_on_n) {
			factor = std::min(iround(n2, N2_MIN)/N2_MIN, factor);
			ndiv *= factor;
		}
		else if (split_on_m) {
			factor = std::min(iround(m2, M2_MIN)/M2_MIN, factor);
			mdiv *= factor;
		}
		else {
			// if we have run out of splits, just use fewer divisions and
			// hence use fewer threads.
			break;
		}
		nthreads /= factor;
	}

	return { mdiv, ndiv };
}

void perflibs::gemm::sgemm_small_nn_tx2(
		const perflibs_trans transa, const perflibs_trans transb, const kernel_inttype m, const kernel_inttype n, const kernel_inttype k,
		const float alpha, const float *const a, const kernel_inttype lda,
		const float *const b, const kernel_inttype ldb, const float beta,
		float *const c, const kernel_inttype ldc) {

	// figure out divisions, number of threads to use
	auto nt = ::perflibs::linalg::omp::get_max_cores();
	const auto p = sgemm_small_nn_mp_plan_tx2(m, n, k, nt);
	nt = p.first * p.second;

	if (nt > 1) {
		#pragma omp parallel default(none) firstprivate(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, p) num_threads(nt)
		{
			sgemm_small_nn_mp_tx2(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, p.first, p.second);
		}
	} else {
		sgemm_small_nn_mp1_tx2(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
	}
}

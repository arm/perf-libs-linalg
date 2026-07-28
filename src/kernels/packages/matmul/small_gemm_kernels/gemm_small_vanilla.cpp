/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "perflibs_assert.hpp"
#include "perflibs_numeric_utils.hpp"
#include "perflibs_util.hpp"
#include "perflibs_blas_types.hpp"
#include "cgemm_unrolled.hpp"

#include <cstdio>
#include <vector>
#include <algorithm>

extern "C" {
void cgemm_small_kernel_NN_nmk(kernel_inttype m, kernel_inttype n, kernel_inttype k, const std::complex<float> * A, kernel_inttype lda, const std::complex<float> * B, kernel_inttype ldb, std::complex<float> * C, kernel_inttype ldc, std::complex<float> alpha, std::complex<float> beta);
void cgemm_small_kernel_TN_nmk(kernel_inttype m, kernel_inttype n, kernel_inttype k, const std::complex<float> * A, kernel_inttype lda, const std::complex<float> * B, kernel_inttype ldb, std::complex<float> * C, kernel_inttype ldc, std::complex<float> alpha, std::complex<float> beta);
void cgemm_small_kernel_CN_nmk(kernel_inttype m, kernel_inttype n, kernel_inttype k, const std::complex<float> * A, kernel_inttype lda, const std::complex<float> * B, kernel_inttype ldb, std::complex<float> * C, kernel_inttype ldc, std::complex<float> alpha, std::complex<float> beta);
void cgemm_small_kernel_TT_mnk(kernel_inttype m, kernel_inttype n, kernel_inttype k, const std::complex<float> * A, kernel_inttype lda, const std::complex<float> * B, kernel_inttype ldb, std::complex<float> * C, kernel_inttype ldc, std::complex<float> alpha, std::complex<float> beta);
void cgemm_small_kernel_TC_mnk(kernel_inttype m, kernel_inttype n, kernel_inttype k, const std::complex<float> * A, kernel_inttype lda, const std::complex<float> * B, kernel_inttype ldb, std::complex<float> * C, kernel_inttype ldc, std::complex<float> alpha, std::complex<float> beta);
void cgemm_small_kernel_CT_mnk(kernel_inttype m, kernel_inttype n, kernel_inttype k, const std::complex<float> * A, kernel_inttype lda, const std::complex<float> * B, kernel_inttype ldb, std::complex<float> * C, kernel_inttype ldc, std::complex<float> alpha, std::complex<float> beta);
void cgemm_small_kernel_CC_mnk(kernel_inttype m, kernel_inttype n, kernel_inttype k, const std::complex<float> * A, kernel_inttype lda, const std::complex<float> * B, kernel_inttype ldb, std::complex<float> * C, kernel_inttype ldc, std::complex<float> alpha, std::complex<float> beta);
void cgemm_small_kernel_NT_nmk(kernel_inttype m, kernel_inttype n, kernel_inttype k, const std::complex<float> * A, kernel_inttype lda, const std::complex<float> * B, kernel_inttype ldb, std::complex<float> * C, kernel_inttype ldc, std::complex<float> alpha, std::complex<float> beta);
void cgemm_small_kernel_NC_nmk(kernel_inttype m, kernel_inttype n, kernel_inttype k, const std::complex<float> * A, kernel_inttype lda, const std::complex<float> * B, kernel_inttype ldb, std::complex<float> * C, kernel_inttype ldc, std::complex<float> alpha, std::complex<float> beta);
}


namespace perflibs::gemm {


static void cgemm_small_vanilla_serial (
	const perflibs_trans transa, const perflibs_trans transb, const kernel_inttype m, const kernel_inttype n,
	const kernel_inttype k, const std::complex<float> alpha, const std::complex<float> *const a,
	const kernel_inttype lda, const std::complex<float> *const b, const kernel_inttype ldb,
	const std::complex<float> beta, std::complex<float> *const c, const kernel_inttype ldc) {

	const kernel_inttype m4 = m - (m % 4);
	const kernel_inttype n2 = n - (n % 2);
	const kernel_inttype n4 = n - (n % 4);

	if (transa == PERFLIBS_NOTRANS && transb == PERFLIBS_NOTRANS) {
		const kernel_inttype mblock = m4 != 0 ? m4 : m;
		const kernel_inttype nblock = n2 != 0 ? n2 : n;

		if (mblock % 4 == 0 && nblock % 2 == 0) {
			cgemm_small_kernel_NN_nmk(mblock, nblock, k, a, lda, b, ldb, c, ldc, alpha, beta);
		} else {
			cgemm_unrolled_NN(mblock, nblock, k, alpha, a, lda, b, ldb, beta, c, ldc);
		}
		if (mblock != m) {
			const auto mm = m - mblock;
			cgemm_unrolled_NN(mm, nblock, k, alpha, &a[mblock], lda, b, ldb, beta, &c[mblock], ldc);
		}
		if (nblock != n) {
			const auto nn = n - nblock;
			cgemm_unrolled_NN(mblock, nn, k, alpha, a, lda, &b[ldb*nblock], ldb, beta, &c[ldc*nblock], ldc);
		}
		if (nblock != n && mblock != m) {
			const auto nn = n - nblock;
			const auto mm = m - mblock;
			cgemm_unrolled_NN(mm, nn, k, alpha, &a[mblock], lda, &b[ldb*nblock], ldb, beta, &c[ldc*nblock + mblock], ldc);
		}
	} else if (transa == PERFLIBS_TRANS && transb == PERFLIBS_NOTRANS) {
		const kernel_inttype mblock = m4 != 0 ? m4 : m;
		const kernel_inttype nblock = n2 != 0 ? n2 : n;

		if (mblock % 4 == 0 && nblock % 2 == 0) {
			cgemm_small_kernel_TN_nmk(mblock, nblock, k, a, lda, b, ldb, c, ldc, alpha, beta);
		} else {
			cgemm_unrolled_TN(mblock, nblock, k, alpha, a, lda, b, ldb, beta, c, ldc);
		}
		if (mblock != m) {
			const auto mm = m - mblock;
			cgemm_unrolled_TN(mm, nblock, k, alpha, &a[lda*mblock], lda, b, ldb, beta, &c[mblock], ldc);
		}
		if (nblock != n) {
			const auto nn = n - nblock;
			cgemm_unrolled_TN(mblock, nn, k, alpha, a, lda, &b[ldb*nblock], ldb, beta, &c[ldc*nblock], ldc);
		}
		if (nblock != n && mblock != m) {
			const auto nn = n - nblock;
			const auto mm = m - mblock;
			cgemm_unrolled_TN(mm, nn, k, alpha, &a[lda*mblock], lda, &b[ldb*nblock], ldb, beta, &c[ldc*nblock + mblock], ldc);
		}
	} else if (transa == PERFLIBS_TRANS && transb == PERFLIBS_TRANS) {
		const kernel_inttype mblock = m4 != 0 ? m4 : m;
		const kernel_inttype nblock = n4 != 0 ? n4 : n;

		if (mblock % 4 == 0 && nblock % 4 == 0) {
			cgemm_small_kernel_TT_mnk(mblock, nblock, k, a, lda, b, ldb, c, ldc, alpha, beta);
		} else {
			cgemm_unrolled_TT(mblock, nblock, k, alpha, a, lda, b, ldb, beta, c, ldc);
		}
		if (mblock != m) {
			const auto mm = m - mblock;
			cgemm_unrolled_TT(mm, nblock, k, alpha, &a[lda*mblock], lda, b, ldb, beta, &c[mblock], ldc);
		}
		if (nblock != n) {
			const auto nn = n - nblock;
			cgemm_unrolled_TT(mblock, nn, k, alpha, a, lda, &b[nblock], ldb, beta, &c[ldc*nblock], ldc);
		}
		if (nblock != n && mblock != m) {
			const auto nn = n - nblock;
			const auto mm = m - mblock;
			cgemm_unrolled_TT(mm, nn, k, alpha, &a[lda*mblock], lda, &b[nblock], ldb, beta, &c[ldc*nblock + mblock], ldc);
		}
	} else if (transa == PERFLIBS_TRANS && transb == PERFLIBS_CONJTRANS) {
		const kernel_inttype mblock = m4 != 0 ? m4 : m;
		const kernel_inttype nblock = n4 != 0 ? n4 : n;

		if (mblock % 4 == 0 && nblock % 4 == 0) {
			cgemm_small_kernel_TC_mnk(mblock, nblock, k, a, lda, b, ldb, c, ldc, alpha, beta);
		} else {
			cgemm_unrolled_TC(mblock, nblock, k, alpha, a, lda, b, ldb, beta, c, ldc);
		}
		if (mblock != m) {
			const auto mm = m - mblock;
			cgemm_unrolled_TC(mm, nblock, k, alpha, &a[lda*mblock], lda, b, ldb, beta, &c[mblock], ldc);
		}
		if (nblock != n) {
			const auto nn = n - nblock;
			cgemm_unrolled_TC(mblock, nn, k, alpha, a, lda, &b[nblock], ldb, beta, &c[ldc*nblock], ldc);
		}
		if (nblock != n && mblock != m) {
			const auto nn = n - nblock;
			const auto mm = m - mblock;
			cgemm_unrolled_TC(mm, nn, k, alpha, &a[lda*mblock], lda, &b[nblock], ldb, beta, &c[ldc*nblock + mblock], ldc);
		}
	} else if (transa == PERFLIBS_CONJTRANS && transb == PERFLIBS_NOTRANS) {
		const kernel_inttype mblock = m4 != 0 ? m4 : m;
		const kernel_inttype nblock = n2 != 0 ? n2 : n;

		if (mblock % 4 == 0 && nblock % 2 == 0) {
			cgemm_small_kernel_CN_nmk(mblock, nblock, k, a, lda, b, ldb, c, ldc, alpha, beta);
		} else {
			cgemm_unrolled_CN(mblock, nblock, k, alpha, a, lda, b, ldb, beta, c, ldc);
		}
		if (mblock != m) {
			const auto mm = m - mblock;
			cgemm_unrolled_CN(mm, nblock, k, alpha, &a[lda*mblock], lda, b, ldb, beta, &c[mblock], ldc);
		}
		if (nblock != n) {
			const auto nn = n - nblock;
			cgemm_unrolled_CN(mblock, nn, k, alpha, a, lda, &b[ldb*nblock], ldb, beta, &c[ldc*nblock], ldc);
		}
		if (nblock != n && mblock != m) {
			const auto nn = n - nblock;
			const auto mm = m - mblock;
			cgemm_unrolled_CN(mm, nn, k, alpha, &a[lda*mblock], lda, &b[ldb*nblock], ldb, beta, &c[ldc*nblock + mblock], ldc);
		}
	} else if (transa == PERFLIBS_CONJTRANS && transb == PERFLIBS_TRANS) {
		const kernel_inttype mblock = m4 != 0 ? m4 : m;
		const kernel_inttype nblock = n4 != 0 ? n4 : n;

		if (mblock % 4 == 0 && nblock % 4 == 0) {
			cgemm_small_kernel_CT_mnk(mblock, nblock, k, a, lda, b, ldb, c, ldc, alpha, beta);
		} else {
			cgemm_unrolled_CT(mblock, nblock, k, alpha, a, lda, b, ldb, beta, c, ldc);
		}
		if (mblock != m) {
			const auto mm = m - mblock;
			cgemm_unrolled_CT(mm, nblock, k, alpha, &a[lda*mblock], lda, b, ldb, beta, &c[mblock], ldc);
		}
		if (nblock != n) {
			const auto nn = n - nblock;
			cgemm_unrolled_CT(mblock, nn, k, alpha, a, lda, &b[nblock], ldb, beta, &c[ldc*nblock], ldc);
		}
		if (nblock != n && mblock != m) {
			const auto nn = n - nblock;
			const auto mm = m - mblock;
			cgemm_unrolled_CT(mm, nn, k, alpha, &a[lda*mblock], lda, &b[nblock], ldb, beta, &c[ldc*nblock + mblock], ldc);
		}
	} else if (transa == PERFLIBS_CONJTRANS && transb == PERFLIBS_CONJTRANS) {
		const kernel_inttype mblock = m4 != 0 ? m4 : m;
		const kernel_inttype nblock = n4 != 0 ? n4 : n;

		if (mblock % 4 == 0 && nblock % 4 == 0) {
			cgemm_small_kernel_CC_mnk(mblock, nblock, k, a, lda, b, ldb, c, ldc, alpha, beta);
		} else {
			cgemm_unrolled_CC(mblock, nblock, k, alpha, a, lda, b, ldb, beta, c, ldc);
		}
		if (mblock != m) {
			const auto mm = m - mblock;
			cgemm_unrolled_CC(mm, nblock, k, alpha, &a[lda*mblock], lda, b, ldb, beta, &c[mblock], ldc);
		}
		if (nblock != n) {
			const auto nn = n - nblock;
			cgemm_unrolled_CC(mblock, nn, k, alpha, a, lda, &b[nblock], ldb, beta, &c[ldc*nblock], ldc);
		}
		if (nblock != n && mblock != m) {
			const auto nn = n - nblock;
			const auto mm = m - mblock;
			cgemm_unrolled_CC(mm, nn, k, alpha, &a[lda*mblock], lda, &b[nblock], ldb, beta, &c[ldc*nblock + mblock], ldc);
		}
	} else if (transa == PERFLIBS_NOTRANS && transb == PERFLIBS_TRANS) {
		const kernel_inttype mblock = m4 != 0 ? m4 : m;
		const kernel_inttype nblock = n2 != 0 ? n2 : n;

		if (mblock % 4 == 0 && nblock % 2 == 0) {
			cgemm_small_kernel_NT_nmk(mblock, nblock, k, a, lda, b, ldb, c, ldc, alpha, beta);
		} else {
			cgemm_unrolled_NT(mblock, nblock, k, alpha, a, lda, b, ldb, beta, c, ldc);
		}
		if (mblock != m) {
			const auto mm = m - mblock;
			cgemm_unrolled_NT(mm, nblock, k, alpha, &a[mblock], lda, b, ldb, beta, &c[mblock], ldc);
		}
		if (nblock != n) {
			const auto nn = n - nblock;
			cgemm_unrolled_NT(mblock, nn, k, alpha, a, lda, &b[nblock], ldb, beta, &c[ldc*nblock], ldc);
		}
		if (nblock != n && mblock != m) {
			const auto nn = n - nblock;
			const auto mm = m - mblock;
			cgemm_unrolled_NT(mm, nn, k, alpha, &a[mblock], lda, &b[nblock], ldb, beta, &c[ldc*nblock + mblock], ldc);
		}
	} else if (transa == PERFLIBS_NOTRANS && transb == PERFLIBS_CONJTRANS) {
		const kernel_inttype mblock = m4 != 0 ? m4 : m;
		const kernel_inttype nblock = n2 != 0 ? n2 : n;

		if (mblock % 4 == 0 && nblock % 2 == 0) {
			cgemm_small_kernel_NC_nmk(mblock, nblock, k, a, lda, b, ldb, c, ldc, alpha, beta);
		} else {
			cgemm_unrolled_NC(mblock, nblock, k, alpha, a, lda, b, ldb, beta, c, ldc);
		}
		if (mblock != m) {
			const auto mm = m - mblock;
			cgemm_unrolled_NC(mm, nblock, k, alpha, &a[mblock], lda, b, ldb, beta, &c[mblock], ldc);
		}
		if (nblock != n) {
			const auto nn = n - nblock;
			cgemm_unrolled_NC(mblock, nn, k, alpha, a, lda, &b[nblock], ldb, beta, &c[ldc*nblock], ldc);
		}
		if (nblock != n && mblock != m) {
			const auto nn = n - nblock;
			const auto mm = m - mblock;
			cgemm_unrolled_NC(mm, nn, k, alpha, &a[mblock], lda, &b[nblock], ldb, beta, &c[ldc*nblock + mblock], ldc);
		}
	}
	else {
		PERFLIBS_ASSERT(0);
	}
}

void cgemm_small_vanilla (
	const kernel_inttype max_threads,
	const perflibs_trans transa, const perflibs_trans transb, const kernel_inttype m, const kernel_inttype n,
	const kernel_inttype k, const std::complex<float> alpha, const std::complex<float> *const a,
	const kernel_inttype lda, const std::complex<float> *const b, const kernel_inttype ldb,
	const std::complex<float> beta, std::complex<float> *const c, const kernel_inttype ldc) {

#ifdef _OPENMP
	auto nthreads = max_threads;
	if (n > 12 && nthreads > 1) {
		const kernel_inttype inc = iround(n, 4*nthreads) / nthreads;
		#pragma omp parallel for default(none) firstprivate(n, inc, transa, transb, m, k, alpha, a, lda, b, ldb, beta, c, ldc) num_threads(nthreads)
		for (kernel_inttype start=0; start < n; start += inc) {
			kernel_inttype work = std::min(n - start, inc);
			if (work > 0) {
				auto local_b = transb == PERFLIBS_NOTRANS ? &b[start*ldb] : &b[start];
				auto local_c = &c[start*ldc];
				cgemm_small_vanilla_serial(transa, transb, m, work, k, alpha, a, lda, local_b, ldb, beta, local_c, ldc);
			}
		}
		return;
	}
#endif
	cgemm_small_vanilla_serial(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}



} //namespace perflibs gemm

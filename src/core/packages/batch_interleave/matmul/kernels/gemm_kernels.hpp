/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_KERNELS_GEMM_KERNELS_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_KERNELS_GEMM_KERNELS_HPP

#include "framework/linalg_util.hpp"
#include "matrix/interleave_batch_matrix.hpp"

namespace perflibs::linalg {

template<typename StrategyTag>
struct batch_interleave_gemm_kernel_tag {
	batch_interleave_gemm_kernel_tag(const StrategyTag&) { }
};

template<
	typename ADataType,
	typename BDataType = ADataType,
	typename CDataType = ADataType,
	typename ScalarType = promote_t<ADataType, BDataType, CDataType>>
using batch_interleave_gemm_kernel_t = void(
	kernel_inttype ninter,
	kernel_inttype nbatch,
	kernel_inttype m,
	kernel_inttype n,
	kernel_inttype k,
	ScalarType alpha,
	const ADataType *a,
	kernel_inttype a_bat_step,
	kernel_inttype a_cntg_step,
	kernel_inttype a_strd_step,
	const BDataType *b,
	kernel_inttype b_bat_step,
	kernel_inttype b_cntg_step,
	kernel_inttype b_strd_step,
	ScalarType beta,
	CDataType *c,
	kernel_inttype c_bat_step,
	kernel_inttype c_cntg_step,
	kernel_inttype c_strd_step);

} // namespace perflibs::linalg

// Forward declarations of the architecture-specific kernels.
namespace perflibs::interleave_batch {
using bi_ger_kernel_t = void(kernel_inttype ninter, kernel_inttype m, kernel_inttype n, double alpha,
                           const double *x, kernel_inttype x_step, const double *y, kernel_inttype y_step,
                           double *a, kernel_inttype a_row_step, kernel_inttype a_column_step);
extern bi_ger_kernel_t ger_neon;
extern bi_ger_kernel_t ger_sve;

extern ::perflibs::linalg::batch_interleave_gemm_kernel_t<double> dgemm_neon_beta_zero;
extern ::perflibs::linalg::batch_interleave_gemm_kernel_t<double> dgemm_neon_beta_nonzero;
extern ::perflibs::linalg::batch_interleave_gemm_kernel_t<double> bi_dgemm_sve;
extern ::perflibs::linalg::batch_interleave_gemm_kernel_t<double> dgemm_sve_vla;
extern ::perflibs::linalg::batch_interleave_gemm_kernel_t<double> dgemm_sve_vls8;
} // namespace perflibs::interleave_batch

namespace perflibs::linalg::batch_interleave::matmul {
using ::perflibs::interleave_batch::dgemm_neon_beta_zero;
using ::perflibs::interleave_batch::dgemm_neon_beta_nonzero;
using ::perflibs::interleave_batch::bi_dgemm_sve;
using ::perflibs::interleave_batch::ger_neon;
using ::perflibs::interleave_batch::ger_sve;

// Scale C without reading A or B, retaining IEEE multiplication when beta is zero.
// ArchitectureSpec prevents differently targeted instantiations from being coalesced by the linker.
template<typename T, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void scal_gemm_shim(kernel_inttype ninter, kernel_inttype nbatch, kernel_inttype m, kernel_inttype n,
                    kernel_inttype k, T alpha, const T *a, kernel_inttype a_bat_step,
                    kernel_inttype a_cntg_step, kernel_inttype a_strd_step, const T *b,
                    kernel_inttype b_bat_step, kernel_inttype b_cntg_step, kernel_inttype b_strd_step,
                    T beta, T *c, kernel_inttype c_bat_step, kernel_inttype c_cntg_step,
                    kernel_inttype c_strd_step) {

	PERFLIBS_ASSERT(alpha == zero<T>, "SCAL kernels require alpha == 0");
	if (ninter == 0 || nbatch == 0 || m == 0 || n == 0) return;

	for (kernel_inttype batch = 0; batch < nbatch; ++batch) {
		const auto matrix = general_matrix {
			interleave_batch_matrix { c + batch * c_bat_step, ninter, m, n, c_cntg_step, c_strd_step }
		};
		for (kernel_inttype column = 0; column < n; ++column) {
			for (kernel_inttype row = 0; row < m; ++row) {
				matrix(row, column, write) *= beta;
			}
		}
	}
}

// Adapt a single-batch GER kernel to the matmul3 kernel signature.
template<auto Kernel>
PERFLIBS_LINALG_INLINE
void ger_gemm_shim(kernel_inttype ninter, kernel_inttype nbatch, kernel_inttype m, kernel_inttype n,
                   kernel_inttype k, double alpha, const double *a, kernel_inttype a_bat_step,
                   kernel_inttype a_cntg_step, kernel_inttype a_strd_step, const double *b,
                   kernel_inttype b_bat_step, kernel_inttype b_cntg_step, kernel_inttype b_strd_step,
                   double beta, double *c, kernel_inttype c_bat_step, kernel_inttype c_cntg_step,
                   kernel_inttype c_strd_step) {

	PERFLIBS_ASSERT(k == 1 && beta == 1.0, "GER kernels require k == 1 and beta == 1");
	for (kernel_inttype batch = 0; batch < nbatch; ++batch) {
		Kernel(ninter, m, n, alpha, a + batch * a_bat_step, a_strd_step,
		       b + batch * b_bat_step, b_strd_step, c + batch * c_bat_step, c_cntg_step, c_strd_step);
	}
}

template<bool BetaIsZero>
PERFLIBS_LINALG_INLINE
void bi_dgemm_neon(
	kernel_inttype ninter, kernel_inttype nbatch, kernel_inttype m, kernel_inttype n,
	kernel_inttype k, double alpha, const double *a, kernel_inttype a_bat_step,
	kernel_inttype a_cntg_step, kernel_inttype a_strd_step, const double *b,
	kernel_inttype b_bat_step, kernel_inttype b_cntg_step, kernel_inttype b_strd_step,
	double beta, double *c, kernel_inttype c_bat_step, kernel_inttype c_cntg_step,
	kernel_inttype c_strd_step) {

	const auto kernel = BetaIsZero ? &dgemm_neon_beta_zero : &dgemm_neon_beta_nonzero;
	kernel_inttype offset = 0;
	const auto compute_interleaves = [&](const kernel_inttype interleave_count) {
		kernel(interleave_count, nbatch, m, n, k, alpha, a + offset, a_bat_step, a_cntg_step,
		       a_strd_step, b + offset, b_bat_step, b_cntg_step, b_strd_step, beta, c + offset,
		       c_bat_step, c_cntg_step, c_strd_step);
		offset += interleave_count;
	};

	while (offset + 16 <= ninter) {
		compute_interleaves(16);
	}
	if (offset + 8 <= ninter) {
		compute_interleaves(8);
	}
	if (offset + 4 <= ninter) {
		compute_interleaves(4);
	}
	if (offset + 2 <= ninter) {
		compute_interleaves(2);
	}
	if (offset < ninter) {
		compute_interleaves(1);
	}
}

} // namespace perflibs::linalg::batch_interleave::matmul

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_KERNELS_GEMM_KERNELS_HPP

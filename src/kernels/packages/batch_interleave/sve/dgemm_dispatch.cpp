/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "packages/batch_interleave/matmul/kernels/gemm_kernels.hpp"

#include <arm_sve.h>

namespace perflibs::interleave_batch {
void bi_dgemm_sve(kernel_inttype ninter, kernel_inttype nbatch, kernel_inttype m, kernel_inttype n,
                  kernel_inttype k, double alpha, const double *a, kernel_inttype a_bat_step,
                  kernel_inttype a_cntg_step, kernel_inttype a_strd_step, const double *b,
                  kernel_inttype b_bat_step, kernel_inttype b_cntg_step, kernel_inttype b_strd_step,
                  double beta, double *c, kernel_inttype c_bat_step, kernel_inttype c_cntg_step,
                  kernel_inttype c_strd_step) {

	auto vec_width = svcntd();
	kernel_inttype ni = 0;
	kernel_inttype ninter_remain = ninter;
	kernel_inttype vec8 = vec_width * 8;

	for (; ni < ninter - vec8 + 1; ni += vec8) {
		perflibs::interleave_batch::dgemm_sve_vls8(vec8, nbatch, m, n, k, alpha, a + ni, a_bat_step, a_cntg_step,
		                                        a_strd_step, b + ni, b_bat_step, b_cntg_step, b_strd_step,
		                                        beta, c + ni, c_bat_step, c_cntg_step, c_strd_step);
		ninter_remain -= vec8;
	}

	if (ninter_remain > 0) {
		perflibs::interleave_batch::dgemm_sve_vla(ninter_remain, nbatch, m, n, k, alpha, a + ni, a_bat_step, a_cntg_step,
                                               a_strd_step, b + ni, b_bat_step, b_cntg_step, b_strd_step,
                                               beta, c + ni, c_bat_step, c_cntg_step, c_strd_step);
	}
}
} // namespace perflibs::interleave_batch

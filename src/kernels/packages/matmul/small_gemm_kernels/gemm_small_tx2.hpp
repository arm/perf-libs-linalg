/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef GEMM_SMALL_TX2_HPP
#define GEMM_SMALL_TX2_HPP

#include "perflibs_complex.hpp"
#include "perflibs_util.hpp"
#include "gemm_small_framework.hpp"
#include "perflibs_blas_types.hpp"

namespace perflibs::gemm {

void sgemm_small_tx2(const kernel_inttype max_threads,
                     const perflibs_trans transa, const perflibs_trans transb, kernel_inttype m, kernel_inttype n, kernel_inttype k,
                     float alpha, const float *a, kernel_inttype lda,
                     const float *b, kernel_inttype ldb, float beta,
                     float *c, kernel_inttype ldc);

void sgemm_small_nn_tx2(const perflibs_trans transa, const perflibs_trans transb, kernel_inttype m, kernel_inttype n, kernel_inttype k,
                        float alpha, const float *a, kernel_inttype lda,
                        const float *b, kernel_inttype ldb, float beta,
                        float *c, kernel_inttype ldc);

template<typename FloatType>
kfunc_t<FloatType> *get_kernel_tx2(trans_t trans_opt, l_order_t loop_o, kernel_inttype rblock_m, kernel_inttype rblock_n,
                                   kernel_inttype rblock_k, FloatType alpha, FloatType beta);

}

#endif // ifndef GEMM_SMALL_TX2_HPP

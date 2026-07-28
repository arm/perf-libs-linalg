/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef GEMM_SMALL_GENERIC_AARCH64_HPP
#define GEMM_SMALL_GENERIC_AARCH64_HPP

#include "perflibs_complex.hpp"
#include "gemm_small_framework.hpp"
#include "perflibs_blas_types.hpp"

namespace perflibs::gemm {

void sgemm_small_generic_aarch64(const kernel_inttype max_threads,
                                 const perflibs_trans transa, const perflibs_trans transb, kernel_inttype m, kernel_inttype n, kernel_inttype k,
                                 float alpha, const float *a, kernel_inttype lda,
                                 const float *b, kernel_inttype ldb, float beta,
                                 float *c, kernel_inttype ldc);

void cgemm_small_generic_aarch64(const kernel_inttype max_threads,
                                 const perflibs_trans transa, const perflibs_trans transb, kernel_inttype m, kernel_inttype n, kernel_inttype k,
                                 complex_float alpha, const complex_float *a, kernel_inttype lda,
                                 const complex_float *b, kernel_inttype ldb, complex_float beta,
                                 complex_float *c, kernel_inttype ldc);

template<typename FloatType>
kfunc_t<FloatType> *get_kernel_generic(trans_t trans_opt, l_order_t loop_o, kernel_inttype rblock_m, kernel_inttype rblock_n,
                                       kernel_inttype rblock_k, FloatType alpha, FloatType beta);

}

#endif // ifndef GEMM_SMALL_GENERIC_AARCH64_HPP

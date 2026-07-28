/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_ATOMIC_KERNELS_HPP
#define PERFLIBS_ATOMIC_KERNELS_HPP

#include "framework/linalg_blas_types.hpp"
#include "perflibs_complex.hpp"

namespace perflibs {
template<typename T>
bool dispatch_atomic_neon(linalg::perflibs_trans transa, linalg::perflibs_trans transb,
                          kernel_inttype m, kernel_inttype n, kernel_inttype k,
                          const T *a, kernel_inttype lda, const T *b, kernel_inttype ldb,
                          T *c, kernel_inttype ldc, T alpha, T beta);

template<typename T>
bool dispatch_atomic_sve(linalg::perflibs_trans transa, linalg::perflibs_trans transb,
                         kernel_inttype m, kernel_inttype n, kernel_inttype k,
                         const T *a, kernel_inttype lda, const T *b, kernel_inttype ldb,
                         T *c, kernel_inttype ldc, T alpha, T beta);
}

#endif // PERFLIBS_ATOMIC_KERNELS_HPP

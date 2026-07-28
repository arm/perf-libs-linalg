/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_VECADD_KERNELS_HPP
#define PERFLIBS_LINALG_VECADD_KERNELS_HPP

#include "framework/linalg_util.hpp"
#include "spec/problem_context.hpp"

namespace perflibs::linalg {

template<typename AType, typename BType=AType, typename CType=AType>
using vecadd_kernel_t = void(kernel_inttype n, const AType *x, kernel_inttype incx, CType *y,
                             kernel_inttype incy);

} //namespace perflibs::linalg

extern "C" {
//Neon
perflibs::linalg::vecadd_kernel_t<float>                svecadd_kernel;
perflibs::linalg::vecadd_kernel_t<double>               dvecadd_kernel;
perflibs::linalg::vecadd_kernel_t<std::complex<float>>  cvecadd_kernel;
perflibs::linalg::vecadd_kernel_t<std::complex<double>> zvecadd_kernel;
perflibs::linalg::vecadd_kernel_t<std::complex<float>>  cvecadd_conj_kernel;
perflibs::linalg::vecadd_kernel_t<std::complex<double>> zvecadd_conj_kernel;
//SVE
perflibs::linalg::vecadd_kernel_t<float>                svecadd_sve_kernel;
perflibs::linalg::vecadd_kernel_t<double>               dvecadd_sve_kernel;
perflibs::linalg::vecadd_kernel_t<std::complex<float>>  cvecadd_sve_kernel;
perflibs::linalg::vecadd_kernel_t<std::complex<double>> zvecadd_sve_kernel;
perflibs::linalg::vecadd_kernel_t<std::complex<float>>  cvecadd_sve_kernel_fcmla;
perflibs::linalg::vecadd_kernel_t<std::complex<double>> zvecadd_sve_kernel_fcmla;
perflibs::linalg::vecadd_kernel_t<std::complex<float>>  cvecadd_conj_sve_kernel;
perflibs::linalg::vecadd_kernel_t<std::complex<double>> zvecadd_conj_sve_kernel;
perflibs::linalg::vecadd_kernel_t<std::complex<float>>  cvecadd_sve_conj_kernel_fcmla;
perflibs::linalg::vecadd_kernel_t<std::complex<double>> zvecadd_sve_conj_kernel_fcmla;

} //extern "C"

#endif // PERFLIBS_LINALG_VECADD_KERNELS_HPP

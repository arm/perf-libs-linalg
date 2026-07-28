/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_GE_KERNELS_HPP
#define PERFLIBS_LINALG_GE_KERNELS_HPP

#include "framework/linalg_util.hpp"

#include "perflibs_float.hpp"

namespace perflibs::linalg {

template<typename T>
using gecpy_f = void(kernel_inttype cntg, kernel_inttype strd, const T *a, kernel_inttype lda, T *b, kernel_inttype ldb);

template<typename T>
using gescal_out_of_place_f = void(T alpha, kernel_inttype cntg, kernel_inttype strd, const T *a, kernel_inttype lda, T *b, kernel_inttype ldb);

template<typename T>
using geset_f = void(T alpha, kernel_inttype cntg, kernel_inttype strd, T *a, kernel_inttype lda);

} //namespace perflibs::linalg

extern "C" {
perflibs::linalg::gecpy_f<double> dgecpy;
perflibs::linalg::gecpy_f<float> sgecpy;
perflibs::linalg::gecpy_f<half> hgecpy;

perflibs::linalg::gescal_out_of_place_f<double> dgescal_out_of_place;
perflibs::linalg::gescal_out_of_place_f<float> sgescal_out_of_place;
perflibs::linalg::gescal_out_of_place_f<half> hgescal_out_of_place;

perflibs::linalg::geset_f<double> dgeset;
perflibs::linalg::geset_f<float> sgeset;
perflibs::linalg::geset_f<half> hgeset;
} //extern "C"

namespace perflibs::linalg {

template<typename T>
inline gecpy_f<T> *gecpy_kernel;
template<>
inline auto gecpy_kernel<double> = dgecpy;
template<>
inline auto gecpy_kernel<float> = sgecpy;
template<>
inline auto gecpy_kernel<half> = hgecpy;

template<typename T>
inline gescal_out_of_place_f<T> *gescal_out_of_place_kernel;
template<>
inline auto gescal_out_of_place_kernel<double> = dgescal_out_of_place;
template<>
inline auto gescal_out_of_place_kernel<float> = sgescal_out_of_place;
template<>
inline auto gescal_out_of_place_kernel<half> = hgescal_out_of_place;

template<typename T>
inline geset_f<T> *geset_kernel;
template<>
inline auto geset_kernel<double> = dgeset;
template<>
inline auto geset_kernel<float> = sgeset;
template<>
inline auto geset_kernel<half> = hgeset;

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_GE_KERNELS_HPP

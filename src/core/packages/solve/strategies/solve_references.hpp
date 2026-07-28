/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SOLVE_STRATEGIES_SOLVE_REFERENCES_HPP
#define PERFLIBS_LINALG_SOLVE_STRATEGIES_SOLVE_REFERENCES_HPP

#include "framework/linalg_blas_types.hpp"
#include "perflibs_complex.hpp"
#include "perflibs_ftn_symb.hpp"

namespace perflibs::linalg::solve::reference {

template<typename T>
using trsm_f = void(const char *side, const char *uplo, const char *transa,
    const char *diag, const int_type *m, const int_type *n, const T *alpha,
    const T *a, const int_type *lda, T *b, const int_type *ldb);

template<typename T>
using trsv_f = void(const char *uplo, const char *transa,
    const char *diag, const int_type *n, const T *a, const int_type *lda,
    T *b, const int_type *incb);

template<typename T>
constexpr trsm_f<T>* trsm { nullptr };

template<typename T>
constexpr trsv_f<T>* trsv { nullptr };

extern "C" {
    trsm_f<r32> FTN_SYMB(strsm_reference);
    trsm_f<r64> FTN_SYMB(dtrsm_reference);
    trsm_f<c32> FTN_SYMB(ctrsm_reference);
    trsm_f<c64> FTN_SYMB(ztrsm_reference);

    trsv_f<r32> FTN_SYMB(strsv_reference);
    trsv_f<r64> FTN_SYMB(dtrsv_reference);
    trsv_f<c32> FTN_SYMB(ctrsv_reference);
    trsv_f<c64> FTN_SYMB(ztrsv_reference);
}

template<> inline constexpr auto trsm<r32> = FTN_SYMB(strsm_reference);
template<> inline constexpr auto trsm<r64> = FTN_SYMB(dtrsm_reference);
template<> inline constexpr auto trsm<c32> = FTN_SYMB(ctrsm_reference);
template<> inline constexpr auto trsm<c64> = FTN_SYMB(ztrsm_reference);

template<> inline constexpr auto trsv<r32> = FTN_SYMB(strsv_reference);
template<> inline constexpr auto trsv<r64> = FTN_SYMB(dtrsv_reference);
template<> inline constexpr auto trsv<c32> = FTN_SYMB(ctrsv_reference);
template<> inline constexpr auto trsv<c64> = FTN_SYMB(ztrsv_reference);

} // namespace perflibs::linalg::solve::reference

#endif // PERFLIBS_LINALG_SOLVE_STRATEGIES_SOLVE_REFERENCES_HPP

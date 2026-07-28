/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_MATMUL_REFERENCES_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_MATMUL_REFERENCES_HPP

#include "framework/linalg_blas_types.hpp"
#include "perflibs_complex.hpp"
#include "perflibs_float.hpp"
#include "perflibs_ftn_symb.hpp"

namespace perflibs::linalg::matmul::reference {
// Local BLAS function type aliases
// gemm
template<typename AType, typename BType, typename CType, typename ScalarType>
using gemm_f = void(const char *transa, const char *transb, const int_type *m,
    const int_type *n, const int_type *k, const ScalarType *alpha, const AType *a, const int_type *lda,
    const BType *b, const int_type *ldb, const ScalarType *beta, CType *c, const int_type *ldc);

// hemm
template<typename T>
using hemm_f = void(const char *side, const char *uplo, const int_type *m,
    const int_type *n, const T *alpha, const T *a, const int_type *lda,
    const T *b, const int_type *ldb, const T *beta, T *c, const int_type *ldc);

// symm
template<typename T>
using symm_f = void(const char *side, const char *uplo,
    const int_type *m, const int_type *n, const T *alpha,
    const T *a, const int_type *lda, const T *b, const int_type *ldb,
    const T *beta, T *c, const int_type *ldc);

// syrk
template<typename T>
using syrk_f = void(const char *uplo, const char *trans, const int_type *n,
    const int_type *k, const T *alpha, const T *a, const int_type *lda,
    const T *beta, T *c, const int_type *ldc);

// herk
template<typename T>
using herk_f = void(const char *uplo, const char *trans, const int_type *n,
    const int_type *k, const perflibs::remove_complex_t<T> *alpha, const T *a, const int_type *lda,
    const perflibs::remove_complex_t<T> *beta, T *c, const int_type *ldc);

// syr2k
template<typename T>
using syr2k_f = void(const char *uplo, const char *trans, const int_type *n,
    const int_type *k, const T *alpha, const T *a, const int_type *lda,
    const T *b, const int_type *ldb, const T *beta, T *c, const int_type *ldc);

// her2k
template<typename T>
using her2k_f = void(const char *uplo, const char *trans, const int_type *n,
    const int_type *k, const T *alpha, const T *a, const int_type *lda,
    const T *b, const int_type *ldb, const perflibs::remove_complex_t<T> *beta, T *c, const int_type *ldc);

// trmm
template<typename T>
using trmm_f = void(const char *side, const char *uplo, const char *transa,
    const char *diag, const int_type *m, const int_type *n, const T *alpha,
    const T *a, const int_type *lda, T *b, const int_type *ldb);

// C declarations for the Fortran BLAS reference symbols
extern "C" {
gemm_f<bf16, bf16, bf16, bf16> FTN_SYMB(bgemm_reference);
gemm_f<bf16, bf16,  r32,  r32> FTN_SYMB(sbgemm_reference);
gemm_f< r16,  r16,  r16,  r16> FTN_SYMB(hgemm_reference);
gemm_f< r32,  r32,  r32,  r32> FTN_SYMB(sgemm_reference);
gemm_f< r64,  r64,  r64,  r64> FTN_SYMB(dgemm_reference);
gemm_f< c32,  c32,  c32,  c32> FTN_SYMB(cgemm_reference);
gemm_f< c64,  c64,  c64,  c64> FTN_SYMB(zgemm_reference);

symm_f<r32> FTN_SYMB(ssymm_reference);
symm_f<r64> FTN_SYMB(dsymm_reference);
symm_f<c32> FTN_SYMB(csymm_reference);
symm_f<c64> FTN_SYMB(zsymm_reference);

hemm_f<c32> FTN_SYMB(chemm_reference);
hemm_f<c64> FTN_SYMB(zhemm_reference);

trmm_f<r32> FTN_SYMB(strmm_reference);
trmm_f<r64> FTN_SYMB(dtrmm_reference);
trmm_f<c32> FTN_SYMB(ctrmm_reference);
trmm_f<c64> FTN_SYMB(ztrmm_reference);

syrk_f<r32> FTN_SYMB(ssyrk_reference);
syrk_f<r64> FTN_SYMB(dsyrk_reference);
syrk_f<c32> FTN_SYMB(csyrk_reference);
syrk_f<c64> FTN_SYMB(zsyrk_reference);

herk_f<c32> FTN_SYMB(cherk_reference);
herk_f<c64> FTN_SYMB(zherk_reference);

syr2k_f<r32> FTN_SYMB(ssyr2k_reference);
syr2k_f<r64> FTN_SYMB(dsyr2k_reference);
syr2k_f<c32> FTN_SYMB(csyr2k_reference);
syr2k_f<c64> FTN_SYMB(zsyr2k_reference);

her2k_f<c32> FTN_SYMB(cher2k_reference);
her2k_f<c64> FTN_SYMB(zher2k_reference);
}

// Reference wrapper declarations using local BLAS type aliases
template<typename AType, typename BType, typename CType, typename ScalarType>
inline constexpr gemm_f<AType, BType, CType, ScalarType>* gemm { nullptr };

template<> inline constexpr auto gemm<bf16, bf16, r32, r32> = FTN_SYMB(sbgemm_reference);
template<> inline constexpr auto gemm<bf16, bf16, bf16, bf16> = FTN_SYMB(bgemm_reference);
template<> inline constexpr auto gemm< r16,  r16, r16, r16> = FTN_SYMB(hgemm_reference);
template<> inline constexpr auto gemm< r32,  r32, r32, r32> = FTN_SYMB(sgemm_reference);
template<> inline constexpr auto gemm< r64,  r64, r64, r64> = FTN_SYMB(dgemm_reference);
template<> inline constexpr auto gemm< c32,  c32, c32, c32> = FTN_SYMB(cgemm_reference);
template<> inline constexpr auto gemm< c64,  c64, c64, c64> = FTN_SYMB(zgemm_reference);

template<typename T>
inline constexpr hemm_f<T>* hemm { nullptr };

template<> inline constexpr auto hemm<c32> = FTN_SYMB(chemm_reference);
template<> inline constexpr auto hemm<c64> = FTN_SYMB(zhemm_reference);

template<typename T>
inline constexpr symm_f<T>* symm { nullptr };

template<> inline constexpr auto symm<r32> = FTN_SYMB(ssymm_reference);
template<> inline constexpr auto symm<r64> = FTN_SYMB(dsymm_reference);
template<> inline constexpr auto symm<c32> = FTN_SYMB(csymm_reference);
template<> inline constexpr auto symm<c64> = FTN_SYMB(zsymm_reference);

template<typename T>
inline constexpr syrk_f<T>* syrk { nullptr };

template<> inline constexpr auto syrk<r32> = FTN_SYMB(ssyrk_reference);
template<> inline constexpr auto syrk<r64> = FTN_SYMB(dsyrk_reference);
template<> inline constexpr auto syrk<c32> = FTN_SYMB(csyrk_reference);
template<> inline constexpr auto syrk<c64> = FTN_SYMB(zsyrk_reference);

template<typename T>
inline constexpr herk_f<T>* herk { nullptr };

template<> inline constexpr auto herk<c32> = FTN_SYMB(cherk_reference);
template<> inline constexpr auto herk<c64> = FTN_SYMB(zherk_reference);

template<typename T>
inline constexpr syr2k_f<T>* syr2k { nullptr };

template<> inline constexpr auto syr2k<r32> = FTN_SYMB(ssyr2k_reference);
template<> inline constexpr auto syr2k<r64> = FTN_SYMB(dsyr2k_reference);
template<> inline constexpr auto syr2k<c32> = FTN_SYMB(csyr2k_reference);
template<> inline constexpr auto syr2k<c64> = FTN_SYMB(zsyr2k_reference);

template<typename T>
inline constexpr her2k_f<T>* her2k { nullptr };

template<> inline constexpr auto her2k<c32> = FTN_SYMB(cher2k_reference);
template<> inline constexpr auto her2k<c64> = FTN_SYMB(zher2k_reference);

template<typename T>
inline constexpr trmm_f<T>* trmm { nullptr };

template<> inline constexpr auto trmm<r32> = FTN_SYMB(strmm_reference);
template<> inline constexpr auto trmm<r64> = FTN_SYMB(dtrmm_reference);
template<> inline constexpr auto trmm<c32> = FTN_SYMB(ctrmm_reference);
template<> inline constexpr auto trmm<c64> = FTN_SYMB(ztrmm_reference);
} // namespace perflibs::linalg::matmul::reference

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_MATMUL_REFERENCES_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LAPACK_H
#define PERFLIBS_LAPACK_H

#include "pl_linalg_int.h"
#include "pl_linalg_complex.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* Selected LAPACK Fortran ABI routines. */

void cgebrd_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, pl_linalg_singlecomplex_t *a, const pl_linalg_int_t *lda, float *d, float *e, pl_linalg_singlecomplex_t *tauq, pl_linalg_singlecomplex_t *taup, pl_linalg_singlecomplex_t *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info);
void cgeqrf_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, pl_linalg_singlecomplex_t *a, const pl_linalg_int_t *lda, pl_linalg_singlecomplex_t *tau, pl_linalg_singlecomplex_t *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info);
void cgetrf_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, pl_linalg_singlecomplex_t *a, const pl_linalg_int_t *lda, pl_linalg_int_t *ipiv, pl_linalg_int_t *info);
void chetrd_(const char *uplo, const pl_linalg_int_t *n, pl_linalg_singlecomplex_t *a, const pl_linalg_int_t *lda, float *d, float *e, pl_linalg_singlecomplex_t *tau, pl_linalg_singlecomplex_t *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info, ... );
void clabrd_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *nb, pl_linalg_singlecomplex_t *a, const pl_linalg_int_t *lda, float *d, float *e, pl_linalg_singlecomplex_t *tauq, pl_linalg_singlecomplex_t *taup, pl_linalg_singlecomplex_t *x, const pl_linalg_int_t *ldx, pl_linalg_singlecomplex_t *y, const pl_linalg_int_t *ldy);
void clarf_(const char *side, const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_singlecomplex_t *v, const pl_linalg_int_t *incv, const pl_linalg_singlecomplex_t *tau, pl_linalg_singlecomplex_t *c, const pl_linalg_int_t *ldc, pl_linalg_singlecomplex_t *work, ... );
void clarfb_(const char *side, const char *trans, const char *direct, const char *storev, const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, const pl_linalg_singlecomplex_t *v, const pl_linalg_int_t *ldv, const pl_linalg_singlecomplex_t *t, const pl_linalg_int_t *ldt, pl_linalg_singlecomplex_t *c, const pl_linalg_int_t *ldc, pl_linalg_singlecomplex_t *work, const pl_linalg_int_t *ldwork, ... );
void clarfg_(const pl_linalg_int_t *n, pl_linalg_singlecomplex_t *alpha, pl_linalg_singlecomplex_t *x, const pl_linalg_int_t *incx, pl_linalg_singlecomplex_t *tau);
void clarft_(const char *direct, const char *storev, const pl_linalg_int_t *n, const pl_linalg_int_t *k, const pl_linalg_singlecomplex_t *v, const pl_linalg_int_t *ldv, const pl_linalg_singlecomplex_t *tau, pl_linalg_singlecomplex_t *t, const pl_linalg_int_t *ldt, ... );
void clatrd_(const char *uplo, const pl_linalg_int_t *n, const pl_linalg_int_t *nb, pl_linalg_singlecomplex_t *a, const pl_linalg_int_t *lda, float *e, pl_linalg_singlecomplex_t *tau, pl_linalg_singlecomplex_t *w, const pl_linalg_int_t *ldw, ... );
void cpotrf_(const char *uplo, const pl_linalg_int_t *n, pl_linalg_singlecomplex_t *a, const pl_linalg_int_t *lda, pl_linalg_int_t *info, ... );
void cunglq_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, pl_linalg_singlecomplex_t *a, const pl_linalg_int_t *lda, const pl_linalg_singlecomplex_t *tau, pl_linalg_singlecomplex_t *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info);
void cungqr_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, pl_linalg_singlecomplex_t *a, const pl_linalg_int_t *lda, const pl_linalg_singlecomplex_t *tau, pl_linalg_singlecomplex_t *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info);
void cunmlq_(const char *side, const char *trans, const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, const pl_linalg_singlecomplex_t *a, const pl_linalg_int_t *lda, const pl_linalg_singlecomplex_t *tau, pl_linalg_singlecomplex_t *c, const pl_linalg_int_t *ldc, pl_linalg_singlecomplex_t *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info, ... );
void cunmqr_(const char *side, const char *trans, const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, const pl_linalg_singlecomplex_t *a, const pl_linalg_int_t *lda, const pl_linalg_singlecomplex_t *tau, pl_linalg_singlecomplex_t *c, const pl_linalg_int_t *ldc, pl_linalg_singlecomplex_t *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info, ... );

void dgebrd_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, double *a, const pl_linalg_int_t *lda, double *d, double *e, double *tauq, double *taup, double *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info);
void dgeqrf_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, double *a, const pl_linalg_int_t *lda, double *tau, double *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info);
void dgetrf_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, double *a, const pl_linalg_int_t *lda, pl_linalg_int_t *ipiv, pl_linalg_int_t *info);
void dlabrd_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *nb, double *a, const pl_linalg_int_t *lda, double *d, double *e, double *tauq, double *taup, double *x, const pl_linalg_int_t *ldx, double *y, const pl_linalg_int_t *ldy);
void dlarf_(const char *side, const pl_linalg_int_t *m, const pl_linalg_int_t *n, const double *v, const pl_linalg_int_t *incv, const double *tau, double *c, const pl_linalg_int_t *ldc, double *work, ... );
void dlarfb_(const char *side, const char *trans, const char *direct, const char *storev, const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, const double *v, const pl_linalg_int_t *ldv, const double *t, const pl_linalg_int_t *ldt, double *c, const pl_linalg_int_t *ldc, double *work, const pl_linalg_int_t *ldwork, ... );
void dlarfg_(const pl_linalg_int_t *n, double *alpha, double *x, const pl_linalg_int_t *incx, double *tau);
void dlarft_(const char *direct, const char *storev, const pl_linalg_int_t *n, const pl_linalg_int_t *k, const double *v, const pl_linalg_int_t *ldv, const double *tau, double *t, const pl_linalg_int_t *ldt, ... );
void dlatrd_(const char *uplo, const pl_linalg_int_t *n, const pl_linalg_int_t *nb, double *a, const pl_linalg_int_t *lda, double *e, double *tau, double *w, const pl_linalg_int_t *ldw, ... );
void dorglq_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, double *a, const pl_linalg_int_t *lda, const double *tau, double *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info);
void dorgqr_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, double *a, const pl_linalg_int_t *lda, const double *tau, double *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info);
void dormlq_(const char *side, const char *trans, const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, const double *a, const pl_linalg_int_t *lda, const double *tau, double *c, const pl_linalg_int_t *ldc, double *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info, ... );
void dormqr_(const char *side, const char *trans, const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, const double *a, const pl_linalg_int_t *lda, const double *tau, double *c, const pl_linalg_int_t *ldc, double *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info, ... );
void dpotrf_(const char *uplo, const pl_linalg_int_t *n, double *a, const pl_linalg_int_t *lda, pl_linalg_int_t *info, ... );
void dsytrd_(const char *uplo, const pl_linalg_int_t *n, double *a, const pl_linalg_int_t *lda, double *d, double *e, double *tau, double *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info, ... );

void sgebrd_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, float *a, const pl_linalg_int_t *lda, float *d, float *e, float *tauq, float *taup, float *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info);
void sgeqrf_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, float *a, const pl_linalg_int_t *lda, float *tau, float *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info);
void sgetrf_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, float *a, const pl_linalg_int_t *lda, pl_linalg_int_t *ipiv, pl_linalg_int_t *info);
void slabrd_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *nb, float *a, const pl_linalg_int_t *lda, float *d, float *e, float *tauq, float *taup, float *x, const pl_linalg_int_t *ldx, float *y, const pl_linalg_int_t *ldy);
void slarf_(const char *side, const pl_linalg_int_t *m, const pl_linalg_int_t *n, const float *v, const pl_linalg_int_t *incv, const float *tau, float *c, const pl_linalg_int_t *ldc, float *work, ... );
void slarfb_(const char *side, const char *trans, const char *direct, const char *storev, const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, const float *v, const pl_linalg_int_t *ldv, const float *t, const pl_linalg_int_t *ldt, float *c, const pl_linalg_int_t *ldc, float *work, const pl_linalg_int_t *ldwork, ... );
void slarfg_(const pl_linalg_int_t *n, float *alpha, float *x, const pl_linalg_int_t *incx, float *tau);
void slarft_(const char *direct, const char *storev, const pl_linalg_int_t *n, const pl_linalg_int_t *k, const float *v, const pl_linalg_int_t *ldv, const float *tau, float *t, const pl_linalg_int_t *ldt, ... );
void slatrd_(const char *uplo, const pl_linalg_int_t *n, const pl_linalg_int_t *nb, float *a, const pl_linalg_int_t *lda, float *e, float *tau, float *w, const pl_linalg_int_t *ldw, ... );
void sorglq_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, float *a, const pl_linalg_int_t *lda, const float *tau, float *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info);
void sorgqr_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, float *a, const pl_linalg_int_t *lda, const float *tau, float *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info);
void sormlq_(const char *side, const char *trans, const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, const float *a, const pl_linalg_int_t *lda, const float *tau, float *c, const pl_linalg_int_t *ldc, float *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info, ... );
void sormqr_(const char *side, const char *trans, const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, const float *a, const pl_linalg_int_t *lda, const float *tau, float *c, const pl_linalg_int_t *ldc, float *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info, ... );
void spotrf_(const char *uplo, const pl_linalg_int_t *n, float *a, const pl_linalg_int_t *lda, pl_linalg_int_t *info, ... );
void ssytrd_(const char *uplo, const pl_linalg_int_t *n, float *a, const pl_linalg_int_t *lda, float *d, float *e, float *tau, float *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info, ... );

void zgebrd_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, pl_linalg_doublecomplex_t *a, const pl_linalg_int_t *lda, double *d, double *e, pl_linalg_doublecomplex_t *tauq, pl_linalg_doublecomplex_t *taup, pl_linalg_doublecomplex_t *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info);
void zgeqrf_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, pl_linalg_doublecomplex_t *a, const pl_linalg_int_t *lda, pl_linalg_doublecomplex_t *tau, pl_linalg_doublecomplex_t *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info);
void zgetrf_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, pl_linalg_doublecomplex_t *a, const pl_linalg_int_t *lda, pl_linalg_int_t *ipiv, pl_linalg_int_t *info);
void zhetrd_(const char *uplo, const pl_linalg_int_t *n, pl_linalg_doublecomplex_t *a, const pl_linalg_int_t *lda, double *d, double *e, pl_linalg_doublecomplex_t *tau, pl_linalg_doublecomplex_t *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info, ... );
void zlabrd_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *nb, pl_linalg_doublecomplex_t *a, const pl_linalg_int_t *lda, double *d, double *e, pl_linalg_doublecomplex_t *tauq, pl_linalg_doublecomplex_t *taup, pl_linalg_doublecomplex_t *x, const pl_linalg_int_t *ldx, pl_linalg_doublecomplex_t *y, const pl_linalg_int_t *ldy);
void zlarf_(const char *side, const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_doublecomplex_t *v, const pl_linalg_int_t *incv, const pl_linalg_doublecomplex_t *tau, pl_linalg_doublecomplex_t *c, const pl_linalg_int_t *ldc, pl_linalg_doublecomplex_t *work, ... );
void zlarfb_(const char *side, const char *trans, const char *direct, const char *storev, const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, const pl_linalg_doublecomplex_t *v, const pl_linalg_int_t *ldv, const pl_linalg_doublecomplex_t *t, const pl_linalg_int_t *ldt, pl_linalg_doublecomplex_t *c, const pl_linalg_int_t *ldc, pl_linalg_doublecomplex_t *work, const pl_linalg_int_t *ldwork, ... );
void zlarfg_(const pl_linalg_int_t *n, pl_linalg_doublecomplex_t *alpha, pl_linalg_doublecomplex_t *x, const pl_linalg_int_t *incx, pl_linalg_doublecomplex_t *tau);
void zlarft_(const char *direct, const char *storev, const pl_linalg_int_t *n, const pl_linalg_int_t *k, const pl_linalg_doublecomplex_t *v, const pl_linalg_int_t *ldv, const pl_linalg_doublecomplex_t *tau, pl_linalg_doublecomplex_t *t, const pl_linalg_int_t *ldt, ... );
void zlatrd_(const char *uplo, const pl_linalg_int_t *n, const pl_linalg_int_t *nb, pl_linalg_doublecomplex_t *a, const pl_linalg_int_t *lda, double *e, pl_linalg_doublecomplex_t *tau, pl_linalg_doublecomplex_t *w, const pl_linalg_int_t *ldw, ... );
void zpotrf_(const char *uplo, const pl_linalg_int_t *n, pl_linalg_doublecomplex_t *a, const pl_linalg_int_t *lda, pl_linalg_int_t *info, ... );
void zunglq_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, pl_linalg_doublecomplex_t *a, const pl_linalg_int_t *lda, const pl_linalg_doublecomplex_t *tau, pl_linalg_doublecomplex_t *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info);
void zungqr_(const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, pl_linalg_doublecomplex_t *a, const pl_linalg_int_t *lda, const pl_linalg_doublecomplex_t *tau, pl_linalg_doublecomplex_t *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info);
void zunmlq_(const char *side, const char *trans, const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, const pl_linalg_doublecomplex_t *a, const pl_linalg_int_t *lda, const pl_linalg_doublecomplex_t *tau, pl_linalg_doublecomplex_t *c, const pl_linalg_int_t *ldc, pl_linalg_doublecomplex_t *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info, ... );
void zunmqr_(const char *side, const char *trans, const pl_linalg_int_t *m, const pl_linalg_int_t *n, const pl_linalg_int_t *k, const pl_linalg_doublecomplex_t *a, const pl_linalg_int_t *lda, const pl_linalg_doublecomplex_t *tau, pl_linalg_doublecomplex_t *c, const pl_linalg_int_t *ldc, pl_linalg_doublecomplex_t *work, const pl_linalg_int_t *lwork, pl_linalg_int_t *info, ... );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* PERFLIBS_LAPACK_H */

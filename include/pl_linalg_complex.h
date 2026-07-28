/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

/*
  A complex datatype for use by the C interfaces to PERFLIBS routines.
  The exact definition can be overridden by manually #define-ing
  pl_linalg_singlecomplex_t and pl_linalg_doublecomplex_t.
*/

/*
  Override complex definitions for C++ code because the C
  complex.h definition of `I` causes problems, e.g. when `I` is
  used as a template parameter. This is only enabled where the
  std::complex ABI is compatible with the C complex ABI.
*/
#if defined(__cplusplus) && !defined(_WIN32)
#ifndef pl_linalg_singlecomplex_t
#include <complex>
#define pl_linalg_singlecomplex_t std::complex<float>
#define PL_LINALG_AVOID_SINGLECOMPLEX_RETURN
#endif
#ifndef pl_linalg_doublecomplex_t
#include <complex>
#define pl_linalg_doublecomplex_t std::complex<double>
#define PL_LINALG_AVOID_DOUBLECOMPLEX_RETURN
#endif
#endif

#ifndef pl_linalg_singlecomplex_t
#include <complex.h>
#if defined(_WIN32)
typedef _Fcomplex pl_linalg_singlecomplex_t;
#else
typedef float _Complex pl_linalg_singlecomplex_t;
#endif
#define pl_linalg_singlecomplex_t pl_linalg_singlecomplex_t
#endif

#ifndef pl_linalg_doublecomplex_t
#include <complex.h>
#if defined(_WIN32)
typedef _Dcomplex pl_linalg_doublecomplex_t;
#else
typedef double _Complex pl_linalg_doublecomplex_t;
#endif
#define pl_linalg_doublecomplex_t pl_linalg_doublecomplex_t
#endif

/*
  The LAPACKE interface uses the macros lapack_complex_float and lapack_complex_double,
  which are compatible with the corresponding perflibs types.
*/
#ifndef lapack_complex_float
#define lapack_complex_float pl_linalg_singlecomplex_t
#endif
#ifndef lapack_complex_double
#define lapack_complex_double pl_linalg_doublecomplex_t
#endif

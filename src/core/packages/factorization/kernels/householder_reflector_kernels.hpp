/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_REFLECTOR_KERNELS_HPP
#define PERFLIBS_LINALG_REFLECTOR_KERNELS_HPP

#include "framework/linalg_util.hpp"
#include "perflibs_ftn_symb.hpp"
// Forward declarations for LARFT
extern "C" {
void FTN_SYMB(slarft)(const char *direct, const char *storev, const pl_linalg_int_t *n, const pl_linalg_int_t *k,
                      float *v, const pl_linalg_int_t *ldv, float *tau, float *t, const pl_linalg_int_t *ldt
                      C_VARARGS_DECL);
void FTN_SYMB(dlarft)(const char *direct, const char *storev, const pl_linalg_int_t *n, const pl_linalg_int_t *k,
                      double *v, const pl_linalg_int_t *ldv, double *tau, double *t, const pl_linalg_int_t *ldt
                      C_VARARGS_DECL);
void FTN_SYMB(clarft)(const char *direct, const char *storev, const pl_linalg_int_t *n, const pl_linalg_int_t *k,
                      complex_float *v, const pl_linalg_int_t *ldv, complex_float *tau,
                      complex_float *t, const pl_linalg_int_t *ldt C_VARARGS_DECL);
void FTN_SYMB(zlarft)(const char *direct, const char *storev, const pl_linalg_int_t *n, const pl_linalg_int_t *k,
                      complex_double *v, const pl_linalg_int_t *ldv, complex_double *tau,
                      complex_double *t, const pl_linalg_int_t *ldt C_VARARGS_DECL);
}


namespace perflibs::linalg::factorization {

template<typename ArchitectureSpec, typename T>
PERFLIBS_LINALG_INLINE
void larft(const char direct, const char storev, const pl_linalg_int_t n, const pl_linalg_int_t k,
           T *v, const pl_linalg_int_t ldv, T *tau, T *t, const pl_linalg_int_t ldt) {

	if constexpr (std::is_same_v<T, float>) {
		FTN_SYMB(slarft)(&direct, &storev, &n, &k, v, &ldv, tau, t, &ldt);
	}
	else if constexpr (std::is_same_v<T, double>) {
		FTN_SYMB(dlarft)(&direct, &storev, &n, &k, v, &ldv, tau, t, &ldt);
	}
	else if constexpr (std::is_same_v<T, std::complex<float>>) {
		FTN_SYMB(clarft)(&direct, &storev, &n, &k, v, &ldv, tau, t, &ldt);
	}
	else if constexpr (std::is_same_v<T, std::complex<double>>) {
		FTN_SYMB(zlarft)(&direct, &storev, &n, &k, v, &ldv, tau, t, &ldt);
	}
}

} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_REFLECTOR_KERNELS_HPP

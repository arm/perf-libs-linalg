/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef BLAS_NFLOP_HPP
#define BLAS_NFLOP_HPP

#include "perflibs_complex.hpp"

#include <cstdint>

namespace perflibs {

/**
 * Get the number of floating-point operations that must be done for each
 * matrix element of the output of BLAS operations.
 *
 * @tparam T        The working precision of the operation (complex arithmetic
 *                  requires more FLOPs).
 * @tparam Talpha   The datatype of the alpha parameter. This can be different
 *                  than T such as in CHPR.
 * @tparam Tbeta    The datatype of the beta parameter. This can be different
 *                  than T such as in CHERK.
 * @param[in] k     The accumulating dimension of the problem.
 * @param[in] alpha The alpha parameter. If alpha is zero/one, skip/add
 *                  FLOP counts accordingly.
 * @param[in] beta  The beta parameter. If beta is zero/one, skip/add
 *                  FLOP counts accordingly.
 */
template<typename T, typename Talpha, typename Tbeta>
inline std::int64_t get_nflops_limit(const std::int64_t k,
                                     const Talpha alpha, const Tbeta beta) {
	const int op_tol_mod_add = is_complex_v<T> ? 2 : 1;
	const int op_tol_mod_mul = is_complex_v<T> ? 6 : 1;
	const int op_tol_mod_mla = is_complex_v<T> ? 8 : 2;

	const bool is_alpha_zero = alpha == Talpha(0);
	const bool is_alpha_one  = alpha == Talpha(1);
	const bool is_beta_zero  = beta  == Tbeta(0);
	const bool is_beta_one   = beta  == Tbeta(1);

	std::int64_t nflops = 0;
	if (!is_alpha_zero && k>0) {
		// k-1 multiply-accumulate
		nflops += op_tol_mod_mla * (k-1);
		// the first one should be a multiply rather than a multiply-accumulate
		nflops += op_tol_mod_mul;
		if (!is_alpha_one) {
			// If T is real, then this always evaluate to 1 which is correct.
			// If T and Talpha is complex, then we add 6 FLOPs for multiplication of
			// complex numbers.
			// If T is complex and Talpha is real, then we add 2 FLOPs for multiplication
			// of complex with real numbers.
			nflops += is_complex_v<Talpha> ? op_tol_mod_mul : op_tol_mod_add;
		}
	}

	if (!is_beta_zero) {
		if (!is_beta_one) {
			// same as the case with !is_alpha_one above
			nflops += is_complex_v<Tbeta> ? op_tol_mod_mul : op_tol_mod_add;
		}
		if (!is_alpha_zero) {
			nflops += op_tol_mod_add;
		}
	}

	return nflops;
}

/**
 * Get the number of floating-point operations that must be done to complete a
 * particular level-3 BLAS operation. Note that you can emulate level-1 and
 * level-2 BLAS functions by setting certain dimensions of the problem to 1 (or
 * 0 in the case of k, if there is no accumulation step).
 *
 * @tparam T The working precision of the operation (complex arithmetic
 *           requires more FLOPs).
 * @param[in] m The first non-accumulating dimension of the problem.
 * @param[in] n The second non-accumulating dimension of the problem.
 * @param[in] k The accumulating dimension of the problem.
 * @param[in] have_alpha Whether we should take a multiplication of the
 *                       accumulation by an alpha parameter into account.
 * @param[in] have_beta  Whether we should take a pre-multiplication of the
 *                       output array by a beta parameter into account.
 */
template<typename T>
inline std::int64_t get_nflops(std::int64_t m, std::int64_t n, std::int64_t k, bool have_alpha, bool have_beta) {
	int op_tol_mod_mul = is_complex_v<T> ? 6 : 1;
	int op_tol_mod_mac = is_complex_v<T> ? 8 : 2;

	std::int64_t nflops = 0;
	if (k > 0) {
		// k multiply-accumulate ops per element, m*n times.
		nflops += m * n * (k-1) * op_tol_mod_mac;
		// the first one should be a multiply rather than a multiply-accumulate
		nflops += m * n * op_tol_mod_mul;
	}

	if (!have_alpha && !have_beta) {
		return nflops;
	}

	// can do one of the two operations as a multiply-accumulate
	// the other one (if present) needs to just be a lone multiply.
	nflops += m * n * op_tol_mod_mac;
	if (have_alpha && have_beta) {
		nflops += m * n * op_tol_mod_mul;
	}
	return nflops;
}

} // namespace perflibs

#endif // BLAS_NFLOP_HPP

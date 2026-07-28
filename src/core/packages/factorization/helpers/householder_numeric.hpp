/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_HOUSEHOLDER_NUMERIC_HPP
#define PERFLIBS_LINALG_FACTORIZATION_HOUSEHOLDER_NUMERIC_HPP

#include <cmath>
#include <limits>

#include "framework/linalg_util.hpp"

namespace perflibs::linalg::factorization {

template<typename T>
PERFLIBS_LINALG_INLINE
remove_complex_t<T> safe_scaling_threshold() {
	using real_type = remove_complex_t<T>;

	return std::numeric_limits<real_type>::min() / std::numeric_limits<real_type>::epsilon();
}

/**
 * Computes the Householder beta value using a safe-scaling norm.
 * @param alpha [in] If T is real, computes hypot(alpha, xnorm).
 *                   If T is complex, computes hypot(real(alpha), imag(alpha), xnorm).
 * @param xnorm [in] The norm of the vector below alpha.
 * @return The computed magnitude with the opposite sign to real(alpha).
 */
template<typename T>
PERFLIBS_LINALG_INLINE
remove_complex_t<T> compute_householder_beta(const T alpha, const remove_complex_t<T> xnorm) {
	using real_type = remove_complex_t<T>;

	const real_type beta = [&] {
		if constexpr (is_complex_v<T>) {
			return std::hypot(perflibs::real(alpha), perflibs::imag(alpha), xnorm);
		}
		else {
			return std::hypot(alpha, xnorm);
		}
	}();

	return -std::copysign(beta, perflibs::real(alpha));
}

} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_HOUSEHOLDER_NUMERIC_HPP

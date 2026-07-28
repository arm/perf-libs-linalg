/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SWAP_AB_2K_HPP
#define PERFLIBS_LINALG_SWAP_AB_2K_HPP

#include "perflibs_complex.hpp"
#include "framework/compute_position.hpp"
#include "matrix/adaptors.hpp"

namespace perflibs::linalg {

/**
 * Operation which performs the (conjugate) transpose operation, for use in
 * the second part of SYR2K (HER2K).
 *
 *   c <- alpha * a * b^H + conj(alpha) * b * a^H + beta * c
 *
 * Assuming the first part:
 *
 *   c <- alpha * a * b^H + beta * c
 *
 * has already been performed, this operation prepares a, b, and alpha for the
 * second part:
 *
 *   c <- conj(alpha) * b * a^H + c
 *
 * That is, it performs the (conjugate) transpose operation:
 *
 *   (alpha * a * b^H)^H = conj(alpha) * b * a^H
 *
 * Note that for the second part: beta is 1.0. This is taken care of in
 * `kernel_exec` by advancing the iteration field of the compute position here
 *
 * For SYR2K, replace "^H" with "^T" and "conj" with "id" above.
 */
template<typename Next>
class transpose_ab_2k {

	Next next_;

public:

	transpose_ab_2k(Next next)
	:	next_ { std::move(next) }
	{	}

	template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename ScalarType, typename... Args>
	PERFLIBS_LINALG_INLINE
	void operator() (MatrixTypeA& a, MatrixTypeB& b, MatrixTypeC& c, compute_position pos, ScalarType alpha, Args&&... args) {
		PERFLIBS_ASSERT(a.cntg() == b.cntg(), "A and B must have an equal ctng dimension");
		PERFLIBS_ASSERT(c.strd() == b.strd(), "A, B, and C must share a strd dimension");
		auto a_parent = a.get_parent();
		auto b_parent = b.get_parent();
		auto a_new = general_matrix { b_parent.sub_matrix(a.absolute_cntg(), a.cntg(), a.absolute_strd(), a.strd()).get_matrix_base(), a.is_conj() };
		auto b_new = general_matrix { a_parent.sub_matrix(b.absolute_cntg(), b.cntg(), b.absolute_strd(), b.strd()).get_matrix_base(), b.is_conj() };

		if constexpr (is_hermitian_matrix_v<std::remove_cv_t<MatrixTypeC>>) {
			next_(a_new, b_new, c, advance(pos, 0, 0, 0, 1), perflibs::conj(alpha), std::forward<Args>(args)...);
		}
		else {
			next_(a_new, b_new, c, advance(pos, 0, 0, 0, 1),             alpha , std::forward<Args>(args)...);
		}
	}
};

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_SWAP_AB_2K_HPP

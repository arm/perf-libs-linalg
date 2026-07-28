/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_HERMITIAN_RANK_TWO_UPDATE_HPP
#define PERFLIBS_LINALG_HERMITIAN_RANK_TWO_UPDATE_HPP

#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"
#include "framework/rank_update_util.hpp"
#include "matrix/matrix.hpp"

namespace perflibs::linalg {
namespace {
/**
 * Stack operator for performing hermitian rank two updates.
 *
 * Hermitian rank two updates are of the form:
 *
 *     A := alpha*x*y**H + conj(alpha)*y*x**H + A
 *
 * Since alpha*x*y**H and conj(alpha)*y*x**H are at most rank one matrices, so alpha*x*y**H +
 * conj(alpha)*y*x**T is at most a rank two matrix. We are 'updating' a hermitian matrix with this (at most)
 * rank two matrix, so this is a 'rank two update'.
 */
template<typename KernelAxpy>
class hermitian_rank_two_update {

	KernelAxpy kernel_axpby_;

public:
	hermitian_rank_two_update(KernelAxpy kernel_axpby) : kernel_axpby_{ kernel_axpby } {
	}

	/**
	 * Perform a hermitian rank two matrix update.
	 *
	 * NOTE: The crop operator (operators/crop.hpp) should be used prior when used
	 * with triangular matrices.
	 *
	 * @tparam A The type of the matrix `a`.
	 * @tparam B The type of the matrix `b`.
	 * @tparam C The type of the matrix `c`.
	 * @param [in] a The first vector.
	 * @param [in] b The second vector.
	 * @param [in, out] c The matrix or submatrix to use.
	 */
	template<typename AType, typename BType, typename CType, typename ScalarType, typename... Args>
	void operator()(AType &a, BType &b, CType &c, const compute_position &pos, ScalarType alpha, Args &&... args) {
		PERFLIBS_ASSERT(a.cntg() == 1, "a must be a row vector.");
		PERFLIBS_ASSERT(b.cntg() == 1, "b must be a row vector.");

		// Get the parent matrices of `a` and `b`.
		auto a2 = a.get_parent();
		auto b2 = b.get_parent();

		// Cut the vectors to the right sizes...
		auto a2_block = a2.sub_matrix(b.absolute_cntg(), b.cntg(), b.absolute_strd(), b.strd());
		auto b2_block = b2.sub_matrix(a.absolute_cntg(), a.cntg(), a.absolute_strd(), a.strd());

		// We grab a general matrix here so that we can get references to the elements.
		auto c_gen = to_general_matrix(c);
		auto [switch_point, cntg_last, cntg_first_step, cntg_last_step] =
		    get_triangular_rank_update_bounding_params(c);

		auto run_kernels = [&](kernel_inttype i, kernel_inttype cntg_first, kernel_inttype cntg_last) {
			if (b(0, i) != zero<ScalarType>) {
				kernel_axpby_(cntg_last - cntg_first, alpha * b(0, i), &a(0, cntg_first, write),
				              one<ScalarType>, &c_gen(cntg_first, i, write), a.strd_step(), c.cntg_step());
			}
			if (a2_block(0, i) != zero<ScalarType>) {
				kernel_axpby_(cntg_last - cntg_first, perflibs::conj(alpha * a2_block(0, i)),
				              &b2_block(0, cntg_first, write), one<ScalarType>, &c_gen(cntg_first, i, write),
				              b2_block.strd_step(), c.cntg_step());
			}
		};

		kernel_inttype i = 0, cntg_first = 0;
		for (; i < switch_point; i++) {
			run_kernels(i, cntg_first, cntg_last);
			cntg_last += cntg_last_step;
		}

		for(; i < c.strd(); i++) {
			run_kernels(i, cntg_first, cntg_last);
			cntg_first += cntg_first_step;
		}
	}

}; // class symmetric_rank_two_update

} // namespace anon
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_HERMITIAN_RANK_TWO_UPDATE_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_OUTER_PRODUCT_HPP
#define PERFLIBS_LINALG_OUTER_PRODUCT_HPP

#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"
#include "matrix/matrix.hpp"
#include "framework/rank_update_util.hpp"

namespace perflibs::linalg {
namespace {

/**
 * Computes the outer product using axpby.
 */
template<typename KernelAxpby>
class outer_product_terminal {

	KernelAxpby kernel_axpby_;

public:
	outer_product_terminal(KernelAxpby kernel_axpby)
	:	kernel_axpby_{ std::move(kernel_axpby) }
	{	}

	/**
	 * Here, we compute the outer product for two vectors and add the result to a matrix. The process is
	 * driven by using our axpby kernel. For general matrices, the process is trivial; we perform full
	 * axpbys for each row in the matrix. For triangular matrices, the process is slightly more complex --
	 * although the same code applies.
	 *
	 * For general matrices, for each row `i` in the matrix:
	 * - we pre-compute `alpha * b(0, i)` for use as our axpby scalar.
	 * - we perform axpby across the column.
	 *
	 * The main difference for triangular matrices is the fact that the diagonal may 'cut across' the
	 * rectangular sub-matrix we're working inside -- perhaps like this:
	 *
	 *     [       ]
	 *     [       ]
	 *     [#      ]
	 *     [##     ]
	 *     [###    ]
	 *     [####   ]
	 *
	 * This means that the columns can be of variable width, which we have to take into account. This is
	 * where get_triangular_rank_update_bounding_params() comes in. Read the relevant doxygen comment for
	 * information on what this function returns.
	 *
	 * In short: when there might be a diagonal intercepting our sub-matrix, we must acknowledge the
	 * position this diagonal and how it affects the sizing of our columns. This varies depending on
	 * whether we have an upper/lower matrix.
	 *
	 * NOTE: The crop operator (operators/crop.hpp) should be used prior when used
	 * with triangular matrices.
	 *
	 * @tparam A The type of the matrix `a`.
	 * @tparam B The type of the matrix `b`.
	 * @tparam C The type of the matrix `c`.
	 * @param [in] a The first vector to use in the outer product.
	 * @param [in] b The second vector to use in the outer product.
	 * @param [in, out] c The matrix to perform the outer product operation on.
	 * @param [in] pos The compute position (unused).
	 * @param [in] alpha The scalar constant for the outer product.
	 */
	template<typename AType, typename BType, typename CType, typename ScalarType, typename... Args>
	PERFLIBS_LINALG_INLINE
	void operator()(const AType &a, const BType &b, CType &c, const compute_position &pos,
		ScalarType alpha, ScalarType beta, Args &&... args) {

		// We grab a general matrix here so that we can get references to the elements.
		auto c_gen = to_general_matrix(c);
		auto [switch_point, cntg_last, cntg_first_step, cntg_last_step] =
		    get_triangular_rank_update_bounding_params(c);

		kernel_inttype i = 0, cntg_first = 0;
		for (; i < switch_point; i++) {
			const auto bval = b(0, i);

			if (bval != zero<ScalarType>) {
				kernel_axpby_(cntg_last - cntg_first, alpha * bval, &a(0, cntg_first, write), beta,
				             &c_gen(cntg_first, i, write), a.strd_step(), c.cntg_step());
			}
			else if(beta != one<ScalarType>) {
				auto c_col = c.sub_matrix(cntg_first, c.cntg() - cntg_first, i, 1);
				scale(beta, c_col);
			}

			cntg_last += cntg_last_step;
		}

		for(; i < c.strd(); i++) {
			const auto bval = b(0, i);

			if (bval != zero<ScalarType>) {
				kernel_axpby_(cntg_last - cntg_first, alpha * bval, &a(0, cntg_first, write), beta,
				             &c_gen(cntg_first, i, write), a.strd_step(), c.cntg_step());
			}
			else if(beta != one<ScalarType>) {
				auto c_col = c.sub_matrix(cntg_first, c.cntg() - cntg_first, i, 1);
				scale(beta, c_col);
			}

			cntg_first += cntg_first_step;
		}
	}
}; // outer_product_terminal

} // namespace anon
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_OUTER_PRODUCT_HPP

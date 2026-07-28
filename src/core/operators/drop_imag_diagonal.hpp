/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_DROP_IMAG_DIAGONAL_HPP
#define PERFLIBS_LINALG_DROP_IMAG_DIAGONAL_HPP

#include "perflibs_numeric_utils.hpp"
#include "perflibs_util.hpp"
#include "framework/compute_position.hpp"
#include "matrix/type_traits.hpp"

namespace perflibs::linalg {
namespace {

/**
 * Stack operator that drop the imaginary component across the diagonal of a triangular matrix (or a submatrix
 * thereof). This is required for numerous level 2 BLAS problems.
 */
template<typename Next>
class drop_imag_diagonal {

	/**
	 * The next operator in the LINALG stack.
	 */
	Next next_;

public:
	drop_imag_diagonal(Next next) : next_{ next } {
	}

	/**
	 * This operator drops the imaginary component across the diagonal of a triangular matrix (or a submatrix
	 * therefore). If the matrix that is passed in is NOT a submatrix, the imaginary component across the
	 * diagonal is dropped as you would expect. If the matrix that is passed in is a submatrix, the imaginary
	 * component across the parent matrix's diagonal is dropped (if it is accessible), making this stack
	 * operator useful in multi-threaded contexts.
	 *
	 * @tparam AType The type of the matrix `a`.
	 * @tparam BType The type of the matrix `b`.
	 * @tparam CType The type of the matrix `c`.
	 * @tparam [in] a Unused, but passed to the next operator.
	 * @tparam [in] b Unused, but passed to the next operator.
	 * @tparam [in, out] c The matrix/submatrix to work on.
	 */
	template<typename AType, typename BType, typename CType, typename... Args>
	PERFLIBS_LINALG_INLINE void operator()(AType &a, BType &b, CType &c, compute_position pos, Args &&... args) {
		static_assert(perflibs::linalg::is_triangular_form_v<CType>);

		using T = typename CType::value_type;
		using Treal = typename T::value_type;

		// We grab a general matrix here so that we can get references to the elements.
		auto c_gen = to_general_matrix(c);

		/*
		 * -- ALGORITHM EXPLAINED --
		 *
		 *  All diagonal elements are of the form (n, n) where n is some integer.
		 *  For regular matrices (i.e. not submatrices) the algorithm is trivial. For submatrices, it is a
		 *  little more involved.
		 *
		 *  We perform this operation in the context of the parent matrix. We start at position (i, i), where
		 *  i is the maximum of the absolute cntg and absolute strd positions of the submatrix. This is
		 *  because, for example, a submatrix starting at (3, 1) cannot possibly contain the diagonal elements
		 *  (1, 1) or (2, 2), because they are 'locked out' by the 3.
		 *
		 *  We continue to iterate until we reach a value of i for which (i, i) is outside of the bounds of
		 *  the submatrix. This value of i is the minimum of the matrix bounds in the cntg and strd
		 *  dimensions. This is because a rectangular submatrix can only possibly contain as many diagonal
		 *  elements as its minimum 'side length'. In other words, a rectangular matrix has to grow into both
		 *  dimensions to have the possibility of accommodating an extra diagonal element -- or, the longest
		 *  diagonal you can make through a submatrix is bounded by the minimum of its cntg and strd
		 *  dimensions.
		 */

		const auto lower_bound = perflibs::max(c.absolute_cntg(), c.absolute_strd());
		const auto upper_bound = perflibs::min(c.absolute_cntg() + c.cntg(), c.absolute_strd() + c.strd());
		for (kernel_inttype i = lower_bound; i < upper_bound; i++) {
			//to use HER2 for HER2K when k=1 we must 0 out imag even if it contains NAN
			auto &v = c_gen(i - c.absolute_cntg(), i - c.absolute_strd(), write);
			v = T{ perflibs::real(v), Treal{ 0 } };
		}

		next_(a, b, c, pos, std::forward<Args>(args)...);
	}
}; // class drop_imag_diagonal

}
} // namespace perflibs::linalg

#endif /* ifndef PERFLIBS_LINALG_DROP_IMAG_DIAGONAL_HPP */

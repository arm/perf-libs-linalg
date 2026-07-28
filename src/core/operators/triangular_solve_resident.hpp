/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_TRIANGULAR_SOLVE_RESIDENT_HPP
#define PERFLIBS_LINALG_TRIANGULAR_SOLVE_RESIDENT_HPP

#include "perflibs_util.hpp"

namespace perflibs::linalg {

/** Block over TRS{M,V}, with the given block size.
 *
 * This takes two 'next' parameters: next_rectangle_ and next_triangle_.
 * next_triangle_ is the stack to use when processing a triangular section -
 * next_rectangle_ is the stack to use when processing a rectangular section.
 * Chances are, next_triangle_ will end up calling a TRS{M,V} kernel, whereas
 * next_rectangle_ will end up calling a GEM{M,V} kernel.
 */
template<typename NextRectangle, typename NextTriangle>
class triangular_solve_resident {
	const kernel_inttype cntg_recursive_block_sizes_;
	const bool           is_lside_;
	NextRectangle        next_rectangle_;
	NextTriangle         next_triangle_;

	template<bool IsLeft, typename AMatrixType, typename BMatrixType>
	PERFLIBS_LINALG_INLINE
	void forward(const AMatrixType& a, BMatrixType& b) {
		// solve problem using block forward substitution
		for (kernel_inttype curr_pos = 0; curr_pos < a.cntg(); curr_pos += cntg_recursive_block_sizes_) {
			auto next_pos = curr_pos + cntg_recursive_block_sizes_;
			next<IsLeft>(a, b, curr_pos, next_pos, a.strd(), next_pos, b.cntg());
		}
	}

	template<bool IsLeft, typename AMatrixType, typename BMatrixType>
	PERFLIBS_LINALG_INLINE
	void backward(const AMatrixType& a, BMatrixType& b) {
		// solve problem using block back substitution
		for (kernel_inttype curr_pos = a.cntg() - cntg_recursive_block_sizes_; curr_pos > -cntg_recursive_block_sizes_; curr_pos -= cntg_recursive_block_sizes_) {
			next<IsLeft>(a, b, curr_pos, 0, curr_pos, 0, curr_pos);
		}
	}

	template<bool IsLeft, typename AMatrixType, typename BMatrixType>
	PERFLIBS_LINALG_INLINE
	void next(const AMatrixType& a, BMatrixType& b,
	          const kernel_inttype curr_pos,
	          const kernel_inttype a_rec_strd_pos, const kernel_inttype a_rec_strd_size,
	          const kernel_inttype b_out_cntg_pos, const kernel_inttype b_out_cntg_size) {
		using value_type = typename AMatrixType::value_type;

		// a_tri is a diagonal matrix of side cntg_recursive_block_sizes obtained from the diagonal of a
		// a_rec is the matrix directly above / below a_tri, depending whether a is upper / lower respectively
		// b_in  is the panel of b used to perform a TRS{M,V} with a_tri
		// b_out is the panel of b which is updated with the result of this TRS{M,V}
		auto a_tri = a.sub_matrix_with_clamp(      curr_pos,       cntg_recursive_block_sizes_,       curr_pos,       cntg_recursive_block_sizes_);
		auto a_rec = a.sub_matrix_with_clamp(      curr_pos,       cntg_recursive_block_sizes_, a_rec_strd_pos, a_rec_strd_size);
		auto b_in  = b.sub_matrix_with_clamp(      curr_pos,       cntg_recursive_block_sizes_,              0,        b.strd());
		auto b_out = b.sub_matrix_with_clamp(b_out_cntg_pos, b_out_cntg_size,              0,        b.strd());

		// perform a TRS{M,V} and a GEM{M,V}
		next_triangle_(a_tri, b_in);
		general_matrix a_rec_gen  { a_rec.get_matrix_base(), a_rec.is_conj() };
		if constexpr (IsLeft) {
			next_rectangle_(a_rec_gen, b_in, b_out, compute_position {0, 0, 0}, -one<value_type>, one<value_type>);
		}
		else {
			b_out = b_out.transpose();
			next_rectangle_(b_in, a_rec_gen, b_out, compute_position {0, 0, 0}, -one<value_type>, one<value_type>);
		}
	}

public:
	triangular_solve_resident(kernel_inttype cntg_recursive_block_sizes,
	                          bool           is_lside,
	                          NextRectangle  next_rectangle,
	                          NextTriangle   next_triangle)
	: cntg_recursive_block_sizes_{cntg_recursive_block_sizes}
	, is_lside_{is_lside}
	, next_rectangle_{next_rectangle}
	, next_triangle_{next_triangle}
	{}

	template<typename AMatrixType, typename BMatrixType>
	inline
	void operator()(const AMatrixType& a, BMatrixType& b) {
		// figure out whether to perform forward or backward substitution
		if (a.is_upper()) is_lside_ ? forward <true>(a, b) : forward <false>(a, b);
		else              is_lside_ ? backward<true>(a, b) : backward<false>(a, b);
	}
}; //class triangular_solve_resident

} //namespace perflibs::linalg

#endif // PERFLIBS_LINALG_TRIANGULAR_SOLVE_RESIDENT_HPP

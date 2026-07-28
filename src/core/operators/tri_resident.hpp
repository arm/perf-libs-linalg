/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_TRI_RESIDENTS_C_HPP
#define PERFLIBS_LINALG_TRI_RESIDENTS_C_HPP

#include "perflibs_numeric_utils.hpp"
#include "perflibs_util.hpp"
#include "matrix/matrix.hpp"
#include "framework/compute_position.hpp"

namespace perflibs::linalg {

/**
 * Given a specified dimension from the GEMM contract, return the value of that dimension for
 * that matrices provides
 */
template<which_dimension WhichDimension, typename AMatType, typename BMatType, typename CMatType>
PERFLIBS_LINALG_INLINE
kernel_inttype get_dimension(const AMatType& a, const BMatType& b, const CMatType& c) {
	if constexpr(WhichDimension == which_dimension::a_strd) {
		return c.cntg();
	}
	else if constexpr(WhichDimension == which_dimension::b_strd) {
		return c.strd();
	}
	else {
		return a.cntg();
	}
}

/**
 * Given two dimensions from the GEMM contract:
 *
 * a.strd() == c.cntg() //a_strd
 * b.strd() == c.strd() //b_strd
 * a.cntg() == b.cntg() //cntg
 *
 * and associated positions and sizes for those dimensions carve up the input
 * matrices, a, b, & c accordingly.
 */
template<which_dimension Dim0, which_dimension Dim1, typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC>
PERFLIBS_LINALG_INLINE
std::tuple<MatrixTypeA, MatrixTypeB, MatrixTypeC, compute_position>
sub_matrix(MatrixTypeA& a, MatrixTypeB& b, MatrixTypeC& c, compute_position pos, kernel_inttype dim0_pos, kernel_inttype dim0_size, kernel_inttype dim1_pos, kernel_inttype dim1_size) {
	/*
	 * even though on a mathematical level the shared dimensions should have the same value
	 * we need to store duplicate values for each pair because if one or both
	 * matrices are interleaved and unrolled then they may have subtly different values
	 */
	kernel_inttype a_strd_size;
	kernel_inttype a_strd_pos;
	kernel_inttype c_cntg_size;
	kernel_inttype c_cntg_pos;

	if constexpr(Dim0 == which_dimension::a_strd) {
		a_strd_pos = dim0_pos;
		a_strd_size = dim0_size;
		c_cntg_pos = dim0_pos;
		c_cntg_size = dim0_size;
	}
	else if constexpr(Dim1 == which_dimension::a_strd) {
		a_strd_pos = dim1_pos;
		a_strd_size = dim1_size;
		c_cntg_pos = dim1_pos;
		c_cntg_size = dim1_size;
	}
	else {
		a_strd_pos = 0;
		a_strd_size = a.strd();
		c_cntg_pos = 0;
		c_cntg_size = c.cntg();
	}

	kernel_inttype b_strd_size;
	kernel_inttype b_strd_pos;
	kernel_inttype c_strd_size;
	kernel_inttype c_strd_pos;
	if constexpr(Dim0 == which_dimension::b_strd) {
		b_strd_pos = dim0_pos;
		b_strd_size = dim0_size;
		c_strd_pos = dim0_pos;
		c_strd_size = dim0_size;
	}
	else if constexpr(Dim1 == which_dimension::b_strd) {
		b_strd_pos = dim1_pos;
		b_strd_size = dim1_size;
		c_strd_pos = dim1_pos;
		c_strd_size = dim1_size;
	}
	else {
		b_strd_pos = 0;
		b_strd_size = b.strd();
		c_strd_pos = 0;
		c_strd_size = c.strd();
	}

	kernel_inttype a_cntg_size;
	kernel_inttype a_cntg_pos;
	kernel_inttype b_cntg_size;
	kernel_inttype b_cntg_pos;
	if constexpr(Dim0 == which_dimension::cntg) {
		a_cntg_pos = dim0_pos;
		a_cntg_size = dim0_size;
		b_cntg_pos = dim0_pos;
		b_cntg_size = dim0_size;
	}
	else if constexpr(Dim1 == which_dimension::cntg) {
		a_cntg_pos = dim1_pos;
		a_cntg_size = dim1_size;
		b_cntg_pos = dim1_pos;
		b_cntg_size = dim1_size;
	}
	else {
		a_cntg_pos = 0;
		a_cntg_size = a.cntg();
		b_cntg_pos = 0;
		b_cntg_size = b.cntg();
	}

	return {
		a.sub_matrix(a_cntg_pos, a_cntg_size, a_strd_pos, a_strd_size),
		b.sub_matrix(b_cntg_pos, b_cntg_size, b_strd_pos, b_strd_size),
		c.sub_matrix(c_cntg_pos, c_cntg_size, c_strd_pos, c_strd_size),
		advance(pos, a_strd_pos, b_strd_pos, a_cntg_pos)
	};
}


/*
 * Iterates over a dimension of matrix that is assumed to have TriangularForm
 * As we progress through that dimension we are able to either shorten or
 * lengthen the other dimension shared with the matrix according to its shape
 * and whether it is upper or lower.
 *
 * @tparam WhichMatrix the matrix that is assumed (not checked) to have TriangularForm
 * @param WhichDimension the dimension (from the GEMM contract) of WhichMatrix we will iterate over
 */
template<which_matrix WhichMatrix, which_dimension WhichDimension, typename Next>
class tri_resident {
	bool           is_lower_;
	kernel_inttype block_size_;
	kernel_inttype shorten_divis_;
	Next           next_;

public:
	/**
	 * @param `is_lower` specifies whether the data in `WhichMatrix` is assumed to be upper or lower
	 * @param `block_size` the size that we wish to iterate over `WhichDimension` of `WhichMatrix` by
	 * @param `shorten_divis` the multiple by which the "shortened" dimension is divisible by
	 */
	PERFLIBS_LINALG_INLINE
	tri_resident(bool is_lower, kernel_inttype block_size, kernel_inttype shorten_divis, Next next)
	:	is_lower_      { is_lower        }
	,	block_size_    { block_size      }
	,	shorten_divis_ { shorten_divis   }
	,	next_          { std::move(next) }
	{	}

	///This constructor is used for constructor template deduction
	PERFLIBS_LINALG_INLINE
	tri_resident(
			which_matrix_constant<WhichMatrix>,
			which_dimension_constant<WhichDimension>,
			bool is_lower, kernel_inttype block_size, kernel_inttype shorten_divis, Next next)
	:	tri_resident { is_lower, block_size, shorten_divis, std::move(next) }
	{	}

	template <typename AType, typename BType, typename CType, typename... Args>
	PERFLIBS_LINALG_INLINE
	void operator()(AType &a, BType &b, CType &c, compute_position pos, Args... args) {
		constexpr auto second_dim =
			corresponding_dimension_v<WhichMatrix, WhichDimension>;

		const auto dim0 = get_dimension<WhichDimension>(a, b, c);
		const auto dim1 = get_dimension<second_dim>(a, b, c);

		/*
		 * In an ideal world is_lower would be extracted from the triangular matrix interface
		 * But as this might be used for triangular matrices that have been packed into
		 * general interleave matrices then we need a way of "assuming" the form of the
		 * data we wish to operate in, thus is_lower is set in the constructor
		 */
		if(is_lower_) {
			for (kernel_inttype i = 0; i < dim0; i += block_size_) {
				const auto dim0_block_size = min(dim0 - i, block_size_);

				const auto dim1_pos        = iround_floor(i, shorten_divis_);
				const auto dim1_size       = dim1 - dim1_pos;

				auto [ a_block, b_block, c_block, pos_block ] =
					sub_matrix<WhichDimension, second_dim>(a, b, c, pos, i, dim0_block_size, dim1_pos, dim1_size);

				next_(a_block, b_block, c_block, pos_block, args...);
			}
		}
		else {
			for (kernel_inttype i = 0; i < dim0; i += block_size_) {
				const auto dim0_block_size = min(dim0 - i, block_size_);
				const kernel_inttype dim1_size = min(iround(i + block_size_, shorten_divis_), dim1);

				auto [ a_block, b_block, c_block, pos_block ] =
					sub_matrix<WhichDimension, second_dim>(a, b, c, pos, i, dim0_block_size, 0, dim1_size);

				next_(a_block, b_block, c_block, pos_block, args...);
			}
		}
	}
}; //class tri_resident

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_TRI_RESIDENTS_C_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATRIX_MATRIX_BASE_HPP
#define PERFLIBS_LINALG_MATRIX_MATRIX_BASE_HPP

#include "types.hpp"

#include "framework/linalg_util.hpp"
#include "perflibs_assert.hpp"
#include "perflibs_util.hpp"

namespace perflibs::linalg {

/**
 * Provides a logical mapping (cntg,strd ie m,n) of a matrix on to a physically stored matrix
 * supporting col and row strides
 */
template<typename T>
class matrix_base {
public:
	using data_type        =T;
	using value_type       =T;
	using const_value_type =const T;
	using reference        =T&;
	using const_reference  =const T&;

private:
	T             *data_;

	kernel_inttype cntg_;
	kernel_inttype strd_;

	kernel_inttype cntg_step_;
	kernel_inttype strd_step_;

	///if this is a sub_matrix, gives the position in the parent matrix
	kernel_inttype absolute_cntg_pos_;
	kernel_inttype absolute_strd_pos_;

	///dimensions of matrix of which this may be a submatrix
	kernel_inttype parent_cntg_;
	kernel_inttype parent_strd_;

public:
	matrix_base()                             =default;
	matrix_base(const matrix_base&)           =default;
	matrix_base(matrix_base&&)                =default;
	matrix_base& operator=(const matrix_base&)=default;
	matrix_base& operator=(matrix_base&&)     =default;

	///Sub matrix constructor, passes parent data pointer
	PERFLIBS_LINALG_INLINE
	matrix_base(value_type *data,
				kernel_inttype cntg,              kernel_inttype strd,
				kernel_inttype cntg_step,         kernel_inttype strd_step,
				kernel_inttype absolute_cntg_pos, kernel_inttype absolute_strd_pos,
				kernel_inttype parent_cntg,       kernel_inttype parent_strd)
	:	data_              { data }
	,	cntg_              { cntg }
	,	strd_              { strd }
	,	cntg_step_         { cntg_step }
	,	strd_step_         { strd_step }
	,	absolute_cntg_pos_ { absolute_cntg_pos }
	,	absolute_strd_pos_ { absolute_strd_pos }
	,	parent_cntg_       { parent_cntg }
	,	parent_strd_       { parent_strd }
	{	}

	PERFLIBS_LINALG_INLINE
	matrix_base(value_type *data,
	            kernel_inttype cntg,                kernel_inttype strd,
	            kernel_inttype cntg_step,           kernel_inttype strd_step)
	:	data_              { data }
	,	cntg_              { cntg }
	,	strd_              { strd }
	,	cntg_step_         { cntg_step }
	,	strd_step_         { strd_step }
	,	absolute_cntg_pos_ { 0 }
	,	absolute_strd_pos_ { 0 }
	,	parent_cntg_       { cntg }
	,	parent_strd_       { strd }
	{	}

	///get an element at a specific logical index
	PERFLIBS_LINALG_INLINE reference get_logical_elem(kernel_inttype cntg, kernel_inttype strd) const;

	///get an element at an absolute position relative to the parent matrix, not local to this sub_matrix
	PERFLIBS_LINALG_INLINE reference get_absolute_logical_elem(kernel_inttype cntg, kernel_inttype strd) const ;

	PERFLIBS_LINALG_INLINE
	matrix_base get_parent() const;

	PERFLIBS_LINALG_INLINE
	matrix_base sub_matrix(kernel_inttype cntg_pos, kernel_inttype cntg_size,
	                       kernel_inttype strd_pos, kernel_inttype strd_size) const;
	PERFLIBS_LINALG_INLINE
	matrix_base sub_matrix_with_clamp(kernel_inttype cntg_pos, kernel_inttype cntg_size,
	                                  kernel_inttype strd_pos, kernel_inttype strd_size) const;
	///returns a matrix representing the same data, but transposed
	PERFLIBS_LINALG_INLINE matrix_base transpose() const;

	///returns a matrix representing the reflected matrix
	PERFLIBS_LINALG_INLINE matrix_base reflect() const;

	///returns a matrix representing the reflected matrix, followed by a transpose
	PERFLIBS_LINALG_INLINE matrix_base reflect_transpose() const;

	///returns a pointer to the first element of the matrix of which this is a sub_matrix
	PERFLIBS_LINALG_INLINE value_type *parent_data() const  {
		return data() - ( absolute_cntg() * cntg_step() + absolute_strd() * strd_step() );
	}

	/// number of contiguous elements in the logical representation of the data
	PERFLIBS_LINALG_INLINE kernel_inttype cntg() const { return cntg_; }
	/// number of elements in the strided dimension of the logical representation of the data
	PERFLIBS_LINALG_INLINE kernel_inttype strd() const { return strd_; }

	/// the cntg component of the coordinates that this submatrix is in the parent matrix (if parent, then 0)
	PERFLIBS_LINALG_INLINE kernel_inttype absolute_cntg() const { return absolute_cntg_pos_; }
	/// the strd component of the coordinates that this submatrix is in the parent matrix (if parent, then 0)
	PERFLIBS_LINALG_INLINE kernel_inttype absolute_strd() const { return absolute_strd_pos_; }

	/// The distance between consecutive elements in the cntg dimension
	kernel_inttype cntg_step() const { return cntg_step_; }
	///The distance between consecutive elements in the strd dimension
	kernel_inttype strd_step() const { return strd_step_; }

	kernel_inttype parent_cntg() const { return parent_cntg_; }

	kernel_inttype parent_strd() const { return parent_strd_; }

	/// pointer to the beginning of the data, top left hand corner of matrix
	PERFLIBS_LINALG_INLINE value_type *data() const  { return data_; }

	void estrange_parent() {
		parent_cntg_ = cntg_;
		parent_strd_ = strd_;
		absolute_cntg_pos_ = 0;
		absolute_strd_pos_ = 0;
	}
}; //class matrix_base

//////////////////////////////////////////////////////////////////////////////
///////  IMPLEMENTATION
//////////////////////////////////////////////////////////////////////////////

template<typename T>
PERFLIBS_LINALG_INLINE
matrix_base<T> matrix_base<T>::sub_matrix(kernel_inttype cntg_idx, kernel_inttype cntg_size,
                                          kernel_inttype strd_idx, kernel_inttype strd_size) const {
	PERFLIBS_ASSERT(cntg_idx >= 0 && cntg_idx + cntg_size <= cntg(),
			"Call sub-matrix with out of bounds cntg dimension");
	PERFLIBS_ASSERT(strd_idx >= 0 && strd_idx + strd_size <= strd(),
			"Call sub-matrix with out of bounds strd dimension");

	const kernel_inttype abs_cntg = absolute_cntg() +  cntg_idx;
	const kernel_inttype abs_strd = absolute_strd() +  strd_idx;

	return {
		&get_logical_elem(cntg_idx, strd_idx),
		cntg_size,    strd_size,
		cntg_step_,   strd_step_,
		abs_cntg,     abs_strd,
		parent_cntg_, parent_strd_
	};
}

template<typename T>
PERFLIBS_LINALG_INLINE
matrix_base<T> matrix_base<T>::sub_matrix_with_clamp(kernel_inttype cntg_idx, kernel_inttype cntg_size,
                                                     kernel_inttype strd_idx, kernel_inttype strd_size) const {
	const auto x0 = min(cntg(), max(0, cntg_idx));
	const auto y0 = min(strd(), max(0, strd_idx));
	const auto x1 = min(cntg(), max(0, cntg_idx+cntg_size));
	const auto y1 = min(strd(), max(0, strd_idx+strd_size));

	cntg_idx = x0;
	strd_idx = y0;
	cntg_size = x1 - x0;
	strd_size = y1 - y0;

	return sub_matrix(cntg_idx, cntg_size, strd_idx, strd_size);
}

template<typename T>
PERFLIBS_LINALG_INLINE
matrix_base<T> matrix_base<T>::transpose() const {
	return {
		data(),
		strd(),             cntg(),
		strd_step_,         cntg_step_,
		absolute_strd_pos_, absolute_cntg_pos_,
		parent_strd_,       parent_cntg_
	};
}

template<typename T>
PERFLIBS_LINALG_INLINE
matrix_base<T> matrix_base<T>::reflect() const {
	const auto parent_cntg_pos = absolute_cntg();
	const auto parent_strd_pos = absolute_strd();

	return {
		&get_absolute_logical_elem(parent_strd_pos, parent_cntg_pos),
		strd(),          cntg(),
		cntg_step_,      strd_step_,
		parent_strd_pos, parent_cntg_pos,
		parent_cntg_,    parent_strd_
	};
}

template<typename T>
PERFLIBS_LINALG_INLINE
matrix_base<T> matrix_base<T>::reflect_transpose() const {
	const auto racntg = absolute_cntg();
	const auto rastrd = absolute_strd();

	const auto cstep  = strd_step();
	const auto sstep  = cntg_step();

	return {
		&get_absolute_logical_elem(rastrd, racntg),
		cntg(),        strd(),
		cstep,         sstep,
		racntg,        rastrd,
		parent_strd(), parent_cntg(),
	};
}

template<typename T>
PERFLIBS_LINALG_INLINE
typename matrix_base<T>::reference
matrix_base<T>::get_absolute_logical_elem(kernel_inttype cntg, kernel_inttype strd) const {
	return parent_data()[ cntg * cntg_step_ + strd * strd_step_ ];
}

template<typename T>
PERFLIBS_LINALG_INLINE
typename matrix_base<T>::reference
matrix_base<T>::get_logical_elem(kernel_inttype cntg, kernel_inttype strd) const {
	return data()[ cntg * cntg_step_ + strd * strd_step_ ];
}

template<typename T>
PERFLIBS_LINALG_INLINE
matrix_base<T> matrix_base<T>::get_parent() const {
	return { parent_data(), parent_cntg(), parent_strd(), cntg_step(), strd_step() };
}

/**
 * Checks to see whether two matrix objects represent the SAME matrix
 * Does not check whether two different matrix object contain the same values
 *
 * We only able this function if the inner value_type are the same
 * after accounting for const-volitileness
 */
template<typename T0, typename T1>
auto quick_equal(const matrix_base<T0>& lhs, const matrix_base<T1>& rhs)
	-> std::enable_if_t<
		std::is_same_v<
			std::remove_cv_t<T0>,
			std::remove_cv_t<T1>>,
		bool>
{
	return lhs.data() == rhs.data()
	    && lhs.cntg() == rhs.cntg()
	    && lhs.strd() == rhs.strd()
	    && lhs.absolute_cntg() == rhs.absolute_cntg()
	    && lhs.absolute_strd() == rhs.absolute_strd();
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_MATRIX_MATRIX_BASE_HPP

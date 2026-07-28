/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BANDED_MATRIX_BASE_HPP
#define PERFLIBS_LINALG_BANDED_MATRIX_BASE_HPP

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
class banded_matrix_base {
public:
	using data_type        =T;
	using value_type       =T;
	using const_value_type =const T;
	using reference        =T&;
	using const_reference  =const T&;

	//these are effectively the same thing, but with different permissions
	friend class banded_matrix_base<const T>;
	friend class banded_matrix_base<std::remove_cv_t<T>>;
private:
	// although parent_data_ can be calculated from data_ and the absolute positions, we don't want to be
	// doing that on every access so cache this param here the matrix types (in terms of dimensions and
	// position) are largely immutable - so it is safe to cache such info here
	T *data_;
	T *parent_data_;

	kernel_inttype physical_cntg_;
	kernel_inttype physical_strd_;

	kernel_inttype physical_cntg_stride_;
	kernel_inttype physical_strd_stride_;

	///if this is a sub_matrix, gives the position in the parent matrix
	kernel_inttype absolute_cntg_pos_;
	kernel_inttype absolute_strd_pos_;

	kernel_inttype parent_cntg_;
	kernel_inttype parent_strd_;

	kernel_inttype kl_;
	kernel_inttype ku_;

	T zero = 0;

public:
	banded_matrix_base() = default;

	template<typename T0>
	banded_matrix_base(const banded_matrix_base<T0>& other)
	:	data_                 { other.data_ }
	,	parent_data_          { other.parent_data_ }
	,	physical_cntg_        { other.physical_cntg_ }
	,	physical_strd_        { other.physical_strd_ }
	,	physical_cntg_stride_ { other.physical_cntg_stride_ }
	,	physical_strd_stride_ { other.physical_strd_stride_ }
	,	absolute_cntg_pos_    { other.absolute_cntg_pos_ }
	,	absolute_strd_pos_    { other.absolute_strd_pos_ }
	,	parent_cntg_          { other.parent_cntg_ }
	,	parent_strd_          { other.parent_strd_ }
	,	kl_                   { other.kl_ }
	,	ku_                   { other.ku_ }
	,	zero                  { 0_ki }
	{
		static_assert(
			std::is_same_v<T, const T0> || std::is_same_v<T, T0>,
			"can not copy construct banded_matrix_base from incompatible types");
	}

	///Standard matrix construct, not assumed to be sub matrix
	banded_matrix_base(value_type *data,
				kernel_inttype logical_cntg,         kernel_inttype logical_strd,
				kernel_inttype physical_cntg_stride, kernel_inttype physical_strd_stride,
				kernel_inttype kl,                   kernel_inttype ku,
				kernel_inttype absolute_cntg_pos=0,  kernel_inttype absolute_strd_pos=0)
	:	data_                 { data }
	,	parent_data_          { data }
	,	physical_cntg_        { logical_cntg }
	,	physical_strd_        { logical_strd }
	,	physical_cntg_stride_ { physical_cntg_stride }
	,	physical_strd_stride_ { physical_strd_stride }
	,	absolute_cntg_pos_    { absolute_cntg_pos }
	,	absolute_strd_pos_    { absolute_strd_pos }
	,	parent_cntg_          { logical_cntg }
	,	parent_strd_          { logical_strd }
	,	kl_                   { kl }
	,	ku_                   { ku }
	{	}

	/*
	 * We expect matrix base to be descended from by adaptors which then
	 * expose a functional interface, so the following functions are protected
	 * as their utility may be need to be wrapped in an adaptor
	 */
protected:
	/*
	 * interleave_batch_matrix needs access to the protected constructor
	 * so that its extract function can have a fully featured and expected
	 * behavior.
	 */
	friend class interleave_batch_matrix<T>;

	///Sub matrix constructor, passes parent data pointer
	banded_matrix_base(value_type *data, value_type *parent_data,
				kernel_inttype logical_cntg,         kernel_inttype logical_strd,
				kernel_inttype physical_cntg_stride, kernel_inttype physical_strd_stride,
				kernel_inttype absolute_cntg_pos,    kernel_inttype absolute_strd_pos,
				kernel_inttype parent_cntg,          kernel_inttype parent_strd,
				kernel_inttype kl,                   kernel_inttype ku)
	:	data_                 { data }
	,	parent_data_          { parent_data }
	,	physical_cntg_        { logical_cntg }
	,	physical_strd_        { logical_strd }
	,	physical_cntg_stride_ { physical_cntg_stride }
	,	physical_strd_stride_ { physical_strd_stride }
	,	absolute_cntg_pos_    { absolute_cntg_pos }
	,	absolute_strd_pos_    { absolute_strd_pos }
	,	parent_cntg_          { parent_cntg }
	,	parent_strd_          { parent_strd }
	,	kl_                   { kl }
	,	ku_                   { ku }
	{	}

	///get an element at a specific logical index
 	PERFLIBS_LINALG_INLINE const_reference get_logical_elem(kernel_inttype cntg, kernel_inttype strd) const;
	PERFLIBS_LINALG_INLINE       reference get_logical_elem(kernel_inttype cntg, kernel_inttype strd);

	PERFLIBS_LINALG_INLINE
	banded_matrix_base get_parent() const;

	PERFLIBS_LINALG_INLINE
	banded_matrix_base sub_matrix(kernel_inttype cntg_pos, kernel_inttype cntg_size,
	                              kernel_inttype strd_pos, kernel_inttype strd_size) const;
	PERFLIBS_LINALG_INLINE
	banded_matrix_base sub_matrix_with_clamp(kernel_inttype cntg_pos, kernel_inttype cntg_size,
	                                         kernel_inttype strd_pos, kernel_inttype strd_size) const;

	///returns a matrix representing the same data, but transposed
	PERFLIBS_LINALG_INLINE banded_matrix_base transpose() const;

	///returns a matrix representing the reflected matrix
	PERFLIBS_LINALG_INLINE banded_matrix_base reflect() const;

	///returns a matrix representing the reflected matrix, followed by a transpose
	PERFLIBS_LINALG_INLINE banded_matrix_base reflect_transpose() const;

	///returns a pointer to the first element of the matrix of which this is a sub_matrix
	PERFLIBS_LINALG_INLINE value_type *parent_data() const  { return parent_data_; }
public:
	/// number of contiguous elements in the logical representation of the data
	PERFLIBS_LINALG_INLINE kernel_inttype cntg() const  { return physical_cntg_; }
	/// number of elements in the strided dimension of the logical representation of the data
	PERFLIBS_LINALG_INLINE kernel_inttype strd() const  { return physical_strd_; }

	/// the cntg component of the coordinates that this submatrix is in the parent matrix (if parent, then 0)
	PERFLIBS_LINALG_INLINE kernel_inttype absolute_cntg() const  { return absolute_cntg_pos_; }
	/// the strd component of the coordinates that this submatrix is in the parent matrix (if parent, then 0)
	PERFLIBS_LINALG_INLINE kernel_inttype absolute_strd() const  { return absolute_strd_pos_; }

	/// The distance between consecutive elements in the cntg dimension
	kernel_inttype cntg_step() const { return physical_cntg_stride_; }
	///The distance between consecutive elements in the strd dimension
	kernel_inttype strd_step() const { return physical_strd_stride_; }

	kernel_inttype parent_cntg() const { return parent_cntg_; }

	kernel_inttype parent_strd() const { return parent_strd_; }

	/// pointer to the beginning of the data, top left hand corner of matrix
	PERFLIBS_LINALG_INLINE value_type *data() const  { return data_; }

	/// size of kl in the logical representation of the data
	PERFLIBS_LINALG_INLINE kernel_inttype kl() const  { return kl_; }
	/// size of ku in the logical representation of the data
	PERFLIBS_LINALG_INLINE kernel_inttype ku() const  { return ku_; }

	/// determine if the current matrix is transposed or not
	PERFLIBS_LINALG_INLINE bool is_trans() const { return physical_strd_stride_ == 1  /* is_strd_contig */; }

	/// determine where the band starts based on the strd index, (on original matrix, not submatrix)
	PERFLIBS_LINALG_INLINE kernel_inttype band_starts_at(kernel_inttype strd) const {
		return max(0,strd-ku_);
	}

	/// determine where the band ends based on the strd index (on original matrix, not submatrix)
	PERFLIBS_LINALG_INLINE kernel_inttype band_ends_at(kernel_inttype strd) const {

		return min(parent_strd_ - 1, strd + kl_);
	}

	/** Determine the absolute start and end position of band j in matrix (or submatrix of) A.
	 *  Meaning the positions computed by these functions are relative to the a origin position (0,0)
	 *  which marks the beginning of the parent matrix of A (or a itself if A is not submatrix)
	 *
	 *     * If we are talking about band j, it means that the diagonal of that band will be
	 *       on strd position jth. Which means its start and end positions will be respectively diag_pos - ku
	 *       (ku positions above the diagonal) and diag_pos + kl (kl positions below the diagonal).
	 *     * This is not enough: we have to remember the space we are confined in the first place
	 *       and prevent this positions from overflowing this space (In this case, our space is bounded by
	 *       0 and parent_strd_ - 1)
	 */
	PERFLIBS_LINALG_INLINE kernel_inttype absolute_band_strd_start(kernel_inttype diag_cntg_pos) const {
		return max(0, diag_cntg_pos - ku_);
	}
	PERFLIBS_LINALG_INLINE kernel_inttype absolute_band_strd_end(kernel_inttype diag_cntg_pos) const {
		return min(parent_strd_ - 1, diag_cntg_pos + kl_);
	}

	/** Determines the start and end position of band j in matrix (or submatrix of) A.
	 *  To get each position we have to get where the band actually starts in the parent matrix (getting the absolute positions)
	 *     (Our cartesian referential has origin position (0,0) of the parent matrix)
	 * And then crop the positions according to the indexes accepted by the submatrix.
	 *    (We shift our cartesian referential to have origin position at (0,0) of the submatrix)
	 * Reason why we need the absolute positions is because our point of reference to get the start and end
	 * positions of the band j is finding the original diagonal and then computing the relevant positions from there.
	 * When we are in the submatrix space, the original diagonal is often not the same as its actual diagonal.
	 * Turns out it is much easier to find the original diagonal in the parent matrix (it is the jth element of that band)
	 * than finding it in its submatrix.
	 */
	PERFLIBS_LINALG_INLINE std::pair<kernel_inttype,kernel_inttype> get_strd_band_pos(kernel_inttype cntg) const {
		auto absolute_band_start_pos = absolute_band_strd_start(cntg);
		auto band_start_pos = max(0, absolute_band_start_pos - absolute_strd_pos_);
		auto absolute_band_end_pos = absolute_band_strd_end(cntg);
		auto band_end_pos = min(physical_strd_ -1, absolute_band_end_pos - absolute_strd_pos_) ;
		return std::make_pair(band_start_pos, band_end_pos);
	}

	/**
	 * Determines the start and end position of jth strd band in matrix (or submatrix of) a
	 * (The band is strded which means its elements have a constant strd pos and a variable cntg pos)
	 * The positions we return are the ones over the cntg dimension
	 * To get each position we have to get where the band actually starts in the parent matrix
	 * (Our cartesian referential has origin position (0,0) of the parent matrix)
	 * And then crop the positions according to the indexes accepted by the submatrix.
	 * (We shift our cartesian referential to have origin position at (0,0) of the submatrix)
	 * Reason why we need the absolute positions is because our point of reference to get the start and end
	 * positions of the band j is finding the diagonal and then computing the relevant positions from there.
	 * Turns out it is much easier to find the diagonal in the parent matrix (it is the jth element of that band)
	 * than finding it in a submatrix.
	 */
	PERFLIBS_LINALG_INLINE std::pair<kernel_inttype,kernel_inttype> get_cntg_band_pos(kernel_inttype strd) const {
		auto band_start_pos = max(0,strd - kl_);
		auto band_end_pos = min(parent_cntg_ - 1, strd + ku_);
		return std::make_pair(band_start_pos, band_end_pos);
	}

	/// determine if the band matrix starts below the bottom of the matrix and none of the
	/// elements of the band are included in the matrix
	PERFLIBS_LINALG_INLINE bool band_starts_below(kernel_inttype cntg) const {
		return cntg - ku_ >= parent_strd_;
	}

	// get a pointer to an element in the parent matrix
	PERFLIBS_LINALG_INLINE value_type *get_absolute_physical_elem(kernel_inttype cntg, kernel_inttype strd) const {
		cntg += absolute_cntg_pos_;
		strd += absolute_strd_pos_;

		if (cntg - kl_ <= strd && cntg + ku_ >= strd) {
			if(!is_trans())
				return parent_data_ + (ku_ + cntg - strd) * physical_cntg_stride_ + strd * physical_strd_stride_;
			else
				return parent_data_ + (kl_ + strd - cntg) * physical_strd_stride_ + cntg * physical_cntg_stride_ ;
		}
		else
			return data_;
	}

	// get a pointer to the top of the current cntg
	PERFLIBS_LINALG_INLINE value_type *get_physical_cntg(kernel_inttype cntg, kernel_inttype strd) const {
		kernel_inttype cntg_start = band_starts_at(cntg);

		if(cntg > cntg_start)
			cntg_start = cntg;

		if(band_starts_below(cntg)) {
			return data_;
		}

		if(!is_trans()) {
			return parent_data_ + cntg_start * physical_strd_stride_ + (strd + ku_ - cntg_start) * physical_cntg_stride_;
		}
		else {
			return parent_data_ + (cntg_start + kl_ - strd) * physical_strd_stride_ +  strd * physical_cntg_stride_;
		}
	}

	/// get an element at an absolute position relative to the parent matrix, not local to this sub_matrix
	PERFLIBS_LINALG_INLINE const_reference get_absolute_logical_elem(kernel_inttype cntg, kernel_inttype strd) const ;
	PERFLIBS_LINALG_INLINE       reference get_absolute_logical_elem(kernel_inttype cntg, kernel_inttype strd);


}; //class banded_matrix_base

//////////////////////////////////////////////////////////////////////////////
///////  IMPLEMENTATION
//////////////////////////////////////////////////////////////////////////////

template<typename T>
PERFLIBS_LINALG_INLINE
banded_matrix_base<T> banded_matrix_base<T>::sub_matrix(kernel_inttype cntg_idx, kernel_inttype cntg_size,
                                                        kernel_inttype strd_idx, kernel_inttype strd_size) const {
	PERFLIBS_ASSERT(cntg_idx >= 0 && cntg_idx + cntg_size <= cntg(),
			"Call sub-matrix with out of bounds cntg dimension");
	PERFLIBS_ASSERT(strd_idx >= 0 && strd_idx + strd_size <= strd(),
			"Call sub-matrix with out of bounds strd dimension");

	const kernel_inttype abs_cntg = absolute_cntg() +  cntg_idx;
	const kernel_inttype abs_strd = absolute_strd() +  strd_idx;

	return {
		get_physical_cntg(abs_cntg, abs_strd),
		parent_data(),
		cntg_size,             strd_size,
		physical_cntg_stride_, physical_strd_stride_,
		abs_cntg,              abs_strd,
		parent_cntg_,          parent_strd_,
		kl_,                   ku_
	};
}

template<typename T>
PERFLIBS_LINALG_INLINE
banded_matrix_base<T> banded_matrix_base<T>::sub_matrix_with_clamp(kernel_inttype cntg_idx, kernel_inttype cntg_size,
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
banded_matrix_base<T> banded_matrix_base<T>::transpose() const {
	return {
		data(),
		parent_data(),
		strd(),                     cntg(),
		physical_strd_stride_,      physical_cntg_stride_,
		absolute_strd_pos_,         absolute_cntg_pos_,
		parent_strd_,               parent_cntg_,
		ku_,                        kl_
	};
}

template<typename T>
PERFLIBS_LINALG_INLINE
banded_matrix_base<T> banded_matrix_base<T>::reflect() const {
	return {
		&get_absolute_logical_elem(physical_strd_, physical_cntg_),
		parent_data(),
		strd(),                cntg(),
		physical_cntg_stride_, physical_strd_stride_,
		absolute_strd_pos_,    absolute_cntg_pos_,
		parent_cntg_,          parent_strd_,
		kl_,                   ku_
	};
}

template<typename T>
PERFLIBS_LINALG_INLINE
banded_matrix_base<T> banded_matrix_base<T>::reflect_transpose() const {
	const auto racntg = absolute_cntg();
	const auto rastrd = absolute_strd();

	const auto cstep  = strd_step();
	const auto sstep  = cntg_step();

	return {
		&get_absolute_logical_elem(rastrd, racntg),
		parent_data(),
		cntg(),        strd(),
		cstep,         sstep,
		racntg,        rastrd,
		parent_cntg(), parent_strd(),
		ku_,           kl_,
		kl_ + ku_
	};
}

template<typename T>
PERFLIBS_LINALG_INLINE
typename banded_matrix_base<T>::reference
banded_matrix_base<T>::get_absolute_logical_elem(kernel_inttype cntg, kernel_inttype strd) {
	if (cntg - ku_ <= strd && cntg + kl_ >= strd){
		if(!is_trans())
			return parent_data()[ (strd + ku_ - cntg) * physical_cntg_stride_ + cntg * physical_strd_stride_ ];
		else
			return parent_data()[ strd * physical_cntg_stride_ + (cntg + kl_ - strd) * physical_strd_stride_ ];
	}
	else
		return zero;
}
template<typename T>
PERFLIBS_LINALG_INLINE
typename banded_matrix_base<T>::const_reference
banded_matrix_base<T>::get_absolute_logical_elem(kernel_inttype cntg, kernel_inttype strd) const {
	return const_cast<banded_matrix_base<T>&>(*this).get_absolute_logical_elem(cntg, strd);
}
template<typename T>
PERFLIBS_LINALG_INLINE
typename banded_matrix_base<T>::reference
banded_matrix_base<T>::get_logical_elem(kernel_inttype cntg, kernel_inttype strd) {
	cntg += absolute_cntg_pos_;
	strd += absolute_strd_pos_;

	// Given a non-transposed banded matrix A:
	// n = 5
	// ku = 1
	// kl = 2
	//
	//   s t r d
	// c D # 0 0 0
	// n # D # 0 0
	// t # # D # 0
	// g X # # D #
	//   X X # # D
	//
	// On any given stride i, the diagonal will be in position i:
	//
	// Stride 2
	// [0] 0
	// [1] #
	// [2] D <-
	// [3] #
	// [4] #
	//
	// Therefore the first element of the stride will be at cntg index strd - ku, and the last at strd + kl,
	// bounded by zero and (n-1). If the desired index is outside of this range for any value of strd, then
	// zero should be returned.
	//
	// The parent data is stored simply as P:
	//
	//   s t r d
	// c X # # # #
	// n D D D D D
	// t # # # # X
	// g # # # X X
	//
	// Since the diagonal has become horizontal, the cntg address in P is affected by both A cntg and strd.
	// if A(i,j) => P(i', j'):
	//  - A(i + 1, j    ) => P(i' + 1, j'    )
	//  - A(i    , j + 1) => P(i' - 1, j' + 1)
	//  - A(i + 1, j + 1) => P(i'    , j' + 1)
	//
	// Incrementing A cntg also increments P cntg, and incrementing A strd decrements P cntg.
	// P strd is always equal to A strd. If A cntg and strd are equal the diagonal is being accessed, which
	// is located ku elements from the top of each stride, hence the ku_ in the non-transposed calculation
	// below.
	//
	// In the transposed case all values are swapped.

	if (cntg - kl_ <= strd && cntg + ku_ >= strd) {
		if(!is_trans())
			return parent_data()[(ku_ + cntg - strd) * physical_cntg_stride_ + strd * physical_strd_stride_ ];
		else
			return parent_data()[(kl_ + strd - cntg) * physical_strd_stride_ + cntg * physical_cntg_stride_ ];
	}
	else
		return zero;
}
template<typename T>
PERFLIBS_LINALG_INLINE
typename banded_matrix_base<T>::const_reference
banded_matrix_base<T>::get_logical_elem(kernel_inttype cntg, kernel_inttype strd) const {
	return const_cast<banded_matrix_base<T>&>(*this).get_logical_elem(cntg, strd);
}

template<typename T>
PERFLIBS_LINALG_INLINE
banded_matrix_base<T> banded_matrix_base<T>::get_parent() const {
	return { parent_data(), parent_cntg(), parent_strd(), cntg_step(), strd_step(), 0, 0, kl(), ku() };
}

/**
 * Checks to see whether two matrix objects represent the SAME matrix
 * Does not check whether two different matrix object contain the same values
 *
 * We only able this function if the inner value_type are the same
 * after accounting for const-volitileness
 */
template<typename T0, typename T1>
auto quick_equal(const banded_matrix_base<T0>& lhs, const banded_matrix_base<T1>& rhs)
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
	    && lhs.absolute_strd() == rhs.absolute_strd()
		&& lhs.kl() == rhs.kl()
		&& lhs.ku() == rhs.ku();
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_banded_matrix_base

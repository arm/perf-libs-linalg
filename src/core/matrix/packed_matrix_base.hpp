/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_PACKED_MATRIX_BASE
#define PERFLIBS_LINALG_PACKED_MATRIX_BASE

#include "framework/linalg_util.hpp"
#include "framework/linalg_blas_types.hpp"
#include "perflibs_assert.hpp"
#include "perflibs_util.hpp"

namespace perflibs::linalg {

template<typename T>
class packed_matrix_base {
public:
	using data_type        = T;
	using value_type       = T;
	using const_value_type = const T;
	using reference        = T&;
	using const_reference  = const T&;
private:
	// pointers to top left corner of matrices
	T *data_;
	T *parent_data_;

	// dimensions of the matrix
	kernel_inttype cntg_;
	kernel_inttype strd_;

	perflibs_uplo uplo_;

	// function pointers for calculating cntg and strd steps
	kernel_inttype (*cntg_step_fn_)(kernel_inttype, kernel_inttype);
	kernel_inttype (*strd_step_fn_)(kernel_inttype, kernel_inttype);

	// if this is a sub_matrix, gives the position in the parent matrix
	kernel_inttype absolute_cntg_pos_;
	kernel_inttype absolute_strd_pos_;

	// dimensions of the parent matrix
	kernel_inttype parent_cntg_;
	kernel_inttype parent_strd_;

	bool is_strd_unit_;
	bool is_trans_;
public:
	packed_matrix_base() = default;

	// Standard matrix construct, not assumed to be sub matrix
	packed_matrix_base(value_type *data,
	                   kernel_inttype cntg, kernel_inttype strd, perflibs_uplo uplo,
	                   bool is_strd_unit=true, bool is_trans=false,
	                   kernel_inttype absolute_cntg_pos=0,
	                   kernel_inttype absolute_strd_pos=0)
	: packed_matrix_base(data, data, cntg, strd, uplo, is_strd_unit, is_trans, absolute_cntg_pos, absolute_strd_pos, cntg, strd)
	{
	}

	// Sub matrix constructor, passes parent data pointer
	packed_matrix_base(value_type *data, value_type *parent_data,
	                   kernel_inttype cntg, kernel_inttype strd, perflibs_uplo uplo,
	                   bool is_strd_unit, bool is_trans,
	                   kernel_inttype absolute_cntg_pos, kernel_inttype absolute_strd_pos,
	                   kernel_inttype parent_cntg, kernel_inttype parent_strd)
	: data_{ data }
	, parent_data_{ parent_data }
	, cntg_{ cntg }
	, strd_{ strd }
	, uplo_{ uplo }
	, cntg_step_fn_{ is_strd_unit ? (is_trans ? packed_matrix_base<T>::unit_step_fn
	                                          : (uplo == PERFLIBS_UPPER ? packed_matrix_base<T>::upper_step_fn
	                                                                 : packed_matrix_base<T>::lower_step_fn))
	                              : (is_trans ? (uplo == PERFLIBS_UPPER ? packed_matrix_base<T>::lower_step_fn
	                                                                 : packed_matrix_base<T>::upper_step_fn)
	                                          : packed_matrix_base<T>::unit_step_fn) }
	, strd_step_fn_{ is_strd_unit ? (is_trans ? (uplo == PERFLIBS_UPPER ? packed_matrix_base<T>::lower_step_fn
	                                                                 : packed_matrix_base<T>::upper_step_fn)
	                                          : packed_matrix_base<T>::unit_step_fn)
	                              : (is_trans ? packed_matrix_base<T>::unit_step_fn
	                                          : (uplo == PERFLIBS_UPPER ? packed_matrix_base<T>::upper_step_fn
	                                                                 : packed_matrix_base<T>::lower_step_fn)) }
	, absolute_cntg_pos_{ absolute_cntg_pos }
	, absolute_strd_pos_{ absolute_strd_pos }
	, parent_cntg_{ parent_cntg }
	, parent_strd_{ parent_strd }
	, is_strd_unit_{ is_strd_unit }
	, is_trans_{ is_trans }
	{
	}
protected:
	// get an element at a specific logical index
	PERFLIBS_LINALG_INLINE reference get_logical_elem(kernel_inttype cntg, kernel_inttype strd) const;


	// get the parent matrix
	PERFLIBS_LINALG_INLINE
	packed_matrix_base get_parent() const;

	PERFLIBS_LINALG_INLINE
	packed_matrix_base sub_matrix(kernel_inttype cntg_pos, kernel_inttype cntg_size,
	                              kernel_inttype strd_pos, kernel_inttype strd_size) const;

	///returns a matrix representing the same data, but transposed
	PERFLIBS_LINALG_INLINE packed_matrix_base transpose() const;
public:
	// get an element at an absolute position relative to the parent matrix, not local to this sub_matrix
	PERFLIBS_LINALG_INLINE reference get_absolute_logical_elem(kernel_inttype cntg, kernel_inttype strd) const ;
	PERFLIBS_LINALG_INLINE value_type *get_absolute_physical_elem(kernel_inttype cntg, kernel_inttype strd) const ;

	static kernel_inttype lower_step_fn(kernel_inttype indx, kernel_inttype cntg_size) { return indx * ( 2*cntg_size - indx - 1 ) / 2; }
	static kernel_inttype upper_step_fn(kernel_inttype indx, kernel_inttype cntg_size) { return indx * (indx + 1) / 2; }
	static kernel_inttype unit_step_fn(kernel_inttype indx, kernel_inttype cntg_size)  { return indx; }

	// pointer to the beginning of the data, top left hand corner of matrix
	PERFLIBS_LINALG_INLINE value_type *data() const { return data_; }

	///returns a pointer to the first element of the matrix of which this is a sub_matrix
	PERFLIBS_LINALG_INLINE value_type *parent_data() const { return parent_data_; }

	/// number of contiguous elements in the logical representation of the data
	PERFLIBS_LINALG_INLINE kernel_inttype cntg() const { return cntg_; }
	/// number of elements in the strided dimension of the logical representation of the data
	PERFLIBS_LINALG_INLINE kernel_inttype strd() const { return strd_; }

	/// The distance between consecutive elements in the cntg dimension
	constexpr
	PERFLIBS_LINALG_INLINE kernel_inttype cntg_step() const {
		// This assert is not always met. These are accessed by
		//               banded_mv_common contrary to the is_strd_unit flag.
		/* PERFLIBS_ASSERT(!is_strd_unit_, "Contiguous dimension stride is not constant"); */
		return 1;
	}

	/// The distance between consecutive elements in the strd dimension
	constexpr
	PERFLIBS_LINALG_INLINE kernel_inttype strd_step() const {
		// This assert is not always met. These are accessed by
		//               banded_mv_common contrary to the is_strd_unit flag.
		/* PERFLIBS_ASSERT(is_strd_unit_, "Stride dimension stride is not constant"); */
		return 1;
	}

	PERFLIBS_LINALG_INLINE perflibs_uplo uplo() const { return uplo_; }

	PERFLIBS_LINALG_INLINE kernel_inttype kl() const { return uplo_ == PERFLIBS_LOWER ? parent_cntg_ : 0; }
	PERFLIBS_LINALG_INLINE kernel_inttype ku() const { return uplo_ == PERFLIBS_UPPER ? parent_cntg_ : 0; }

	// assumptions: 1) absolute_cntg_pos_ == 0 because we parallelise only in the strd dimension
	//              2) this function is called within a loop from first_band_cntg and last_band_cntg
	PERFLIBS_LINALG_INLINE std::pair<kernel_inttype,kernel_inttype> get_strd_band_pos(kernel_inttype cntg)const {
		PERFLIBS_ASSERT(absolute_cntg_pos_ == 0, "absolute_cntg_pos_ must be 0");

		auto band_start_pos = uplo_ == PERFLIBS_UPPER ? 0 : max(cntg - absolute_strd_pos_, 0);
		auto band_end_pos = uplo_ == PERFLIBS_UPPER ? min(cntg - absolute_strd_pos_, strd_-1) : strd_-1;
		return std::make_pair(band_start_pos, band_end_pos);
	}

	// assumptions: 1) absolute_strd_pos_ == 0 because we parallelise only in the cntg dimension
	PERFLIBS_LINALG_INLINE std::pair<kernel_inttype,kernel_inttype> get_cntg_band_pos(kernel_inttype strd) const {
		PERFLIBS_ASSERT(absolute_strd_pos_ == 0, "absolute_cntg_pos_ must be 0");

		auto band_start_pos = uplo_ == PERFLIBS_LOWER ? 0 : max(strd - absolute_cntg_pos_, 0);
		auto band_end_pos = uplo_ == PERFLIBS_LOWER ? min(strd - absolute_cntg_pos_, cntg_-1) : cntg_-1;
		return std:: make_pair(band_start_pos, band_end_pos);
	}

	/// the cntg component of the coordinates that this submatrix is in the parent matrix (if parent, then 0)
	PERFLIBS_LINALG_INLINE kernel_inttype absolute_cntg() const { return absolute_cntg_pos_; }
	/// the strd component of the coordinates that this submatrix is in the parent matrix (if parent, then 0)
	PERFLIBS_LINALG_INLINE kernel_inttype absolute_strd() const { return absolute_strd_pos_; }

	/// number of contiguous elements of the parent matrix
	PERFLIBS_LINALG_INLINE kernel_inttype parent_cntg() const { return parent_cntg_; }
	/// number of elements in the strided dimension of the parent matrix
	PERFLIBS_LINALG_INLINE kernel_inttype parent_strd() const { return parent_strd_; }

	PERFLIBS_LINALG_INLINE bool is_strd_unit() const { return is_strd_unit_; }
	PERFLIBS_LINALG_INLINE bool is_trans() const { return is_trans_; }
}; // class packed_matrix_base

//////////////////////////////////////////////////////////////////////////////
///////  IMPLEMENTATION
//////////////////////////////////////////////////////////////////////////////

template<typename T>
packed_matrix_base<T> packed_matrix_base<T>::sub_matrix(kernel_inttype cntg_idx, kernel_inttype cntg_size,
	                                                    kernel_inttype strd_idx, kernel_inttype strd_size) const {

	PERFLIBS_ASSERT(cntg_idx >= 0 && cntg_idx + cntg_size <= cntg(),
	                "Call sub-matrix with out of bounds cntg dimension");
	PERFLIBS_ASSERT(strd_idx >= 0 && strd_idx + strd_size <= strd(),
	                "Call sub-matrix with out of bounds strd dimension");

	const int abs_cntg = absolute_cntg() + cntg_idx;
	const int abs_strd = absolute_strd() + strd_idx;

	return {
	        &get_logical_elem(cntg_idx, strd_idx),
	        parent_data(),
	        cntg_size, strd_size, uplo_,
	        is_strd_unit(), is_trans(),
	        abs_cntg, abs_strd,
	        parent_cntg_, parent_strd_
	};
}

template<typename T>
PERFLIBS_LINALG_INLINE
typename packed_matrix_base<T>::reference
packed_matrix_base<T>::get_logical_elem(kernel_inttype cntg, kernel_inttype strd) const {
	auto abs_cntg = cntg + absolute_cntg();
	auto abs_strd = strd + absolute_strd();
	return parent_data()[ cntg_step_fn_(abs_cntg, parent_cntg_) + strd_step_fn_(abs_strd, parent_cntg_) ];
}

template<typename T>
PERFLIBS_LINALG_INLINE
typename packed_matrix_base<T>::reference
packed_matrix_base<T>::get_absolute_logical_elem(kernel_inttype cntg, kernel_inttype strd) const {
	return parent_data()[ cntg_step_fn_(cntg, parent_cntg_) + strd_step_fn_(strd, parent_cntg_) ];
}

// This is a temporary fix and will be revisited.
template<typename T>
PERFLIBS_LINALG_INLINE
typename packed_matrix_base<T>::value_type*
packed_matrix_base<T>::get_absolute_physical_elem(kernel_inttype cntg, kernel_inttype strd) const {
	return &get_absolute_logical_elem(cntg, strd);
}


template<typename T>
packed_matrix_base<T> packed_matrix_base<T>::get_parent() const {
	return { parent_data(), parent_data(), parent_cntg(), parent_strd(), uplo(), is_strd_unit(), is_trans(),
	         0, 0, parent_cntg(), parent_strd() };
}

template<typename T>
packed_matrix_base<T> packed_matrix_base<T>::transpose() const {
	// flip uplo and is_trans, swap cntg with strd, and swap absolute_cntg with absolute_strd
	return { data(), parent_data(), strd(), cntg(), lower_flip(uplo()), is_strd_unit(), !is_trans(),
	         absolute_strd(), absolute_cntg(), parent_cntg(), parent_strd() };
}

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_PACKED_MATRIX_BASE

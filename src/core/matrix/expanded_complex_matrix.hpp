/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATRIX_EXPANDED_COMPLEX_MATRIX_HPP
#define PERFLIBS_LINALG_MATRIX_EXPANDED_COMPLEX_MATRIX_HPP

#include "expanded_complex_ref.hpp"
#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

// In expanded-complex format, each complex value is stored as a real 2x2 block.
// For example, the complex matrix:
// [ a + bi   c + di ]
// [ e + fi   g + hi ]

// Is expanded into the real matrix:
// [ a  -b   c  -d ]
// [ b   a   d   c ]
// [ e  -f   g  -h ]
// [ f   e   h   g ]

// Dimensions and strides below are measured in complex elements.
template<typename T>
class expanded_complex_matrix {
	static_assert(is_complex_v<T>, "type of expanded_complex_matrix must be complex");

public:
	using value_type = T;
	using const_value_type = const T;
	using precision_t = typename value_type::value_type;
	using reference = expanded_complex_ref<precision_t>;

private:
	T *data_;
	kernel_inttype interleave_;
	kernel_inttype cntg_;
	kernel_inttype strd_;
	kernel_inttype strd_step_;

	PERFLIBS_LINALG_INLINE
	kernel_inttype get_elem_idx(kernel_inttype in_cntg, kernel_inttype in_strd) const {
		return in_strd / interleave_ * strd_step_
			+ in_cntg * 2 * interleave_
			+ in_strd % interleave_;
	}

public:
	expanded_complex_matrix(T *data, kernel_inttype interleave, kernel_inttype cntg,
			kernel_inttype strd, kernel_inttype strd_step)
	:data_ { data }
	, interleave_ { interleave }
	, cntg_ { cntg }
	, strd_ { strd }
	, strd_step_ { strd_step }
	{}

	PERFLIBS_LINALG_INLINE kernel_inttype interleave() const { return interleave_; }
	PERFLIBS_LINALG_INLINE kernel_inttype cntg() const { return cntg_; }
	PERFLIBS_LINALG_INLINE kernel_inttype strd() const { return strd_; }
	PERFLIBS_LINALG_INLINE kernel_inttype strd_step() const { return strd_step_; }
	PERFLIBS_LINALG_INLINE value_type *data() const { return data_; }

	PERFLIBS_LINALG_INLINE kernel_inttype strd_interleave() const { return interleave_; }
	PERFLIBS_LINALG_INLINE kernel_inttype cntg_interleave() const { return 1_ki; }
	PERFLIBS_LINALG_INLINE kernel_inttype cntg_interleave_step() const { return 1_ki; }
	PERFLIBS_LINALG_INLINE kernel_inttype strd_interleave_step() const { return 1_ki; }
	PERFLIBS_LINALG_INLINE kernel_inttype cntg_tile_step() const { return 2 * interleave_; }
	PERFLIBS_LINALG_INLINE kernel_inttype strd_tile_step() const { return strd_step_; }

	PERFLIBS_LINALG_INLINE expanded_complex_matrix sub_matrix(kernel_inttype cntg_pos,
			kernel_inttype cntg_size, kernel_inttype strd_pos, kernel_inttype strd_size) const;

	reference operator()(kernel_inttype cntg, kernel_inttype strd) const;
}; //class expanded_complex_matrix

template<typename T>
PERFLIBS_LINALG_INLINE
expanded_complex_matrix<T> expanded_complex_matrix<T>::sub_matrix(kernel_inttype cntg_pos,
		kernel_inttype cntg_size, kernel_inttype strd_pos, kernel_inttype strd_size) const {
	PERFLIBS_ASSERT(cntg_pos + cntg_size <= cntg(),
		"sub_matrix cntg dimensions outside of parent range");
	PERFLIBS_ASSERT(strd_pos + strd_size <= strd(),
		"sub_matrix strd dimensions outside of parent range");
	PERFLIBS_ASSERT(strd_pos % interleave_ == 0, "strd_pos must be divisible by interleave");

	return { data_ + get_elem_idx(cntg_pos, strd_pos), interleave_, cntg_size, strd_size,
			strd_step_ };
}

template<typename T>
PERFLIBS_LINALG_INLINE
typename expanded_complex_matrix<T>::reference
expanded_complex_matrix<T>::operator()(kernel_inttype cntg, kernel_inttype strd) const {
	const auto idx = get_elem_idx(cntg, strd);
	return { data_[idx], data_[idx + interleave_]};
}

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_EXPANDED_COMPLEX_MATRIX

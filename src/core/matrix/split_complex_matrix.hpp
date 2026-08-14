/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SPLIT_COMPLEX_MATRIX
#define PERFLIBS_LINALG_SPLIT_COMPLEX_MATRIX

#include "split_complex_ref.hpp"
#include "framework/linalg_util.hpp"

#include <perflibs_complex.hpp>

namespace perflibs::linalg {


/**
 * An interleave packed matrix format for complex numbers
 * instead of storing the number ririri, they are stored rrrriiiirrrriiii
 * where the block length is the interleave factor
 */
template<typename T>
class split_complex_matrix {
	static_assert(is_complex_v<T>, "type of split_complex_matrix must be complex");
public:
	using value_type       =T;
	using const_value_type =const T;
	using precision_t      =typename value_type::value_type;
	using reference        =split_complex_ref<precision_t>;
	using const_reference  =split_complex_ref<const precision_t>;
private:
	T *data_;
	kernel_inttype interleave_;
	kernel_inttype split_factor_;
	kernel_inttype cntg_;
	kernel_inttype strd_;
	kernel_inttype strd_step_;

	PERFLIBS_LINALG_INLINE
	std::pair<kernel_inttype, kernel_inttype> get_elem_idx(kernel_inttype in_cntg, kernel_inttype in_strd) const;

	/**
	 * @returns a pointer to the data as a real data type rather than complex
	 */
	precision_t *data_as_real() const { return reinterpret_cast<precision_t*>(data_); }
public:
	split_complex_matrix(T *data, kernel_inttype interleave, kernel_inttype split_factor, kernel_inttype cntg, kernel_inttype strd, kernel_inttype strd_step)
	:	data_         { data         }
	,	interleave_   { interleave   }
	,	split_factor_ { split_factor }
	,	cntg_         { cntg         }
	,	strd_         { strd         }
	,	strd_step_    { strd_step    }
	{	}

	PERFLIBS_LINALG_INLINE kernel_inttype    interleave()   const { return interleave_;   }
	PERFLIBS_LINALG_INLINE kernel_inttype    split_factor() const { return split_factor_; }
	PERFLIBS_LINALG_INLINE kernel_inttype    cntg()         const { return cntg_;         }
	PERFLIBS_LINALG_INLINE kernel_inttype    strd()         const { return strd_;         }
	PERFLIBS_LINALG_INLINE kernel_inttype    strd_step()    const { return strd_step_;    }
	PERFLIBS_LINALG_INLINE value_type       *data()         const { return data_;         }

	PERFLIBS_LINALG_INLINE kernel_inttype    strd_interleave()      const { return interleave_;   }
	PERFLIBS_LINALG_INLINE kernel_inttype    cntg_interleave()      const { return 1_ki;   }
	PERFLIBS_LINALG_INLINE kernel_inttype    cntg_interleave_step() const { return 1_ki;   }
	PERFLIBS_LINALG_INLINE kernel_inttype    strd_interleave_step() const { return 1_ki;   }
	PERFLIBS_LINALG_INLINE kernel_inttype    cntg_tile_step()       const { return strd_interleave(); }
	PERFLIBS_LINALG_INLINE kernel_inttype    strd_tile_step()       const { return strd_step();   }

	PERFLIBS_LINALG_INLINE split_complex_matrix sub_matrix(kernel_inttype cntg_pos, kernel_inttype cntg_size,
                                           kernel_inttype strd_pos, kernel_inttype strd_size) const;

	reference operator()(kernel_inttype cntg, kernel_inttype strd) const;
}; //class split_complex_matrix

template<typename T>
PERFLIBS_LINALG_INLINE
std::pair<kernel_inttype, kernel_inttype> split_complex_matrix<T>::get_elem_idx(kernel_inttype cntg, kernel_inttype strd) const {
	/*
	 * Gets the index of the first element of the interleaved row that strd_in designates
	 */
	const kernel_inttype strd_start_index = strd / interleave_ * strd_step_;

	/*
	 * Get the index of the interleaved block in that interleaved row
	 */
	const kernel_inttype cntg_start_index = cntg * interleave_;

	/*
	 * Get the index of the real component of the split pair in the interleaved block
	 */
	const kernel_inttype real_block_position = strd % interleave_ / split_factor_ * split_factor_ * 2 + strd % split_factor_;

	/*
	 * accumulate the indices denoting the block start and convert from complex<T> to T>
	 */
	const kernel_inttype real_index = 2 * ( strd_start_index +  cntg_start_index ) + real_block_position;

	/*
	 * Add the split factor to get the index of the imaginary component.
	 */
	const kernel_inttype imag_index = real_index + split_factor_;

	return { real_index, imag_index };
}

template<typename T>
PERFLIBS_LINALG_INLINE
split_complex_matrix<T> split_complex_matrix<T>::sub_matrix(kernel_inttype cntg_pos, kernel_inttype cntg_size,
                                kernel_inttype strd_pos, kernel_inttype strd_size) const {

	PERFLIBS_ASSERT((cntg_pos + cntg_size) <= cntg(), "sub_matrix cntg dimensions outside of parent range");
	PERFLIBS_ASSERT((strd_pos + strd_size) <= strd(), "sub_matrix strd dimensions outside of parent range");
	// This assert is not always met. This will be
	//               resolved when we generalize this matrix type.
	/* PERFLIBS_ASSERT(strd_pos % interleave_ == 0, "strd pos must be divisible by interleave"); */

	auto new_data = reinterpret_cast<value_type*>(
		data_as_real() + get_elem_idx(cntg_pos, strd_pos).first);

	return { new_data, interleave_, split_factor_, cntg_size, strd_size, strd_step_ };
}

template<typename T>
PERFLIBS_LINALG_INLINE
typename split_complex_matrix<T>::reference
split_complex_matrix<T>::operator()(kernel_inttype cntg, kernel_inttype strd) const {
	const auto [ r, i ] = get_elem_idx(cntg, strd);

	auto das = data_as_real();

	return { das[r], das[i] };
}

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_SPLIT_COMPLEX_MATRIX

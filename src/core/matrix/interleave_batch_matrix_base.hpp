/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_INTERLEAVE_BATCH_MATRIX_BASE
#define PERFLIBS_LINALG_INTERLEAVE_BATCH_MATRIX_BASE

#include "interleave_batch.hpp"
#include "framework/linalg_util.hpp"
#include "matrix_base.hpp"

namespace perflibs::linalg {

/**
 * This is base matrix type for interleave_batch operations (see PERFLIBS_ROOT/libraries/interleave_batch)
 *
 * When you index any element element in this matrix, rather than referring to an single scalar element
 * you are actually referencing a "batch" of elements the size of which is defined by the `batch_cntg`
 * param the matrix is initialized with
 *
 * We expect the elements to be stored contiguous in memory, these contiguous sequences of memory are
 * then traversed with a given strd & cntg step value - these values are given in terms of the underlying
 * values type (ie double, float, not interleave_batch_val<double> interleave_batch_ref<reference>
 *
 * So if you batch-cntg was 8, and is_cntg_contig(matrix) is true, cntg_step must be at least 8
 */
template<typename T>
class interleave_batch_matrix {
	T *data_;
	kernel_inttype batch_cntg_;
	kernel_inttype cntg_;
	kernel_inttype strd_;
	kernel_inttype cntg_step_;
	kernel_inttype strd_step_;
public:
	//The underlying data type of the operation, ie double, float
	using data_type            = T;
	//The type to which a reference of the batch can be assigned to which will then own that data
	using value_type           = interleave_batch_val<T>;
	using const_value_type     = const interleave_batch_val<T>;
	//A type which encapsulate matrix data as though it was a batch
	using reference            = interleave_batch_ref<T>;
	using const_reference      = interleave_batch_ref<const T>;

	/*
	 * Constructor
	 *
	 * @param data       [in,out] a pointer the data, values in this matrix may be modified via other class member functions
	 * @param batch_cntg [in]     the size of the batch in number of elements
	 * @param cntg       [in]     the size of the cntg dimension, for GEMM like operations cntg is the 'k' component (for both A & B) by convention - M for C
	 * @param strd       [in]     the size of the strd dimension, for GEMM like operations strd is the non-'k' component (for both A & B) by convention - N for C
	 * @param cntg_step  [in]     the distance between consequentive batches in the cntg dimension represented as number of data_type
	 * @param strd_step  [in]     the distance between consequentive batches in the strd dimension represented as number of data_type
	 *
	 * @note the "transposed" property of matrix is controlled by switching cntg_step and strd_step
	 */
	interleave_batch_matrix(T *data, kernel_inttype batch_cntg, kernel_inttype cntg, kernel_inttype strd, kernel_inttype cntg_step, kernel_inttype strd_step)
	:	data_       { data       }
	,	batch_cntg_ { batch_cntg }
	,	cntg_       { cntg       }
	,	strd_       { strd       }
	,	cntg_step_  { cntg_step  }
	,	strd_step_  { strd_step  }
	{	}

	/**
	 * @returns a pointer to the the first element of the first batch
	 */
	PERFLIBS_LINALG_INLINE       data_type *data()       { return data_; }
	PERFLIBS_LINALG_INLINE const data_type *data() const { return data_; }

	/**
	 * @param cntg position of the batch (not element)
	 * @param strd position of the batch (not element)
	 * @returns a pointer to the beginning of a batch
	 */
	PERFLIBS_LINALG_INLINE
	T *batch_data(kernel_inttype cntg, kernel_inttype strd)
	{ return data_ + cntg * cntg_step_ + strd * strd_step_; }

	PERFLIBS_LINALG_INLINE
	const T *batch_data(kernel_inttype cntg, kernel_inttype strd) const
	{ return data_ + cntg * cntg_step_ + strd * strd_step_; }

	/// interleaving factor
	PERFLIBS_LINALG_INLINE kernel_inttype batch_cntg() const { return batch_cntg_; }

	/// number of batches in the cntg dimension of the logical representation of the data
	PERFLIBS_LINALG_INLINE kernel_inttype cntg() const { return cntg_; }
	/// number of batches in the strd dimension of the logical representation of the data
	PERFLIBS_LINALG_INLINE kernel_inttype strd() const { return strd_; }

	/// The distance between consecutive elements in the cntg dimension
	PERFLIBS_LINALG_INLINE kernel_inttype cntg_step() const { return cntg_step_; }
	///The distance between consecutive elements in the strd dimension
	PERFLIBS_LINALG_INLINE kernel_inttype strd_step() const { return strd_step_; }

	PERFLIBS_LINALG_INLINE interleave_batch_matrix sub_matrix(kernel_inttype cntg_pos, kernel_inttype cntg_size,
                                                         kernel_inttype strd_pos, kernel_inttype strd_size) {
		PERFLIBS_ASSERT((cntg_pos + cntg_size) <= cntg(), "sub_matrix cntg dimensions outside of parent range");
		PERFLIBS_ASSERT((strd_pos + strd_size) <= strd(), "sub_matrix strd dimensions outside of parent range");

		return {
			batch_data(cntg_pos, strd_pos),
			batch_cntg_,
			cntg_size,
			strd_size
		};
	}

	PERFLIBS_LINALG_INLINE reference get_logical_elem(kernel_inttype cntg, kernel_inttype strd) const { return { data_ + cntg * cntg_step_ + strd * strd_step_, batch_cntg_ }; }

	PERFLIBS_LINALG_INLINE matrix_base<data_type> extract(kernel_inttype inter) {
		PERFLIBS_ASSERT(inter < batch_cntg_, "inter is out of range");
		auto d = data() + inter;
		return matrix_base<data_type> {
			d, cntg(), strd(), cntg_step(), strd_step()
		};
	}
}; //class interleave_batch_matrix

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_INTERLEAVE_BATCH_MATRIX_BASE

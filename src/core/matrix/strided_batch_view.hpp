/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATRIX_STRIDED_BATCH_VIEW_HPP
#define PERFLIBS_LINALG_MATRIX_STRIDED_BATCH_VIEW_HPP

#include "framework/linalg_util.hpp"

#include "perflibs_assert.hpp"

#include <utility>

namespace perflibs::linalg {

/**
 * A strided view over a batch of matrices with identical shape and layout.
 *
 * The stored matrix describes one problem. The batch step is expressed in
 * elements of the matrix's underlying data type.
 */
template<typename MatrixType>
class strided_batch_view {
	MatrixType matrix_;
	kernel_inttype batch_count_;
	kernel_inttype batch_step_;

public:
	using matrix_type       = MatrixType;
	using data_type         = typename matrix_type::data_type;
	using value_type        = typename matrix_type::value_type;
	using const_value_type  = typename matrix_type::const_value_type;
	using reference         = typename matrix_type::reference;
	using const_reference   = typename matrix_type::const_reference;

	PERFLIBS_LINALG_INLINE
	strided_batch_view(matrix_type matrix, kernel_inttype batch_count, kernel_inttype batch_step)
	: matrix_      { std::move(matrix) }
	, batch_count_ { batch_count       }
	, batch_step_  { batch_step        }
	{ }

	PERFLIBS_LINALG_INLINE const matrix_type& matrix() const { return matrix_; }

	PERFLIBS_LINALG_INLINE kernel_inttype batch_count() const { return batch_count_; }
	PERFLIBS_LINALG_INLINE kernel_inttype batch_step() const { return batch_step_; }

	PERFLIBS_LINALG_INLINE
	matrix_type operator[](kernel_inttype index) const {
		PERFLIBS_ASSERT(index >= 0 && index < batch_count(), "batch index is out of range");
		return matrix_.rebased(index * batch_step());
	}
}; // class strided_batch_view

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_MATRIX_STRIDED_BATCH_VIEW_HPP

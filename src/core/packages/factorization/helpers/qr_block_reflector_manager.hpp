/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_QR_BLOCK_REFLECTOR_MANAGER_HPP
#define PERFLIBS_LINALG_FACTORIZATION_QR_BLOCK_REFLECTOR_MANAGER_HPP

#include "matrix/adaptors.hpp"
#include "perflibs_util.hpp"
#include <vector>
#include <algorithm>

namespace perflibs::linalg::factorization {

template<typename T, typename ArchitectureSpec>
class block_reflector_manager {
private:
	kernel_inttype max_reflector_elts_;
	std::vector<T> memory_pool_;
	triangular_matrix<matrix_base<T>> matrix_a_;
	triangular_matrix<matrix_base<T>> matrix_b_;
	bool a_is_current_;

public:
	block_reflector_manager(kernel_inttype max_panel_size)
	:	max_reflector_elts_{ max_panel_size * max_panel_size }
	,	memory_pool_(2 * max_reflector_elts_)
	,	matrix_a_{ PERFLIBS_UPPER, PERFLIBS_NOUNIT, matrix_base<T>{ nullptr, 0, 0, 1, 0 } }
	,	matrix_b_{ PERFLIBS_UPPER, PERFLIBS_NOUNIT, matrix_base<T>{ nullptr, 0, 0, 1, 0 } }
	,	a_is_current_{ true }
	{	}

	auto& get_current_reflector() {
		return a_is_current_ ? matrix_a_ : matrix_b_;
	}

	const auto& get_previous_reflector() const {
		return a_is_current_ ? matrix_b_ : matrix_a_;
	}

	void allocate_current_reflector(kernel_inttype panel_width) {
		T *memory_start = memory_pool_.data() + (a_is_current_ ? 0 : max_reflector_elts_);

		get_current_reflector() = triangular_matrix<matrix_base<T>>(
		    PERFLIBS_UPPER, PERFLIBS_NOUNIT,
		    matrix_base<T>{ memory_start, panel_width, panel_width, 1, panel_width });
	}

	void update_previous_reflectors() {
		a_is_current_ = !a_is_current_;
	}
};


template<typename T, typename ArchitectureSpec>
class householder_block_manager {
private:
	kernel_inttype max_panel_length_;
	kernel_inttype max_panel_width_;
	kernel_inttype max_block_elts_;
	std::vector<T> memory_pool_;
	general_matrix<matrix_base<T>> matrix_a_;
	general_matrix<matrix_base<T>> matrix_b_;
	bool a_is_current_;

public:
	householder_block_manager(kernel_inttype max_panel_length, kernel_inttype max_panel_width)
	:	max_panel_length_{ max_panel_length }
	,	max_panel_width_{ max_panel_width }
	,	max_block_elts_{ max_panel_length * max_panel_width }
	,	memory_pool_(2 * max_block_elts_)
	,	matrix_a_{ matrix_base<T>{ nullptr, 0, 0, 1, 1 } }
	,	matrix_b_{ matrix_base<T>{ nullptr, 0, 0, 1, 1 } }
	,	a_is_current_{ true }
	{	}

	auto& get_current_block() {
		return a_is_current_ ? matrix_a_ : matrix_b_;
	}

	const auto& get_previous_block() const {
		return a_is_current_ ? matrix_b_ : matrix_a_;
	}

	void allocate_current_block(kernel_inttype panel_length, kernel_inttype panel_width) {
		// Ensure we don't exceed the maximum allocated space
		assert(panel_length <= max_panel_length_ && panel_width <= max_panel_width_);

		T* memory_start = memory_pool_.data() + (a_is_current_ ? 0 : max_block_elts_);

		get_current_block() = general_matrix<matrix_base<T>>(
			matrix_base<T>{ memory_start, panel_length, panel_width, 1, panel_length }
		);
	}

	// Copy from source matrix to current block storage
	void copy_to_current_block(const general_matrix<matrix_base<T>>& source) {
		auto& current = get_current_block();
		copy(source, current);
	}

	void update_previous_blocks() {
		a_is_current_ = !a_is_current_;
	}

	// Get the actual dimensions of the previous block
	kernel_inttype get_previous_panel_length() const {
		return get_previous_block().cntg();
	}

	kernel_inttype get_previous_panel_width() const {
		return get_previous_block().strd();
	}
};

} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_QR_BLOCK_REFLECTOR_MANAGER_HPP
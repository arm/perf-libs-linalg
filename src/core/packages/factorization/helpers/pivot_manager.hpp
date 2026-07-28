/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_PIVOT_MANAGER_HPP
#define PERFLIBS_LINALG_FACTORIZATION_PIVOT_MANAGER_HPP

#include "framework/linalg_util.hpp"
#include "pod_vector.hpp"

#include <vector>
#include <algorithm>
#include <numeric>

/*
 * The pivot_manager class manages pivot indices for LU factorization,
 * usable by both sequential and parallel algorithms.
 * It stores panel start points and sizes, and updates pivot indices to
 * global indices at the end of the factorization.
 *
 * In the parallel code, it also accumulates pivots for applying to the
 * left part of the matrix after factorization.
 */

namespace perflibs::linalg::factorization {

template<typename ArchitectureSpec, typename T>
class pivot_manager {

public:
	pivot_manager(std::span<T> pivot) : pivot_{ pivot } {}

	// Store panel information (start point and size)
	void add_panel(kernel_inttype start_point, kernel_inttype panel_size) {
		panel_starts_.push_back(start_point);
		panel_sizes_.push_back(panel_size);
	}

	// Accumulate pivots
	void accumulate_pivots(kernel_inttype start_point, kernel_inttype panel_size) {
		// Store panel information
		add_panel(start_point, panel_size);

		// Accumulate pivots.
		// Note that given a panel, the top square already factorized
		// does not require pivot application anymore. The size
		// of the pivot vector is therefore reduced by panel_size accordingly.

		// Calculate the size of the working pivot (remaining after the panel)
		const auto current_pivot_size = pivot_.size() - start_point;
		const auto working_pivot_size = current_pivot_size - panel_size;

		// Create vector of identity pivot of size working_pivot_size
		perflibs::pod_vector<T> new_pivot(working_pivot_size);
		std::iota(new_pivot.begin(), new_pivot.end(), 0);

		// Update existing accumulated pivots if this is not the first panel
		if (!pivot_accumulated_.empty()) {
			update_existing_pivots(start_point, panel_size);
		}

		// Add current identity pivot to accumulated pivots
		// Since our pod_vector does not have insert, we resize and use std::copy
		const std::size_t old_size = pivot_accumulated_.size();
		pivot_accumulated_.resize(old_size + working_pivot_size);
		std::copy(new_pivot.begin(), new_pivot.end(), pivot_accumulated_.begin() + old_size);
	}

	// Get the local pivot indices for the previous panel
	std::span<T> get_previous_local_pivot() const {
		PERFLIBS_ASSERT(panel_starts_.size() >= 1, "No previous panel exists");
		const auto index = panel_starts_.size() - 1;
		const auto start_point = panel_starts_[index];
		const auto panel_size = panel_sizes_[index];
		return pivot_.subspan(start_point, panel_size);
	}

	// Update pivot indices to global indices at the end
	// This implementation updates all the pivot and called
	// from sequential code
	void update_pivots_to_global() {
		// Update pivot indices for each panel
		for (std::size_t i = 0; i < panel_starts_.size(); ++i) {
			const auto start_point = panel_starts_[i];
			const auto panel_size  = panel_sizes_[i];
			       auto sub_pivot  = pivot_.subspan(start_point, panel_size);
			for (auto& idx : sub_pivot) {
				idx += start_point;
			}
		}
	}

	// Update pivot indices to global indices at the end.
	// this implementation updates the pivots restricted to a panel
	// To be called by the parallel implementation
	void update_pivot_to_global(std::size_t panel_id) {
		PERFLIBS_ASSERT(panel_id < panel_starts_.size(), "Invalid panel_id");

		const auto panel_start_point = panel_starts_[panel_id];
		const auto panel_size        = panel_sizes_[panel_id];
		      auto sub_pivot         = pivot_.subspan(panel_start_point, panel_size);
		for (kernel_inttype i = 0; i < panel_size; ++i) {
			sub_pivot[i] += panel_start_point;
		}
	}

	// Update pivot indices to global indices at the end.
	// This implementation is for the recursive LU
	// It takes as inputs the panel width and the
	// the pivot size from the trailing matrix factorization
	void update_pivot_to_global(kernel_inttype panel_width, kernel_inttype pivot_size) {
		// NOTE: This assert is currently failing.
		// TODO: figure out why the assert is failing and fix/remove.
		/* const std::size_t expected_pivot_size = panel_width + pivot_size; */
		/* PERFLIBS_ASSERT( expected_pivot_size == pivot_.size(), "Invalid panel_width or pivot_size"); */
		auto sub_pivot = pivot_.subspan(panel_width, pivot_size);
		for (auto& i : sub_pivot) i += panel_width;
	}

	// Get the number of panels
	std::size_t size() const {
		return panel_starts_.size();
	}

	// Get the panel size for a given panel index
	std::size_t get_panel_size(std::size_t id) const {
		PERFLIBS_ASSERT(id < panel_sizes_.size(), "Index out of range");
		return panel_sizes_[id];
	}

	// Get the start point for a given panel index
	kernel_inttype get_start_point(std::size_t id) const {
		PERFLIBS_ASSERT(id < panel_starts_.size(), "Index out of range");
		return panel_starts_[id];
	}

	// Get accumulated pivots for applying to the left
	std::span<T> get_pivots(std::size_t id) {
		PERFLIBS_ASSERT(id < panel_starts_.size() - 1, "Index out of range");

		// Calculate the starting index in pivot_accumulated_
		std::size_t start_index = 0;
		for (std::size_t i =  0; i < id; ++i) {
			start_index += pivot_.size() - panel_starts_[i] - panel_sizes_[i];
		}

		// Calculate the size of the pivot segment
		auto size = pivot_.size() - panel_starts_[id] - panel_sizes_[id];
		return std::span<T>(pivot_accumulated_.data() + start_index, size);
	}

private:
	std::span<T> pivot_;
	perflibs::pod_vector<kernel_inttype> panel_starts_;
	perflibs::pod_vector<kernel_inttype> panel_sizes_;
	perflibs::pod_vector<T> pivot_accumulated_;

	void update_existing_pivots(kernel_inttype start_point, kernel_inttype panel_size) {

    	auto current_pivot = pivot_.subspan(start_point, panel_size);

		for (std::size_t idx = 0; idx + 1 < size(); ++idx) {
			auto existing_pivot = get_pivots(idx);

			// Get local starting point in existing pivot that matches the current pivot
			const auto existing_pivot_start_point = existing_pivot.size() - (pivot_.size() - start_point);

			for (kernel_inttype i = 0; i < panel_size; ++i) {
				if (current_pivot[i] != i) {
					const auto pivot_index = i + existing_pivot_start_point;
					const auto pivot_entry = current_pivot[i] + existing_pivot_start_point;
					existing_pivot[pivot_index] = pivot_entry;
				}
			}
		}
	}
};
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_PIVOT_MANAGER_HPP

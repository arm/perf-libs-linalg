/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_MATRIX_DISTRIBUTION_HPP
#define PERFLIBS_LINALG_FACTORIZATION_MATRIX_DISTRIBUTION_HPP

#include "perflibs_util.hpp"
#include <utility>
#include <algorithm>
#include <cmath>

namespace perflibs::linalg::factorization {

// This function calculates the range of columns each thread should process
// in a parallel trailing matrix update for QR or LU factorization
// Thread 0 has additional responsibilities (panel factorization)
// and may contribute less to the trailing matrix update.
// The goal is to balance the workload among threads.
//
// - total_columns: Total number of columns in the trailing matrix
// - panel_size: Size of the panel to factorized
// - num_threads: Total number of available threads
// - thread_id: ID of the current thread (0 to num_threads-1)
// - min_columns_per_thread: Minimum number of columns that each thread should process
// - weight_factor: Factor to account for the extra work done by thread 0.
//                  Thread 0 contribute to the trailing matrix update only if the trailing matrix
//                  is large enough so that the number of columns distritubed by per thread
//                  exceeds weight_factor * panel_size.
template<typename ArchitectureSpec>
std::pair<kernel_inttype, kernel_inttype> distribute_trailing_matrix_update(
	kernel_inttype total_columns,
	kernel_inttype panel_size,
	kernel_inttype num_threads,
	kernel_inttype thread_id,
	kernel_inttype min_columns_per_thread,
	double weight_factor) {


	// Handle the case where panel_size is 0
	// This is the case where the matrix
	// is already fully factorized by we need an
	// extra iteration to update the trailing matrix.
	if (panel_size == 0) {
		// If total_columns is less than min_columns_per_thread, only thread 0 gets the work
		if (total_columns < min_columns_per_thread) {
			if (thread_id == 0) {
				return {0, total_columns};
			}
			else {
				return {total_columns, total_columns};
			}
		}

		const auto active_threads = min(num_threads, total_columns / min_columns_per_thread);

		if (thread_id >= active_threads) {
			return {total_columns, total_columns};
		}

		const auto columns_per_thread = total_columns / active_threads;
		const auto leftover_columns = total_columns % active_threads;

		const auto start = thread_id * columns_per_thread + min(thread_id, leftover_columns);
		const auto end = start + columns_per_thread + (thread_id < leftover_columns ? 1 : 0);

		return {start, min(end, total_columns)};
	}

	// Calculate the weighted panel size,
	//rounding up to ensure integer values
	const auto weighted_panel_size = static_cast<kernel_inttype>(std::ceil(weight_factor * panel_size));

	// Calculate remaining columns after the panel.
	// This is important as the current panel is statically
	// mapped to thread 0, it is the remaining that should be distributed
	const auto remaining_columns = total_columns - panel_size;

	// Calculate the workload for threads other than thread 0
	const auto other_threads_workload = weighted_panel_size * (num_threads - 1);

	// Determine if thread 0 contribute to trailing matrix update
	const bool thread0_contributes = remaining_columns > other_threads_workload;

	// Calculate extra work if thread 0 contributes
	auto extra_work = thread0_contributes ? remaining_columns - other_threads_workload : 0;

	if (thread0_contributes) {
		// Adjust the extra work to account for rounding mismatch
		extra_work += (weighted_panel_size - static_cast<kernel_inttype>(weight_factor * panel_size)) * (num_threads - 1);

		// Distribute extra work among all threads
		const auto extra_columns_per_thread = extra_work / num_threads;
		const auto extra_columns_leftover   = extra_work % num_threads;

		if (thread_id == 0) {
			// Thread 0 gets its portion of extra work plus panel factorization
			return {panel_size, panel_size + extra_columns_per_thread + (extra_columns_leftover > 0 ? 1 : 0)};
		}

		// Calculate start position for other threads
		const auto base_start = panel_size + extra_columns_per_thread + (extra_columns_leftover > 0 ? 1 : 0);

		// Calculate columns assigned to this thread
		const auto thread_assigned_columns = weighted_panel_size + extra_columns_per_thread + (thread_id <= extra_columns_leftover ? 1 : 0);

		// Calculate start position for this thread
		const auto start = base_start + (thread_id - 1) * (weighted_panel_size + extra_columns_per_thread) + min(thread_id - 1, extra_columns_leftover);

		return {start, min(start + thread_assigned_columns, total_columns)};
	}
	else {
		// If thread 0 do not contribute, it only handles the panel
		if (thread_id == 0) return {panel_size, panel_size};

		// Calculate the number of active threads based on remaining work
		const auto active_threads = min(num_threads - 1, (remaining_columns + min_columns_per_thread - 1) / min_columns_per_thread);

		// If this thread is not needed, return an empty range
		if (thread_id > active_threads) return {total_columns, total_columns};

		// Distribute remaining columns among active threads
		const auto base_columns_per_thread = remaining_columns / active_threads;
		const auto leftover_columns = remaining_columns % active_threads;

		// Calculate start position for this thread
		const auto start = panel_size + (thread_id - 1) * base_columns_per_thread + min(thread_id - 1, leftover_columns);

		// Calculate end position for this thread
		const auto end = start + base_columns_per_thread + (thread_id <= leftover_columns ? 1 : 0);

		return {start, min(end, total_columns)};
	}
}
// This function is specifically designed for distributing the work of applying
// block reflectors in parallel, where thread 0 has additional work of generating
// the next reflector but can still participate in applying the current reflector.
//
// - total_size: Total number of columns/rows to process in the application
// - reflector_size: Size of the reflector being applied
// - num_threads: Total number of available threads
// - thread_id: ID of the current thread (0 to num_threads-1)
// - min_work_per_thread: Minimum amount of work each thread should receive
// - weight_factor: Ratio between reflector generation cost and applying it to one column/row
//                  Higher values mean reflector generation is more expensive relative to application
template<typename ArchitectureSpec>
std::pair<kernel_inttype, kernel_inttype> distribute_reflector_application_work(
	kernel_inttype total_size,
	kernel_inttype reflector_size,
	kernel_inttype num_threads,
	kernel_inttype thread_id,
	kernel_inttype min_work_per_thread,
	double weight_factor) {

	// Handle trivial case of no work
	if (total_size <= 0) {
		return {0, 0};
	}

	// Calculate how much application work is equivalent to generating a reflector
	// This is based on the weight factor and reflector size
	const auto reflector_gen_cost = static_cast<kernel_inttype>(std::ceil(weight_factor * reflector_size));

	// Calculate how much work other threads can perform while thread 0 generates the reflector
	const auto other_threads_capacity = reflector_gen_cost * (num_threads - 1);

	// Determine if thread 0 will need to help with application work
	const bool thread0_helps = (total_size > other_threads_capacity);

	// If there's not enough work or thread 0 doesn't need to help
	if (!thread0_helps) {
		// Thread 0 only generates the reflector (no application work)
		if (thread_id == 0) {
			return {0, 0};
		}

		// Calculate the number of active threads based on remaining work
		const auto active_threads = min(num_threads - 1, max(1_ki, total_size / min_work_per_thread));

		// If this thread is not needed, return an empty range
		if (thread_id > active_threads) {
			return {total_size, total_size};
		}

		// Distribute work among active threads (excluding thread 0)
		const auto work_per_thread = total_size / active_threads;
		const auto leftover_work   = total_size % active_threads;

		// Adjust for thread numbering (thread 1 is the first application thread)
		const auto adj_thread_id = thread_id - 1;

		const auto start = adj_thread_id * work_per_thread + min(adj_thread_id, leftover_work);
		const auto end = start + work_per_thread + (adj_thread_id < leftover_work ? 1 : 0);

		return {start, min(end, total_size)};
	}
	else {
		// We're in the case where there's more work than other threads can handle
		// Thread 0 will generate the reflector and then help with application

		// Calculate the excess work (beyond what other threads can handle during generation)
		const auto excess_work = total_size - other_threads_capacity;

		// Distribute this excess work among all threads
		const auto excess_per_thread = excess_work / num_threads;
		const auto excess_remainder = excess_work % num_threads;

		if (thread_id == 0) {
			// Thread 0 gets its portion of the excess work
			const auto thread0_work = excess_per_thread + (0 < excess_remainder ? 1 : 0);
			return {0, thread0_work};
		}

		// For other threads, they get their portion during reflector generation
		// plus their share of the excess
		const auto base_work = reflector_gen_cost;
		const auto excess_portion      = excess_per_thread + (thread_id < excess_remainder ? 1 : 0);

		// Calculate the starting point
		// Each thread except 0 starts at the end of thread 0's work, plus previous threads' work
		const auto thread0_work = excess_per_thread + (0 < excess_remainder ? 1 : 0);

		auto preceding_extra = 0_ki;
		if (excess_remainder > 0) {
			// only threads with id < excess_remainder actually get the extra, and
			// we exclude thread 0 from this count in the "preceding threads" block:
			preceding_extra = std::max<kernel_inttype>(0, std::min(thread_id, excess_remainder) - 1);
		}
		const auto preceding_threads_work = (thread_id - 1) * (base_work + excess_per_thread) + preceding_extra;

		const auto start = thread0_work + preceding_threads_work;
		const auto end   = start + base_work + excess_portion;

		return {start, min(end, total_size)};
	}
}
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_MATRIX_DISTRIBUTION_HPP
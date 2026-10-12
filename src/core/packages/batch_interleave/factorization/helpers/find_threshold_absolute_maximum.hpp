/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_HELPERS_FIND_THRESHOLD_ABSOLUTE_MAXIMUM_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_HELPERS_FIND_THRESHOLD_ABSOLUTE_MAXIMUM_HPP

#include "matrix/interleave_batch_matrix.hpp"

#include <cmath>
#include <limits>

namespace perflibs::linalg::batch_interleave::factorization {

/**
 * Find the first value in each interleaved lane whose absolute value reaches
 * a threshold, or the lane's absolute maximum if no value reaches it.
 */
template<typename T, typename MaximumRange, typename PositionRange>
PERFLIBS_LINALG_INLINE
void find_threshold_absolute_maximum(
	const general_interleave_batch_matrix<const T>& x, T threshold,
	MaximumRange& maximum, PositionRange& position) {
	using position_type = typename PositionRange::value_type;

	const auto ninter = x.interleave_count();
	if (ninter == 0) return;

	const auto absolute_threshold = std::abs(threshold);
	const auto first_values = x.get_logical_elem(0, 0);
	bool all_lanes_reached_threshold = true;

	for (kernel_inttype lane = 0; lane < ninter; lane++) {
		maximum[lane] = std::abs(first_values[lane]);
		position[lane] = position_type(0);
		all_lanes_reached_threshold &= maximum[lane] >= absolute_threshold;
	}
	if (all_lanes_reached_threshold) return;

	for (kernel_inttype index = 1; index < x.cntg(); index++) {
		const auto values = x.get_logical_elem(index, 0);
		for (kernel_inttype lane = 0; lane < ninter; lane++) {
			if (maximum[lane] >= absolute_threshold) continue;

			const auto value = std::abs(values[lane]);
			if (value > maximum[lane]) {
				maximum[lane] = value;
				position[lane] = static_cast<position_type>(index);
			}
		}
	}
}

/** Find the absolute maximum in each interleaved lane. */
template<typename T, typename MaximumRange, typename PositionRange>
PERFLIBS_LINALG_INLINE
void find_threshold_absolute_maximum(
	const general_interleave_batch_matrix<const T>& x,
	MaximumRange& maximum, PositionRange& position) {
	find_threshold_absolute_maximum(x, std::numeric_limits<T>::max(), maximum, position);
}

} // namespace perflibs::linalg::batch_interleave::factorization

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_HELPERS_FIND_THRESHOLD_ABSOLUTE_MAXIMUM_HPP

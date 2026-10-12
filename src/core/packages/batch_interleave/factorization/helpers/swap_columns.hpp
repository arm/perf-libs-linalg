/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_HELPERS_SWAP_COLUMNS_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_HELPERS_SWAP_COLUMNS_HPP

#include "framework/linalg_util.hpp"
#include "matrix/interleave_batch_matrix.hpp"

#include <span>
#include <utility>

namespace perflibs::linalg::batch_interleave::factorization {

/**
 * Swap a source column with a lane-specific target column, including its
 * pivot and column-norm metadata.
 */
template<typename T, typename IntType>
PERFLIBS_LINALG_INLINE
void swap_columns(
	const general_interleave_batch_matrix<T>& a,
	const general_interleave_batch_matrix<IntType>& pivot,
	const general_interleave_batch_matrix<T>& norm,
	std::span<const IntType> target_columns,
	kernel_inttype source_column) {
	PERFLIBS_ASSERT(
		a.strd() == pivot.cntg() && a.strd() == norm.cntg() &&
		pivot.strd() == 1 && norm.strd() == 1,
		"column-swap matrices have incompatible dimensions");
	PERFLIBS_ASSERT(
		a.interleave_count() == pivot.interleave_count() &&
		a.interleave_count() == norm.interleave_count() &&
		a.interleave_count() == static_cast<kernel_inttype>(target_columns.size()),
		"column-swap matrices have incompatible interleave counts");
	PERFLIBS_ASSERT(
		source_column >= 0 && source_column < a.strd(),
		"column-swap source column is out of range");

	const auto ninter = a.interleave_count();

	for (kernel_inttype lane = 0; lane < ninter; lane++) {
		const auto target = target_columns[lane];

		PERFLIBS_ASSERT(
			target >= 0 && target < a.strd(),
			"column-swap target column is out of range");

		std::swap(pivot(source_column, 0, write)(lane),
		          pivot(target, 0, write)(lane));
		std::swap(norm(source_column, 0, write)(lane),
		          norm(target, 0, write)(lane));

		for (kernel_inttype row = 0; row < a.cntg(); row++) {
			std::swap(a(row, source_column, write)(lane),
			          a(row, target, write)(lane));
		}
	}
}

} // namespace perflibs::linalg::batch_interleave::factorization

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_HELPERS_SWAP_COLUMNS_HPP

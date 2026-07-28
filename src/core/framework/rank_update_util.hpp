/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef RANK_UPDATE_UTIL_HPP
#define RANK_UPDATE_UTIL_HPP

#include "framework/linalg_util.hpp"
#include "matrix/matrix.hpp"
#include "matrix/type_traits.hpp"

#include <array>

namespace perflibs::linalg {

/**
 * Gets four parameters required for computing rank updates with triangular matrices.
 *
 * @tparam MatrixType The type of matrix `c`.
 * @param [in] c The matrix to get the parameters for.
 * @return A std::array of length four, where:
 * - index 0 is the 'switch point' -- this is the point where we start changing our bounds in the driver
 *   algorithm to match the shape of the matrix.
 * - index 1 is cntg_last -- this is the upper bound for the current stretch of values within the matrix
 *   that we're considering.
 * - index 2 is cntg_first_step -- this is the cntg step after we reach the switch point.
 * - index 3 is cntg_last_step -- this is the step for cntg_last before we reach the switch point.
 */
template<typename MatrixType>
PERFLIBS_LINALG_INLINE std::array<kernel_inttype, 4>
get_triangular_rank_update_bounding_params(const MatrixType &c) {
	if constexpr (!is_triangular_form_v<MatrixType>) {
		return { c.strd(), c.cntg(), 0, 0 };
	}
	else if (c.is_upper()) {
		auto [strd_first, _] = get_non_virtual_strd_bounds_for_cntg(c, c.cntg() - 1);
		PERFLIBS_UNUSED(_);
		return { strd_first, c.cntg() - strd_first, 0, 1 };
	}
	else {
		auto [_, strd_last] = get_non_virtual_strd_bounds_for_cntg(c, 0);
		PERFLIBS_UNUSED(_);
		return { strd_last - 1, c.cntg(), 1, 0 };
	}
}

}; // namespace perflibs::linalg

#endif /* ifndef RANK_UPDATE_UTIL_HPP */

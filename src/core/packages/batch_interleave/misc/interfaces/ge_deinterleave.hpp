/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_MISC_INTERFACES_GE_DEINTERLEAVE_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_MISC_INTERFACES_GE_DEINTERLEAVE_HPP

#include "pl_linalg_status.h"
#include "matrix/matrix.hpp"

namespace perflibs::linalg::batch_interleave::misc {

/**
 * This function takes as input an interleaved matrix src and extracts the matrix in interleaved position inter into the dst buffer.
 *
 * @param ninter      [in] The number of matrices interleaved.
 * @param inter       [in] The index of the matrix.
 * @param m           [in] The number of rows of each matrix.
 * @param n           [in] The number of columns of each matrix.
 * @param dst         [out] Pointer to the output matrix.
 * @param istrdDst    [in] The stride between successive elements in the same column of the output matrix.
 * @param jstrdDst    [in] The stride between successive elements in the same row of the output matrix.
 * @param src         [in] Pointer to the start of the interleaved matrices.
 * @param istrdSrc    [in] The stride between successive elements in the same column of the same matrix in the interleaved input.
 * @param jstrdSrc    [in] The stride between successive elements in the same row of the same matrix in the interleaved input.
 * @return Success/fail status.
 */
template<typename IntType, typename T, typename ArchitectureSpec = void>
pl_linalg_status_t ge_deinterleave(IntType ninter, IntType inter,
                               IntType m, IntType n,
                               T *dst, IntType istrdDst, IntType jstrdDst,
                               const T *src, IntType istrdSrc, IntType jstrdSrc) {

	if (m == 0 || n == 0 || ninter == 0) {
		return PL_LINALG_STATUS_SUCCESS;
	}

	if (m < 0 || n < 0 || ninter < 0 || inter < 0) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	if (inter >= ninter) return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;

	if (istrdDst < 1 || jstrdDst < 1) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	// Check we always have `ninter` contiguous elements in interleaved batch
	if (istrdSrc < ninter || jstrdSrc < ninter) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	const general_matrix srcm { matrix_base { src + inter, m, n, istrdSrc, jstrdSrc } };
	// copy() selects its general-matrix path only for a non-const destination view.
	general_matrix dstm { matrix_base { dst, m, n, istrdDst, jstrdDst } };

	copy(srcm, dstm);

	return PL_LINALG_STATUS_SUCCESS;
}

} // namespace perflibs::linalg::batch_interleave::misc

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_MISC_INTERFACES_GE_DEINTERLEAVE_HPP

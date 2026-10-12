/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_MISC_INTERFACES_GE_INTERLEAVE_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_MISC_INTERFACES_GE_INTERLEAVE_HPP

#include "pl_linalg_status.h"
#include "matrix/matrix.hpp"

namespace perflibs::linalg::batch_interleave::misc {

/**
 * This function takes as input a standard matrix and places it into an existing batch of interleaved matrices.
 * The input matrix is placed into interleaving position inter.
 *
 * @param ninter   [in] The number of matrices to interleave.
 * @param inter    [in] The index of the matrix in the interleaving.
 * @param m        [in] The number of rows of each matrix.
 * @param n        [in] The number of columns of each matrix.
 * @param src      [in] Pointer to the start of the input matrix.
 * @param istrdSrc [in] The stride between successive elements in the same column of the input matrix.
 * @param jstrdSrc [in] The stride between successive elements in the same row of the input matrix.
 * @param dst      [out] Pointer to the start of the interleaved matrices.
 * @param istrdDst [in] The stride between successive elements in the same column of the same matrix in the interleaved output.
 * @param jstrdDst [in] The stride between successive elements in the same row of the same matrix in the interleaved output.
 * @return Success/fail status.
 */
template<typename IntType, typename T, typename ArchitectureSpec = void>
pl_linalg_status_t ge_interleave(IntType ninter, IntType inter,
                             IntType m, IntType n,
                             const T *src, IntType istrdSrc, IntType jstrdSrc,
                                   T *dst, IntType istrdDst, IntType jstrdDst) {

	if (m == 0 || n == 0 || ninter == 0) {
		return PL_LINALG_STATUS_SUCCESS;
	}

	if (m < 0 || n < 0 || ninter < 0 || inter < 0) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	if (inter >= ninter) return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;

	if (istrdSrc < 1 || jstrdSrc < 1) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	// Check we always have `ninter` contiguous elements in interleaved batch
	if (istrdDst < ninter || jstrdDst < ninter) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	const general_matrix srcm { matrix_base { src, m, n, istrdSrc, jstrdSrc } };
	// copy() selects its general-matrix path only for a non-const destination view.
	general_matrix dstm { matrix_base { dst + inter, m, n, istrdDst, jstrdDst } };

	copy(srcm, dstm);

	return PL_LINALG_STATUS_SUCCESS;
}

} // namespace perflibs::linalg::batch_interleave::misc

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_MISC_INTERFACES_GE_INTERLEAVE_HPP

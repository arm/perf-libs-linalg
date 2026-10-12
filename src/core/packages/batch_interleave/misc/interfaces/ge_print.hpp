/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_MISC_INTERFACES_GE_PRINT_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_MISC_INTERFACES_GE_PRINT_HPP

#include "pl_linalg_status.h"
#include "matrix/matrix.hpp"
#include "matrix/print.hpp"

#include <iostream>

namespace perflibs::linalg::batch_interleave::misc {

/**
 * Print one matrix from an interleaved batch to standard output.
 *
 * @param ninter         [in] The number of interleaved matrices in each batch.
 * @param inter          [in] The zero-based index of the matrix within the batch.
 * @param nbatch         [in] The number of batches.
 * @param batch          [in] The zero-based index of the batch to print.
 * @param m              [in] The number of rows of each matrix.
 * @param n              [in] The number of columns of each matrix.
 * @param src            [in] Pointer to the start of the interleaved batches.
 * @param batch_step_src [in] The stride between batches, in elements.
 * @param istrdSrc       [in] The stride between rows of the same matrix, in elements.
 * @param jstrdSrc       [in] The stride between columns of the same matrix, in elements.
 * @return Success/fail status. Empty inputs produce no output.
 */
template<typename IntType, typename T, typename ArchitectureSpec = void>
pl_linalg_status_t batch_interleave_ge_print(IntType ninter, IntType inter, IntType nbatch, IntType batch,
                                         IntType m, IntType n, const T *src, IntType batch_step_src,
                                         IntType istrdSrc, IntType jstrdSrc) {

	if (ninter < 0 || inter < 0 || nbatch < 0 || batch < 0 || m < 0 || n < 0) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	if (ninter == 0 || nbatch == 0 || m == 0 || n == 0) {
		return PL_LINALG_STATUS_SUCCESS;
	}

	if (inter >= ninter || batch >= nbatch) {
		return PL_LINALG_STATUS_INPUT_PARAMETER_ERROR;
	}

	const auto src_batch_ptr = src + kernel_inttype { batch } * batch_step_src + inter;
	const general_matrix srcm { matrix_base { src_batch_ptr, m, n, istrdSrc, jstrdSrc } };

	print(std::cout, srcm);
	return PL_LINALG_STATUS_SUCCESS;
}

} // namespace perflibs::linalg::batch_interleave::misc

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_MISC_INTERFACES_GE_PRINT_HPP

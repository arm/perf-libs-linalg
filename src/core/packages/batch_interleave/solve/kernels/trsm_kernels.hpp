/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_SOLVE_KERNELS_TRSM_KERNELS_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_SOLVE_KERNELS_TRSM_KERNELS_HPP

#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

template<typename ADataType, typename BDataType = ADataType,
         typename ScalarType = promote_t<ADataType, BDataType>>
using batch_interleave_trsm_kernel_t = void(
	kernel_inttype ninter,
	const ADataType *a,
	kernel_inttype a_row_step,
	kernel_inttype a_column_step,
	BDataType *b,
	kernel_inttype b_row_step,
	kernel_inttype b_column_step,
	kernel_inttype m,
	kernel_inttype n,
	bool nounit,
	ScalarType alpha);

} // namespace perflibs::linalg

namespace perflibs::interleave_batch {
extern ::perflibs::linalg::batch_interleave_trsm_kernel_t<double> dtrsm_neon_kernel;
extern ::perflibs::linalg::batch_interleave_trsm_kernel_t<double> dtrsm_sve_kernel;
} // namespace perflibs::interleave_batch

namespace perflibs::linalg::batch_interleave::solve {
using ::perflibs::interleave_batch::dtrsm_neon_kernel;
using ::perflibs::interleave_batch::dtrsm_sve_kernel;
} // namespace perflibs::linalg::batch_interleave::solve

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_SOLVE_KERNELS_TRSM_KERNELS_HPP

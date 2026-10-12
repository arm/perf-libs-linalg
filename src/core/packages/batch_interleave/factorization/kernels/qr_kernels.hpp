/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_KERNELS_QR_KERNELS_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_KERNELS_QR_KERNELS_HPP

#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"
#include "matrix/interleave_batch_matrix.hpp"
#include "naive/qr.hpp"

namespace perflibs::linalg {

template<typename T>
using gemv_qr_kernel_t = void(
	kernel_inttype ninter, kernel_inttype m, kernel_inttype n,
	const T *a, kernel_inttype a_cntg_step, kernel_inttype a_strd_step,
	const T *x, kernel_inttype x_cntg_step, T *y, kernel_inttype y_cntg_step);

template<typename T>
using ger_qr_kernel_t = void(
	kernel_inttype ninter, kernel_inttype m, kernel_inttype n,
	const T *alpha, const T *x, kernel_inttype x_cntg_step,
	const T *y, kernel_inttype y_cntg_step,
	T *a, kernel_inttype a_cntg_step, kernel_inttype a_strd_step);

namespace {

template<typename T>
PERFLIBS_LINALG_INLINE
void gemv_qr_fallback(
	kernel_inttype ninter, kernel_inttype m, kernel_inttype n,
	const T *a, kernel_inttype a_cntg_step, kernel_inttype a_strd_step,
	const T *x, kernel_inttype x_cntg_step, T *y, kernel_inttype y_cntg_step) {
	const auto a_matrix = general_matrix {
		interleave_batch_matrix { a, ninter, n, m, a_cntg_step, a_strd_step }
	};
	const auto x_matrix = general_matrix {
		interleave_batch_matrix { x, ninter, n, 1, x_cntg_step, 0 }
	};
	auto y_matrix = general_matrix {
		interleave_batch_matrix { y, ninter, m, 1, y_cntg_step, 0 }
	};

	naive_gemv_qr {}(a_matrix, x_matrix, y_matrix, compute_position {});
}

} // namespace

} // namespace perflibs::linalg

namespace perflibs::interleave_batch {

extern ::perflibs::linalg::gemv_qr_kernel_t<double> dgemv_qr_neon;
extern ::perflibs::linalg::ger_qr_kernel_t<double> ger_qr_neon;
extern ::perflibs::linalg::ger_qr_kernel_t<double> ger_qr_sve;

} // namespace perflibs::interleave_batch

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_KERNELS_QR_KERNELS_HPP

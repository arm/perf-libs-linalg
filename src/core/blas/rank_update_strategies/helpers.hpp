/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_RK_HELPERS_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_RK_HELPERS_HPP

#include "matrix/adaptors.hpp"

namespace perflibs::linalg {

namespace {

template<typename MatrixType>
PERFLIBS_LINALG_INLINE
void zero_diag_imag(MatrixType& mat) {
	auto gen_mat = to_general_matrix( mat );
	for(kernel_inttype i=0; i!=mat.cntg(); ++i) {
		gen_mat(i, i, write) = mat(i,i);
	}
}
} // Anon Namespace
} // perflibs::linalg

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_RK_HELPERS_HPP

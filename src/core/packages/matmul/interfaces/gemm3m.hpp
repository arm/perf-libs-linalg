/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_GEMM3M_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_GEMM3M_HPP

#include "packages/matmul/interfaces/gemm.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename AType, typename BType, typename CType>
inline constexpr std::string_view gemm3m_name = "?GEMM3M";

template<> inline constexpr std::string_view gemm3m_name<c32, c32, c32> = "CGEMM3M";
template<> inline constexpr std::string_view gemm3m_name<c64, c64, c64> = "ZGEMM3M";

template<bool ParamCheck, typename IntType, typename AType, typename BType, typename CType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void gemm3m(
	const char *transa, const char *transb,
	const IntType *m, const IntType *n, const IntType *k,
	const promote_t<AType, BType, CType> *alpha,
	const AType *a, const IntType *lda,
	const BType *b, const IntType *ldb,
	const promote_t<AType, BType, CType> *beta,
	      CType *c, const IntType *ldc) {

	gemm<ParamCheck, IntType, AType, BType, CType, ArchitectureSpec>(
		transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
		gemm3m_name<AType, BType, CType>);
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_GEMM3M_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_PROBLEM_CONTEXT_HELPERS_HPP
#define PERFLIBS_LINALG_PROBLEM_CONTEXT_HELPERS_HPP

#include "perflibs_complex.hpp"
#include "perflibs_util.hpp"
#include "matrix/matrix.hpp"
#include "spec/strategy_tag.hpp"
#include "framework/linalg_util.hpp"

#include "packages/solve/problem_context_bases.hpp"
#include "packages/matmul/problem_context_bases.hpp"
#include "packages/solve/fwd.hpp"

namespace perflibs::linalg {

namespace spec {

template<typename ProblemContext> PERFLIBS_LINALG_INLINE kernel_inttype syrk_lda(const ProblemContext& pc) {
	return is_cntg_contig(pc.a) ? pc.a.strd_step() : pc.a.cntg_step();
}

template<typename ProblemContext> PERFLIBS_LINALG_INLINE kernel_inttype syr2k_ldb(const ProblemContext& pc) {
	return is_cntg_contig(pc.a) ? pc.b.strd_step() : pc.b.cntg_step();
}

template<typename ProblemContext>
kernel_inttype gemm_lda(const ProblemContext& pctx) {
	if(is_cntg_contig(pctx.a) || pctx.b.cntg() == 1) {
		return pctx.a.strd() == 1
		     ? pctx.a.cntg()
		     : pctx.a.strd_step();
	}
	return pctx.a.cntg_step();
}

template<typename ProblemContext>
kernel_inttype gemm_ldb(const ProblemContext& pctx) {
	if(is_strd_contig(pctx.b) || pctx.b.strd() == 1) {
		return pctx.b.cntg() == 1
		     ? pctx.b.strd()
		     : pctx.b.cntg_step();
	}
	return pctx.b.strd_step();
}

template<typename ProblemContext>
char gemm_transa(const ProblemContext& pctx) {
	return is_cntg_contig(pctx.a) || pctx.b.cntg() == 1
	     ? pctx.a.is_conj() ? 'C' : 'T'
	     : 'N';
}

template<typename ProblemContext>
char gemm_transb(const ProblemContext& pctx) {
	return is_strd_contig(pctx.b) || pctx.b.strd() == 1
	     ? pctx.b.is_conj() ? 'C' : 'T'
	     : 'N';
}

/**
 * GEMV accessors
 */
template<typename ProblemContext>
PERFLIBS_LINALG_INLINE
kernel_inttype gemv_incx(const ProblemContext& pc) {
	if(pc.b.strd() == 1) { //A is our matrix, B is our vector
		return pc.b.cntg_step();
	}
	else {
		return pc.a.cntg_step();
	}
}

template<typename ProblemContext>
PERFLIBS_LINALG_INLINE
kernel_inttype gemv_incy(const ProblemContext& pc) {
	if(pc.b.strd() == 1) { //A is our matrix, B is our vector
		return pc.c.cntg_step();
	}
	else {
		return pc.c.strd_step();
	}
}

template<typename ProblemContext>
PERFLIBS_LINALG_INLINE
kernel_inttype dot_incx(const ProblemContext& pctx) {
	return pctx.a.cntg_step();
}

template<typename ProblemContext>
PERFLIBS_LINALG_INLINE
kernel_inttype dot_incy(const ProblemContext& pctx) {
	return pctx.b.cntg_step();
}

template<typename... T>
PERFLIBS_LINALG_INLINE
bool dot_is_conj(const solve::solve<T...>& pctx) {
	return pctx.a.is_conj();
}

template<typename AMatrixType, typename... T>
PERFLIBS_LINALG_INLINE
bool dot_is_conj(const matmul::matmul3<AMatrixType, T...>& pctx) {
	if constexpr (is_symmetric_matrix_v<AMatrixType>) {
		return false;
	}
	else if constexpr (is_hermitian_matrix_v<AMatrixType>) {
		return true;
	}
	else {
		return pctx.a.is_conj();
	}
}

template<typename... T>
PERFLIBS_LINALG_INLINE
bool dot_is_conj(const matmul::matmul2<T...>& pctx) {
	return pctx.a.is_conj();
}

/**
 * TRMM accessors
 */
template<typename ProblemContext> PERFLIBS_LINALG_INLINE kernel_inttype trmm_m(const ProblemContext& pc) { return pc.side == PERFLIBS_LEFT ? pc.a_strd : pc.b_strd; }
template<typename ProblemContext> PERFLIBS_LINALG_INLINE kernel_inttype trmm_n(const ProblemContext& pc) { return pc.side == PERFLIBS_LEFT ? pc.b_strd : pc.a_strd; }

template<typename ProblemContext> PERFLIBS_LINALG_INLINE char trmm_uplo(const ProblemContext& pc) {
	const bool is_left               = pc.side == PERFLIBS_LEFT;
	const bool is_trans_a            = is_trans(pc.transa) ^ (!is_left);

	return uplo_to_c(is_trans_a ? pc.uplo : lower_flip(pc.uplo));
}

template<typename ProblemContext>
PERFLIBS_LINALG_INLINE kernel_inttype trmm_lda(const ProblemContext& pc) {
	const bool is_left               = pc.side == PERFLIBS_LEFT;
	const bool is_trans_a            = is_trans(pc.transa) ^ (!is_left);

	return is_trans_a ? pc.a_strd_step : pc.a_cntg_step;
}

template<typename ProblemContext>
PERFLIBS_LINALG_INLINE kernel_inttype trmm_ldb(const ProblemContext& pc) {
	const bool is_left               = pc.side == PERFLIBS_LEFT;

	return is_left ? pc.b_strd_step : pc.b_cntg_step;
}

} //namespace spec

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_PROBLEM_CONTEXT_HELPERS_HPP

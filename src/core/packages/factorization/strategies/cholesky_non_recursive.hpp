/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_CHOLESKY_NON_RECURSIVE_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_CHOLESKY_NON_RECURSIVE_HPP

#include "matrix/matrix.hpp"
#include "spec/problem_context.hpp"
#include "packages/factorization/problem_context_bases.hpp"
#include "packages/factorization/kernels/cholesky_non_recursive_kernel.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class cholesky_non_recursive {

	template<template<typename> typename MatrixAdaptorType,
		typename ADataType,
		typename IntType,
		typename ArchitectureSpec
	>
	using cholesky_non_recursive_pctx_t  = spec::problem_context<
		cholesky_factorization<MatrixAdaptorType<matrix_base<ADataType>>, IntType>,
		ArchitectureSpec
	>;
public:
	static constexpr std::string_view name() { return "cholesky_non_recursive"; }

	template<template<typename> typename MatrixAdaptorType,
		typename ADataType,
		typename IntType,
		typename ArchitectureSpec
	>
	PERFLIBS_LINALG_INLINE
	bool operator()(const cholesky_non_recursive_pctx_t<MatrixAdaptorType, ADataType, IntType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		// Todo: Get cholesky_non_recursive_kernel from spec as kernel
		pctx.info = cholesky_non_recursive_kernel<ArchitectureSpec>(pctx.a.uplo(), pctx.a.cntg(), pctx.a.data(), pctx.a.strd_step());
		return true;
	}

	template<typename ProblemContext>
	constexpr bool operator()(const ProblemContext&) const {
		return false;
	}

	template<template<typename> typename MatrixAdaptorType,
		typename ADataType,
		typename IntType,
		typename ArchitectureSpec
	>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(
	    const cholesky_non_recursive_pctx_t<MatrixAdaptorType, ADataType, IntType, ArchitectureSpec>& pctx) const {

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class cholesky_non_recursive
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_CHOLESKY_NON_RECURSIVE_HPP

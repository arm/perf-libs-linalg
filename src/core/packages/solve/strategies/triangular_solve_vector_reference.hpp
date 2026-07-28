/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_VECTOR_REFERENCE_HPP
#define PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_VECTOR_REFERENCE_HPP

#include "packages/solve/problem_context_bases.hpp"
#include "spec/problem_context.hpp"
#include "framework/linalg_util.hpp"

#include "solve_references.hpp"

#include <string_view>

namespace perflibs::linalg::solve {

class triangular_solve_vector_reference {
	template<typename ADataType, typename BDataType, typename ScalarType, typename ArchitectureSpec>
	using pctx_t = spec::problem_context<
		solve<
			triangular_matrix<matrix_base<const ADataType>>,
			general_matrix<   matrix_base<      BDataType>>,
			ScalarType
		>,
		ArchitectureSpec
	>;

public:
	static constexpr std::string_view name() { return "triangular_solve_vector_reference"; }

	// Actual computation. this function will be called only if
	// the triangular_solve_matrix_reference strategy is valid to solve
	// the problem described by the ProblemContext
	template<typename ADataType, typename BDataType, typename ScalarType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator() (const pctx_t<ADataType, BDataType, ScalarType, ArchitectureSpec>& pctx) const {
		if( !this->can_compute(pctx) ) return false;

		// recover BLAS parameters from problem context
		const auto trans    = trans_to_c(pctx.transa);
		const auto uplo     = uplo_to_c(is_trans(pctx.transa) ? pctx.a.uplo() : lower_flip(pctx.a.uplo()));
		const auto diag     = diag_to_c(pctx.a.diag());
		const int_type n    = pctx.a.strd();
		const auto *a       = pctx.a.data();
		const int_type lda  = is_trans(pctx.transa) ? pctx.a.strd_step() : pctx.a.cntg_step();
		      auto *x       = pctx.b.data();
		const int_type incx = pctx.b.cntg_step();

		// pointer is moved to end of data in interfaces/trsv.hpp
		// when incx is negative, so reset it if necessary
		if (incx < 0)
			x += incx * (n - 1);

		// Todo: find a way to use constexpr here
		if (pctx.alpha != one<ScalarType>) {
			scale(pctx.alpha, pctx.b);
		}

		reference::trsv<ScalarType>(&uplo, &trans, &diag, &n, a, &lda, x, &incx);

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }

	template<typename ADataType, typename BDataType, typename ScalarType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const pctx_t<ADataType, BDataType, ScalarType, ArchitectureSpec>& pctx) const {
		return is_left(pctx.side) && pctx.b.strd() == 1;
	}

}; // class triangular_solve_vector_reference

} // namespace perflibs::linalg::solve

#endif //PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_VECTOR_REFERENCE_HPP

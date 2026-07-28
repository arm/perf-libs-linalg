/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_MATRIX_REFERENCE_HPP
#define PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_MATRIX_REFERENCE_HPP

#include "spec/problem_context.hpp"

#include "packages/solve/problem_context_bases.hpp"

#include "framework/linalg_util.hpp"
#include "solve_references.hpp"

#include <string_view>

namespace perflibs::linalg::solve {

class triangular_solve_matrix_reference {
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
	static constexpr std::string_view name() { return "triangular_solve_matrix_reference"; }

	// Actual computation. this function will be called only if
	// the triangular_solve_matrix_reference strategy is valid to solve
	// the problem described by the ProblemContext
	template<typename ADataType, typename BDataType, typename ScalarType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator() (const pctx_t<ADataType, BDataType, ScalarType, ArchitectureSpec>& pctx) const {
		if ( !this->can_compute(pctx) ) return false;

		// recover BLAS parameters from problem context
		const auto side       = side_to_c(pctx.side);
		const auto transa     = trans_to_c(pctx.transa);
		const auto diag       = diag_to_c(pctx.a.diag());
		const auto alpha      = pctx.alpha;
		const auto *a         = pctx.a.data();
		      auto *b         = pctx.b.data();

		const auto is_lside   = is_left(pctx.side);
		const auto is_trans_a = is_trans(pctx.transa);

		char uplo;
		int_type m, n, lda, ldb;
		if (is_lside) {
			uplo = uplo_to_c(is_trans_a ? pctx.a.uplo() : lower_flip(pctx.a.uplo()));
			m = pctx.a.strd();
			n = pctx.b.strd();
			lda = is_trans_a ? pctx.a.strd_step() : pctx.a.cntg_step();
			ldb = pctx.b.strd_step();
		}
		else {
			uplo = uplo_to_c(is_trans_a ? lower_flip(pctx.a.uplo()) : pctx.a.uplo());
			m = pctx.b.strd();
			n = pctx.a.strd();
			lda = is_trans_a ? pctx.a.cntg_step() : pctx.a.strd_step();
			ldb = pctx.b.cntg_step();
		}

		reference::trsm<ScalarType>(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);

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
		// reference requires positive ldb
		return is_left(pctx.side) ? pctx.b.strd_step() > 0 : pctx.b.cntg_step() > 0;
	}
}; // class triangular_solve_matrix_reference
} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_MATRIX_REFERENCE_HPP

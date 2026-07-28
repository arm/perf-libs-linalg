/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_PANEL_UPDATE_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_PANEL_UPDATE_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"

#include "packages/factorization/problem_context_bases.hpp"
#include "packages/solve/problem_context_bases.hpp"
#include "packages/matmul/problem_context_bases.hpp"
#include "packages/solve/fwd.hpp"
#include "packages/matmul/fwd.hpp"

#include "matrix/adaptors.hpp"

namespace perflibs::linalg::factorization {

class panel_update {

	template<typename ADataType, typename IntType, typename ArchitectureSpec>
	using lu_panel_update_pctx_t = spec::problem_context<
		lu_panel_update<general_matrix<matrix_base<ADataType>>, IntType>,
		ArchitectureSpec
	>;

public:
	template<typename ADataType, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE bool
	operator()(const lu_panel_update_pctx_t<ADataType, IntType, ArchitectureSpec>& pctx) const {

		if (!can_compute(pctx)) return false;

		PERFLIBS_ASSERT(pctx.a.cntg() == pctx.b.cntg(), "Panel A and B must contain the same number of rows");

		// This algorithm updates the panel B using data from the factorized panel A.
		// The matrix is subdivided in submatrices below
		//	[ A11 ]   [ B11 ]
		//	[ A21 ]   [ B21 ]
		//	[ A31 ]   [ B31 ]
		// The algorithm :
		// 1. apply the pivot to B
		// 2. update B11 by a triangular solve using the lower part of A11 as the triangular matrix
		// 3. Update B21 and B31 with rank update: [B21] = [B21] - [A21] * [B11]
		//                                         [B31]   [B31]   [A31]


		// 1. Apply row interchanges to the panel B
		auto apply_pivot_pctx = spec::problem_context{
			lu_apply_pivot{pctx.b, pctx.pivot },
			pctx.architecture_spec
		};
		compute(apply_pivot_pctx);

		// 2. Create the problem context to solve to update B11
		// Note that PERFLIBS_UPPER is used instead of PERFLIBS_LOWER
		// and cntg_step and strd_step of A_solve are interchanged accordingly.

		// Create rhs submatrix (B11) for
		auto RHS_solve     = get_cntg_panel(pctx.b, 0, pctx.a.strd());
		// Create the solve coefficient submatix (A11) for the solve
		auto A_solve_panel = get_cntg_panel(pctx.a, 0, pctx.a.strd());
		auto A_solve = to_const(triangular_matrix{
		    PERFLIBS_UPPER, PERFLIBS_UNIT, A_solve_panel.transpose().get_matrix_base(), A_solve_panel.is_conj() });

		auto solve_pctx = spec::problem_context{
			solve::solve{ PERFLIBS_LEFT, PERFLIBS_NOTRANS, A_solve, RHS_solve, one<ADataType> },
			pctx.architecture_spec
		};
		compute(solve_pctx);

		if (pctx.b.cntg() > pctx.a.strd()) {
			// 3. Update of the trailing matrix
			//  Create submatrices to update trailing matrix C = C - A*B, step 3 in algo above
			// B = RHS_solve
			auto A_gemm = get_cntg_panel(pctx.a, pctx.a.strd(), pctx.a.cntg() - pctx.a.strd() ).transpose();
			auto C_gemm = get_cntg_panel(pctx.b, pctx.a.strd(), pctx.b.cntg() - pctx.a.strd() );

			// Create gemm problem context and compute
			using data_value = decltype(pctx.a)::value_type;
			auto gemm_pctx = spec::problem_context{
				matmul::matmul3{ to_const(A_gemm),to_const(RHS_solve), C_gemm, data_value{ -1 }, one<ADataType> },
				pctx.architecture_spec
			};
			compute(gemm_pctx);
		}
		return true;
	}

	template<typename ProblemContext>
	constexpr bool operator()(const ProblemContext&) const {
		return false;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return true; }

}; // class lu_panel_update
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_PANEL_UPDATE_HPP

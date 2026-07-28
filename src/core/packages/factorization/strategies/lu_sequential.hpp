/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_SEQUENTIAL_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_SEQUENTIAL_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"

#include "packages/solve/fwd.hpp"
#include "packages/matmul/fwd.hpp"
#include "packages/factorization/fwd.hpp"

#include "packages/factorization/helpers/pivot_manager.hpp"
#include "matrix/adaptors.hpp"

#include "perflibs_util.hpp"
#include "perflibs_numeric_utils.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class lu_sequential {

	template<typename ADataType, typename IntType, typename ArchitectureSpec>
	using lu_pctx_t = spec::problem_context<
		lu_factorization<general_matrix<matrix_base<ADataType>>, IntType>,
	    ArchitectureSpec
	>;

public:
	static constexpr std::string_view name() { return "lu_sequential"; }

	template<typename ADataType, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const lu_pctx_t<ADataType, IntType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;
		// Ensures there are at least two blocks, and then adjusts the block size
		// so that we have near-equal block sizes rather than runt blocks.
		// this has the effect that once the problem size is smaller than the blocksize
		// we are effectively doing a recursive factorisation
		const auto spec    = get_spec(spec::strategy_tag<lu_factorization_generic>{}, pctx);
		const auto npanels = max(iround_div(min(pctx.a.strd(), pctx.a.cntg()), spec.block_size), 2);
		const auto nb      = iround_div(min(pctx.a.strd(), pctx.a.cntg()), npanels);

		// This algorithm is recursive and the matrix be subdivided in submatrices below
		//	[ A11 | A12  | A13 ]
		//	[ A21 | A22  | A23 ]
		//	[ A31 | A32  | A33 ]
		// Where we assume A[:, 1] the first panel is already factorized and will be denoted A_left, as
		// at the left of the panel.
		// The algorithm :
		// 1. factorizes the panel A[:,2],then
		// 2. Apply row swaps to A_left
		// 3. Apply row swaps to the columns at the right of the panel, denoted A_right
		// 4. Update A13 which becomes U13 with a solve: A12 * U13 = A13. Note that A12 is L12.
		// 5. Update A23 and A33 with rank update: [A23] = [A23]- [A22] * [U13]
		//                                         [A33]   [A33]  [A32]
		// At the end the same strategy is called on
		//[A21 | A22 | A23]
		//[A32 | A32 | A33]
		// Note that the full clock rows passed in as row interchanges must be applied to the left
		// of the current panel.


		// Initialize the matrix to factorize
		auto a                             = pctx.a;
		// Initial a_panel_start_point, the position of the sub matrix in parent
		kernel_inttype a_panel_start_point = 0;
		const auto minmn                   = min(a.strd(), a.cntg());
		pivot_manager<ArchitectureSpec, IntType> pivot_mgr(pctx.pivot);

		while (a_panel_start_point < minmn) {

			// Get panel width
			const kernel_inttype current_nb_columns =  a.strd() - a_panel_start_point;
			const kernel_inttype panel_width        = min(nb, min(a.cntg(), current_nb_columns));

			// 1. Create problem context and factorise the panel
			//  Create the panel submatrix to factorize
			auto panel   = get_strd_panel(a, a_panel_start_point, panel_width);
			int_type iinfo = 0;
			auto panel_factorization_pctx = spec::problem_context{
				lu_factorization{ panel, pctx.pivot.subspan(a_panel_start_point), iinfo },
				pctx.architecture_spec
			};
			compute(panel_factorization_pctx);

			// Adjust info
			if (pctx.info == 0 && iinfo > 0) {
				pctx.info = iinfo + a_panel_start_point;
			}

			// 2. Apply row interchanges at left of the panel
			// Create the submatrix to apply pivots at the left of the panel

			if (a.cntg() < a.get_parent().cntg()) {
				auto a_left = get_strd_panel(a, 0, a_panel_start_point);
				auto apply_pivot_left_pctx = spec::problem_context{
					lu_apply_pivot{ a_left, pctx.pivot.subspan(a_panel_start_point, panel_width) },
					pctx.architecture_spec
				};
				compute(apply_pivot_left_pctx);
			}

			if (current_nb_columns > panel_width) {
				// 3. Apply row interchanges at the right of the panel
				// Create the right of panel submatrix to apply row interchange
				const kernel_inttype trailing_matrix_start = a_panel_start_point + panel_width;
				auto trailing_matrix = get_strd_panel(a, trailing_matrix_start, a.strd() - trailing_matrix_start);
				auto apply_pivot_right_pctx = spec::problem_context {
					lu_apply_pivot { trailing_matrix, pctx.pivot.subspan(a_panel_start_point, panel_width) },
					pctx.architecture_spec
				};
				compute(apply_pivot_right_pctx);

				// 4. Create the problem context to solve to update A13
				//  Note that PERFLIBS_UPPER is used instead of PERFLIBS_LOWER
				//  and cntg_step and strd_step of A_solve are interchanged accordingly.

				// Create rhs submatrix (A13) for solve
				auto a13 = get_cntg_panel(trailing_matrix, 0, panel_width);
				// Create the solve coefficient submatix (A12) for the solve
				auto a_solve_panel = get_cntg_panel(panel, 0, panel_width);
				auto a12 = triangular_matrix{
					PERFLIBS_UPPER, PERFLIBS_UNIT,
				    a_solve_panel.transpose().get_matrix_base(),
				    a_solve_panel.is_conj()
				};

				auto solve_pctx = spec::problem_context{
					solve::solve{ PERFLIBS_LEFT, PERFLIBS_NOTRANS, to_const(a12), a13, one<ADataType> },
					pctx.architecture_spec
				};
				compute(solve_pctx);

				if (a.cntg() > panel_width) {
					// 5. Update of the trailing matrix
					//  Create submatrices to update trailing matrix C = C - A*B, step 5 in algo above
					auto a_gemm = to_const(a.sub_matrix(panel_width, a.cntg() - panel_width, a_panel_start_point, panel_width).transpose());
					auto c_gemm = get_cntg_panel(trailing_matrix, panel_width, trailing_matrix.cntg() - panel_width);

					// Create gemm problem context and compute
					using data_value = decltype(pctx.a)::value_type;
					auto gemm_pctx = spec::problem_context{
						matmul::matmul3{ a_gemm, to_const(a13) , c_gemm, data_value{ -1 }, one<ADataType> },
						pctx.architecture_spec
					};
					compute(gemm_pctx);
				}
			}
			// Store panel information to update pivot later
			pivot_mgr.add_panel(a_panel_start_point, panel_width);

			// Prepare for iteration
			a                    = get_cntg_panel(a, panel_width, a.cntg() - panel_width);
			a_panel_start_point += panel_width;
		}
		// Update pivot indices to global indices
		pivot_mgr.update_pivots_to_global();

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext&) const { return false; }

	template<typename ADataType, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const lu_pctx_t<ADataType, IntType, ArchitectureSpec>& pctx) const {
		return pctx.a.strd() > 1;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class lu_sequential
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_SEQUENTIAL_HPP

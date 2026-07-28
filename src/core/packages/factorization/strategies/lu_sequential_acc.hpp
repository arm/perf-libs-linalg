/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_SEQUENTIAL_ACC_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_SEQUENTIAL_ACC_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"

#include "packages/factorization/problem_context_bases.hpp"
#include "packages/solve/problem_context_bases.hpp"
#include "packages/matmul/problem_context_bases.hpp"
#include "packages/factorization/helpers/pivot_manager.hpp"

#include "matrix/adaptors.hpp"
#include "perflibs_util.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class lu_sequential_acc {

	template<typename ADataType, typename IntType, typename ArchitectureSpec>
	using lu_pctx_t = spec::problem_context<
		lu_factorization<general_matrix<matrix_base<ADataType>>, IntType>,
		ArchitectureSpec
	>;

public:
	static constexpr std::string_view name() { return "lu_sequential_acc"; }

	template<typename ADataType, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const lu_pctx_t<ADataType, IntType, ArchitectureSpec>& pctx) const {

		// This algorithm performs a sequential LU factorization with left pivot application
		// accumulated and applied at the end. The matrix can be subdivided in submatrices below
		//	[ A11 | A12  | A13 ]
		//	[ A21 | A22  | A23 ]
		//	[ A31 | A32  | A33 ]
		//
		// A the first iteration, the algorithm
		// 1. factorizes the panel A[1,:]
		// 2. Update all the panel at right.
		//     2.1 applies the row interchange
		//     2.2 update the top square using trsm
		//     2.3 update the remaining of the panel using gemm
		// 3. Create a pivot of identity permutation for the for panel,
		//   which will be updated later by accumulating the future pivots
		// At the second iteration, the matrix left to factorize is
		//   [A21 | A22 | A23]
		//   [A32 | A32 | A33]
		// but during the iterations this algorithm only focuses on
		//   [A22 | A23]
		//   [A32 | A33]
		// as
		//   [A21]
		//   [A31]
		// only needs to apply pivots from the factorization of the other panels.
		// This approach helps applying accumulated pivots to each left panel only once.
		// This is different from the standard algorithm which applies pivots to each
		// left panel whenever a new panel is factorized. Accumulating pivots and applying
		// once has a performance benefit as the pivot application is a memory bandwidth
		// bound operation.

		const auto spec = get_spec(spec::strategy_tag<lu_factorization_generic>{}, pctx);
		const auto nb   = spec.block_size;

		// Initialize accumulate_pivot
		const auto minmn = min(pctx.a.strd(), pctx.a.cntg());
		pivot_manager<ArchitectureSpec, IntType> pivot_mgr(pctx.pivot);

		// Initialize the remaining submatrix A to factorize
		auto a = pctx.a;

		// Get the row starting position of the sub matrix in parent
		kernel_inttype a_panel_start_point = 0;

		while (a_panel_start_point < minmn) {

			// Get panel width
			const auto sub_matrix_columns =  pctx.a.get_parent().strd() - a_panel_start_point;
			const auto panel_width        = min(nb, min(a.cntg(), sub_matrix_columns));

			// 1. Create problem context and factorise the panel
			//  Create the panel submatrix to factorize
			auto panel = get_strd_panel(a, 0, panel_width);
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

			//2.  Update the trailing matrix
			const auto trailing_matrix   = get_strd_panel(a, panel_width, a.strd() - panel_width);
			auto panel_update_pctx = spec::problem_context{
				lu_panel_update{ panel, trailing_matrix, pctx.pivot.subspan(a_panel_start_point, panel_width)},
				pctx.architecture_spec
			};
			compute(panel_update_pctx);

			// 3. Accumulate pivots
			pivot_mgr.accumulate_pivots(a_panel_start_point, panel_width);

			//Update the remaining submatrix to factorize
			a = a.sub_matrix(panel_width, a.cntg() - panel_width, panel_width, a.strd() - panel_width);

			// Update a_panel_start_point
			a_panel_start_point += panel_width;
		}

		// Apply pivots to left
		auto nb_panel_to_update = static_cast<kernel_inttype>(pivot_mgr.size()) -1;
		for(auto panel_id = 0_ki; panel_id <nb_panel_to_update; panel_id++){

			const auto panel_width       = pivot_mgr.get_panel_size(panel_id);
			const auto panel_start_point = pivot_mgr.get_start_point(panel_id);
			// Starting point of the panel block to update.
			// Note that the full panel does not need updating
			// as the top block is already fully factorized
			const auto subpanel_row_start = panel_start_point + panel_width;
			const auto pivots             = pivot_mgr.get_pivots(panel_id);
			auto panel                    = pctx.a.sub_matrix(subpanel_row_start, pctx.a.cntg() - subpanel_row_start, panel_start_point, panel_width);
			auto apply_pivot_pctx         = spec::problem_context {
				lu_apply_pivot { panel, pivots }, pctx.architecture_spec
			};
			compute(apply_pivot_pctx);
		}

		// Update pivot indices to global indices
		pivot_mgr.update_pivots_to_global();

		return true;
	}

	template<typename ProblemContext>
	constexpr bool operator()(const ProblemContext&) const {
		return false;
	}

	template<typename ADataType, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const lu_pctx_t<ADataType, IntType, ArchitectureSpec>& pctx) const {
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class lu_sequential_acc
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_SEQUENTIAL_ACC_HPP

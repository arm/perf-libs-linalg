/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_PARALLEL_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_PARALLEL_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"

#include "packages/factorization/problem_context_bases.hpp"
#include "packages/solve/problem_context_bases.hpp"
#include "packages/matmul/problem_context_bases.hpp"
#include "packages/factorization/helpers/pivot_manager.hpp"
#include "packages/factorization/helpers/matrix_distribution.hpp"

#include "matrix/adaptors.hpp"

#include "detect/omp.hpp"
#include "perflibs_util.hpp"
#include "perflibs_numeric_utils.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class lu_parallel {

	template<typename ADataType, typename IntType, typename ArchitectureSpec>
	using lu_pctx_t = spec::problem_context<
		lu_factorization<general_matrix<matrix_base<ADataType>>, IntType>,
		ArchitectureSpec
	>;

public:
	static constexpr std::string_view name() { return "lu_parallel"; }

	template<typename ADataType, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE bool
	operator()(const lu_pctx_t<ADataType, IntType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		// This algorithm performs a parallel LU factorization where the matrix is subdivided in submatrices
		//	[ A11 | A12  | A13 ]
		//	[ A21 | A22  | A23 ]
		//	[ A31 | A32  | A33 ]
		//
		// The first iteration, the algorithm
		// 1. factorizes the panel A[1,:]
		// 2. Update all the panel at right. Note the panel update happens in parallel, and each thread
		//     2.1 applies the row interchange
		//     2.2 update the top square using trsm
		//     2.3 update the remaining of the panel using gemm
		// 3. Create a pivot of identity permutation for the for panel, which will be updated later by accumulating the future pivots
		// At the second iteration, the matrix left to factorize is
		//   [A21 | A22 | A23]
		//   [A32 | A32 | A33]
		// but during the iterations this algorithm only focuses on
		//   [A22 | A23]
		//   [A32 | A33]
		// as
		//   [A21]
		//   [A31]
		// only needs to apply pivots from the factorization of the other panel, these
		// pivots are accumulated and applied at the end. This approach help visiting
		// left panel once to apply the accumulated pivot, instead of applying the pivot each time
		// a panel is factorized
		// The algorithm then

		// Initialize accumulate_pivot
		const auto minmn = min(pctx.a.strd(), pctx.a.cntg());

		// Initialize pivot_manager with the pivot array
		pivot_manager<ArchitectureSpec, IntType> pivot_mgr(pctx.pivot);

		// Initialize the remained submatrix A to factorize
		auto a = pctx.a;

		// Ensures there are at least two blocks, and then adjusts the block size
		// so that we have near-equal block sizes rather than runt blocks.
		// this has the effect that once the problem size is smaller than the blocksize
		// we are effectively doing a recursive factorisation
		const auto spec    = get_spec(spec::strategy_tag<lu_factorization_generic>{}, pctx);
		const auto npanels = max(iround_div(min(pctx.a.strd(), pctx.a.cntg()), spec.block_size), 2);
		const auto nb      = iround_div(min(pctx.a.strd(), pctx.a.cntg()), npanels);

		const auto nt      = spec.max_threads;

		// Get the row starting position of the sub matrix in parent
		kernel_inttype a_panel_start_point = 0;
		// Initialize previous and current factorized_panel
		auto previous_panel                = get_strd_panel(a, 0, 0);
		auto current_panel                 = get_strd_panel(a, 0, 0);
		kernel_inttype panel_width         = 0;
		kernel_inttype pivot_size          = 0;
		bool continue_factorization        = true;


		#pragma omp parallel default(none) shared(pctx, a, previous_panel, current_panel, panel_width, pivot_size, minmn, a_panel_start_point, pivot_mgr, continue_factorization, nb, nt, spec) num_threads(nt)
		{
			const auto thread_id = omp::get_thread_num();

		   // First iteration: Only one thread contributes, as there is only a panel factorization
			#pragma omp single
			{
				panel_width    = min(nb, min(a.cntg(), a.strd()));
				auto panel     = get_strd_panel(a, 0, panel_width);
				int_type iinfo = 0;
				auto panel_factorization_pctx = spec::problem_context{
					lu_factorization{ panel, pctx.pivot, iinfo },
					pctx.architecture_spec
				};
				compute(panel_factorization_pctx);

				if (pctx.info == 0 && iinfo > 0) {
					pctx.info = iinfo;
				}

				// In case all the rows have been factorized but
				// there are more columns than rows,
				// only one update of the columns is required,
				if(a.cntg() == panel_width && a.strd( ) > panel_width) {
					auto trailing_matrix  = get_strd_panel(a, panel_width, a.strd() - panel_width);
					auto panel_update_pctx = spec::problem_context{
						lu_panel_update{ panel, trailing_matrix, pctx.pivot},
						pctx.architecture_spec
					};
					compute(panel_update_pctx);

					continue_factorization = false;
				}
				else {

					pivot_mgr.accumulate_pivots(a_panel_start_point, panel_width);

					// Prepare for next iteration
					previous_panel      = panel;
					a                   = get_strd_panel(a, panel_width, a.strd() - panel_width);
					a_panel_start_point = panel_width;
					pivot_size          = panel_width;
				}
			}

			// From here factorization and trailing matrix overlap
			while (continue_factorization) {

				const auto min_columns_per_thread   = spec.min_update_block_size;
				const auto weight_factor            = spec.weight_factor;
				const auto current_panel_length     = a.cntg() - previous_panel.strd();
				const auto panel_width              = min(nb, min(current_panel_length, a.strd()));
    			      auto only_trailing_update     = panel_width != 0 && panel_width == current_panel_length && a.strd() > panel_width;

				auto [start_col, end_col] = distribute_trailing_matrix_update<ArchitectureSpec>(a.strd(), panel_width, nt, thread_id, min_columns_per_thread, weight_factor);
				const auto trailing_update_size = end_col - start_col;
				auto previous_pivot = pivot_mgr.get_previous_local_pivot();

				//Thread 0: Panel update, factorization, and partial trailing update
				if (thread_id == 0){

					// Update the panel
					if(panel_width > 0) {
						const auto panel       = get_strd_panel(a, 0, panel_width);
						auto panel_update_pctx = spec::problem_context{
							lu_panel_update{ previous_panel, panel, previous_pivot},
							pctx.architecture_spec
						};
						compute(panel_update_pctx);

						//  Factorize the panel decreased by previous_panel_width rows
						current_panel                 = get_cntg_panel(panel, previous_panel.strd(), panel.cntg() - previous_panel.strd());
						int_type iinfo                = 0;
						auto panel_factorization_pctx = spec::problem_context{
							lu_factorization{ current_panel, pctx.pivot.subspan(a_panel_start_point), iinfo },
							pctx.architecture_spec
						};
						compute(panel_factorization_pctx);

						if (pctx.info == 0 && iinfo > 0) {
							pctx.info = iinfo + a_panel_start_point;
						}
					}
					// Help with trailing matrix update
					if (trailing_update_size > 0) {
						auto trailing_contribution = get_strd_panel(a, start_col, trailing_update_size);
						auto panel_update_pctx          = spec::problem_context{
							lu_panel_update{ previous_panel, trailing_contribution, previous_pivot },
							pctx.architecture_spec
						};
						compute(panel_update_pctx);
					}
				}
				// Other threads perform trailing matrix update only
				else if (trailing_update_size > 0) {
					auto trailing_contribution = get_strd_panel(a, start_col, trailing_update_size);
					auto panel_update_pctx     = spec::problem_context{
						lu_panel_update{ previous_panel, trailing_contribution, previous_pivot },
						pctx.architecture_spec
					};
					compute(panel_update_pctx);
				}

				#pragma omp barrier

				#pragma omp single
				{
					// Accumulate pivot
					 if (panel_width > 0) {
						pivot_size  = min(current_panel.cntg(), current_panel.strd());
						pivot_mgr.accumulate_pivots(a_panel_start_point, pivot_size);
					 }

					// Prepare for next iteration
					previous_panel                  = a.sub_matrix(previous_panel.strd(), a.cntg() - previous_panel.strd(), 0, panel_width);
					const auto start_trailing_col   = a_panel_start_point + panel_width;
					const auto trailing_col_length  = pctx.a.strd() - start_trailing_col;
					a                               = pctx.a.sub_matrix(a_panel_start_point, pctx.a.cntg() - a_panel_start_point, start_trailing_col, trailing_col_length);
					a_panel_start_point             += panel_width;
					continue_factorization          = a_panel_start_point < minmn || only_trailing_update;
				}
			}

			// Apply pivots to left
			const auto total_panels = static_cast<kernel_inttype>(pivot_mgr.size());
			const auto nb_panel_to_update = total_panels - 1;

			#pragma omp for
			for(kernel_inttype panel_id = 0; panel_id < total_panels; panel_id++) {

				// Update the pivot indices for this panel to global indices
				pivot_mgr.update_pivot_to_global(panel_id);

				if (panel_id < nb_panel_to_update) {
					const auto panel_width        = pivot_mgr.get_panel_size(panel_id);
					const auto panel_start_point  = pivot_mgr.get_start_point(panel_id);

					// Starting point of the panel block to update.
					// Note that the full panel does not need updating
					// as the top block is already fully factorized
					const auto subpanel_row_start = panel_start_point + panel_width;
					auto pivots = pivot_mgr.get_pivots(panel_id);

					auto panel = pctx.a.sub_matrix(subpanel_row_start, pctx.a.cntg() - subpanel_row_start, panel_start_point,  panel_width);

					auto apply_pivot_pctx = spec::problem_context {
						lu_apply_pivot { panel, pivots },
						pctx.architecture_spec
					};
					compute(apply_pivot_pctx);
				}
			}
		}
		return true;
	}

	template<typename ProblemContext>
	constexpr bool operator()(const ProblemContext&) const {
		return false;
	}

	template<typename ADataType, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool can_compute(const lu_pctx_t<ADataType, IntType, ArchitectureSpec>& pctx) const {
		const auto spec = get_spec(spec::strategy_tag<lu_factorization_generic>{}, pctx);
		return !omp::in_parallel() && spec.max_threads > 1;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class lu_parallel
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_PARALLEL_HPP

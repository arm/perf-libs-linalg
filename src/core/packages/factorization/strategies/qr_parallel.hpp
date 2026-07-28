/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_QR_PARALLEL_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_QR_PARALLEL_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"
#include "matrix/adaptors.hpp"
#include "detect/omp.hpp"
#include "perflibs_util.hpp"
#include "perflibs_numeric_utils.hpp"

#include "packages/factorization/problem_context_bases.hpp"
#include "packages/factorization/helpers/qr_block_reflector_manager.hpp"
#include "packages/factorization/helpers/matrix_distribution.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class qr_parallel {
	template<typename MatrixType, typename ArchitectureSpec>
	using qr_pctx_t = spec::problem_context<qr_factorization<MatrixType>, ArchitectureSpec>;

public:
	static constexpr std::string_view name() { return "qr_parallel"; }

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const qr_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		      auto a              = pctx.a;

		// Ensures there are at least two blocks, and then adjusts the block size
		// so that we have near-equal block sizes rather than runt blocks.
		// this has the effect that once the problem size is smaller than the blocksize
		// we are effectively doing a recursive factorisation
		const auto spec           = get_spec(spec::strategy_tag<qr_factorization_generic>{}, pctx);
		const auto npanels        = max(iround_div(min(pctx.a.strd(), pctx.a.cntg()), spec.block_size), 2);
		      auto panel_width    = iround_div(min(pctx.a.strd(), pctx.a.cntg()), npanels);

		const auto nt             = spec.max_threads;
		const auto max_block_size = spec.max_block_size;

		// Use our own workspace in case the
		// provided is not big enough
		using value_type = typename MatrixType::value_type;
		std::span<value_type> effective_workspace;
		perflibs::pod_vector<value_type> our_workspace;
		const std::size_t max_workspace_size = max_block_size * pctx.a.strd();
		if (pctx.work.size() < max_workspace_size) {
			our_workspace.resize(max_workspace_size);
			effective_workspace = std::span(our_workspace);
		}
		else {
			effective_workspace = pctx.work;
		}

		kernel_inttype a_panel_start_point = 0;
		auto previous_factorized_panel = get_strd_panel(a, 0, 0);
		auto remaining_columns_to_process = 0_ki;

		// Initialize block_reflector_manager
		block_reflector_manager<value_type, ArchitectureSpec> reflector_mgr(max_block_size);

		// Allocate and compute T matrix for the first panel
		reflector_mgr.allocate_current_reflector(panel_width);

		// Main parallel region
		#pragma omp parallel default(none) shared(pctx, a, previous_factorized_panel, panel_width, a_panel_start_point, nt, reflector_mgr, remaining_columns_to_process, effective_workspace) num_threads(nt)
		{
			auto thread_id = omp::get_thread_num();

			#pragma omp single
			{
				// First iteration: Only one thread contributes, as there is only a panel factorization
				auto panel       = get_strd_panel(a, 0, panel_width);
				auto panel_factorization_pctx = spec::problem_context{
					qr_factorization{ panel, pctx.tau, effective_workspace },
					pctx.architecture_spec
				};
				compute(panel_factorization_pctx);


				auto larft_pctx = spec::problem_context{
					form_block_reflector_factor{ PERFLIBS_FORWARD, PERFLIBS_COLUMNWISE, panel, pctx.tau, reflector_mgr.get_current_reflector() },
					pctx.architecture_spec
				};
				compute(larft_pctx);

				previous_factorized_panel = panel;
				a = get_strd_panel(a, panel_width, a.strd() - panel_width);
				a_panel_start_point = panel_width;
				reflector_mgr.update_previous_reflectors();
				remaining_columns_to_process = a.strd();
			}

			while (remaining_columns_to_process > 0) {

				// Create a local problem context reduced to the remaining matrix
				// for tuning purpose
				const auto tuning_pctx = spec::problem_context{
					qr_factorization{a, pctx.tau, effective_workspace},
					pctx.architecture_spec
				};
				const auto spec                   = get_spec(spec::strategy_tag<qr_factorization_generic>{},tuning_pctx);
				const auto min_columns_per_thread = spec.min_update_block_size;
				const auto weight_factor          = spec.weight_factor;

				auto previous_panel_width = previous_factorized_panel.strd();
				panel_width = min(panel_width, min(a.cntg() - previous_panel_width, a.strd()));

				auto [start_col, end_col] = distribute_trailing_matrix_update<ArchitectureSpec>(a.strd(), panel_width, nt, thread_id, min_columns_per_thread, weight_factor);
				auto trailing_update_size = end_col - start_col;

				// Thread 0: Panel update, factorization, and partial trailing update
				if (thread_id == 0) {

					if (panel_width > 0) {
						// Apply the block reflector to update the current panel
						auto current_panel = get_strd_panel(a, 0, panel_width);
						const auto ldwork = current_panel.strd();
						auto work_space = general_matrix {
							matrix_base { effective_workspace.data(), ldwork, previous_panel_width, 1, ldwork }
						};
						auto reflector = factorization::block_reflector {
							to_const(previous_factorized_panel),
							to_const(adjoint(reflector_mgr.get_previous_reflector())),
							block_reflector_layout::leading_unit_lower
						};
						auto larfb_pctx = spec::problem_context {
							apply_block_reflector {
								reflector,
								current_panel,
								work_space
							},
							pctx.architecture_spec
						};
						compute(larfb_pctx);

						// Factorize the current panel
						if (a_panel_start_point < pctx.a.cntg()) {
							auto factorized_panel = get_cntg_panel(current_panel, previous_panel_width, current_panel.cntg() - previous_panel_width);

							auto panel_factorization_pctx = spec::problem_context{
								qr_factorization{ factorized_panel,
									pctx.tau.subspan(a_panel_start_point, panel_width),
									effective_workspace },
								pctx.architecture_spec
							};
							compute(panel_factorization_pctx);

							// Allocate and compute a new block reflector T matrix for the next iteration
							reflector_mgr.allocate_current_reflector(panel_width);
							auto larft_pctx = spec::problem_context{
								form_block_reflector_factor{
									PERFLIBS_FORWARD, PERFLIBS_COLUMNWISE,
									factorized_panel, pctx.tau.subspan(a_panel_start_point, panel_width),
									reflector_mgr.get_current_reflector()
								},
								pctx.architecture_spec
							};
							compute(larft_pctx);
						}
					}

					// Help with trailing matrix update
					if (trailing_update_size > 0) {
						auto update_contribution = get_strd_panel(a, start_col, trailing_update_size);
						const auto ldwork = update_contribution.strd();
						auto work_space = general_matrix {
							matrix_base { effective_workspace.data(), ldwork, previous_panel_width, 1, ldwork }
						};
						auto reflector = factorization::block_reflector {
							to_const(previous_factorized_panel),
							to_const(adjoint(reflector_mgr.get_previous_reflector())),
							block_reflector_layout::leading_unit_lower
						};
						auto larfb_pctx = spec::problem_context {
							apply_block_reflector {
								reflector,
								update_contribution,
								work_space
							},
							pctx.architecture_spec
						};
						compute(larfb_pctx);
					}
				}
				// Other threads perform trailing matrix update only
				else if (trailing_update_size > 0) {
					auto update_contribution = get_strd_panel(a, start_col, trailing_update_size);
					const auto ldwork = update_contribution.strd();
					size_t work_offset = start_col * previous_panel_width;
					auto work_space = general_matrix {
						matrix_base { effective_workspace.data() + work_offset, ldwork, previous_panel_width, 1, ldwork }
					};
					auto reflector = factorization::block_reflector {
						to_const(previous_factorized_panel),
						to_const(adjoint(reflector_mgr.get_previous_reflector())),
						block_reflector_layout::leading_unit_lower
					};
					auto larfb_pctx = spec::problem_context {
						apply_block_reflector {
							reflector,
							update_contribution,
							work_space
						},
						pctx.architecture_spec
					};
					compute(larfb_pctx);
				}

				#pragma omp barrier

				#pragma omp single
				{
					// Prepare for next iteration
					if (panel_width == 0) {
						remaining_columns_to_process = 0;
					}
					else {
						previous_factorized_panel = a.sub_matrix(previous_panel_width, a.cntg() - previous_panel_width, 0, panel_width);
						a = a.sub_matrix(previous_panel_width, a.cntg() - previous_panel_width, panel_width, a.strd() - panel_width);
						reflector_mgr.update_previous_reflectors();
						a_panel_start_point += panel_width;
						remaining_columns_to_process -= panel_width;
					}
				}
			}
		}
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext&) const { return false; }

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool can_compute(const qr_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		const auto spec = get_spec(spec::strategy_tag<qr_factorization_generic>{}, pctx);
		return !omp::in_parallel() && spec.max_threads > 1;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
};
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_QR_PARALLEL_HPP

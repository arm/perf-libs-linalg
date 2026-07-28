/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_GENERATE_Q_FROM_QR_PARALLEL_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_GENERATE_Q_FROM_QR_PARALLEL_HPP

#include "framework/compute.hpp"
#include "framework/parallel.hpp"
#include "spec/problem_context.hpp"
#include "packages/factorization/fwd.hpp"
#include "packages/factorization/helpers/qr_block_reflector_manager.hpp"
#include "packages/factorization/helpers/matrix_distribution.hpp"
#include "matrix/adaptors.hpp"

#include "detect/omp.hpp"
#include "perflibs_util.hpp"
#include "perflibs_numeric_utils.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class generate_q_from_qr_parallel {
	template<typename MatrixType, typename ArchitectureSpec>
	using generate_q_pctx_t = spec::problem_context<generate_q_from_qr<MatrixType>, ArchitectureSpec>;

public:
	static constexpr std::string_view name() { return "generate_q_from_qr_parallel"; }

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const generate_q_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		// Quick return
		if (pctx.a.strd() == 0) return true;

		using value_type = typename MatrixType::value_type;

		// Ensures there are at least two blocks, and then adjusts the block size
		// so that we have near-equal block sizes rather than runt blocks.
		// this has the effect that once the problem size is smaller than the blocksize
		const auto spec    = get_spec(spec::strategy_tag<generate_q_from_qr_generic>{}, pctx);
		const auto npanels = max(iround_div(pctx.a.strd(), spec.block_size), 2);
		const auto nb      = iround_div(pctx.a.strd(), npanels);

		// Extract all tuning parameters from spec
		const auto nt                     = spec.max_threads;
		const auto min_columns_per_thread = spec.min_update_block_size;
		const auto weight_factor          = spec.weight_factor;

		// Determine workspace requirements
		const auto k                             = pctx.tau.cntg();
		const auto required_workspace_per_thread = pctx.a.strd() * nb;
		const auto required_workspace_size       = required_workspace_per_thread * nt;

		// Setup workspace
		perflibs::pod_vector<value_type> scratch_work;
		auto work_memory = pctx.work.data();
		if (pctx.work.cntg() < required_workspace_size) {
			// Provided workspace is not large enough, allocate our own
			scratch_work.resize(static_cast<std::size_t>(required_workspace_size));
			work_memory = scratch_work.data();
		}

		// Determine the last block start index
		const auto last_block_start = ((k - 1) / nb) * nb;

		// Initialize managers
		block_reflector_manager<value_type, ArchitectureSpec> reflector_mgr(nb);
		householder_block_manager<value_type, ArchitectureSpec> householder_mgr(pctx.a.cntg(), nb);

		// Track current block index
		auto current_block_idx = last_block_start;
		auto total_blocks      = (last_block_start / nb) + 1;

		// Main parallel region
		#pragma omp parallel default(none) shared(pctx, current_block_idx, nb, k, reflector_mgr, householder_mgr, nt, work_memory, total_blocks, min_columns_per_thread, required_workspace_per_thread, weight_factor) num_threads(nt)
		{
			auto thread_id = omp::get_thread_num();

			// Initialize columns k+1:n to columns of the unit matrix
			#pragma omp for schedule(static)
			for (auto j = k; j < pctx.a.strd(); ++j) {
				for (auto i = 0_ki; i < pctx.a.cntg(); ++i) {
					pctx.a(i, j, write) = zero<value_type>;
				}
				pctx.a(j, j, write) = one<value_type>;
			}

			// Zero the upper triangular part of the first k columns
			#pragma omp for schedule(static)
			for (auto j = 1_ki; j < k; ++j) {
				for (auto i = 0_ki; i < j; ++i) {
					pctx.a(i, j, write) = zero<value_type>;
				}
			}

			// First iteration - only master thread prepares the last block
			#pragma omp single
			{
				const auto block_size   = min(nb, k - current_block_idx);
				const auto panel_length = pctx.a.cntg() - current_block_idx;

				// Allocate and copy current Householder block
				householder_mgr.allocate_current_block(panel_length, block_size);
				auto block_v = pctx.a.sub_matrix(current_block_idx, panel_length, current_block_idx, block_size);
				householder_mgr.copy_to_current_block(block_v);

				// Generate triangular factor T
				reflector_mgr.allocate_current_reflector(block_size);
				std::span<value_type> tau_span(pctx.tau.data() + current_block_idx, static_cast<size_t>(block_size));

				auto larft_pctx = spec::problem_context{
					form_block_reflector_factor{
						PERFLIBS_FORWARD,
						PERFLIBS_COLUMNWISE,
						householder_mgr.get_current_block(),
						tau_span,
						reflector_mgr.get_current_reflector()
					},
					pctx.architecture_spec
				};
				compute(larft_pctx);

				// Set current block to identity
				// Note that the upper triangular has already been
				// set to zero in the initialization phase.
				for (kernel_inttype j = 0; j < block_size; ++j) {
					pctx.a(current_block_idx + j, current_block_idx + j, write) = one<value_type>;
					for (kernel_inttype i = j + 1; i < panel_length; ++i) {
						pctx.a(current_block_idx + i, current_block_idx + j, write) = zero<value_type>;
					}
				}

				// Move to next block
				reflector_mgr.update_previous_reflectors();
				householder_mgr.update_previous_blocks();
				current_block_idx -= nb;
			}

			// Process remaining blocks
			for (kernel_inttype block_count = 1; block_count < total_blocks; ++block_count) {
				// Prepare current block
				if (thread_id == 0) {
					const auto panel_length = pctx.a.cntg() - current_block_idx;

					// Allocate and copy current Householder block
					householder_mgr.allocate_current_block(panel_length, nb);
					auto block_v = pctx.a.sub_matrix(current_block_idx, panel_length, current_block_idx, nb);
					householder_mgr.copy_to_current_block(block_v);

					// Generate triangular factor T
					reflector_mgr.allocate_current_reflector(nb);
					std::span<value_type> tau_span(pctx.tau.data() + current_block_idx, static_cast<size_t>(nb));

					auto larft_pctx = spec::problem_context{
						form_block_reflector_factor{
							PERFLIBS_FORWARD,
							PERFLIBS_COLUMNWISE,
							householder_mgr.get_current_block(),
							tau_span,
							reflector_mgr.get_current_reflector()
						},
						pctx.architecture_spec
					};
					compute(larft_pctx);

					// Set current block to identity
					// Note that the upper triangular has already been
					// set to zero in the initialization phase.
					for (auto j = 0_ki; j < nb; ++j) {
						pctx.a(current_block_idx + j, current_block_idx + j, write) = one<value_type>;
						for (kernel_inttype i = j + 1; i < panel_length; ++i) {
							pctx.a(current_block_idx + i, current_block_idx + j, write) = zero<value_type>;
						}
					}
				}

				// Apply previous reflector to the trailing matrix, the matrix at the right
				// of the current block
				const auto prev_panel_length = householder_mgr.get_previous_panel_length();
				const auto prev_block_size   = householder_mgr.get_previous_panel_width();
				      auto prev_block_idx    = current_block_idx + nb;
				      auto trailing_cols     = pctx.a.strd() - prev_block_idx;

				// Distribute work among threads
				auto [start, end] = distribute_reflector_application_work<ArchitectureSpec>(trailing_cols, nb, nt, thread_id, min_columns_per_thread, weight_factor);

				if (start < end) {
					// Each thread updates part of trailing matrix
					auto thread_update = pctx.a.sub_matrix(
						prev_block_idx, prev_panel_length,
						prev_block_idx + start, end - start
					);

					// Get workspace for this thread
					auto work_space = general_matrix{
						matrix_base{
							work_memory + thread_id * required_workspace_per_thread,
							thread_update.strd(),
							prev_block_size,
							1,
							thread_update.strd()
						}
					};
					auto reflector = factorization::block_reflector {
						to_const(householder_mgr.get_previous_block()),
						to_const(reflector_mgr.get_previous_reflector()),
						block_reflector_layout::leading_unit_lower
					};

					// Apply block reflector to the block
					auto larfb_pctx = spec::problem_context {
						apply_block_reflector {
							reflector,
							thread_update,
							work_space
						},
						pctx.architecture_spec
					};
					compute(larfb_pctx);
				}

				#pragma omp barrier

				// Update managers and move to next block
				#pragma omp single
				{
					reflector_mgr.update_previous_reflectors();
					householder_mgr.update_previous_blocks();
					current_block_idx -= nb;
				}
			}

			// Apply the final reflector (block 0)
			// It is only application, the block reflectors have been generated
			// in the previous iteration.
			if (total_blocks > 0) {
				// Apply reflector of the first block
				const auto prev_panel_length = householder_mgr.get_previous_panel_length();
				const auto prev_block_size   = householder_mgr.get_previous_panel_width();
				      auto trailing_cols           = pctx.a.strd();

				// Distribute work among threads
				auto split    = make_parallel_split_min_work(trailing_cols, nt, min_columns_per_thread, split_options::chunk_fill);

				if (thread_id < split.threads) {
					auto [ start, size ] = work_distribution(thread_id, split);
					auto thread_update = pctx.a.sub_matrix(
						0, prev_panel_length,
						start, size
					);

					auto work_space = general_matrix{
						matrix_base{
							work_memory + thread_id * required_workspace_per_thread,
							thread_update.strd(),
							prev_block_size,
							1,
							thread_update.strd()
						}
					};
					auto reflector = factorization::block_reflector {
						to_const(householder_mgr.get_previous_block()),
						to_const(reflector_mgr.get_previous_reflector()),
						block_reflector_layout::leading_unit_lower
					};

					auto larfb_pctx = spec::problem_context {
						apply_block_reflector {
							reflector,
							thread_update,
							work_space
						},
						pctx.architecture_spec
					};
					compute(larfb_pctx);
				}
			}
		}

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext&) const {
		return false;
	}

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool can_compute(const generate_q_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		// Only use parallel implementation when we're not already in a parallel region,
		// have more than one thread available
		const auto spec = get_spec(spec::strategy_tag<generate_q_from_qr_generic>{}, pctx);
		return !omp::in_parallel() && spec.max_threads > 1;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const {
		return false;
	}
}; // class generate_q_from_qr_parallel

} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_GENERATE_Q_FROM_QR_PARALLEL_HPP

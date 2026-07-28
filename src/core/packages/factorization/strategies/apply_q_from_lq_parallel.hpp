/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_APPLY_Q_FROM_LQ_PARALLEL_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_APPLY_Q_FROM_LQ_PARALLEL_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"
#include "framework/parallel.hpp"
#include "packages/factorization/problem_context_bases.hpp"
#include "packages/factorization/helpers/qr_block_reflector_manager.hpp"
#include "packages/factorization/helpers/matrix_distribution.hpp"
#include "matrix/adaptors.hpp"

#include "detect/omp.hpp"
#include "perflibs_util.hpp"
#include "perflibs_numeric_utils.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class apply_q_from_lq_parallel {
	template<typename MatrixType, typename ArchitectureSpec>
	using apply_q_pctx_t = spec::problem_context<apply_q_from_lq<MatrixType>, ArchitectureSpec>;

public:
	static constexpr std::string_view name() { return "apply_q_from_lq_parallel"; }

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const apply_q_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		// Quick return
		if (empty(pctx.c)) return true;

		using value_type = typename MatrixType::value_type;

		const bool left   = (pctx.side == PERFLIBS_LEFT);
		const bool notran = !is_trans(pctx.trans);

		const auto spec    = get_spec(spec::strategy_tag<apply_q_from_lq_generic>{}, pctx);
		const auto npanels = max(iround_div(pctx.a.cntg(), spec.block_size), 2);
		const auto nb      = iround_div(pctx.a.cntg(), npanels);

		// Extract all tuning parameters from spec
		const auto nt                  = spec.max_threads;
		const auto min_work_per_thread = spec.min_update_block_size;
		const auto weight_factor       = spec.weight_factor;

		// Calculate workspace requirements
		const auto ldwork                  = left ? pctx.c.strd() : pctx.c.cntg();
		const auto required_workspace_size = (ldwork * nb) * nt;

		// Check if provided workspace is sufficient
		perflibs::pod_vector<value_type> scratch_work;
		auto work_memory = pctx.work.data();
		if (pctx.work.cntg() < required_workspace_size) {
			// Provided workspace is not large enough, allocate our own
			scratch_work.resize(static_cast<std::size_t>(required_workspace_size));
			work_memory = scratch_work.data();
		}

		// Determine the iteration direction and limits
		kernel_inttype start_idx, end_idx, increment;
		if ((left && notran) || (!left && !notran)) {
			start_idx = 0;
			end_idx = pctx.a.cntg() - 1;  // Iterate over k (rows of A)
			increment = nb;
		}
		else {
			// Start from the last block
			start_idx = ((pctx.a.cntg() - 1) / nb) * nb;
			end_idx = 0;
			increment = -nb;
		}

		// Initialize the block reflector manager
		block_reflector_manager<value_type, ArchitectureSpec> reflector_mgr(nb);

		// Variable to track current block index
		auto current_idx      = start_idx;
		auto remaining_blocks = 1 + ((end_idx - start_idx) / increment);
		auto prev_block_size  = nb;

		// Main parallel region
		#pragma omp parallel default(none) \
		shared(pctx, reflector_mgr, current_idx, prev_block_size, remaining_blocks, left) \
		firstprivate(min_work_per_thread, weight_factor, nb, nt, ldwork, increment, work_memory, notran) \
		num_threads(nt)
		{
			const int thread_id = omp::get_thread_num();

			// First iteration only generate the reflector, there is no application
			#pragma omp single
			{
				// Get the first block size
				const kernel_inttype block_size = std::min(nb, pctx.a.cntg() - current_idx);

				// Initialize reflector for the first block
				reflector_mgr.allocate_current_reflector(block_size);

				// Calculate reflector dimensions
				const auto nq             = left ? pctx.c.cntg() : pctx.c.strd();
				const auto reflector_cols = nq - current_idx;

				// Form the triangular factor for the first block
				std::span<value_type> tau_span(pctx.tau.data() + current_idx, static_cast<size_t>(block_size));
				auto larft_pctx = spec::problem_context{
					form_block_reflector_factor{
						PERFLIBS_FORWARD,
						PERFLIBS_ROWWISE,
						pctx.a.sub_matrix(current_idx, block_size, current_idx, reflector_cols).transpose(),
						tau_span,
						reflector_mgr.get_current_reflector()
					},
					pctx.architecture_spec
				};
				compute(larft_pctx);

				// Update for next iteration
				reflector_mgr.update_previous_reflectors();
				current_idx    += increment;
				remaining_blocks--;
				prev_block_size = block_size;
			}

			while (remaining_blocks > 0) {
				// Get the current block size
				const kernel_inttype block_size = std::min(nb, pctx.a.cntg() - current_idx);

				// Thread 0 generates the next reflector
				if (thread_id == 0) {
					// Allocate storage for the current reflector
					reflector_mgr.allocate_current_reflector(block_size);

					// Calculate reflector dimensions for current block
					const auto nq             = left ? pctx.c.cntg() : pctx.c.strd();
					const auto reflector_cols = nq - current_idx;

					// Form the triangular factor for the current block
					std::span<value_type> tau_span(pctx.tau.data() + current_idx, static_cast<size_t>(block_size));
					auto larft_pctx = spec::problem_context{
						form_block_reflector_factor{
							PERFLIBS_FORWARD,
							PERFLIBS_ROWWISE,
							pctx.a.sub_matrix(current_idx, block_size, current_idx, reflector_cols).transpose(),
							tau_span,
							reflector_mgr.get_current_reflector()
						},
						pctx.architecture_spec
					};
					compute(larft_pctx);
				}

				// All threads contribute to the application of the previous
				// reflector, including thread 0 if needed for load balancing

				// Set up dimensions for reflector application
				kernel_inttype row_idx, nb_rows, col_idx, nb_cols;
				kernel_inttype work_dimension;

				if (left) {
					row_idx        = current_idx - increment;
					nb_rows        = pctx.c.cntg() - row_idx;
					col_idx        = 0;
					nb_cols        = pctx.c.strd();
					work_dimension = nb_cols;
				}
				else {
					row_idx        = 0;
					nb_rows        = pctx.c.cntg();
					col_idx        = current_idx - increment;
					nb_cols        = pctx.c.strd() - col_idx;
					work_dimension = nb_rows;
				}

				auto [start_idx, end_idx] = distribute_reflector_application_work<ArchitectureSpec>(
					work_dimension, prev_block_size, nt, thread_id,
					min_work_per_thread, weight_factor);

				if (end_idx > start_idx) {
					// Get thread-local workspace
					auto thread_workspace = general_matrix {
						matrix_base { work_memory + thread_id * prev_block_size * ldwork, ldwork, prev_block_size, 1, ldwork }
					};

					// Apply the previous block reflector
					const auto prev_idx            = current_idx - increment;
					const auto prev_nq             = left ? pctx.c.cntg() : pctx.c.strd();
					const auto prev_reflector_cols = prev_nq - prev_idx;
					// LQ stores reflector rows in A, so use the adjoint view as V in H = I - V T V^H.
					auto block_v = adjoint(pctx.a.sub_matrix(prev_idx, prev_block_size, prev_idx, prev_reflector_cols));
					auto c_update = left ? pctx.c.sub_matrix(row_idx, nb_rows, col_idx + start_idx, end_idx - start_idx)
					                     : pctx.c.sub_matrix(row_idx + start_idx, end_idx - start_idx, col_idx, nb_cols);
					auto reflector = factorization::block_reflector {
						to_const(block_v),
						is_trans(pctx.trans) ? to_const(reflector_mgr.get_previous_reflector())
						                     : to_const(adjoint(reflector_mgr.get_previous_reflector())),
						block_reflector_layout::leading_unit_lower
					};

					if (left) {
						auto larfb_pctx = spec::problem_context {
							apply_block_reflector {
								reflector,
								c_update,
								thread_workspace
							},
							pctx.architecture_spec
						};
						compute(larfb_pctx);
					}
					else {
						auto larfb_pctx = spec::problem_context {
							apply_block_reflector {
								c_update,
								reflector,
								thread_workspace
							},
							pctx.architecture_spec
						};
						compute(larfb_pctx);
					}
				}


				// Synchronize all threads before moving to the next block
				#pragma omp barrier

				// Update for the next iteration (done by a single thread)
				#pragma omp single
				{
					reflector_mgr.update_previous_reflectors();
					current_idx += increment;
					remaining_blocks--;
					prev_block_size = block_size;
				}
			}

			// Applies the last generated reflector
			kernel_inttype final_idx = current_idx - increment;

			// Set up dimensions for the final application
			kernel_inttype row_idx, nb_rows, col_idx, nb_cols;
			kernel_inttype work_dimension;

			if (left) {
				row_idx        = final_idx;
				nb_rows        = pctx.c.cntg() - row_idx;
				col_idx        = 0;
				nb_cols        = pctx.c.strd();
				work_dimension = nb_cols;
			}
			else {
				row_idx        = 0;
				nb_rows        = pctx.c.cntg();
				col_idx        = final_idx;
				nb_cols        = pctx.c.strd() - col_idx;
				work_dimension = nb_rows;
			}

			// Create an even distribution for all threads for the final application
			auto split = make_parallel_split_min_work(work_dimension, nt, min_work_per_thread, split_options::chunk_fill);

			// Only apply if this thread is needed
			if (thread_id < split.threads) {
				auto [start, size] = work_distribution(thread_id, split);

				// Get thread-local workspace
				auto thread_workspace = general_matrix{
					matrix_base{ work_memory + thread_id * prev_block_size * ldwork, ldwork, prev_block_size, 1, ldwork }
				};

				// Calculate reflector dimensions for final application
				const auto final_nq             = left ? pctx.c.cntg() : pctx.c.strd();
				const auto final_reflector_cols = final_nq - final_idx;

				// Apply the last block reflector
				// LQ stores reflector rows in A, so use the adjoint view as V in H = I - V T V^H.
				auto block_v = adjoint(pctx.a.sub_matrix(final_idx, prev_block_size, final_idx, final_reflector_cols));
				auto c_update = left ? pctx.c.sub_matrix(row_idx, nb_rows, col_idx + start, size)
				                     : pctx.c.sub_matrix(row_idx + start, size, col_idx, nb_cols);
				auto reflector = factorization::block_reflector {
					to_const(block_v),
					is_trans(pctx.trans) ? to_const(reflector_mgr.get_previous_reflector())
					                     : to_const(adjoint(reflector_mgr.get_previous_reflector())),
					block_reflector_layout::leading_unit_lower
				};
				if (left) {
					auto larfb_pctx = spec::problem_context {
						apply_block_reflector {
							reflector,
							c_update,
							thread_workspace
						},
						pctx.architecture_spec
					};
					compute(larfb_pctx);
				}
				else {
					auto larfb_pctx = spec::problem_context {
						apply_block_reflector {
							c_update,
							reflector,
							thread_workspace
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
	constexpr bool operator()(const ProblemContext&) const {
		return false;
	}

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const apply_q_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		const auto spec = get_spec(spec::strategy_tag<apply_q_from_lq_generic>{}, pctx);
		return !omp::in_parallel() && spec.max_threads > 1;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const {
		return false;
	}
};

} // namespace perflibs::linalg::factorization
#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_APPLY_Q_FROM_LQ_PARALLEL_HPP

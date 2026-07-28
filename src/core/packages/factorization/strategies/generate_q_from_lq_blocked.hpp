/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_GENERATE_Q_FROM_LQ_BLOCKED_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_GENERATE_Q_FROM_LQ_BLOCKED_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"
#include "packages/factorization/fwd.hpp"
#include "matrix/adaptors.hpp"
#include "perflibs_numeric_utils.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class generate_q_from_lq_blocked {
	template<typename MatrixType, typename ArchitectureSpec>
	using generate_q_pctx_t = spec::problem_context<generate_q_from_lq<MatrixType>, ArchitectureSpec>;

public:
	static constexpr std::string_view name() { return "generate_q_from_lq_blocked"; }

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const generate_q_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		// Quick return
		if (pctx.a.cntg() == 0) return true;

		using value_type = typename MatrixType::value_type;

		// Determine the block size
		const auto spec     = get_spec(spec::strategy_tag<generate_q_from_lq_generic>{}, pctx);
		const auto npanels  = max(iround_div(pctx.a.cntg(), spec.block_size), 2);
		const auto nb       = iround_div(pctx.a.cntg(), npanels);

		// Determine how many rows will be handled by blocked code
		// The first kk rows are handled by the block method
		const auto last_block_start = ((pctx.tau.cntg() - 1) / nb) * nb;
		const auto kk               = min(pctx.tau.cntg(), last_block_start + nb);

		// Zero the lower triangular part
		for (auto j = 0_ki; j < pctx.a.strd(); ++j) {
			for (auto i = j + 1; i < pctx.a.cntg(); ++i) {
				pctx.a(i, j, write) = zero<value_type>;
			}
		}

		// Setup workspace for blocked operations
		const auto ldwork                  = pctx.a.cntg();
		const auto required_workspace_size = nb * (nb + 1) + ldwork * nb;

		perflibs::pod_vector<value_type> scratch_work;
		auto work_memory = pctx.work.data();
		if (pctx.work.cntg() < required_workspace_size) {
			scratch_work.resize(static_cast<std::size_t>(required_workspace_size));
			work_memory = scratch_work.data();
		}
		auto workspace = general_matrix{ matrix_base{ work_memory, required_workspace_size, 1, 1, 1}};


		// Use unblocked code for the last or only block (the tail)
		// Handle A(kk:m, kk:n) with remaining reflectors k-kk
		if (kk < pctx.a.cntg()) {
			auto tail_pctx = spec::problem_context{
				generate_q_from_lq{
					pctx.a.sub_matrix(kk, pctx.a.cntg() - kk, kk, pctx.a.strd() - kk),
					pctx.tau.sub_matrix(kk, pctx.tau.cntg() - kk, 0, 1),
					pctx.work.sub_matrix(0, pctx.a.cntg() - kk, 0, 1)
				},
				pctx.architecture_spec
			};
			factorization::generate_q_from_lq_unblocked{}(tail_pctx);
		}

		// Return is there is not reflector to apply
		if (pctx.tau.cntg() == 0) return true;

		// Use blocked code for the first kk rows
		// Process blocks in reverse order: from last_block_start down to 0
		for (auto i = last_block_start; i >= 0; i -= nb) {
			const auto block_size = min(nb, pctx.tau.cntg() - i);
			const auto ldt        = block_size + 1;

			// Apply block reflector to rows below if they exist
			if (i + block_size < pctx.a.cntg()) {
				// Form the triangular factor T of the block reflector
				// H = H(i) H(i+1) . . . H(i+block_size-1)
				auto block_reflector = triangular_matrix{
					PERFLIBS_UPPER, PERFLIBS_NOUNIT,
					matrix_base{ workspace.data() + ldwork * block_size, block_size, block_size, 1, ldt }
				};

				// Form the triangular factor of the block reflector
				std::span<value_type> tau_span(pctx.tau.data() + i, static_cast<size_t>(block_size));
				auto larft_pctx = spec::problem_context{
					form_block_reflector_factor{
						PERFLIBS_FORWARD,
						PERFLIBS_ROWWISE,
						pctx.a.sub_matrix(i, block_size, i, pctx.a.strd() - i).transpose(),
						tau_span,
						block_reflector
					},
					pctx.architecture_spec
				};
				compute(larft_pctx);

				// Apply H**H to A(i+block_size:m, i:n) from the right
				auto work_space = general_matrix {
					matrix_base { workspace.data(), ldwork, block_size, 1, ldwork }
				};
				// LQ stores reflector rows in A, so use the adjoint view as V in H = I - V T V^H.
				auto block_v = adjoint(pctx.a.sub_matrix(i, block_size, i, pctx.a.strd() - i));
				auto reflector = factorization::block_reflector {
					to_const(block_v),
					to_const(adjoint(block_reflector)),
					block_reflector_layout::leading_unit_lower
				};
				auto larfb_pctx = spec::problem_context {
					apply_block_reflector {
						pctx.a.sub_matrix(i + block_size, pctx.a.cntg() - i - block_size, i, pctx.a.strd() - i),
						reflector,
						work_space
					},
					pctx.architecture_spec
				};
				compute(larfb_pctx);
			}

			// Apply H**H to columns i:n of current block
			// Generate Q for the current block using unblocked method
			auto block_pctx = spec::problem_context{
				generate_q_from_lq{
					pctx.a.sub_matrix(i, block_size, i, pctx.a.strd() - i),
					pctx.tau.sub_matrix(i, block_size, 0, 1),
					pctx.work,
				},
				pctx.architecture_spec
			};
			factorization::generate_q_from_lq_unblocked{}(block_pctx);
		}
		return true;
	}

	template<typename ProblemContext>
	constexpr bool operator()(const ProblemContext&) const {
		return false;
	}

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const generate_q_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const {
		return false;
	}

}; // class generate_q_from_lq_blocked

} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_GENERATE_Q_FROM_LQ_BLOCKED_HPP

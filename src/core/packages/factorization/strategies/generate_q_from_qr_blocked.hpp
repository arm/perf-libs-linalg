/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_GENERATE_Q_FROM_QR_BLOCKED_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_GENERATE_Q_FROM_QR_BLOCKED_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"
#include "packages/factorization/fwd.hpp"
#include "matrix/adaptors.hpp"

#include "perflibs_util.hpp"
#include "perflibs_numeric_utils.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class generate_q_from_qr_blocked {
	template<typename MatrixType, typename ArchitectureSpec>
	using generate_q_pctx_t = spec::problem_context<generate_q_from_qr<MatrixType>, ArchitectureSpec>;

public:
	static constexpr std::string_view name() { return "generate_q_from_qr_blocked"; }

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const generate_q_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		// Quick return
		if (pctx.a.strd() == 0) return true;

		using value_type = typename MatrixType::value_type;

		// Ensures there are at least two blocks, and then adjusts the block size
		// so that we have near-equal block sizes rather than runt blocks.
		const auto spec    = get_spec(spec::strategy_tag<generate_q_from_qr_generic>{}, pctx);
		const auto npanels = max(iround_div(pctx.a.strd(), spec.block_size), 2);
		const auto nb      = iround_div(pctx.a.strd(), npanels);

		// Determine how many columns will be handled by blocked code
		// The first kk columns are handled by the block method
		const auto k                = pctx.tau.cntg();
		const auto last_block_start = ((k - 1) / nb) * nb;
		const auto kk               = min(k, last_block_start + nb);

		// Set A(1:kk, kk+1:n) to zero
		// This is the upper triangular part that will be handled by blocked operations
		for (auto j = kk; j < pctx.a.strd(); ++j) {
			for (auto i = 0_ki; i < kk; ++i) {
				pctx.a(i, j, write) = zero<value_type>;
			}
		}

		// Setup workspace for blocked operations
		const kernel_inttype ldwork = pctx.a.strd();
		const kernel_inttype required_workspace_size = nb * (nb + 1) + ldwork * nb;

		perflibs::pod_vector<value_type> scratch_work;
		auto work_memory = pctx.work.data();
		if (pctx.work.cntg() < required_workspace_size) {
			scratch_work.resize(static_cast<std::size_t>(required_workspace_size));
			work_memory = scratch_work.data();
		}
		auto workspace = general_matrix{ matrix_base{ work_memory, required_workspace_size, 1, 1, 1}};

		// Use unblocked code for the last or only block (the tail)
		// Handle A(kk:m, kk:n) with remaining reflectors k-kk
		if (kk < pctx.a.strd()) {
			auto tail_pctx = spec::problem_context{
				generate_q_from_qr{
					pctx.a.sub_matrix(kk, pctx.a.cntg() - kk, kk, pctx.a.strd() - kk),
					pctx.tau.sub_matrix(kk, k - kk, 0, 1),
					pctx.work.sub_matrix(0, pctx.a.strd() - kk, 0, 1)
				},
				pctx.architecture_spec
			};
			factorization::generate_q_from_qr_unblocked{}(tail_pctx);
		}

		// Zero the upper triangular part of the first kk columns
		for (auto j = 1_ki; j < kk; ++j) {
			for (auto i = 0_ki; i < j; ++i) {
				pctx.a(i, j, write) = zero<value_type>;
			}
		}

		// Use blocked code for the first kk columns
		// Process blocks in reverse order: from last_block_start down to 0
		for (kernel_inttype i = last_block_start; i >= 0; i -= nb) {
			const auto block_size = min(nb, k - i);
			const auto ldt = block_size + 1;

			// Apply block reflector to columns to the right if they exist
			if (i + block_size < pctx.a.strd()) {
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
						PERFLIBS_COLUMNWISE,
						pctx.a.sub_matrix(i, pctx.a.cntg() - i, i, block_size),
						tau_span,
						block_reflector
					},
					pctx.architecture_spec
				};
				compute(larft_pctx);

				// Apply H to A(i:m, i+block_size:n) from the left
				auto work_space = general_matrix {
					matrix_base { workspace.data(), ldwork, block_size, 1, ldwork }
				};
				auto reflector = factorization::block_reflector {
					to_const(pctx.a.sub_matrix(i, pctx.a.cntg() - i, i, block_size)),
					to_const(block_reflector),
					block_reflector_layout::leading_unit_lower
				};

				auto larfb_pctx = spec::problem_context {
					apply_block_reflector {
						reflector,
						pctx.a.sub_matrix(i, pctx.a.cntg() - i, i + block_size, pctx.a.strd() - i - block_size),
						work_space
					},
					pctx.architecture_spec
				};
				compute(larfb_pctx);
			}

			// Apply H to rows i:m of current block
			// Generate Q for the current block using unblocked method
			auto block_pctx = spec::problem_context{
				generate_q_from_qr{
					pctx.a.sub_matrix(i, pctx.a.cntg() - i, i, block_size),
					pctx.tau.sub_matrix(i, block_size, 0, 1),
					pctx.work,
				},
				pctx.architecture_spec
			};
			factorization::generate_q_from_qr_unblocked{}(block_pctx);
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
}; // class generate_q_from_qr_blocked

} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_GENERATE_Q_FROM_QR_BLOCKED_HPP

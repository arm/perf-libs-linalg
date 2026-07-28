/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_APPLY_Q_FROM_QR_BLOCKED_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_APPLY_Q_FROM_QR_BLOCKED_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"
#include "packages/factorization/fwd.hpp"
#include "matrix/adaptors.hpp"

#include "perflibs_util.hpp"
#include "perflibs_numeric_utils.hpp"

#include <string_view>
namespace perflibs::linalg::factorization {

class apply_q_from_qr_blocked {
	template<typename MatrixType, typename ArchitectureSpec>
	using apply_q_pctx_t = spec::problem_context<apply_q_from_qr<MatrixType>, ArchitectureSpec>;

public:
	static constexpr std::string_view name() { return "apply_q_from_qr_blocked"; }

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const apply_q_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		// Quick return
		if (empty(pctx.c)) return true;

		using value_type = typename MatrixType::value_type;

		const  bool left   = (pctx.side == PERFLIBS_LEFT);
		const  bool notran = !is_trans(pctx.trans);

		// Determine the block size
		// Ensures there are at least two blocks, and then adjusts the block size
		// so that we have near-equal block sizes rather than runt blocks.
		const auto spec    = get_spec(spec::strategy_tag<apply_q_from_qr_generic>{}, pctx);
		const auto npanels = max(iround_div(min(pctx.a.strd(), pctx.a.cntg()), spec.block_size), 2);
		const auto nb      = iround_div(min(pctx.a.strd(), pctx.a.cntg()), npanels);

		const kernel_inttype ldwork = left ? pctx.c.strd(): pctx.c.cntg();
		const kernel_inttype required_workspace_size = nb * (nb + 1) + ldwork * nb;

		// Check if provided workspace is sufficient
		perflibs::pod_vector<value_type> scratch_work;
		auto work_memory = pctx.work.data();
		if (pctx.work.cntg() < required_workspace_size) {
			// Provided workspace is not large enough, allocate our own
			scratch_work.resize(static_cast<std::size_t>(required_workspace_size));
			work_memory = scratch_work.data();
		}
		auto workspace = general_matrix{ matrix_base{ work_memory, required_workspace_size, 1, 1, 1}};

		// Determine the start index, end index, and increment based on side and trans
		kernel_inttype start_idx, end_idx, increment;
		if ((left && !notran) || (!left && notran)) {
			start_idx = 0;
			end_idx   = pctx.a.strd() - 1;
			increment = nb;
		}
		else {
			// Start from the last block
			start_idx = ((pctx.a.strd() - 1) / nb) * nb;
			end_idx   = 0;
			increment = -nb;
		}

		// Initialize dimensions that will be adjusted based on side
		kernel_inttype nb_rows_left = 0, nb_cols_left = 0, row_idx = 0, col_idx = 0;
		if (left) {
			nb_cols_left = pctx.c.strd();
			col_idx      = 0;
		}
		else {
			nb_rows_left = pctx.c.cntg();
			row_idx      = 0;
		}

		// Iterate through blocks
		for (kernel_inttype i = start_idx; increment > 0 ? i <= end_idx : i >= end_idx; i += increment) {
			// Adjust the actual block size
			const kernel_inttype block_size = std::min(nb, pctx.a.strd() - i);
			const kernel_inttype ldt        = block_size + 1;

			// Adjust dimensions based on the side
			if (left) {
				// H(i:i+block_size-1) or H(i:i+block_size-1)**H is applied to C(i:m, 0:n-1)
				nb_rows_left = pctx.c.cntg() - i;
				row_idx      = i;
			}
			else {
				// H(i:i+block_size-1) or H(i:i+block_size-1)**H is applied to C(0:m-1, i:n)
				nb_cols_left = pctx.c.strd() - i;
				col_idx      = i;
			}

			// Form the triangular factor T of the block reflector H = I - V*T*V**H
			// We use the workspace for T.
			auto block_reflector = triangular_matrix{
				PERFLIBS_UPPER, PERFLIBS_NOUNIT,
				matrix_base{ workspace.data() + ldwork * block_size, block_size, block_size, 1, ldt }
			};

			// Form the triangular factor of the block reflector
			// Todo: ensure span in changed to matrix in the class
			std::span<value_type> tau_span(pctx.tau.data()+i, static_cast<size_t>(block_size));
			auto larft_pctx = spec::problem_context{
				form_block_reflector_factor{
					PERFLIBS_FORWARD,
					PERFLIBS_COLUMNWISE,
					pctx.a.sub_matrix(i, pctx.a.cntg()-i, i, block_size),
					//pctx.tau.sub_matrix(i, block_size, 0, 1),
					tau_span,
					block_reflector
				},
				pctx.architecture_spec
			};
			compute(larft_pctx);

			// Apply the block reflector H or H**H
			auto work_space = general_matrix{ matrix_base{ workspace.data(), ldwork, block_size, 1, ldwork } };
			auto block_v    = pctx.a.sub_matrix(i, pctx.a.cntg() - i, i, block_size);
			auto c_update   = pctx.c.sub_matrix(row_idx, nb_rows_left, col_idx, nb_cols_left);
			auto reflector = factorization::block_reflector {
				to_const(block_v),
				is_trans(pctx.trans) ? to_const(adjoint(block_reflector)) : to_const(block_reflector),
				block_reflector_layout::leading_unit_lower
			};

			if (left) {
				auto larfb_pctx = spec::problem_context {
					apply_block_reflector {
						reflector,
						c_update,
						work_space
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
						work_space
					},
					pctx.architecture_spec
				};
				compute(larfb_pctx);
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
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const {
		return false;
	}
}; // class apply_q_from_qr_blocked
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_APPLY_Q_FROM_QR_BLOCKED_HPP

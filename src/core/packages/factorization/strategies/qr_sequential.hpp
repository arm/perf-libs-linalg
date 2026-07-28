/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_QR_SEQUENTIAL_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_QR_SEQUENTIAL_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"
#include "packages/factorization/fwd.hpp"
#include "matrix/adaptors.hpp"
#include "perflibs_util.hpp"
#include "perflibs_numeric_utils.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class qr_sequential {
	template<typename MatrixType, typename ArchitectureSpec>
	using qr_pctx_t = spec::problem_context<qr_factorization<MatrixType>, ArchitectureSpec>;

public:
	static constexpr std::string_view name() { return "qr_sequential"; }

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const qr_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		// Ensures there are at least two blocks, and then adjusts the block size
		// so that we have near-equal block sizes rather than runt blocks.
		const auto spec    = get_spec(spec::strategy_tag<qr_factorization_generic>{}, pctx);
		const auto npanels = max(iround_div(min(pctx.a.strd(), pctx.a.cntg()), spec.block_size), 2);
		const auto nb      = iround_div(min(pctx.a.strd(), pctx.a.cntg()), npanels);

		// Initialize workspace
		using value_type = typename MatrixType::value_type;
		const auto block_cntg                     = min(pctx.a.cntg(), spec.block_size);
		const std::size_t required_workspace_size = pctx.a.strd() * block_cntg;
		auto work                                 = pctx.work;
		perflibs::pod_vector<value_type> scratch_work;

		if (work.size() < required_workspace_size) {
			scratch_work.resize(required_workspace_size);
			work = std::span<value_type>(scratch_work.data(), static_cast<std::size_t>(required_workspace_size));
		}

		// Keep track of the current panel position
		auto a           = pctx.a;
		const auto minmn = min(a.strd(), a.cntg());

		for (auto a_panel_start_point = 0_ki; a_panel_start_point < minmn; a_panel_start_point += nb) {

			const kernel_inttype panel_width = min(nb, minmn - a_panel_start_point);

			// 1. Factorize the current panel
			auto panel = get_strd_panel(a, 0, panel_width);
			auto panel_factorization_pctx = spec::problem_context{
				qr_factorization{ panel, pctx.tau.subspan(a_panel_start_point, panel_width), work },
				pctx.architecture_spec
			};
			compute(panel_factorization_pctx);

			// Break if this was the last panel
			if (a.strd() <= panel_width) {
				break;
			}

			// 2. Apply the transformations to the trailing matrix
			auto trailing_matrix      = get_strd_panel(a, panel_width, a.strd() - panel_width);
			auto trailing_update_pctx = spec::problem_context{
				qr_panel_update{ panel, pctx.tau.subspan(a_panel_start_point, panel_width), trailing_matrix, work },
				pctx.architecture_spec
			};
			compute(trailing_update_pctx);

			// 3. Update matrix for next iteration by removing the factorized part
			a = get_cntg_panel(trailing_matrix, panel_width, trailing_matrix.cntg() - panel_width);
		}
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext&) const { return false; }

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const qr_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		return min(pctx.a.strd(), pctx.a.cntg()) > 1;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class qr_sequential
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_QR_SEQUENTIAL_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_QR_PANEL_UPDATE_SEQUENTIAL_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_QR_PANEL_UPDATE_SEQUENTIAL_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"
#include "packages/factorization/problem_context_bases.hpp"
#include "packages/factorization/strategies.hpp"
#include "matrix/adaptors.hpp"

namespace perflibs::linalg::factorization {

class qr_panel_update_sequential {
	template<typename MatrixType, typename ArchitectureSpec>
	using qr_panel_update_pctx_t = spec::problem_context< qr_panel_update<MatrixType>, ArchitectureSpec>;

public:
	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const qr_panel_update_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		// Form the block reflector T using work array
		auto T = triangular_matrix{
			PERFLIBS_UPPER, PERFLIBS_NOUNIT,
			matrix_base{ pctx.work.data(), pctx.a.strd(), pctx.a.strd(), 1, pctx.a.strd() }
		};

		auto larft_pctx = spec::problem_context{
			form_block_reflector_factor{ PERFLIBS_FORWARD, PERFLIBS_COLUMNWISE, pctx.a, pctx.tau, T },
			pctx.architecture_spec
		};
		compute(larft_pctx);

		// Apply the block reflector to the trailing matrix
		const auto ldwork = pctx.b.strd();
		auto work_space = general_matrix{
			matrix_base{ pctx.work.data() + pctx.a.strd() * pctx.a.strd(), ldwork, pctx.a.strd(), 1, ldwork }
		};
		auto reflector = factorization::block_reflector {
			to_const(pctx.a),
			to_const(adjoint(T)),
			block_reflector_layout::leading_unit_lower
		};

		auto larfb_pctx = spec::problem_context {
			apply_block_reflector {
				reflector,
				pctx.b,
				work_space
			},
			pctx.architecture_spec
		};
		compute(larfb_pctx);

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return true; }
};

} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_QR_PANEL_UPDATE_SEQUENTIAL_HPP

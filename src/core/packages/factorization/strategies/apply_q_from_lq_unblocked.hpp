/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_APPLY_Q_FROM_LQ_UNBLOCKED_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_APPLY_Q_FROM_LQ_UNBLOCKED_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"
#include "packages/factorization/fwd.hpp"
#include "matrix/adaptors.hpp"

#include "perflibs_util.hpp"

#include <string_view>


namespace perflibs::linalg::factorization {

class apply_q_from_lq_unblocked {
	template<typename MatrixType, typename ArchitectureSpec>
	using apply_q_pctx_t = spec::problem_context<apply_q_from_lq<MatrixType>, ArchitectureSpec>;

public:
static constexpr std::string_view name() { return "apply_q_from_lq_unblocked"; }

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const apply_q_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		using value_type = typename MatrixType::value_type;

		// Quick return
		if (empty(pctx.c)) return true;

		const bool left   = (pctx.side == PERFLIBS_LEFT);
		const bool notran = !is_trans(pctx.trans);

		kernel_inttype start_idx, end_idx, increment;
		if ((left && notran) || (!left && !notran)) {
			start_idx = 0;
			end_idx   = pctx.a.cntg() - 1;
			increment = 1;
		}
		else {
			start_idx = pctx.a.cntg() - 1;
			end_idx   = 0;
			increment = -1;
		}

		kernel_inttype nb_rows_left = 0, nb_cols_left = 0, row_idx = 0, col_idx = 0;
		if (left) {
			nb_cols_left = pctx.c.strd();
			col_idx      = 0;
		}
		else {
			nb_rows_left = pctx.c.cntg();
			row_idx      = 0;
		}

		for (kernel_inttype i = start_idx; increment > 0 ? i <= end_idx : i >= end_idx; i += increment) {
			if (left) {
				// H(i) or H(i)**H is applied to C(i:m, 0:n-1)
				nb_rows_left = pctx.c.cntg() - i;
				row_idx = i;
			}
			else {
				// H(i) or H(i)**H is applied to C(0:m-1, i:n)
				nb_cols_left = pctx.c.strd() - i;
				col_idx = i;
			}

			// Save and update A(i,i)
			const auto aii      = pctx.a(i, i);
			pctx.a(i, i, write) = one<value_type>;

			const auto nq               = left ? pctx.c.cntg() : pctx.c.strd();
			const auto reflector_length = nq - i;

			auto householder_vector = to_const(to_conj(pctx.a.sub_matrix(i, 1, i, reflector_length).transpose()));
			auto larf_pctx = spec::problem_context{
				apply_elementary_reflector{
					pctx.side,
					householder_vector,
					pctx.c.sub_matrix(row_idx, nb_rows_left, col_idx, nb_cols_left),
					pctx.work,
					pctx.tau(i, 0)
				},
				pctx.architecture_spec
			};
			compute(larf_pctx);

			// Restore A(i,i)
			pctx.a(i, i, write) = aii;
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
}; // class apply_q_from_lq_unblocked

} // namespace perflibs::linalg::factorization
#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_APPLY_Q_FROM_LQ_UNBLOCKED_HPP
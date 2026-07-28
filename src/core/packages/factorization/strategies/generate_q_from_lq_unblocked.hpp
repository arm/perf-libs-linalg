/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_GENERATE_Q_FROM_LQ_UNBLOCKED_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_GENERATE_Q_FROM_LQ_UNBLOCKED_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"
#include "packages/matmul/strategies.hpp"
#include "packages/factorization/fwd.hpp"
#include "matrix/adaptors.hpp"

#include "perflibs_util.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class generate_q_from_lq_unblocked {

template<typename MatrixType, typename ArchitectureSpec>
using generate_q_pctx_t = spec::problem_context<generate_q_from_lq<MatrixType>, ArchitectureSpec>;

public:
	static constexpr std::string_view name() { return "generate_q_from_lq_unblocked"; }

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const generate_q_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		using value_type = typename MatrixType::value_type;

		// Quick return
		if (pctx.a.cntg() == 0) return true;

		// Step 1.1: Initialize rows k+1:m to rows of the unit matrix
		if (pctx.tau.cntg() <  pctx.a.cntg()) {
			for (auto j = 0_ki; j < pctx.a.strd(); ++j) {
				for (auto l = pctx.tau.cntg(); l < pctx.a.cntg(); ++l) {
					pctx.a(l, j, write) = zero<value_type>;
				}
				if (j > pctx.tau.cntg() - 1 && j < pctx.a.cntg()) {
					pctx.a(j, j, write) = one<value_type>;
				}
			}
		}

		// Step 1.2: Set the lower triangular part to zero
		for (auto j = 0_ki; j < pctx.a.strd(); ++j) {
			for (auto i = j + 1_ki; i < pctx.a.cntg(); ++i) {
				pctx.a(i, j, write) = zero<value_type>;
			}
		}
		// Return is there is not reflector to apply
		if (pctx.tau.cntg() == 0) return true;

		// Step 2: Apply the Householder reflectors in reverse order
		// Starting from the last reflector (k-1) and going backwards to 0
		// Each reflector is applied from the right with conjugate transpose
		for (auto i = pctx.tau.cntg() - 1; i >= 0; --i) {

			const auto householder_size       = pctx.a.strd() - i;
			const auto rows_below_i           = pctx.a.cntg() - i - 1;
			const auto cols_right_of_diagonal = pctx.a.strd() - i - 1;

			// Apply H(i)**H to A(i:m, i:n) from the right
			if (i < pctx.a.strd() - 1) {
				if (i < pctx.a.cntg() - 1) {
					// Save and set A(i,i) = 1 temporarily for the reflector
					pctx.a(i, i, write) = one<value_type>;

					// Apply H(i)**H to A(i+1:m-1, i:n-1) from the right
					auto larf_pctx = spec::problem_context{
						apply_elementary_reflector{
							PERFLIBS_RIGHT,
							to_const(to_conj(pctx.a.sub_matrix(i, 1, i, householder_size).transpose())),
							pctx.a.sub_matrix(i + 1, rows_below_i, i, householder_size),
							pctx.work.sub_matrix(0, rows_below_i, 0, 1),
							pctx.tau(i, 0)
						},
						pctx.architecture_spec
					};
					compute(larf_pctx);
				}

				// Scale A(i, i+1:n-1) by -conj( tau(i) )
				const auto one_ptr = &one<value_type>;
				auto scale_pctx = spec::problem_context{
					matmul::matmul3{
						general_matrix{ matrix_base{ one_ptr, 1_ki, cols_right_of_diagonal, 0_ki, 1_ki } },
						general_matrix{ matrix_base{ one_ptr, 1_ki, 1_ki,                   0_ki, 0_ki } },
						pctx.a.sub_matrix(i, 1, i + 1, cols_right_of_diagonal).transpose(),
						zero<value_type>, - pctx.tau(i, 0),
						zero_mode::scale
					},
					pctx.architecture_spec
				};
				matmul::matmul3_vector_scalar{}(scale_pctx);
			}

			// Set the diagonal element
			pctx.a(i, i, write) = one<value_type> - pctx.tau(i, 0);
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

}; // class generate_q_from_lq_unblocked

} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_GENERATE_Q_FROM_LQ_UNBLOCKED_HPP

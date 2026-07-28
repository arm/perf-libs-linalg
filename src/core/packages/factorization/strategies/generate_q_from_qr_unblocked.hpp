/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_GENERATE_Q_FROM_QR_UNBLOCKED_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_GENERATE_Q_FROM_QR_UNBLOCKED_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"
#include "packages/matmul/strategies.hpp"
#include "packages/factorization/fwd.hpp"
#include "matrix/adaptors.hpp"

#include "perflibs_util.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class generate_q_from_qr_unblocked {

template<typename MatrixType, typename ArchitectureSpec>
using generate_q_pctx_t = spec::problem_context<generate_q_from_qr<MatrixType>, ArchitectureSpec>;

public:
	static constexpr std::string_view name() { return "generate_q_from_qr_unblocked"; }

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const generate_q_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		using value_type = typename MatrixType::value_type;

		// Quick return
		if (pctx.a.strd() == 0) return true;

		// Step 1: Initialize columns k+1:n to columns of the unit matrix
		for (auto j = pctx.tau.cntg(); j < pctx.a.strd(); ++j) {
			// Zero out column j
			for (auto l = 0_ki; l < pctx.a.cntg(); ++l) {
				pctx.a(l, j, write) = zero<value_type>;
			}
			// Set diagonal element to 1
			pctx.a(j, j, write) = one<value_type>;
		}

		// Step 2: Apply the Householder reflectors in reverse order
		// Starting from the last reflector (k-1) and going backwards to 0
		for (auto i = pctx.tau.cntg() - 1; i >= 0; --i) {

			// Apply H(i) to A(i:m, i:n) from the left
			if (i < pctx.a.strd() - 1) {
				// Save and set A(i,i) = 1 temporarily for the reflector
				pctx.a(i, i, write) = one<value_type>;

				// Apply H(i) to A(i:m, i+1:n) from the left
				auto larf_pctx = spec::problem_context{
					apply_elementary_reflector{
						PERFLIBS_LEFT,
						to_const(pctx.a.sub_matrix(i, pctx.a.cntg() - i, i, 1)),
						pctx.a.sub_matrix(i, pctx.a.cntg() - i, i + 1, pctx.a.strd() - i - 1),
						pctx.work.sub_matrix(0, pctx.a.strd() - i - 1, 0, 1),
						pctx.tau(i, 0)
					},
					pctx.architecture_spec
				};
				compute(larf_pctx);
			}

			// Scale A(i+1:m, i) by -tau(i)
			// This implements: A(i+1:m, i) = -tau(i) * A(i+1:m, i)
			if (i < pctx.a.cntg() - 1) {
				const auto one_ptr =  &one<value_type>;
				const auto num_elements = pctx.a.cntg() - i - 1;
				auto scale_pctx = spec::problem_context{
					matmul::matmul3{
						general_matrix{ matrix_base{ one_ptr, 1, num_elements, 0, 1 } },
						general_matrix{ matrix_base{ one_ptr, 1, 1,            0, 0 } },
						pctx.a.sub_matrix(i + 1, num_elements, i, 1),
						zero<value_type>, -pctx.tau(i, 0),
						zero_mode::scale
					},
					pctx.architecture_spec
				};
				matmul::matmul3_vector_scalar{}(scale_pctx);
			}

			// Set the diagonal element
			pctx.a(i, i, write) = one<value_type> - pctx.tau(i, 0);

			// Set A(0:i-1, i) to zero
			for (auto l = 0_ki; l < i; ++l) {
				pctx.a(l, i, write) = zero<value_type>;
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
	constexpr bool can_compute(const generate_q_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const {
		return false;
	}

}; // class generate_q_from_qr_unblocked

} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_GENERATE_Q_FROM_QR_UNBLOCKED_HPP

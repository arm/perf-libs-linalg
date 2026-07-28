/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_BIDIAGONALIZE_UNBLOCKED_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_BIDIAGONALIZE_UNBLOCKED_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"
#include "packages/factorization/problem_context_bases.hpp"
#include "matrix/adaptors.hpp"
#include "perflibs_complex.hpp"
#include "perflibs_util.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class bidiagonalize_unblocked {
	template<typename MatrixType, typename ArchitectureSpec>
	using bidiag_pctx_t = spec::problem_context<bidiagonalization<MatrixType>, ArchitectureSpec>;

public:
	static constexpr std::string_view name() { return "bidiagonalize_unblocked"; }

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const bidiag_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		using value_type = typename MatrixType::value_type;

		// Quick return
		if (empty(pctx.a)) return true;

		if (pctx.a.cntg() >= pctx.a.strd()) {
			// Reduce to upper bidiagonal form
			for (auto i = 0_ki; i < pctx.a.strd(); ++i) {
				// Generate elementary reflector H(i) to annihilate A(i+1:m,i)
				auto alpha = pctx.a(i, i);
				auto reflector_pctx = spec::problem_context{
					generate_reflector{
						alpha,
						pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1),
						pctx.tauq(i, 0, write)
					},
					pctx.architecture_spec
				};
				compute(reflector_pctx);

				pctx.d(i, 0, write) = real(alpha);
				pctx.a(i, i, write) = one<value_type>;

				// Apply H(i)^H to A(i:m,i+1:n) from the left
				if (i < pctx.a.strd() - 1) {
					const auto tauq = conj(pctx.tauq(i, 0));
					auto larf_pctx = spec::problem_context{
						apply_elementary_reflector{
							PERFLIBS_LEFT,
							to_const(pctx.a.sub_matrix(i, pctx.a.cntg()-i, i, 1)),
							pctx.a.sub_matrix(i, pctx.a.cntg()-i, i+1, pctx.a.strd()-i-1),
							pctx.work,
							tauq
						},
						pctx.architecture_spec
					};
					compute(larf_pctx);
				}
				pctx.a(i, i, write) = pctx.d(i, 0);

				if (i < pctx.a.strd() - 1) {
					// Generate elementary reflector G(i) to annihilate A(i,i+2:n)
					const auto col_start    = min(i+2, pctx.a.strd() - 1);
					const auto col_length   = pctx.a.strd()-i-2;

					auto householder_vector = pctx.a.sub_matrix(i, 1, i+1, pctx.a.strd() -i -1).transpose();
					conjugate_in_place(householder_vector);
				    auto alpha          = pctx.a(i, i+1);
					auto reflector_pctx = spec::problem_context{
						generate_reflector{
							alpha,
							pctx.a.sub_matrix(i, 1, col_start, col_length).transpose(),
							pctx.taup(i, 0, write)
						},
						pctx.architecture_spec
					};
					compute(reflector_pctx);

					pctx.e(i, 0, write)   = real(alpha);
					pctx.a(i, i+1, write) = one<value_type>;

					// Apply G(i) to A(i+1:m,i+1:n) from the right
					const auto taup = pctx.taup(i, 0);
					auto larf_pctx  = spec::problem_context{
						apply_elementary_reflector{
							PERFLIBS_RIGHT,
							to_const(householder_vector),
							pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i+1, pctx.a.strd()-i-1),
							pctx.work,
							taup
						},
						pctx.architecture_spec
					};
					compute(larf_pctx);
					conjugate_in_place(householder_vector);
					pctx.a(i, i+1, write) = pctx.e(i, 0);
				}
				else {
					pctx.taup(i, 0, write) = zero<value_type>;
				}
			}
		}
		else {
			// Reduce to lower bidiagonal form
			for (auto i = 0_ki; i < pctx.a.cntg(); ++i) {
				// Generate elementary reflector G(i) to annihilate A(i,i+1:n)
				auto householder_vector = pctx.a.sub_matrix(i, 1, i, pctx.a.strd()-i).transpose();
				conjugate_in_place(householder_vector);
				auto alpha = pctx.a(i, i);

				auto reflector_pctx = spec::problem_context{
					generate_reflector{
						alpha,
						pctx.a.sub_matrix(i, 1, i+1, pctx.a.strd()-i-1).transpose(),
						pctx.taup(i, 0, write)
					},
					pctx.architecture_spec
				};
				compute(reflector_pctx);

				pctx.d(i, 0, write) = real(alpha);
				pctx.a(i, i, write) = one<value_type>;

				// Apply G(i) to A(i+1:m,i:n) from the right
				if (i < pctx.a.cntg() - 1) {
					const auto taup = pctx.taup(i, 0);
					auto larf_pctx  = spec::problem_context{
						apply_elementary_reflector{
							PERFLIBS_RIGHT,
							to_const(householder_vector),
							pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i, pctx.a.strd()-i),
							pctx.work,
							taup
						},
						pctx.architecture_spec
					};
					compute(larf_pctx);
				}
				conjugate_in_place(householder_vector);
				pctx.a(i, i, write) = pctx.d(i, 0);

				if (i < pctx.a.cntg() - 1) {
					// Generate elementary reflector H(i) to annihilate A(i+2:m,i)
					      auto alpha      = pctx.a(i+1, i);
					const auto row_start  = min(i+2, pctx.a.cntg()-1);
					const auto row_length = pctx.a.cntg()-i-2;

					auto reflector_pctx  = spec::problem_context{
						generate_reflector{
							alpha,
							pctx.a.sub_matrix(row_start, row_length, i, 1),
							pctx.tauq(i, 0, write)
						},
						pctx.architecture_spec
					};
					compute(reflector_pctx);

					pctx.e(i, 0, write)   = real(alpha);
					pctx.a(i+1, i, write) = one<value_type>;

					// Apply H(i)^H to A(i+1:m,i+1:n) from the left
					const auto tauq = conj(pctx.tauq(i, 0));
					auto larf_pctx  = spec::problem_context{
						apply_elementary_reflector{
							PERFLIBS_LEFT,
							to_const(pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1)),
							pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i+1, pctx.a.strd()-i-1),
							pctx.work,
							tauq
						},
						pctx.architecture_spec
					};
					compute(larf_pctx);
					pctx.a(i+1, i, write) = pctx.e(i, 0);
				}
				else {
					pctx.tauq(i, 0, write) = zero<value_type>;
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
	constexpr bool can_compute(const bidiag_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const {
		return false;
	}
private:
	// This function is required to conjugate
	template<typename MatrixType>
	static void conjugate_in_place(MatrixType& mat) {
		for (kernel_inttype i = 0; i < mat.cntg(); ++i) {
			for (kernel_inttype j = 0; j < mat.strd(); ++j) {
				mat(i, j, write) = conj(mat(i, j));
			}
		}
	}
}; // class bidiagonalize_unblocked
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_BIDIAGONALIZE_UNBLOCKED_HPP

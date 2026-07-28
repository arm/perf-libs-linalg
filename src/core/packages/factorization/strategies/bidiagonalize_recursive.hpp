/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_BIDIAGONALIZE_RECURSIVE_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_BIDIAGONALIZE_RECURSIVE_HPP

#include "perflibs_numeric_utils.hpp"
#include "framework/compute.hpp"
#include "spec/problem_context.hpp"
#include "packages/factorization/fwd.hpp"
#include "packages/matmul/fwd.hpp"
#include "matrix/adaptors.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class bidiagonalize_recursive {
	template<typename MatrixType, typename ArchitectureSpec>
	using bidiag_pctx_t = spec::problem_context<bidiagonalization<MatrixType>, ArchitectureSpec>;

public:
	static constexpr std::string_view name() { return "bidiagonalize_recursive"; }

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const bidiag_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		using value_type = typename MatrixType::value_type;

		// Quick return if empty
		if (empty(pctx.a)) return true;

		// Ensures there are at least two blocks, and then adjusts the block size
		// so that we have near-equal block sizes rather than runt blocks.
		// When the problem size is smaller than the specified block_size:
		// - We limit the actual block size to at most 16 (if problem size permits)
		// - This gives better performance for small matrices while still maintaining
		//   the recursive factorization approach for very small problems
		const auto spec              = get_spec(spec::strategy_tag<bidiagonalization_generic>{}, pctx);
		const auto problem_size      = min(pctx.a.strd(), pctx.a.cntg());
		const auto target_block_size = (problem_size < spec.block_size) ? min(16, problem_size) : spec.block_size;
		const auto npanels           = max(iround_div(problem_size, target_block_size), 2);
		const auto nb                = iround_div(problem_size, npanels);

		const auto minmn          = min(pctx.a.cntg(), pctx.a.strd());
		const auto x_size         = pctx.a.cntg() * nb;
		const auto y_size         = pctx.a.strd() * nb;
		const auto workspace_size = x_size + y_size;

		// Allocate necessary workspace, in case
		// the provided is not enough.
		auto work        = pctx.work.data();
		auto work_matrix = pctx.work;
		perflibs::pod_vector<value_type> scratch_work;

		if (pctx.work.cntg() * pctx.work.strd() < workspace_size) {
			scratch_work.resize(workspace_size);
			work = scratch_work.data();
			work_matrix = general_matrix{matrix_base{work, workspace_size, 1, 1, 1}};
		}

		auto x = general_matrix{ matrix_base{ work,          pctx.a.cntg(), nb, 1, pctx.a.cntg() } };
		auto y = general_matrix{ matrix_base{ work + x_size, pctx.a.strd(), nb, 1, pctx.a.strd() } };

		// Step 1: Reduce first nb columns/rows using block algorithm
		auto block_pctx = spec::problem_context{
			bidiagonalization_block{
				pctx.a, pctx.d, pctx.e, pctx.tauq, pctx.taup, x, y
			},
			pctx.architecture_spec
		};
		compute(block_pctx);

		// Step 2: Update trailing matrix
		// Update A(nb:m, nb:n)
		// 2.1: A_trailing := A_trailing - V*Y**H
		auto V_update_pctx = spec::problem_context{
			matmul::matmul3{
				to_const(pctx.a.sub_matrix(nb, pctx.a.cntg()-nb, 0, nb).transpose()),
				to_const(to_conj(get_cntg_panel(y, nb, y.cntg()-nb).transpose())),
				pctx.a.sub_matrix(nb, pctx.a.cntg()-nb, nb, pctx.a.strd()-nb),
				-one<value_type>, one<value_type>
			},
			pctx.architecture_spec
		};
		compute(V_update_pctx);

		// 2.2 A_trailing := A_trailing - X*U**H
		auto U_update_pctx = spec::problem_context{
			matmul::matmul3{
				to_const(get_cntg_panel(x, nb, x.cntg() -nb).transpose()),
				to_const(pctx.a.sub_matrix(0, nb, nb, pctx.a.strd()-nb)),
				pctx.a.sub_matrix(nb, pctx.a.cntg()-nb, nb, pctx.a.strd()-nb),
				-one<value_type>, one<value_type>
			},
			pctx.architecture_spec
		};
		compute(U_update_pctx);

		// In NETLIB terms, if m >= n, B is upper bidiagonal
		if (pctx.a.cntg() >= pctx.a.strd()) {
			for (auto i = 0_ki; i < nb; ++i) {
				pctx.a(i, i, write)     = pctx.d(i, 0);
				pctx.a(i, i + 1, write) = pctx.e(i, 0);
			}
		}
		// In NETLIB terms, if m < n, B is lower bidiagonal
		else {
			for (auto i = 0_ki; i < nb; ++i) {
				pctx.a(i, i, write)     = pctx.d(i, 0);
				pctx.a(i + 1, i, write) = pctx.e(i, 0);
			}
		}

		// Step 3: Recursively reduce remaining matrix
		auto recursive_pctx = spec::problem_context{
			bidiagonalization{
				pctx.a.sub_matrix(nb, pctx.a.cntg()-nb, nb, pctx.a.strd()-nb),
				get_cntg_panel(pctx.d, nb, minmn-nb),
				get_cntg_panel(pctx.e, nb, minmn-nb-1),
				get_cntg_panel(pctx.tauq, nb, minmn-nb),
				get_cntg_panel(pctx.taup, nb, minmn-nb),
				work_matrix
			},
			pctx.architecture_spec
		};
		compute(recursive_pctx);
		return true;
	}

	template<typename ProblemContext>
	constexpr bool operator()(const ProblemContext&) const {
		return false;
	}

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const bidiag_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		return min(pctx.a.strd(), pctx.a.cntg()) > 1;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const {
		return false;
	}
}; // class bidiagonalize_recursive
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_BIDIAGONALIZE_RECURSIVE_HPP
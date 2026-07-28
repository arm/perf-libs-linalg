/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_TRIDIAGONALIZE_RECURSIVE_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_TRIDIAGONALIZE_RECURSIVE_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"
#include "packages/factorization/fwd.hpp"
#include "matrix/adaptors.hpp"
#include "perflibs_complex.hpp"
#include "perflibs_util.hpp"
#include "perflibs_numeric_utils.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class tridiagonalize_recursive {
	template<typename MatrixType, typename ArchitectureSpec>
	using tridiagonalize_pctx_t = spec::problem_context<tridiagonalization<MatrixType>, ArchitectureSpec>;

public:
	static constexpr std::string_view name() { return "tridiagonalize_recursive"; }

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const tridiagonalize_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		using value_type = typename MatrixType::value_type;

		// Ensures there are at least two blocks, and then adjusts the block size
		// so that we have near-equal block sizes rather than runt blocks.
		// this has the effect that once the problem size is smaller than the blocksize
		// we are effectively doing a recursive factorisation
		const auto spec              = get_spec(spec::strategy_tag<tridiagonalization_generic>{}, pctx);
		const auto target_block_size = (pctx.a.strd() < spec.block_size) ? min(16, pctx.a.strd()) : spec.block_size;
		const auto npanels           = max(iround_div(pctx.a.strd(), target_block_size), 2);
		const auto panel_size        = iround_div(pctx.a.strd(), npanels);

		// Allocate workspace in case the provided one is not large enough
		const std::size_t optimal_w_size = pctx.a.cntg() * panel_size;
		auto w_matrix                    = pctx.work;
		auto work                        = pctx.work.data();
		perflibs::pod_vector<value_type> scratch_work;

		if (w_matrix.strd() != panel_size ) {
			if (w_matrix.strd()*w_matrix.cntg() < (kernel_inttype)optimal_w_size ){
				scratch_work.resize(optimal_w_size);
				work = scratch_work.data();
			}
			w_matrix = general_matrix{ matrix_base{work, pctx.a.cntg(), panel_size, 1, pctx.a.cntg()}};
		}

		const auto trailing_size  = pctx.a.cntg() - panel_size;
		      auto a_gen          = to_general_matrix(pctx.a);

		if (pctx.a.is_lower()) {
			// Call LATRD on the current matrix
			auto latrd_pctx = spec::problem_context{
				tridiagonalization_block{
					pctx.a,
					get_cntg_panel(pctx.off_diagonal, 0, panel_size),
					get_cntg_panel(pctx.tau, 0, panel_size),
					w_matrix
				},
				pctx.architecture_spec
			};
			compute(latrd_pctx);

			auto trailing_matrix = pctx.a.sub_matrix(panel_size, trailing_size, panel_size, trailing_size);

			// Update the trailing matrix using HER2K
			auto her2k_pctx = spec::problem_context{
				matmul::rank_update_2k{
					to_const(general_matrix{ matrix_base {a_gen.data()+panel_size,    panel_size, trailing_size, pctx.a.strd_step(),   1}    }),
					to_const(general_matrix{ matrix_base {w_matrix.data()+panel_size, panel_size, trailing_size, w_matrix.strd_step(), 1}, true}),
					         trailing_matrix, -one<value_type>, one<value_type>
				},
				pctx.architecture_spec
			};
			compute(her2k_pctx);

			// Copy diagonal elements to D and off-diagonal elements back to A
			copy_elements(pctx, 0_ki, panel_size);

			// Recursive call on the trailing matrix
			auto next_pctx = spec::problem_context{
				tridiagonalization{
					trailing_matrix,
					get_cntg_panel(pctx.diagonal, panel_size, trailing_size),
					get_cntg_panel(pctx.off_diagonal, panel_size, trailing_size -1),
					get_cntg_panel(pctx.tau, panel_size, trailing_size -1),
					w_matrix
				},
				pctx.architecture_spec
			};
			compute(next_pctx);
		}
		else { // Upper case

			// Call LATRD on the current matrix
			auto latrd_pctx = spec::problem_context{
				tridiagonalization_block{
					pctx.a,
					get_cntg_panel(pctx.off_diagonal, 0, pctx.a.cntg()-1),
					get_cntg_panel(pctx.tau, 0, pctx.a.cntg()-1),
					w_matrix
				},
				pctx.architecture_spec
			};
			compute(latrd_pctx);

			auto trailing_matrix = pctx.a.sub_matrix(0, trailing_size, 0, trailing_size);

			auto her2k_pctx = spec::problem_context{
				matmul::rank_update_2k{
					to_const(general_matrix { matrix_base {pctx.a.data()+ pctx.a.strd_step()*trailing_size,  panel_size, trailing_size, pctx.a.strd_step(), 1 }}),
					to_const(general_matrix { matrix_base {w_matrix.data(),                                  panel_size, trailing_size, w_matrix.strd_step(), 1 }, true}),
					         trailing_matrix, -one<value_type>, one<value_type>
				},
				pctx.architecture_spec
			};
			compute(her2k_pctx);

			// Copy diagonal elements to D and off-diagonal elements back to A
			copy_elements(pctx, trailing_size, panel_size);

			// Recursive call on the trailing matrix
			auto next_pctx = spec::problem_context{
				tridiagonalization{
					trailing_matrix,
					get_cntg_panel(pctx.diagonal, 0, trailing_size),
					get_cntg_panel(pctx.off_diagonal, 0, trailing_size - 1),
					get_cntg_panel(pctx.tau, 0, trailing_size - 1),
					w_matrix
				},
				pctx.architecture_spec
			};
			compute(next_pctx);
		}
		return true;
	}

	template<typename ProblemContext>
	constexpr bool operator()(const ProblemContext&) const { return false; }

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const tridiagonalize_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		return pctx.a.strd() > 1;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }

private:
	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	static void copy_elements(const ProblemContext& pctx, const kernel_inttype i,
	                         const kernel_inttype nb) {

		auto a_gen = to_general_matrix(pctx.a);

		if (pctx.a.is_lower()) {
			for (auto j = i; j < i + nb; ++j) {
				// Copy subdiagonal back to A
				a_gen(j + 1, j, write) = pctx.off_diagonal(j, 0);
				// Copy diagonal to D array
				pctx.diagonal(j, 0, write) = real(a_gen(j, j));
			}
		}
		else {
			for (auto j = i; j < i + nb; ++j) {
				// Copy superdiagonal back to A
				a_gen(j - 1, j, write) = pctx.off_diagonal(j - 1, 0);
				// Copy diagonal to D array
				pctx.diagonal(j, 0, write) = real(a_gen(j, j));
			}
		}
	}
}; // class tridiagonalize_recursive
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_TRIDIAGONALIZE_RECURSIVE_HPP

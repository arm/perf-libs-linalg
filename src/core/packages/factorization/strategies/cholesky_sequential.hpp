/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_CHOLESKY_SEQUENTIAL_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_CHOLESKY_SEQUENTIAL_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"

#include "packages/factorization/fwd.hpp"
#include "packages/solve/fwd.hpp"
#include "packages/matmul/fwd.hpp"

#include "matrix/adaptors.hpp"
#include "perflibs_numeric_utils.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class cholesky_sequential {

	template<template<typename> typename MatrixAdaptorType,
		typename ADataType, typename IntType,
		typename ArchitectureSpec
	>
	using cholesky_pctx_t = spec::problem_context<
		cholesky_factorization<MatrixAdaptorType<matrix_base<ADataType>>, IntType>,
		ArchitectureSpec
	>;

public:
	static constexpr std::string_view name() { return "cholesky_sequential"; }

	template<template<typename> typename MatrixAdaptorType,
		typename ADataType, typename IntType,
		typename ArchitectureSpec
	>
	PERFLIBS_LINALG_INLINE
	bool operator()(const cholesky_pctx_t<MatrixAdaptorType, ADataType, IntType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;
		// Ensures there are at least two blocks, and then adjusts the block size
		// so that we have near-equal block sizes rather than runt blocks.
		// this has the effect that once the problem size is smaller than the blocksize
		// we are effectively doing a recursive factorisation
		const auto spec    = get_spec(spec::strategy_tag<cholesky_factorization_generic>{}, pctx);
		const auto npanels = max(iround_div(pctx.a.cntg(), spec.block_size), 2);
		const auto nb      = iround_div(pctx.a.cntg(), npanels);

		// The matrix is subdivided in submatrices below
		//	[ A11 |  *   | *   ]
		//	[ A21 | A22  | *   ]
		//	[ A31 | A32  | A33 ]
		// The algorithm :
		// 1. factorizes the panel A11
		// 2. Update the block column below A11, ([A21, A31]) using using TRSM
		// 3. Update A22 using SYRK: A22 = A22 - A21*A21^T
		// 4. Update A32 with GEMM:  A32 = A32 - A21 * A31^T
		//
		// At the end the same strategy is called on
		//[A21 | A22 | *  ]
		//[A32 | A32 | A33]
		// Note that the full block rows passed to the next iteration as the A32 is
		// required to update A33

		const auto trans = is_hermitian_matrix_v<decltype(pctx.a)> ? PERFLIBS_CONJTRANS : PERFLIBS_TRANS;

		// We unapack the matrix a and reduce it to the
		// submatrix of interest throughout the iterations
		auto a = pctx.a;

		if (a.uplo() == PERFLIBS_LOWER) {
			for(auto a_start_point = 0_ki; a_start_point < pctx.a.cntg(); a_start_point += nb) {
				// Get tile size
				const auto tile_size = min(nb, a.cntg());

				// 1. Create problem context and factorise the top tile
				auto panel = get_strd_panel(a, a_start_point, tile_size);
				auto tile  = get_cntg_panel(panel, 0, tile_size);

				auto tile_factorization_pctx = spec::problem_context{
					cholesky_factorization{ tile, pctx.info },
					pctx.architecture_spec
				};
				compute(tile_factorization_pctx);

				// Update info to global index
				if (pctx.info != 0 ) {
					pctx.info += a_start_point;
					return true;
				}

				// Return if there is no work
				if(a.cntg() <= tile_size) {
					return true;
				}

				// 2. Create problem context and upade with TRSM
				auto b = to_general_matrix(get_cntg_panel(panel, tile_size, a.cntg() - tile_size)).transpose();
				auto a_solve = to_const(triangular_matrix{
					PERFLIBS_UPPER, PERFLIBS_NOUNIT,
					tile.transpose().get_matrix_base(), true
				});

				auto solve_pctx = spec::problem_context{
					solve::solve{ PERFLIBS_RIGHT, trans, a_solve, b, one<ADataType> },
					pctx.architecture_spec
				};
				compute(solve_pctx);

				// Update the next panel using SYRK and GEMM
				const auto start_point_next_tile = a_start_point + tile_size;
				const auto next_tile_size = min(nb, a.cntg() - tile_size);
				auto next_panel = to_general_matrix(a.sub_matrix(tile_size, a.cntg() - tile_size, start_point_next_tile, next_tile_size));

				// Get the matrix A to use in the update of the next panel
				auto a21 = a.sub_matrix(tile_size, next_tile_size, 0, start_point_next_tile).transpose();
				auto a22 = a.sub_matrix(tile_size, next_tile_size, start_point_next_tile, next_tile_size);

				// 3. Update the next tile with SYRK/HERK
				// A22 = A22 - A21*A21^H
				auto herk_pctx = spec::problem_context{
					matmul::matmul3{
						to_const(to_general_matrix(a21)),
						to_const(to_conj(to_general_matrix(a21))), a22,
						-one<ADataType>, one<ADataType>
					},
					pctx.architecture_spec
				};
				compute(herk_pctx);

				// 4. Update the remaining of the next panel with GEMM
				// A32 = A32 - A31 * A21^H
				if (next_panel.cntg() > next_tile_size) {
					auto gemm_start_row = tile_size + next_tile_size;

					auto a31 = to_general_matrix(a.sub_matrix(gemm_start_row, a.cntg() - gemm_start_row, 0, start_point_next_tile).transpose());
					auto a32 = get_cntg_panel(next_panel, next_tile_size, next_panel.cntg() - next_tile_size);

					auto gemm_pctx = spec::problem_context{
						matmul::matmul3{
							to_const(a31),
							to_const(to_conj( to_general_matrix(a21))), a32,
							-one<ADataType>, one<ADataType>
						},
						pctx.architecture_spec
					};
					compute(gemm_pctx);
				}
				// Prepare for next iteration
				// At this stage the remaining matrix to factorize
				// is reduced by the current block row. All the
				// columns are however needed
				a = get_cntg_panel(a, tile_size, a.cntg() - tile_size);
			}
		}
		else { // PERFLIBS_UPPER
			for(auto a_start_point = 0_ki; a_start_point < pctx.a.cntg(); a_start_point += nb) {

				// Get tile size
				const auto tile_size = min(nb, a.strd());

				// 1. Create problem context and factorise the top tile
				// Note that here the stripe (equivalent of panel) is a block row
				auto stripe = get_cntg_panel(a, a_start_point, tile_size);
				auto tile   = get_strd_panel(stripe, 0, tile_size);

				auto tile_factorization_pctx = spec::problem_context{
					cholesky_factorization{ tile, pctx.info },
					pctx.architecture_spec
				};
				compute(tile_factorization_pctx);

				// Update info to global index
				if (pctx.info != 0) {
					pctx.info += a_start_point;
					return true;
				}

				// Return if there is no more work
				if (a.strd() <= tile_size) {
					return true;
				}

				// 2. Create problem context and upade with TRSM
				auto b      = to_general_matrix(get_strd_panel(stripe, tile_size, a.strd() - tile_size));
				auto a_solve = triangular_matrix{ PERFLIBS_UPPER, PERFLIBS_NOUNIT, tile.get_matrix_base(), true };

				auto solve_pctx = spec::problem_context{
					solve::solve{ PERFLIBS_LEFT, trans, to_const(a_solve), b, one<ADataType> },
					pctx.architecture_spec
				};
				compute(solve_pctx);

				// Update the next stripe using SYRK and GEMM
				const auto start_point_next_tile = a_start_point + tile_size;
				const auto next_tile_size        = min(nb, a.strd() - tile_size);
				      auto next_stripe           = to_general_matrix(a.sub_matrix(start_point_next_tile, next_tile_size, tile_size, a.strd() - tile_size));

				// Get the matrix A12 to use in the update of the next stripe
				auto a12 = a.sub_matrix(0, start_point_next_tile, tile_size, next_tile_size);
				auto a22 = a.sub_matrix(start_point_next_tile, next_tile_size, tile_size, next_tile_size);

				// 3 Update the next tile with SYRK/HERK
				// A22 = A22 - A12^H * A12
				auto herk_pctx = spec::problem_context{
					matmul::matmul3{
						to_const(to_conj(to_general_matrix(a12))),
						to_const(to_general_matrix(a12)), a22,
						-one<ADataType>, one<ADataType>
					},
					pctx.architecture_spec
				};
				compute(herk_pctx);

				// 4. Update the remaining of the next stripe using GEMM
				// A23 = A23 - A12^H * A13
				if (next_stripe.strd() > next_tile_size) {
					auto gemm_start_col = tile_size + next_tile_size;

					auto a13 = to_const(to_general_matrix(a.sub_matrix(0, start_point_next_tile, gemm_start_col, a.strd() - gemm_start_col)));
					auto a23 = get_strd_panel(next_stripe, next_tile_size, next_stripe.strd() - next_tile_size);

					auto gemm_pctx = spec::problem_context{
						matmul::matmul3{
							to_const(to_conj(to_general_matrix(a12))),
							a13, a23, -one<ADataType>, one<ADataType>
						},
						pctx.architecture_spec
					};
					compute(gemm_pctx);
				}
				// Prepare for next iteration
				// At this stage the remaining matrix to factorize
				// is reduced by the current block columns. All the
				// columns are however needed
				a = get_strd_panel(a, tile_size, a.strd() - tile_size);
			}
		}
		return true;
	}

	template<typename ProblemContext>
	constexpr bool operator()(const ProblemContext&) const { return false; }

	template<template<typename> typename MatrixAdaptorType,
		typename ADataType,
		typename IntType,
		typename ArchitectureSpec
	>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const cholesky_pctx_t<MatrixAdaptorType, ADataType, IntType, ArchitectureSpec>& pctx) const {
		return pctx.a.strd() > 1;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class cholesky_sequential
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_CHOLESKY_SEQUENTIAL_HPP

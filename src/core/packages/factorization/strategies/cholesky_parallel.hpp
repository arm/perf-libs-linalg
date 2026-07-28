/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_CHOLESKY_PARALLEL_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_CHOLESKY_PARALLEL_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"

#include "packages/factorization/problem_context_bases.hpp"
#include "packages/solve/problem_context_bases.hpp"
#include "packages/matmul/problem_context_bases.hpp"
#include "packages/solve/fwd.hpp"
#include "packages/matmul/fwd.hpp"

#include "matrix/adaptors.hpp"
#include "detect/omp.hpp"
#include "perflibs_numeric_utils.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class cholesky_parallel {

	template<template<typename> typename MatrixAdaptorType,
		typename ADataType,
		typename IntType,
		typename ArchitectureSpec
	>
	using cholesky_pctx_t = spec::problem_context<
		cholesky_factorization<MatrixAdaptorType<matrix_base<ADataType>>, IntType>,
		ArchitectureSpec
	>;

public:
	static constexpr std::string_view name() { return "cholesky_parallel"; }

	template<template<typename> typename MatrixAdaptorType,
		typename ADataType,
		typename IntType,
		typename ArchitectureSpec
	>
	PERFLIBS_LINALG_INLINE bool
	operator()(const cholesky_pctx_t<MatrixAdaptorType, ADataType, IntType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		// Ensures there are at least two blocks, and then adjusts the block size
		// so that we have near-equal block sizes rather than runt blocks.
		// this has the effect that once the problem size is smaller than the blocksize
		// we are effectively doing a recursive factorisation
		const auto spec    = get_spec(spec::strategy_tag<cholesky_factorization_generic>{}, pctx);
		const auto npanels = max(iround_div(pctx.a.cntg(), spec.block_size), 2);
		const auto nb      = iround_div(pctx.a.cntg(), npanels);

		const auto trans = is_hermitian_matrix_v<decltype(pctx.a)> ? PERFLIBS_CONJTRANS : PERFLIBS_TRANS;

		if (pctx.a.uplo() == PERFLIBS_LOWER) {
			#pragma omp parallel default(none) shared(pctx, nb, trans)
			{
				#pragma omp single
				#pragma omp taskgroup
				{
					auto a = pctx.a;
					for(auto a_start_point = 0_ki; a_start_point < pctx.a.cntg(); a_start_point += nb) {

						const auto tile_size = min(nb, a.cntg());
						auto panel           = get_strd_panel(a, a_start_point, tile_size);
						auto tile            = get_cntg_panel(panel, 0, tile_size);
						auto *tile_ptr       = tile.data();

						#pragma omp task depend(inout: tile_ptr[0:tile_size])
						{
							if (pctx.info == 0) {
								auto tile_factorization_pctx = spec::problem_context{
									cholesky_factorization{ tile, pctx.info },
									pctx.architecture_spec
								};
								compute(tile_factorization_pctx);

								if (pctx.info > 0) {
									pctx.info += a_start_point;
									#pragma omp cancel taskgroup
								}
							}
						}

						if(a.cntg() <= tile_size) {
							break;
						}

						auto b_tmp   = get_cntg_panel(panel, tile_size, panel.cntg() - tile_size);
						auto b       = to_general_matrix(b_tmp).transpose();
						auto a_solve = triangular_matrix{
							PERFLIBS_UPPER, PERFLIBS_NOUNIT,
							tile.transpose().get_matrix_base(),
							true
						};

						// Tiled TRSM
						for (auto i = 0_ki; i < b.strd(); i += tile_size) {
							auto  b_tile     = get_strd_panel(b, i, min(tile_size, b.strd() - i));
							auto *b_tile_ptr = b_tile.data();
							#pragma omp task depend(in: tile_ptr[0:tile_size]) depend(inout: b_tile_ptr[0:tile_size])
							{
								auto solve_pctx = spec::problem_context{
									solve::solve{ PERFLIBS_RIGHT, trans, to_const(a_solve), b_tile, ADataType{1} },
									pctx.architecture_spec
								};
								compute(solve_pctx);
							}
						}

						const auto start_point_next_tile = a_start_point + tile_size;
						const auto next_tile_size        = min(nb, a.cntg() - tile_size);
						auto next_panel                  = to_general_matrix(
							a.sub_matrix(tile_size, a.cntg() - tile_size, start_point_next_tile, next_tile_size));

						auto a21 = a.sub_matrix(tile_size, next_tile_size, 0, start_point_next_tile).transpose();
						auto a22 = a.sub_matrix(tile_size, next_tile_size, start_point_next_tile, next_tile_size);

						// Tiled HERK/SYRK
						auto *a22_ptr      = a22.data();
						for (auto i = 0_ki; i < a21.cntg(); i += tile_size) {
							auto  a21_tile     = get_cntg_panel(a21, i, min(tile_size, a21.cntg() - i));
							auto *a21_tile_ptr = a21_tile.data();
							#pragma omp task depend(in: a21_tile_ptr[0:tile_size]) depend(inout: a22_ptr[0:tile_size])
							{
								auto herk_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(to_general_matrix(a21_tile)),
										to_const(to_conj(to_general_matrix(a21_tile))), a22,
										ADataType{-1}, ADataType{1}
									},
									pctx.architecture_spec
								};
								compute(herk_pctx);
							}
						}

						if (next_panel.cntg() > next_tile_size) {
							auto gemm_start_row = tile_size + next_tile_size;

							auto a31 = to_general_matrix(a.sub_matrix(gemm_start_row, a.cntg() - gemm_start_row, 0, start_point_next_tile).transpose());
							auto a32 = get_cntg_panel(next_panel, next_tile_size, next_panel.cntg() - next_tile_size);

							// Tiled GEMM
							// A32 = A32 - A31 * A21^H
							for (auto i = 0_ki; i < a32.cntg(); i += tile_size) {
								auto a32_tile      = get_cntg_panel(a32, i, min(tile_size, a32.cntg() - i));
								auto *a32_tile_ptr = a32_tile.data();
								auto a31_panel     = get_strd_panel(a31, i, min(tile_size, a31.strd() - i));

								for (auto k = 0_ki; k < a31.cntg(); k += tile_size) {
									auto a21_tile      = get_cntg_panel(a21, k, min(tile_size, a21.cntg() - k));
									auto a31_tile      = get_cntg_panel(a31_panel, k, min(tile_size, a31_panel.cntg() - k));
									auto *a21_tile_ptr = a21_tile.data();
									auto *a31_tile_ptr = a31_tile.data();

									#pragma omp task depend(in: a31_tile_ptr[0:tile_size], a21_tile_ptr[0:tile_size]) depend(inout: a32_tile_ptr[0:tile_size])
									{
										auto gemm_pctx = spec::problem_context{
											matmul::matmul3{
												to_const(a31_tile),
												to_const(to_conj(to_general_matrix(a21_tile))),
												a32_tile,
												ADataType{-1}, ADataType{1}
											},
											pctx.architecture_spec
										};
										compute(gemm_pctx);
									}
								}
							}
						}
						// Prepare for next iteration.
						// At this stage the remaining matrix to factorize
						// is reduced by the current block row. All the
						// columns are however needed
						a = get_cntg_panel(a, tile_size, a.cntg() - tile_size);
					}
				}
			}
		}
		else { // PERFLIBS_UPPER
			#pragma omp parallel default(none) shared(pctx, nb, trans)
			{
				#pragma omp single
				#pragma omp taskgroup
				{
					auto a = pctx.a;
					for(auto a_start_point = 0_ki; a_start_point < pctx.a.cntg(); a_start_point += nb) {

						const auto tile_size = min(nb, a.strd());

						auto  stripe     = get_cntg_panel(a, a_start_point, tile_size);
						auto  tile       = get_strd_panel(stripe, 0, tile_size);
						auto *tile_ptr   = tile.data();

						#pragma omp task depend(inout: tile_ptr[0:tile_size])
						{
							if (pctx.info == 0) {
								auto tile_factorization_pctx = spec::problem_context{
									cholesky_factorization{tile, pctx.info },
									pctx.architecture_spec
								};
								compute(tile_factorization_pctx);

								if (pctx.info != 0) {
									pctx.info += a_start_point;
									#pragma omp cancel taskgroup
								}
							}
						}

						if (a.strd() <= tile_size) {
							break;
						}

						// Tiled TRSM
						auto b       = to_general_matrix(get_strd_panel(stripe, tile_size, a.strd() - tile_size));
						auto a_solve = triangular_matrix{ PERFLIBS_UPPER, PERFLIBS_NOUNIT, tile.get_matrix_base(), true };
						for (auto j = 0_ki; j < b.strd(); j += tile_size) {
							auto  b_tile     = get_strd_panel(b, j, min(tile_size, b.strd() - j));
							auto *b_tile_ptr = b_tile.data();

							#pragma omp task depend(in: tile_ptr[0:tile_size]) depend(inout: b_tile_ptr[0:tile_size])
							{
								auto solve_pctx = spec::problem_context{
									solve::solve{ PERFLIBS_LEFT, trans, to_const(a_solve), b_tile, ADataType{1} },
									pctx.architecture_spec
								};
								compute(solve_pctx);
							}
						}

						const auto start_point_next_tile = a_start_point + tile_size;
						const auto next_tile_size        = min(nb, a.strd() - tile_size);

						auto next_stripe = to_general_matrix(a.sub_matrix(start_point_next_tile, next_tile_size, tile_size, a.strd() - tile_size));
						auto a12         = a.sub_matrix(0, start_point_next_tile, tile_size, next_tile_size);
						auto a22         = a.sub_matrix(start_point_next_tile, next_tile_size, tile_size, next_tile_size);

						// Tiled HERK/SYRK
						auto *a22_ptr      = a22.data();
						for (auto i = 0_ki; i < a12.cntg(); i += tile_size) {
							auto  a12_tile     = get_cntg_panel(a12, i, min(tile_size, a12.cntg() - i));
							auto *a12_tile_ptr = a12_tile.data();
							#pragma omp task depend(in: a12_tile_ptr[0:tile_size]) depend(inout: a22_ptr[0:tile_size])
							{
								auto herk_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(to_conj(to_general_matrix(a12_tile))),
										to_const(to_general_matrix(a12_tile)), a22,
										ADataType{-1}, ADataType{1}
									},
									pctx.architecture_spec
								};
								compute(herk_pctx);
							}
						}

						if (next_stripe.strd() > next_tile_size) {
							auto gemm_start_col = tile_size + next_tile_size;
							auto a13            = to_general_matrix(a.sub_matrix(0, start_point_next_tile, gemm_start_col, a.strd() - gemm_start_col));
							auto a23            = get_strd_panel(next_stripe, next_tile_size, next_stripe.strd() - next_tile_size);

							// Tiled GEMM
							// A23 = A23 - A12^H * A13
							for (auto j = 0_ki; j < a23.strd(); j += tile_size) {
									auto a23_tile      = get_strd_panel(a23, j, min(tile_size, a23.strd() -j));
									auto a13_panel     = get_strd_panel(a13, j, min(tile_size, a23.strd() -j));
									auto *a23_tile_ptr = a23_tile.data();

								for (auto i = 0_ki; i < a13.cntg(); i += tile_size) {
									auto  a13_tile     = get_cntg_panel(a13_panel, i, min(tile_size, a13.cntg() -i));
									auto  a12_tile     = get_cntg_panel(a12,       i, min(tile_size, a12.cntg() -i));
									auto *a13_tile_ptr = a13_tile.data();
									auto *a12_tile_ptr = a12_tile.data();

									#pragma omp task depend(in: a13_tile_ptr[0:tile_size]) depend(in: a12_tile_ptr[0:tile_size]) depend(inout: a23_tile_ptr[0:tile_size])
									{
										auto gemm_pctx = spec::problem_context{
											matmul::matmul3{
												to_const(to_conj(to_general_matrix(a12_tile))),
												to_const(a13_tile), a23_tile,
												ADataType{-1}, ADataType{1}
											},
											pctx.architecture_spec
										};
										compute(gemm_pctx);
									}
								}
							}
						}
						// Prepare for next iteration
						// At this stage the remaining matrix to factorize
						// is reduced by the current block columns. All the
						// rows are however needed
						a = get_strd_panel(a, tile_size, a.strd() - tile_size);
					}
				}
			}
		}
		return true;
	}
	template<typename ProblemContext>
	constexpr bool operator()(const ProblemContext&) const {
		return false;
	}

	template<template<typename> typename MatrixAdaptorType,
		typename ADataType,
		typename IntType,
		typename ArchitectureSpec
	>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(
		const cholesky_pctx_t<MatrixAdaptorType, ADataType, IntType, ArchitectureSpec>& pctx) const {
		const auto spec = get_spec(spec::strategy_tag<cholesky_factorization_generic>{}, pctx);
		return !omp::in_parallel() && spec.max_threads > 1;
	}
}; // class cholesky_parallel
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_CHOLESKY_PARALLEL_HPP

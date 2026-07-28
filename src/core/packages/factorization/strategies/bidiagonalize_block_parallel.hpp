/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_BIDIAGONALIZE_BLOCK_PARALLEL_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_BIDIAGONALIZE_BLOCK_PARALLEL_HPP

#include "framework/compute.hpp"
#include "framework/parallel.hpp"
#include "spec/problem_context.hpp"

#include "matrix/adaptors.hpp"
#include "packages/matmul/fwd.hpp"
#include "packages/factorization/problem_context_bases.hpp"
#include "packages/factorization/kernels/vectors_reduction_kernel.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class bidiagonalize_block_parallel {
	template<typename MatrixType, typename ArchitectureSpec>
	using bidiag_pctx_t = spec::problem_context<bidiagonalization_block<MatrixType>, ArchitectureSpec>;

public:
	static constexpr std::string_view name() { return "bidiagonalize_block_parallel"; }

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const bidiag_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		using value_type = typename MatrixType::value_type;

		// These are declared here because pointers to
		// zero and one are required for the scal strategy
		constexpr value_type one{1.0};
		constexpr value_type zero{0.0};

		// Extra memory for the contiguous copy of row i
		// for GEMV operations
		perflibs::pod_vector<value_type> row_i_vec(pctx.a.strd());
		auto row_i = general_matrix { matrix_base {row_i_vec.data(), pctx.a.strd(), 1, 1, 1 } };

		const auto nb             = pctx.x.strd();
		const auto spec           = get_spec(spec::strategy_tag<bidiagonalization_block_generic>{}, pctx);
		const auto min_block_size = spec.min_block_size;
		const auto nt             = spec.max_threads;

		// Allocate temporary memory for GEMVs
		perflibs::pod_vector<value_type> thread_workspace(max(pctx.a.strd(), pctx.a.cntg())*(nt+1));
		value_type* tmp_data  = thread_workspace.data() + nt * max(pctx.a.strd(), pctx.a.cntg());

		// Reduce to upper bidiagonal form
		if (pctx.a.cntg() >= pctx.a.strd()) {

			#pragma omp parallel default(none) firstprivate(one, zero, nb, min_block_size) shared(pctx, row_i, nt, thread_workspace, tmp_data) num_threads(nt)
			{
				const auto thread_id = omp::get_thread_num();
				for (auto i = 0_ki; i < nb; ++i) {
					if (i > 0) {
						//1. Update A(i:m,i)
						// Using a temporary memory space since both GEMVs write to the same output
						const auto a1 = pctx.a.sub_matrix(i, pctx.a.cntg()-i, 0, i).transpose();
						const auto a2 = pctx.x.sub_matrix(i, pctx.a.cntg()-i, 0, i).transpose();
						const auto y  = pctx.a.sub_matrix(i, pctx.a.cntg()-i, i, 1);
						auto y_tmp    = general_matrix { matrix_base { thread_workspace.data(), pctx.a.cntg()-i, 1, 1, 1 } };
						auto split    = make_parallel_split_min_work(y.cntg(), nt, min_block_size, split_options::chunk_fill);

						if (thread_id < split.threads) {
							// 1.1 A(i:m,i) -= A(i:m,1:i) * conj(Y(i,1:i))^T
							auto [ start, size ] = work_distribution(thread_id, split);
							const auto gemv_pctx1 = spec::problem_context{
								matmul::matmul3{
									to_const(get_strd_panel(a1, start, size)),
									to_const(to_conj(pctx.y.sub_matrix(i, 1, 0, i).transpose())),
									get_cntg_panel(y, start, size),
									-one, one
								},
								pctx.architecture_spec
							};
							compute(gemv_pctx1);

							// 1.2 A(i:m,i) -= X(i:m,1:i-1) * A(1:i-1,i)
							const auto gemv_pctx2 = spec::problem_context{
								matmul::matmul3{
									to_const(get_strd_panel(a2, start, size)),
									to_const(pctx.a.sub_matrix(0, i, i, 1)),
									get_cntg_panel(y_tmp, start, size),
									-one, zero
								},
								pctx.architecture_spec
							};
							compute(gemv_pctx2);

							// Add the results of the 2 GEMVs using axpy
							const auto axpy_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(get_cntg_panel(y_tmp, start, size).transpose()),
									general_matrix { matrix_base { &one, 1, 1, 0, 0 } },
									get_cntg_panel(y, start, size),
									one, one
								},
								pctx.architecture_spec
							};
							compute(axpy_pctx);
						}
					}
					#pragma omp barrier
					#pragma omp single
					{
						// Generate reflection Q(i) to annihilate A(i+1:m,i)
						auto alpha          = pctx.a(i, i);
						auto reflector_pctx = spec::problem_context{
							generate_reflector{
								alpha,
								pctx.a.sub_matrix(min(i+1, pctx.a.cntg() -1), pctx.a.cntg()-i-1, i, 1),
								pctx.tauq(i, 0, write)
							},
							pctx.architecture_spec
						};
						compute(reflector_pctx);
						pctx.d(i, 0, write) = real(alpha);
						if (i < pctx.a.strd()-1) {
							pctx.a(i, i, write) = one;
						}
					}
					if (i < pctx.a.strd()-1) {
						{
							// Compute Y(i+1:n,i)
							const auto a  = to_conj(pctx.a.sub_matrix(i, pctx.a.cntg()-i, i+1, pctx.a.strd()-i-1));
							const auto y  = pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, i, 1);
							auto split    = make_parallel_split_min_work(y.cntg(), nt, min_block_size, split_options::chunk_fill);
							if (thread_id < split.threads) {
								auto [ start, size ] = work_distribution(thread_id, split);
								const auto gemv_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(get_strd_panel(a, start, size)),
										to_const(pctx.a.sub_matrix(i, pctx.a.cntg()-i, i, 1)),
										get_cntg_panel(y, start, size),
										one, zero
									},
									pctx.architecture_spec
								};
								compute(gemv_pctx);
							}
						}
						#pragma omp barrier

						if (i > 0) {
							auto y_tmp = general_matrix { matrix_base { thread_workspace.data(), i, 1, 1, 1 } };
							{
								// Y(1:i,i) = A(i:m,1:i)'*A(i:m,i)
								const auto a = to_conj(pctx.a.sub_matrix(i, pctx.a.cntg()-i, 0, i));
								auto y       = pctx.y.sub_matrix(0, i, i, 1);
								auto split   = make_parallel_split_min_work(y.cntg(), nt, min_block_size, split_options::chunk_fill);

								if (thread_id < split.threads){
									auto [ start, size ] = work_distribution(thread_id, split);
									const auto gemv_pctx = spec::problem_context{
										matmul::matmul3{
											to_const(get_strd_panel(a, start, size)),
											to_const(pctx.a.sub_matrix(i, pctx.a.cntg()-i, i, 1)),
											get_cntg_panel(y, start, size),
											one, zero
										},
										pctx.architecture_spec
									};
									compute(gemv_pctx);
								}
							}
							{
								// Y(1:i,i) = A(i:m,1:i)'*A(i:m,i)
								const auto a = to_conj(pctx.x.sub_matrix(i, pctx.a.cntg()-i, 0, i));
								auto y_tmp   = general_matrix { matrix_base { thread_workspace.data(), i, 1, 1, 1 } };
								auto split   = make_parallel_split_min_work(y_tmp.cntg(), nt, min_block_size, split_options::chunk_fill);

								if (thread_id < split.threads){
									auto [ start, size ] = work_distribution(thread_id, split);
									const auto gemv_pctx = spec::problem_context{
										matmul::matmul3{
											to_const(get_strd_panel(a, start, size)),
											to_const(pctx.a.sub_matrix(i, pctx.a.cntg()-i, i, 1)),
											get_cntg_panel(y_tmp, start, size),
											one, zero
										},
										pctx.architecture_spec
									};
									compute(gemv_pctx);
								}
							}
							#pragma omp barrier
							{
								// Y(i+1:n,i) -= Y(i+1:n,1:i)*Y(1:i,i)
								const auto a = pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, 0, i).transpose();
								auto y       = pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, i, 1);
								auto split   = make_parallel_split_min_work(y.cntg(), nt, min_block_size, split_options::chunk_fill);

								if (thread_id < split.threads){
									auto [ start, size ] = work_distribution(thread_id, split);
									const auto gemv_pctx = spec::problem_context{
										matmul::matmul3{
											to_const(get_strd_panel(a, start, size)),
											to_const(pctx.y.sub_matrix(0, i, i, 1)),
											get_cntg_panel(y, start, size),
											-one, one
										},
										pctx.architecture_spec
									};
									compute(gemv_pctx);
								}
							}
							#pragma omp barrier
							{
								// Y(i+1:n,i) -= A(1:i,i+1:n)'*Y(1:i,i)
								const auto a = to_conj(pctx.a.sub_matrix(0, i, i+1, pctx.a.strd()-i-1));
								auto y       = pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, i, 1);
								auto split   = make_parallel_split_min_work(y.cntg(), nt, min_block_size, split_options::chunk_fill);

								if (thread_id < split.threads){
									auto [ start, size ] = work_distribution(thread_id, split);
									const auto gemv_pctx = spec::problem_context{
										matmul::matmul3{
											to_const(get_strd_panel(a, start, size)),
											to_const(y_tmp),
											get_cntg_panel(y, start, size),
											-one, one
										},
										pctx.architecture_spec
									};
									compute(gemv_pctx);
								}
							}
						}
						#pragma omp barrier
						#pragma omp single
						{
							// Scale Y
							auto scal_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(general_matrix{matrix_base{&zero, 1, pctx.a.strd()-i-1, 0, 1}}),
									to_const(general_matrix{matrix_base{&zero, 1, 1, 0, 0}}),
									pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, i, 1),
									zero, pctx.tauq(i, 0),
									zero_mode::scale
								},
								pctx.architecture_spec
							};
							compute(scal_pctx);

							// Update A(i,i+1:n)
							// A(i,i+1:n) -= Y(i+1:n,1:i) * A(i,1:i)^H
							// explicit conjugation of the householder vector
							// Make a copy of row i into a contiguous memory
							const auto copy_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(get_cntg_panel(pctx.a, i, 1)),
									to_const(general_matrix{matrix_base{&one, 1, 1, 0, 0}}),
									row_i, one, zero
								},
								pctx.architecture_spec
							};
							compute(copy_pctx);
							auto householder_vec = get_cntg_panel(row_i, i+1, pctx.a.strd()-i-1 );
							conjugate_in_place<ArchitectureSpec>(householder_vec);
							const auto gemv_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, 0, i+1).transpose()),
									to_const(to_conj(get_cntg_panel(row_i, 0, i+1))),
									householder_vec,
									-one, one
								},
								pctx.architecture_spec
							};
							compute(gemv_pctx);

						}
						if(i > 0){
							// A(i,i+1:n) -= A(1:i-1,i+1:n) * X(i,1:i-1)^H
							const auto a         = to_conj(pctx.a.sub_matrix(0, i, i+1, pctx.a.strd()-i-1));
							auto householder_vec = get_cntg_panel(row_i, i+1, pctx.a.strd()-i-1 );
							auto split           = make_parallel_split_min_work(a.strd(), nt, min_block_size, split_options::chunk_fill);

							if (thread_id < split.threads) {
								auto [ start, size ] = work_distribution(thread_id, split);
								const auto gemv_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(get_strd_panel(a, start, size)),
										to_const(to_conj(pctx.x.sub_matrix(i, 1, 0, i).transpose())),
										get_cntg_panel(householder_vec, start, size),
										-one, one
									},
									pctx.architecture_spec
								};
								compute(gemv_pctx);
							}
						}
						#pragma omp barrier
						#pragma omp single
						{
							// Generate reflection P(i)
							auto householder_vec = get_cntg_panel(row_i, i+1, pctx.a.strd()-i-1 );
							auto alpha = householder_vec(0, 0);
							auto reduced_householder_vec = get_cntg_panel(row_i, min(i+2, pctx.a.strd()-1), pctx.a.strd()-i-2);
							auto reflector_pctx = spec::problem_context{
								generate_reflector{
									alpha,
									reduced_householder_vec,
									pctx.taup(i, 0, write)
								},
								pctx.architecture_spec
							};
							compute(reflector_pctx);

							pctx.e(i, 0, write)   = real(alpha);
							//pctx.a(i, i+1, write) = one;
							householder_vec(0, 0, write) = one;
						}
						// Compute X(i+1:m,i)
						{
							// X(i+1:m,i) = A(i+1:m,i+1:n) * A(i,i+1:n)^H
							const auto rows            = pctx.a.cntg()-i-1;
							const auto cols            = pctx.a.strd()-i-1;
							const auto a               = pctx.a.sub_matrix(i+1, rows, i+1, cols).transpose();
							const auto x               = pctx.x.sub_matrix(i+1, rows, i, 1);
							auto householder_vec       = get_cntg_panel(row_i, i+1, cols);
							value_type* thread_section = thread_workspace.data() + thread_id * rows;
							auto thread_results        = general_matrix { matrix_base { thread_section, rows, 1, 1, 1 } };
							auto split                 = make_parallel_split_min_work(a.cntg(), nt, min_block_size, split_options::chunk_fill);

							std::fill(thread_section, thread_section + rows, zero);
							if (thread_id < split.threads) {
								auto [ start, size ] = work_distribution(thread_id, split);
								const auto gemv_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(get_cntg_panel(a, start, size)),
										to_const(get_cntg_panel(householder_vec, start, size)),
										thread_results,
										one, zero
									},
									pctx.architecture_spec
								};
								compute(gemv_pctx);
							}
							#pragma omp barrier
							vectors_reduction<ArchitectureSpec>(thread_workspace.data(), x, rows, nt);
						}
						#pragma omp barrier
						{
							// X(1:i,i) = Y(i+1:n,1:i)^H * A(i,i+1:n)^H
							auto householder_vec = get_cntg_panel(row_i, i+1, pctx.a.strd()-i-1 );
							const auto a         = to_conj(pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, 0, i+1));
							auto  tmp_x          = general_matrix { matrix_base { tmp_data, i+1, 1, 1, 1 } };
							auto split           = make_parallel_split_min_work(tmp_x.cntg(), nt, min_block_size, split_options::chunk_fill);

							if (thread_id < split.threads){
								auto [ start, size ] = work_distribution(thread_id, split);
								const auto gemv_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(get_strd_panel(a, start, size)),
										to_const(householder_vec),
										get_cntg_panel(tmp_x, start, size),
										one, zero
									},
									pctx.architecture_spec
								};
								compute(gemv_pctx);
							}
						}
						if(i > 0) {
							// X(1:i,i) = A(1:i-1,i+1:n) * A(i,i+1:n)^H
							{
								const auto rows            = i;
								const auto cols            = pctx.a.strd()-i-1;
								const auto a               = pctx.a.sub_matrix(0, rows, i+1, cols).transpose();
								const auto x               = pctx.x.sub_matrix(0, rows, i, 1);
								auto householder_vec       = get_cntg_panel(row_i, i+1, cols);
								value_type* thread_section = thread_workspace.data() + thread_id * rows;
								auto thread_results        = general_matrix { matrix_base { thread_section, rows, 1, 1, 1 } };
								auto split                 = make_parallel_split_min_work(a.cntg(), nt, min_block_size, split_options::chunk_fill);

								std::fill(thread_section, thread_section + rows, zero);
								if (thread_id < split.threads) {
									auto [ start, size ] = work_distribution(thread_id, split);
									const auto gemv_pctx = spec::problem_context{
										matmul::matmul3{
											to_const(get_cntg_panel(a, start, size)),
											to_const(get_cntg_panel(householder_vec, start, size)),
											thread_results,
											one, zero
										},
										pctx.architecture_spec
									};
									compute(gemv_pctx);
								}
								#pragma omp barrier
								vectors_reduction<ArchitectureSpec>(thread_workspace.data(), x, rows, nt);
							}
						}
						#pragma omp barrier
						{
							// X(i+1:m,i) -= A(i+1:m,1:i) * X(1:i,i)
							const auto a      = pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, 0, i+1).transpose();
							       auto tmp_x = general_matrix { matrix_base { tmp_data, i+1, 1, 1, 1 } };
							       auto x     = pctx.x.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1);
							       auto split = make_parallel_split_min_work(x.cntg(), nt, min_block_size, split_options::chunk_fill);

							if (thread_id < split.threads){
								auto [ start, size ] = work_distribution(thread_id, split);
								const auto gemv_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(get_strd_panel(a, start, size)),
										to_const(tmp_x),
										get_cntg_panel(x, start, size),
										-one, one
									},
									pctx.architecture_spec
								};
								compute(gemv_pctx);
							}
						}
						#pragma omp barrier
						if(i > 0) {
							#pragma omp single
							{
								// X(i+1:m,i) -= X(i+1:m,1:i-1) * X(1:i-1,i)
								const auto gemv_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(pctx.x.sub_matrix(i+1, pctx.a.cntg()-i-1, 0, i).transpose()),
										to_const(pctx.x.sub_matrix(0, i, i, 1)),
										pctx.x.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1),
										-one, one
									},
									pctx.architecture_spec
								};
								compute(gemv_pctx);
							}
						}
						#pragma omp single
						{
							// Scale X(i+1:m,i) by taup(i)
							const auto scal_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(general_matrix{matrix_base{&zero, 1, pctx.a.cntg()-i-1, 0, 1}}),
									to_const(general_matrix{matrix_base{&zero, 1, 1, 0, 0}}),
									pctx.x.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1),
									zero, pctx.taup(i, 0),
									zero_mode::scale
								},
								pctx.architecture_spec
							};
							compute(scal_pctx);

							// explicit conjugation of the householder vector
							// and copy back to matrix A
							{
								auto householder_vec = get_cntg_panel(row_i, i+1, pctx.a.strd()-i-1 );
								conjugate_in_place<ArchitectureSpec>(householder_vec);
								const auto copy_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(householder_vec.transpose()),
										to_const(general_matrix{matrix_base{&one, 1, 1, 0, 0}}),
										pctx.a.sub_matrix(i, 1, i+1, pctx.a.strd()-i-1).transpose(),
										one, zero
									},
									pctx.architecture_spec
								};
								compute(copy_pctx);
							}
						} // pragma single
					} // end if (i < pctx.a.strd()-1)
				} // main for loop
			} // pragma omp parallel
		}
		else {
			// Reduce to lower bidiagonal form
			#pragma omp parallel default(none) firstprivate(one, zero, nb, min_block_size) shared(pctx, row_i, nt, thread_workspace, tmp_data) num_threads(nt)
			{
				const auto thread_id = omp::get_thread_num();
				for (auto i = 0_ki; i < nb; ++i) {
					// Update A(i,i:n)
					// A(i,i:n) -= Y(i:n,1:i) * A(i,1:i)^H
					// Explicit conjugation of the Householder vector
					// Make a copy of the row into a contiguous memory
					#pragma omp single
					{
						const auto copy_pctx = spec::problem_context{
							matmul::matmul3{
								to_const(get_cntg_panel(pctx.a, i, 1)),
								to_const(general_matrix{matrix_base{&one, 1, 1, 0, 0}}),
								row_i, one, zero
							},
							pctx.architecture_spec
						};
						compute(copy_pctx);
						auto householder_vec = get_cntg_panel(row_i, i, pctx.a.strd() -i);
						conjugate_in_place<ArchitectureSpec>(householder_vec);
					}

					if (i > 0) {
						// First GEMV: A(i,i:n) -= Y(i:n,1:i) * A(i,1:i)^H
						{
							const auto a               = pctx.y.sub_matrix(i, pctx.a.strd()-i, 0, i).transpose();
							      auto householder_vec = get_cntg_panel(row_i, i, pctx.a.strd() -i);
							      auto split           = make_parallel_split_min_work(householder_vec.cntg(), nt, min_block_size, split_options::chunk_fill);

							if (thread_id < split.threads) {
								auto [ start, size ] = work_distribution(thread_id, split);
								const auto gemv_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(get_strd_panel(a, start, size)),
										to_const(to_conj(get_cntg_panel(row_i, 0, i))),
										get_cntg_panel(householder_vec, start, size),
										-one, one
									},
									pctx.architecture_spec
								};
								compute(gemv_pctx);
							}
						}
						#pragma omp barrier
						// Second GEMV: A(i,i:n) -= A(1:i,i:n) * X(i,1:i)^H
						{
							const auto a               = to_conj(pctx.a.sub_matrix(0, i, i, pctx.a.strd()-i));
							      auto householder_vec = get_cntg_panel(row_i, i, pctx.a.strd() -i);
							      auto split           = make_parallel_split_min_work(householder_vec.cntg(), nt, min_block_size, split_options::chunk_fill);

							if (thread_id < split.threads) {
								auto [ start, size ] = work_distribution(thread_id, split);
								const auto gemv_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(get_strd_panel(a, start, size)),
										to_const(to_conj(pctx.x.sub_matrix(i, 1, 0, i).transpose())),
										get_cntg_panel(householder_vec, start, size),
										-one, one
									},
									pctx.architecture_spec
								};
								compute(gemv_pctx);
							}
						}
					}
					#pragma omp barrier
					#pragma omp single
					{
						// Generate reflection P(i) to annihilate A(i,i+1:n)
						auto householder_vec = get_cntg_panel(row_i, i, pctx.a.strd() -i);
						auto alpha           = householder_vec(0, 0);
						auto reflector_pctx  = spec::problem_context{
							generate_reflector{
								alpha,
								get_cntg_panel(householder_vec, 1, householder_vec.cntg()-1),
								pctx.taup(i, 0, write)
							},
							pctx.architecture_spec
						};
						compute(reflector_pctx);
						pctx.d(i, 0, write) = real(alpha);
						if (i < pctx.a.cntg()-1) {
							householder_vec(0, 0, write) = one;
						}
					}

					if (i < pctx.a.cntg()-1) {

						// Compute X(i+1:m,i)
						{
							// X(i+1:m,i) = A(i+1:m,i:n) * A(i,i:n)^H
							// Parallelize by block column
							const auto rows            = pctx.a.cntg()-i-1;
							const auto cols            = pctx.a.strd()-i;
							const auto a               = pctx.a.sub_matrix(i+1, rows, i, cols).transpose();
							const auto x               = pctx.x.sub_matrix(i+1, rows, i, 1);
							      auto householder_vec = get_cntg_panel(row_i, i, cols);

							value_type* thread_section = thread_workspace.data() + thread_id * rows;
							std::fill(thread_section, thread_section + rows, zero);
							auto thread_results = general_matrix { matrix_base { thread_section, rows, 1, 1, 1 } };
							auto split = make_parallel_split_min_work(a.cntg(), nt, min_block_size, split_options::chunk_fill);

							if (thread_id < split.threads) {
								auto [ start, size ] = work_distribution(thread_id, split);
								const auto gemv_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(get_cntg_panel(a, start, size)),
										to_const(get_cntg_panel(householder_vec, start, size)),
										thread_results,
										one, zero
									},
									pctx.architecture_spec
								};
								compute(gemv_pctx);
							}
							#pragma omp barrier
							vectors_reduction<ArchitectureSpec>(thread_workspace.data(), x, rows, nt);
						}
						#pragma omp barrier

						if ( i > 0) {
							{
								// X(1:i,i) = Y(i:n,1:i)^H * A(i,i:n)^H
								const auto a               = to_conj(pctx.y.sub_matrix(i, pctx.a.strd()-i, 0, i));
								      auto householder_vec = get_cntg_panel(row_i, i, pctx.a.strd() -i);
								      auto tmp_x           = general_matrix { matrix_base { tmp_data, i, 1, 1, 1 } };
								      auto split           = make_parallel_split_min_work(tmp_x.cntg(), nt, min_block_size, split_options::chunk_fill);

								if (thread_id < split.threads) {
									auto [ start, size ] = work_distribution(thread_id, split);
									const auto gemv_pctx = spec::problem_context{
										matmul::matmul3{
											to_const(get_strd_panel(a, start, size)),
											to_const(householder_vec),
											get_cntg_panel(tmp_x, start, size),
											one, zero
										},
										pctx.architecture_spec
									};
									compute(gemv_pctx);
								}
							}
							{
								// X(1:i,i) = A(1:i,i:n) * A(i,i:n)^H
								const auto rows            = i;
								const auto cols            = pctx.a.strd()-i;
								const auto a               = pctx.a.sub_matrix(0, rows, i, cols).transpose();
								const auto x               = pctx.x.sub_matrix(0, rows, i, 1);
								      auto householder_vec = get_cntg_panel(row_i, i, cols);

								value_type* thread_section = thread_workspace.data() + thread_id * rows;
								std::fill(thread_section, thread_section + rows, zero);
								auto thread_results = general_matrix { matrix_base { thread_section, rows, 1, 1, 1 } };
								auto split = make_parallel_split_min_work(a.cntg(), nt, min_block_size, split_options::chunk_fill);

								if (thread_id < split.threads) {
									auto [ start, size ] = work_distribution(thread_id, split);
									const auto gemv_pctx = spec::problem_context{
										matmul::matmul3{
											to_const(get_cntg_panel(a, start, size)),
											to_const(get_cntg_panel(householder_vec, start, size)),
											thread_results,
											one, zero
										},
										pctx.architecture_spec
									};
									compute(gemv_pctx);
								}

								#pragma omp barrier
								vectors_reduction<ArchitectureSpec>(thread_workspace.data(), x, rows, nt);
							}
							#pragma omp barrier
							{
								// X(i+1:m,i) -= A(i+1:m,1:i) * X(1:i,i)
								const auto a     = pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, 0, i).transpose();
								      auto x     = pctx.x.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1);
								      auto tmp_x = general_matrix { matrix_base { tmp_data, i, 1, 1, 1 } };
								      auto split = make_parallel_split_min_work(x.cntg(), nt, min_block_size, split_options::chunk_fill);

								if (thread_id < split.threads) {
									auto [ start, size ] = work_distribution(thread_id, split);
									const auto gemv_pctx = spec::problem_context{
										matmul::matmul3{
											to_const(get_strd_panel(a, start, size)),
											to_const(tmp_x),
											get_cntg_panel(x, start, size),
											-one, one
										},
										pctx.architecture_spec
									};
									compute(gemv_pctx);
								}
							}
							#pragma omp barrier
							{
								// X(i+1:m,i) -= X(i+1:m,1:i) * X(1:i,i)
								const auto a     = pctx.x.sub_matrix(i+1, pctx.a.cntg()-i-1, 0, i).transpose();
								      auto x     = pctx.x.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1);
								      auto split = make_parallel_split_min_work(x.cntg(), nt, min_block_size, split_options::chunk_fill);

								if (thread_id < split.threads) {
									auto [ start, size ] = work_distribution(thread_id, split);
									const auto gemv_pctx = spec::problem_context{
										matmul::matmul3{
											to_const(get_strd_panel(a, start, size)),
											to_const(pctx.x.sub_matrix(0, i, i, 1)),
											get_cntg_panel(x, start, size),
											-one, one
										},
										pctx.architecture_spec
									};
									compute(gemv_pctx);
								}
							}
						}
						#pragma omp barrier
						#pragma omp single
						{
							// Scale X(i+1:m,i)
							auto scal_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(general_matrix{matrix_base{&zero, 1, pctx.a.cntg()-i-1, 0, 1}}),
									to_const(general_matrix{matrix_base{&zero, 1, 1, 0, 0}}),
									pctx.x.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1),
									zero, pctx.taup(i, 0),
									zero_mode::scale
								},
								pctx.architecture_spec
							};
							compute(scal_pctx);

							// Explicit conjugation of the Householder vector
							auto householder_vec = get_cntg_panel(row_i, i, pctx.a.strd() -i);
							conjugate_in_place<ArchitectureSpec>(householder_vec);

							//Copy Householder vector back a
							const auto copy_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(householder_vec.transpose()),
									to_const(general_matrix{matrix_base{&one, 1, 1, 0, 0}}),
									pctx.a.sub_matrix(i, 1, i, pctx.a.strd()-i).transpose(),
									one, zero
								},
								pctx.architecture_spec
							};
							compute(copy_pctx);
						}

						if (i > 0) {
							// Update A(i+1:m,i)
							// A(i+1:m,i) -= A(i+1:m,1:i) * conj(Y(i,1:i))^T
							const auto a      = pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, 0, i).transpose();
							      auto result = pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1);
							      auto split  = make_parallel_split_min_work(result.cntg(), nt, min_block_size, split_options::chunk_fill);

							if (thread_id < split.threads) {
								auto [ start, size ] = work_distribution(thread_id, split);
								const auto gemv_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(get_strd_panel(a, start, size)),
										to_const(to_conj(pctx.y.sub_matrix(i, 1, 0, i).transpose())),
										get_cntg_panel(result, start, size),
										-one, one
									},
									pctx.architecture_spec
								};
								compute(gemv_pctx);
							}
						}
						#pragma omp barrier
						{
							// A(i+1:m,i) -= X(i+1:m,1:i) * A(1:i,i)
							const auto a      = pctx.x.sub_matrix(i+1, pctx.a.cntg()-i-1, 0, i+1).transpose();
							      auto result = pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1);
							      auto split  = make_parallel_split_min_work(result.cntg(), nt, min_block_size, split_options::chunk_fill);

							if (thread_id < split.threads) {
								auto [ start, size ] = work_distribution(thread_id, split);
								const auto gemv_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(get_strd_panel(a, start, size)),
										to_const(pctx.a.sub_matrix(0, i+1, i, 1)),
										get_cntg_panel(result, start, size),
										-one, one
									},
									pctx.architecture_spec
								};
								compute(gemv_pctx);
							}
						}
						#pragma omp barrier
						#pragma omp single
						{
							// Generate reflection Q(i) to annihilate A(i+2:m,i)
							auto alpha          = pctx.a(i+1, i);
							auto reflector_pctx = spec::problem_context{
								generate_reflector{
									alpha,
									pctx.a.sub_matrix(min(pctx.a.cntg()-1, i+2), pctx.a.cntg()-i-2, i, 1),
									pctx.tauq(i, 0, write)
								},
								pctx.architecture_spec
							};
							compute(reflector_pctx);

							pctx.e(i, 0, write) = real(alpha);
							pctx.a(i+1, i, write) = one;
						}
						// Compute Y(i+1:n,i)
						{
							// Y(i+1:n,i) = A(i+1:m,i+1:n)^H * A(i+1:m,i)
							const auto a     = to_conj(pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i+1, pctx.a.strd()-i-1));
							      auto y     = pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, i, 1);
							      auto split = make_parallel_split_min_work(y.cntg(), nt, min_block_size, split_options::chunk_fill);

							if (thread_id < split.threads) {
								auto [ start, size ] = work_distribution(thread_id, split);
								const auto gemv_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(get_strd_panel(a, start, size)),
										to_const(pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1)),
										get_cntg_panel(y, start, size),
										one, zero
									},
									pctx.architecture_spec
								};
								compute(gemv_pctx);
							}
						}
						#pragma omp barrier
						if (i > 0) {
							{
								// Y(1:i,i) = A(i+1:m,1:i)^H * A(i+1:m,i)
								const auto a     = to_conj(pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, 0, i));
								      auto tmp_y = general_matrix { matrix_base { tmp_data, i, 1, 1, 1 } };
								      auto split = make_parallel_split_min_work(tmp_y.cntg(), nt, min_block_size, split_options::chunk_fill);

								if (thread_id < split.threads) {
									auto [ start, size ] = work_distribution(thread_id, split);
									const auto gemv_pctx = spec::problem_context{
										matmul::matmul3{
											to_const(get_strd_panel(a, start, size)),
											to_const(pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1)),
											get_cntg_panel(tmp_y, start, size),
											one, zero
										},
										pctx.architecture_spec
									};
									compute(gemv_pctx);
								}
							}
						}
						{
							// Y(1:i,i) = X(i+1:m,1:i)^H * A(i+1:m,i)
							const auto a    = to_conj(pctx.x.sub_matrix(i+1, pctx.a.cntg()-i-1, 0, i+1));
							      auto y    = pctx.y.sub_matrix(0, i+1, i, 1);
							      auto split = make_parallel_split_min_work(y.cntg(), nt, min_block_size, split_options::chunk_fill);

							if (thread_id < split.threads) {
								auto [ start, size ] = work_distribution(thread_id, split);
								const auto gemv_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(get_strd_panel(a, start, size)),
										to_const(pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1)),
										get_cntg_panel(y, start, size),
										one, zero
									},
									pctx.architecture_spec
								};
								compute(gemv_pctx);
							}
						}
						#pragma omp barrier
						if(i > 0) {
							{
								// Y(i+1:n,i) -= Y(i+1:n,1:i) * Y(1:i,i)
								const auto a     = pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, 0, i).transpose();
								      auto y     = pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, i, 1);
								      auto tmp_y = general_matrix { matrix_base { tmp_data, i, 1, 1, 1 } };
								      auto split = make_parallel_split_min_work(y.cntg(), nt, min_block_size, split_options::chunk_fill);

								if (thread_id < split.threads) {
									auto [ start, size ] = work_distribution(thread_id, split);
									const auto gemv_pctx = spec::problem_context{
										matmul::matmul3{
											to_const(get_strd_panel(a, start, size)),
											to_const(tmp_y),
											get_cntg_panel(y, start, size),
											-one, one
										},
										pctx.architecture_spec
									};
									compute(gemv_pctx);
								}
							}
							#pragma omp barrier
						}
						#pragma omp barrier
						{
							// Y(i+1:n,i) -= A(1:i,i+1:n)^H * Y(1:i,i)
							const auto a = to_conj(pctx.a.sub_matrix(0, i+1, i+1, pctx.a.strd()-i-1));
							auto y       = pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, i, 1);
							auto split   = make_parallel_split_min_work(y.cntg(), nt, min_block_size, split_options::chunk_fill);

							if (thread_id < split.threads) {
								auto [ start, size ] = work_distribution(thread_id, split);
								const auto gemv_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(get_strd_panel(a, start, size)),
										to_const(pctx.y.sub_matrix(0, i+1, i, 1)),
										get_cntg_panel(y, start, size),
										-one, one
									},
									pctx.architecture_spec
								};
								compute(gemv_pctx);
							}
						}
						#pragma omp barrier
						#pragma omp single
						{
							// Scale Y(i+1:n,i)
							auto scal_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(general_matrix{matrix_base{&zero, 1, pctx.a.strd()-i-1, 0, 1}}),
									to_const(general_matrix{matrix_base{&zero, 1, 1, 0, 0}}),
									pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, i, 1),
									zero, pctx.tauq(i, 0),
									zero_mode::scale
								},
								pctx.architecture_spec
							};
							compute(scal_pctx);
						} // single
					}
					else {
						#pragma omp single
						{
							// Explicit conjugation of the Householder vector
							auto householder_vec = get_cntg_panel(row_i, i, pctx.a.strd() -i);
							conjugate_in_place<ArchitectureSpec>(householder_vec);
							{
								//Copy Householder vector back a
								const auto copy_pctx = spec::problem_context{
									matmul::matmul3{
										to_const(householder_vec.transpose()),
										to_const(general_matrix{matrix_base{&one, 1, 1, 0, 0}}),
										pctx.a.sub_matrix(i, 1, i, pctx.a.strd()-i).transpose(),
										one, zero
									},
									pctx.architecture_spec
								};
								compute(copy_pctx);
							}
						} // single
					} // if (i < pctx.a.cntg()-1)
				} // main for loop
			} // #pragma parallel
		} // end main else condition
		return true;
	}

	template<typename ProblemContext>
	constexpr bool operator()(const ProblemContext&) const {
		return false;
	}

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const bidiag_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		const auto spec = get_spec(spec::strategy_tag<bidiagonalization_block_generic>{}, pctx);
		return !omp::in_parallel() && spec.max_threads > 1;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const {
		return false;
	}

private:
	// This function is required to conjugate
	template<typename ArchitectureSpec, typename MatrixType>
	static void conjugate_in_place(MatrixType& mat) {
		if constexpr (perflibs::is_complex_v<typename MatrixType::value_type>) {

			for (kernel_inttype i = 0; i < mat.cntg(); ++i) {
				for (kernel_inttype j = 0; j < mat.strd(); ++j) {
					mat(i, j, write) = conj(mat(i, j));
				}
			}
		}
	}
}; // class bidiagonalize_block_parallel
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_BIDIAGONALIZE_BLOCK_PARALLEL_HPP

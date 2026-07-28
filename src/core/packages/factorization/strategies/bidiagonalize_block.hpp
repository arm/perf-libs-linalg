/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_BIDIAGONALIZE_BLOCK_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_BIDIAGONALIZE_BLOCK_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"
#include "packages/factorization/fwd.hpp"
#include "packages/matmul/fwd.hpp"
#include "matrix/adaptors.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class bidiagonalize_block {
	template<typename MatrixType, typename ArchitectureSpec>
	using bidiag_pctx_t = spec::problem_context<bidiagonalization_block<MatrixType>, ArchitectureSpec>;

public:
	static constexpr std::string_view name() { return "bidiagonalize_block"; }

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

		const auto nb = pctx.x.strd();

		if (pctx.a.cntg() >= pctx.a.strd()) {
			// Reduce to upper bidiagonal form
			for (auto i = 0_ki; i < nb; ++i) {
				if (i > 0) {
					//1. Update A(i:m,i)
					{
						//1.1 A(i:m,i) -= A(i:m,1:i) * conj(Y(i,1:i))^T
						const auto gemv_pctx = spec::problem_context{
							matmul::matmul3{
								to_const(pctx.a.sub_matrix(i, pctx.a.cntg()-i, 0, i).transpose()),
								to_const(to_conj(pctx.y.sub_matrix(i, 1, 0, i).transpose())),
								pctx.a.sub_matrix(i, pctx.a.cntg()-i, i, 1),
								-one, one
							},
							pctx.architecture_spec
						};
						compute(gemv_pctx);
					}
					{
						//1.2 A(i:m,i) -= X(i:m,1:i-1) * A(1:i-1,i)
						const auto gemv_pctx = spec::problem_context{
							matmul::matmul3{
								to_const(pctx.x.sub_matrix(i, pctx.a.cntg()-i, 0, i).transpose()),
								to_const(pctx.a.sub_matrix(0, i, i, 1)),
								pctx.a.sub_matrix(i, pctx.a.cntg()-i, i, 1),
								-one, one
							},
							pctx.architecture_spec
						};
						compute(gemv_pctx);
					}
				}

				// Generate reflection Q(i) to annihilate A(i+1:m,i)
				auto alpha = pctx.a(i, i);
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
					{
						// Compute Y(i+1:n,i)
						const auto gemv_pctx = spec::problem_context{
							matmul::matmul3{
								to_const(to_conj(pctx.a.sub_matrix(i, pctx.a.cntg()-i, i+1, pctx.a.strd()-i-1))),
								to_const(pctx.a.sub_matrix(i, pctx.a.cntg()-i, i, 1)),
								pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, i, 1),
								one, zero
							},
							pctx.architecture_spec
						};
						compute(gemv_pctx);
					}
					if (i > 0) {
						{
							// Y(1:i,i) = A(i:m,1:i)'*A(i:m,i)
							const auto gemv_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(to_conj(pctx.a.sub_matrix(i, pctx.a.cntg()-i, 0, i))),
									to_const(pctx.a.sub_matrix(i, pctx.a.cntg()-i, i, 1)),
									pctx.y.sub_matrix(0, i, i, 1),
									one, zero
								},
								pctx.architecture_spec
							};
							compute(gemv_pctx);
						}
						{
							// Y(i+1:n,i) -= Y(i+1:n,1:i)*Y(1:i,i)
							const auto gemv_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, 0, i).transpose()),
									to_const(pctx.y.sub_matrix(0, i, i, 1)),
									pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, i, 1),
									-one, one
								},
								pctx.architecture_spec
							};
							compute(gemv_pctx);
						}
						{
							// X(1:i,i) = A(i:m,1:i)'*A(i:m,i)
							const auto gemv_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(to_conj(pctx.x.sub_matrix(i, pctx.a.cntg()-i, 0, i))),
									to_const(pctx.a.sub_matrix(i, pctx.a.cntg()-i, i, 1)),
									pctx.y.sub_matrix(0, i, i, 1),
									one, zero
								},
								pctx.architecture_spec
							};
							compute(gemv_pctx);
						}
						{
							// Y(i+1:n,i) -= A(1:i,i+1:n)'*Y(1:i,i)
							const auto gemv_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(to_conj(pctx.a.sub_matrix(0, i, i+1, pctx.a.strd()-i-1))),
									to_const(pctx.y.sub_matrix(0, i, i, 1)),
									pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, i, 1),
									-one, one
								},
								pctx.architecture_spec
							};
							compute(gemv_pctx);
						}
					}
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
					if(i > 0){
						// A(i,i+1:n) -= A(1:i-1,i+1:n) * X(i,1:i-1)^H
						const auto gemv_pctx = spec::problem_context{
							matmul::matmul3{
								to_const(to_conj(pctx.a.sub_matrix(0, i, i+1, pctx.a.strd()-i-1))),
								to_const(to_conj(pctx.x.sub_matrix(i, 1, 0, i).transpose())),
								householder_vec,
								-one, one
							},
							pctx.architecture_spec
						};
						compute(gemv_pctx);
					}
					// Generate reflection P(i)
					alpha = householder_vec(0, 0);
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
					// Compute X(i+1:m,i)
					{
						// X(i+1:m,i) = A(i+1:m,i+1:n) * A(i,i+1:n)^H
						const auto gemv_pctx = spec::problem_context{
							matmul::matmul3{
								to_const(pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i+1, pctx.a.strd()-i-1).transpose()),
								to_const(householder_vec),
								pctx.x.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1),
								one, zero
							},
							pctx.architecture_spec
						};
						compute(gemv_pctx);
					}
					{
						// X(1:i,i) = Y(i+1:n,1:i)^H * A(i,i+1:n)^H
						const auto gemv_pctx = spec::problem_context{
							matmul::matmul3{
								to_const(to_conj(pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, 0, i+1))),
								to_const(householder_vec),
								pctx.x.sub_matrix(0, i+1, i, 1),
								one, zero
							},
							pctx.architecture_spec
						};
						compute(gemv_pctx);
					}
					{
						// X(i+1:m,i) -= A(i+1:m,1:i) * X(1:i,i)
						const auto gemv_pctx = spec::problem_context{
							matmul::matmul3{
								to_const(pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, 0, i+1).transpose()),
								to_const(pctx.x.sub_matrix(0, i+1, i, 1)),
								pctx.x.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1),
								-one, one
							},
							pctx.architecture_spec
						};
						compute(gemv_pctx);
					}
					if(i > 0) {
						{
							// X(1:i,i) = A(1:i-1,i+1:n) * A(i,i+1:n)^H
							const auto gemv_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(pctx.a.sub_matrix(0, i, i+1, pctx.a.strd()-i-1).transpose()),
									to_const(householder_vec),
									pctx.x.sub_matrix(0, i, i, 1),
									one, zero
								},
								pctx.architecture_spec
							};
							compute(gemv_pctx);
						}
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
					// Scale X(i+1:m,i) by taup(i)
					scal_pctx = spec::problem_context{
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
				}
			}
		}
		else {
			 // Reduce to lower bidiagonal form
			for (auto i = 0_ki; i < nb; ++i) {
				// Update A(i,i:n)
				// A(i,i:n) -= Y(i:n,1:i) * A(i,1:i)^H
				// Explicit conjugation of the Householder vector
				// Make a copy of the row into a contiguous memory

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
				if( i > 0) {
					{
						const auto gemv_pctx = spec::problem_context{
							matmul::matmul3{
								to_const(pctx.y.sub_matrix(i, pctx.a.strd()-i, 0, i).transpose()),
								to_const(to_conj(get_cntg_panel(row_i, 0, i))),
								householder_vec,
								-one, one
							},
							pctx.architecture_spec
						};
						compute(gemv_pctx);
					}
					{
						// A(i,i:n) -= A(1:i,i:n) * X(i,1:i)^H
						const auto gemv_pctx = spec::problem_context{
							matmul::matmul3{
								to_const(to_conj(pctx.a.sub_matrix(0, i, i, pctx.a.strd()-i))),
								to_const(to_conj(pctx.x.sub_matrix(i, 1, 0, i).transpose())),
								householder_vec,
								-one, one
							},
							pctx.architecture_spec
						};
						compute(gemv_pctx);
					}
				}
				// Generate reflection P(i) to annihilate A(i,i+1:n)
				auto alpha = householder_vec(0, 0);
				auto reflector_pctx = spec::problem_context{
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

					// Compute X(i+1:m,i)
					// X(i+1:m,i) = A(i+1:m,i:n) * A(i,i:n)^H
					const auto gemv_pctx = spec::problem_context{
						matmul::matmul3{
							to_const(pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i, pctx.a.strd()-i).transpose()),
							to_const(householder_vec),
							pctx.x.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1),
							one, zero
						},
						pctx.architecture_spec
					};
					compute(gemv_pctx);
					if ( i > 0) {
						{
							// X(1:i,i) = Y(i:n,1:i)^H * A(i,i:n)^H
							const auto gemv_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(to_conj(pctx.y.sub_matrix(i, pctx.a.strd()-i, 0, i))),
									to_const(householder_vec),
									pctx.x.sub_matrix(0, i, i, 1),
									one, zero
								},
								pctx.architecture_spec
							};
							compute(gemv_pctx);
						}
						{
							// X(i+1:m,i) -= A(i+1:m,1:i) * X(1:i,i)
							const auto gemv_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, 0, i).transpose()),
									to_const(pctx.x.sub_matrix(0, i, i, 1)),
									pctx.x.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1),
									-one, one
								},
								pctx.architecture_spec
							};
							compute(gemv_pctx);
						}
						{
							// X(1:i,i) = A(1:i,i:n) * A(i,i:n)^H
							const auto gemv_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(pctx.a.sub_matrix(0, i, i, pctx.a.strd()-i).transpose()),
									to_const(householder_vec),
									pctx.x.sub_matrix(0, i, i, 1),
									one, zero
								},
								pctx.architecture_spec
							};
							compute(gemv_pctx);
						}
						{
							// X(i+1:m,i) -= X(i+1:m,1:i) * X(1:i,i)
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

					if ( i > 0){
						// Update A(i+1:m,i)
						// A(i+1:m,i) -= A(i+1:m,1:i) * conj(Y(i,1:i))^T
						const auto gemv_pctx = spec::problem_context{
							matmul::matmul3{
								to_const(pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, 0, i).transpose()),
								to_const(to_conj(pctx.y.sub_matrix(i, 1, 0, i).transpose())),
								pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1),
								-one, one
							},
							pctx.architecture_spec
						};
						compute(gemv_pctx);
					}
					{
						// A(i+1:m,i) -= X(i+1:m,1:i) * A(1:i,i)
						const auto gemv_pctx = spec::problem_context{
							matmul::matmul3{
								to_const(pctx.x.sub_matrix(i+1, pctx.a.cntg()-i-1, 0, i+1).transpose()),
								to_const(pctx.a.sub_matrix(0, i+1, i, 1)),
								pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1),
								-one, one
							},
							pctx.architecture_spec
						};
						compute(gemv_pctx);
					}
					// Generate reflection Q(i) to annihilate A(i+2:m,i)
					alpha = pctx.a(i+1, i);
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

					// Compute Y(i+1:n,i)
					{
						// Y(i+1:n,i) = A(i+1:m,i+1:n)^H * A(i+1:m,i)
						const auto gemv_pctx = spec::problem_context{
							matmul::matmul3{
								to_const(to_conj(pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i+1, pctx.a.strd()-i-1))),
								to_const(pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1)),
								pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, i, 1),
								one, zero
							},
							pctx.architecture_spec
						};
						compute(gemv_pctx);
					}
					if (i > 0) {
						{
							// Y(1:i,i) = A(i+1:m,1:i)^H * A(i+1:m,i)
							const auto gemv_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(to_conj(pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, 0, i))),
									to_const(pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1)),
									pctx.y.sub_matrix(0, i, i, 1),
									one, zero
								},
								pctx.architecture_spec
							};
							compute(gemv_pctx);
						}
						{
							// Y(i+1:n,i) -= Y(i+1:n,1:i) * Y(1:i,i)
							const auto gemv_pctx = spec::problem_context{
								matmul::matmul3{
									to_const(pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, 0, i).transpose()),
									to_const(pctx.y.sub_matrix(0, i, i, 1)),
									pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, i, 1),
									-one, one
								},
								pctx.architecture_spec
							};
							compute(gemv_pctx);
						}
					}
					{
						// Y(1:i,i) = X(i+1:m,1:i)^H * A(i+1:m,i)
						const auto gemv_pctx = spec::problem_context{
							matmul::matmul3{
								to_const(to_conj(pctx.x.sub_matrix(i+1, pctx.a.cntg()-i-1, 0, i+1))),
								to_const(pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1)),
								pctx.y.sub_matrix(0, i+1, i, 1),
								one, zero
							},
							pctx.architecture_spec
						};
						compute(gemv_pctx);
					}
					{
						// Y(i+1:n,i) -= A(1:i,i+1:n)^H * Y(1:i,i)
						const auto gemv_pctx = spec::problem_context{
							matmul::matmul3{
								to_const(to_conj(pctx.a.sub_matrix(0, i+1, i+1, pctx.a.strd()-i-1))),
								to_const(pctx.y.sub_matrix(0, i+1, i, 1)),
								pctx.y.sub_matrix(i+1, pctx.a.strd()-i-1, i, 1),
								-one, one
							},
							pctx.architecture_spec
						};
						compute(gemv_pctx);
					}
					// Scale Y(i+1:n,i)
					scal_pctx = spec::problem_context{
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
				}
				else {
					// Explicit conjugation of the Householder vector
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
}; // class bidiagonalize_block
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_BIDIAGONALIZE_BLOCK_HPP
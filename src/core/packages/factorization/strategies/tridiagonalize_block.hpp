/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_TRIDIAGONALIZE_BLOCK_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_TRIDIAGONALIZE_BLOCK_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"
#include "packages/factorization/fwd.hpp"
#include "packages/matmul/fwd.hpp"
#include "matrix/adaptors.hpp"
#include "perflibs_complex.hpp"

namespace perflibs::linalg::factorization {

class tridiagonalize_block {
template<typename MatrixType, typename ArchitectureSpec>
using tridiagonal_pctx_t = spec::problem_context <tridiagonalization_block<MatrixType>, ArchitectureSpec>;

public:
template<typename MatrixType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
bool operator()(const tridiagonal_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
	if (!can_compute(pctx)) return false;

	using value_type = typename MatrixType::value_type;

	constexpr value_type zero{0.0};
	constexpr value_type one{1.0};
	constexpr value_type half{0.5};

	// Extra memory for the contiguous copy of row i
	// for GEMV operations
	perflibs::pod_vector<value_type> row_w_vec(pctx.a.cntg());
	perflibs::pod_vector<value_type> row_a_vec(pctx.a.cntg());
	auto row_w = general_matrix { matrix_base {row_w_vec.data(), pctx.a.cntg(), 1, 1, 1 } };
	auto row_a = general_matrix { matrix_base {row_a_vec.data(), pctx.a.cntg(), 1, 1, 1 } };

	auto a_gen = to_general_matrix(pctx.a);

	if (pctx.a.is_upper()) {
		for (auto i = pctx.a.cntg() - 1; i >= pctx.a.cntg() - pctx.w.strd(); --i) {
			auto iw = i - (pctx.a.cntg() - pctx.w.strd());

			if (i < pctx.a.cntg()-1) {

				auto len          = pctx.a.cntg()-i-1;
				a_gen(i,i, write) = real(a_gen(i,i));

				{
					// Make a copy of ith row of w and a into a contiguous memory
					const auto copy_pctx = spec::problem_context{
						matmul::matmul3{
							to_const(to_conj(pctx.w.sub_matrix(i, 1, iw+1, len))),
							to_const(general_matrix{matrix_base{&one, 1, 1, 0, 0}}),
							get_cntg_panel(row_w, 0, len),
							one, zero
						},
						pctx.architecture_spec
					};
					compute(copy_pctx);
				}
				{
					// Make a copy of ith row of a and a into a contiguous memory
					const auto copy_pctx = spec::problem_context{
						matmul::matmul3{
							to_const(to_conj(a_gen.sub_matrix(i, 1, i+1, len))),
							to_const(general_matrix{matrix_base{&one, 1, 1, 0, 0}}),
							get_cntg_panel(row_a, 0, len),
							one, zero
						},
						pctx.architecture_spec
					};
					compute(copy_pctx);
				}

				// A(0:i, i) -= A(0:i, i+1:n) * conj(W(i, iw+1:iw+len))^T
				auto gemv_pctx = spec::problem_context{
					matmul::matmul3{
						to_const(a_gen.sub_matrix(0, i+1, i+1, len).transpose()),
						to_const(get_cntg_panel(row_w, 0, len)),
						         a_gen.sub_matrix(0, i+1, i, 1), -one, one
					},
					pctx.architecture_spec
				};
				compute(gemv_pctx);

				//  A(0:i, i) -= W(0:i, iw+1:iw+len) * conj(A(i, i+1:n))^T
				gemv_pctx = spec::problem_context{
					matmul::matmul3{
						to_const(pctx.w.sub_matrix(0, i+1, iw+1, len).transpose()),
						to_const(get_cntg_panel(row_a, 0, len)),
						        a_gen.sub_matrix(0, i+1, i, 1), -one, one
					},
					pctx.architecture_spec
				};
				compute(gemv_pctx);

				// Set diagonal to real again
				a_gen(i,i, write) = real(a_gen(i,i));
			}

			if (i > 0) {
				// Generate elementary reflector
				auto alpha = a_gen(i-1,i);
				auto len   = pctx.a.cntg()-i-1;


				auto reflector_pctx = spec::problem_context{
					generate_reflector{ alpha, a_gen.sub_matrix(0, i-1, i, 1), pctx.tau(i-1, 0, write)},
					pctx.architecture_spec
				};
				compute(reflector_pctx);

				pctx.off_diagonal(i-1, 0, write) = real(alpha);
				a_gen(i-1,i, write)              = one;

				auto householder_vec = a_gen.sub_matrix(0, i, i, 1);

				// W(0:i,iw) := A(0:i,0:i) * A(0:i,i)
				auto hemv_pctx = spec::problem_context {
					matmul::matmul3 {
						to_const(MatrixType { PERFLIBS_LOWER, pctx.a.sub_matrix(0, i, 0, i).transpose().get_matrix_base() }),
						to_const(householder_vec), pctx.w.sub_matrix(0, i, iw, 1), one, zero
					},
					ArchitectureSpec{ }
				};
				compute(hemv_pctx);

				if (i < pctx.a.cntg()-1) {

					// W(i+1:i+len,iw) := conj(A(0:i-1,i+1:i+len))^T * A(0:i-1,i)
					auto gemv_pctx = spec::problem_context{
						matmul::matmul3{
							to_const(to_conj(pctx.w.sub_matrix(0, i, iw+1, len))),
							to_const(householder_vec),
							pctx.w.sub_matrix(i+1, len, iw, 1),
							one, zero
						},
						pctx.architecture_spec
					};
					compute(gemv_pctx);

					 // W(0:i,iw) := W(0:i,iw) - A(0:i,i+1:i+len) * W(i+1:i+len,iw)
					gemv_pctx = spec::problem_context{
						matmul::matmul3{
							to_const(a_gen.sub_matrix(0, i, i+1, len).transpose()),
							to_const(pctx.w.sub_matrix(i+1, len, iw, 1)),
							         pctx.w.sub_matrix(0, i, iw, 1), -one, one
						},
						pctx.architecture_spec
					};
					compute(gemv_pctx);

					// W(i+1:i+len,iw) := conj(A(0:i-1,i+1:i+len))^T * A(0:i-1,i)
					gemv_pctx = spec::problem_context{
						matmul::matmul3{
							to_const(to_conj(a_gen.sub_matrix(0, i, i+1, len))),
							to_const(householder_vec),
							         pctx.w.sub_matrix(i+1, len, iw, 1), one, zero
						},
						pctx.architecture_spec
					};
					compute(gemv_pctx);

					 // W(0:i,iw) := W(0:i,iw) - W(0:i,iw+1:iw+len-1) * W(i+1:i+len,iw)
					gemv_pctx = spec::problem_context{
						matmul::matmul3{
							to_const(pctx.w.sub_matrix(0,   i,   iw+1, len).transpose()),
							to_const(pctx.w.sub_matrix(i+1, len, iw,   1)),
							         pctx.w.sub_matrix(0,   i,   iw,   1), -one, one
						},
						pctx.architecture_spec
					};
					compute(gemv_pctx);
				}

				// Scale W(0:i,iw) by tau(i-1)
				auto scale_pctx = spec::problem_context{
					matmul::matmul3{
						to_const(general_matrix{matrix_base{&zero, 1, i, 0, 1}}),
						to_const(general_matrix{matrix_base{&zero, 1, 1, 0, 0}}),
						         pctx.w.sub_matrix(0, i, iw, 1),
						         zero, pctx.tau(i-1, 0), zero_mode::scale
					},
					pctx.architecture_spec
				};
				compute(scale_pctx);

				// alpha = -0.5 * tau(i-1) * (W(0:i,iw)^H * A(0:i,i))
				alpha = zero;
				auto dot_pctx = spec::problem_context{
					matmul::matmul3{
						to_const(to_conj(pctx.w.sub_matrix(0, i, iw, 1))),
						to_const(a_gen.sub_matrix(0, i, i, 1)),
						         general_matrix{matrix_base{&alpha, 1, 1, 0, 0}}, one, zero
					},
					pctx.architecture_spec
				};
				compute(dot_pctx);

				alpha *= -half * pctx.tau(i-1, 0);

				// W(0:i,iw) := W(0:i,iw) + alpha * A(0:i,i)
				auto axpy_pctx = spec::problem_context{
					matmul::matmul3{
						to_const(householder_vec.transpose()),
						to_const(general_matrix{matrix_base{&one, 1, 1, 0, 0}}),
						         pctx.w.sub_matrix(0, i, iw, 1), alpha, one
					},
					pctx.architecture_spec
				};
				compute(axpy_pctx);
			}
		}
	}
	else{
		// Reduce first NB columns of lower triangle
		for (auto i = 0; i < pctx.w.strd(); ++i) {
			auto len = pctx.a.cntg()-i-1;

			// Set diagonal element of A to real part
			a_gen(i,i, write) = real(a_gen(i,i));

			if (i > 0) {

				{
					// Make a copy of ith row of w and a into a contiguous memory
					const auto copy_pctx = spec::problem_context{
						matmul::matmul3{
							to_const(to_conj(pctx.w.sub_matrix(i, 1, 0, i))),
							to_const(general_matrix{matrix_base{&one, 1, 1, 0, 0}}),
							get_cntg_panel(row_w, 0, i),
							one, zero
						},
						pctx.architecture_spec
					};
					compute(copy_pctx);
				}
				{
					// Make a copy of ith row of a and a into a contiguous memory
					const auto copy_pctx = spec::problem_context{
						matmul::matmul3{
							to_const(to_conj(a_gen.sub_matrix(i, 1, 0, i))),
							to_const(general_matrix{matrix_base{&one, 1, 1, 0, 0}}),
							get_cntg_panel(row_a, 0, i),
							one, zero
						},
						pctx.architecture_spec
					};
					compute(copy_pctx);
				}

				// A(i:n-1,i) := A(i:n-1,i) - A(i:n-1,0:i-1) * conj(W(i,0:i-1))^T
				auto gemv_pctx = spec::problem_context{
					matmul::matmul3{
						to_const(a_gen.sub_matrix(i, pctx.a.cntg()-i, 0, i).transpose()),
						to_const(get_cntg_panel(row_w, 0, i)),
								a_gen.sub_matrix(i, pctx.a.cntg()-i, i, 1), -one, one
					},
					pctx.architecture_spec
				};
				compute(gemv_pctx);

				// A(i:n-1,i) := A(i:n-1,i) - W(i:n-1,0:i-1) * conj(A(i,0:i-1))^T
				gemv_pctx = spec::problem_context{
					matmul::matmul3{
						to_const(pctx.w.sub_matrix(i, pctx.a.cntg()-i, 0, i).transpose()),
						to_const(get_cntg_panel(row_a, 0, i)),
								a_gen.sub_matrix(i, pctx.a.cntg()-i, i, 1), -one, one
					},
					pctx.architecture_spec
				};
				compute(gemv_pctx);
			}

			// Set diagonal to real again
			a_gen(i,i, write) = real(a_gen(i,i));

			if (i < pctx.a.cntg() - 1) {
				// Generate elementary reflector H(i)
				auto alpha = a_gen(i+1, i);
				auto reflector_pctx = spec::problem_context {
					generate_reflector{ alpha, a_gen.sub_matrix(i+2, pctx.a.cntg()-i-2, i, 1), pctx.tau(i, 0, write) },
					pctx.architecture_spec
				};
				compute(reflector_pctx);

				pctx.off_diagonal(i, 0, write) = real(alpha);
				a_gen(i+1,i, write)            = one;

				// W(i+1:n-1,i) := A(i+1:n-1,i+1:n-1) * A(i+1:n-1,i)
				auto hemv_pctx = spec::problem_context{
					matmul::matmul3{
						to_const(MatrixType{ PERFLIBS_UPPER, pctx.a.sub_matrix(i+1, len, i+1, len).transpose().get_matrix_base()}),
						to_const(a_gen.sub_matrix(i+1, len, i, 1)),
						         pctx.w.sub_matrix(i+1, len, i, 1), one, zero
					},
					pctx.architecture_spec
				};
				compute(hemv_pctx);

				if (i > 0) {
					// W(0:i-1,i) = conj(W(i+1:n-1,0:i-1))^T *  A(i+1:n-1,i)
					auto gemv_pctx = spec::problem_context{
						matmul::matmul3{
							to_const(to_conj(pctx.w.sub_matrix(i+1, len, 0, i))),
							to_const(a_gen.sub_matrix(i+1, len, i, 1)),
									pctx.w.sub_matrix(0, i, i, 1), one, zero
						},
						pctx.architecture_spec
					};
					compute(gemv_pctx);

					// W(i+1:n-1,i) := W(i+1:n-1,i) - A(i+1:n-1,0:i) * W(0:i,i)
					gemv_pctx = spec::problem_context{
						matmul::matmul3{
							to_const(a_gen.sub_matrix(i+1, len, 0, i).transpose()),
							to_const(pctx.w.sub_matrix(0, i, i, 1)),
									pctx.w.sub_matrix(i+1, len, i, 1), -one, one
						},
						pctx.architecture_spec
					};
					compute(gemv_pctx);

					// W(0:i,i) := conj(A(i+1:n-1,0:i))^T * A(i+1:n-1,i)
					gemv_pctx = spec::problem_context{
						matmul::matmul3{
							to_const(to_conj(a_gen.sub_matrix(i+1, len, 0, i))),
							to_const(a_gen.sub_matrix(i+1, len, i, 1)),
									pctx.w.sub_matrix(0, i, i, 1), one, zero
						},
						pctx.architecture_spec
					};
					compute(gemv_pctx);

					// W(i+1:n-1,i) := W(i+1:n-1,i) - W(i+1:n-1,0:i) * W(0:i,i)
					gemv_pctx = spec::problem_context{
						matmul::matmul3{
							to_const(pctx.w.sub_matrix(i+1, len, 0, i).transpose()),
							to_const(pctx.w.sub_matrix(0, i, i, 1)),
									pctx.w.sub_matrix(i+1, len, i, 1), -one, one
						},
						pctx.architecture_spec
					};
					compute(gemv_pctx);
				}

				// Scale W(i+1:n-1,i) by tau(i)
				auto scale_pctx = spec::problem_context{
					matmul::matmul3{
						to_const(general_matrix{matrix_base{&zero, 1, len, 0, 1}}),
						to_const(general_matrix{matrix_base{&zero, 1, 1, 0, 0}}),
						         pctx.w.sub_matrix(i+1, len, i, 1), zero, pctx.tau(i, 0), zero_mode::scale
					},
					pctx.architecture_spec
				};
				compute(scale_pctx);

				// alpha = -0.5 * tau(i) * (W(i+1:n-1,i)^H * A(i+1:n-1,i))
				alpha = zero;
				auto dot_pctx = spec::problem_context{
					matmul::matmul3{
						to_const(to_conj(pctx.w.sub_matrix(i+1, len, i, 1))),
						to_const(a_gen.sub_matrix(i+1, len, i, 1)),
						         general_matrix{matrix_base{&alpha, 1, 1, 0, 0}}, one, zero
					},
					pctx.architecture_spec
				};
				compute(dot_pctx);

				alpha *= -half * pctx.tau(i, 0);

				// W(i+1:n-1,i) := W(i+1:n-1,i) + alpha * A(i+1:n-1,i)
				auto axpy_pctx = spec::problem_context{
					matmul::matmul3{
						to_const(a_gen.sub_matrix(i+1, len, i, 1).transpose()),
						to_const(general_matrix{matrix_base{&one, 1, 1, 0, 0}}),
						         pctx.w.sub_matrix(i+1, len, i, 1), alpha, one
					},
					pctx.architecture_spec
				};
				compute(axpy_pctx);
			}
		}
	}
	return true;
}

template<typename ProblemContext>
PERFLIBS_LINALG_INLINE
constexpr bool operator()(const ProblemContext&) const { return false; }

template<typename MatrixType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
constexpr bool can_compute(const tridiagonal_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
	return true;
}

template<typename ProblemContext>
PERFLIBS_LINALG_INLINE
constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class tridiagonalize_block
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_TRIDIAGONALIZE_BLOCK_HPP

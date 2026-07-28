/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_TRIDIAGONALIZE_UNBLOCKED_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_TRIDIAGONALIZE_UNBLOCKED_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"
#include "packages/factorization/fwd.hpp"
#include "matrix/adaptors.hpp"
#include "perflibs_complex.hpp"
#include "perflibs_util.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class tridiagonalize_unblocked {
	template<typename AMatrixType, typename ArchitectureSpec>
	using tridiag_pctx_t = spec::problem_context<tridiagonalization<AMatrixType>, ArchitectureSpec>;

public:
	static constexpr std::string_view name() { return "tridiagonalize_unblocked"; }

	template<typename AMatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const tridiag_pctx_t<AMatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		using value_type          = typename AMatrixType::value_type;
		constexpr value_type one  = value_type(1.0);
		constexpr value_type half = value_type(0.5);

		// Quick return
		if (pctx.a.cntg() <= 0) return true;

		// Allocate workspace in case the provided one is not large enough
		const kernel_inttype needed_work = max<kernel_inttype>(1, pctx.a.cntg()-1);
		auto work_ptr                    = pctx.work.data();
		perflibs::pod_vector<value_type> scratch_work;
		if (pctx.work.cntg() < needed_work) {
			scratch_work.resize(static_cast<std::size_t>(needed_work));
			work_ptr = scratch_work.data();
		}

		// Convert A to a general matrix
		auto a_gen = to_general_matrix(pctx.a);

		if (pctx.a.uplo() == PERFLIBS_UPPER) {
			// Set last diagonal element
			a_gen(pctx.a.cntg()-1,pctx.a.cntg()-1, write) = real(a_gen(pctx.a.cntg()-1,pctx.a.cntg()-1));

			// Reduce the upper triangle, working backwards
			for (auto i = pctx.a.cntg()-1; i > 0; --i) {
				// Generate elementary reflector to annihilate A(1:i-1,i+1)
				auto alpha      = a_gen(i-1, i);
				value_type& tau = pctx.tau(i-1, 0, write);

				auto reflector_pctx = spec::problem_context {
					generate_reflector { alpha, a_gen.sub_matrix(0, i-1, i, 1), tau },
					ArchitectureSpec{ }
				};
				compute(reflector_pctx);

				pctx.off_diagonal(i-1, 0, write) = real(alpha);

				if (tau != zero<value_type>) {
					a_gen(i-1,i, write) = one;

					// Householder vector and work matrices
					auto householder_vec = a_gen.sub_matrix(0, i, i, 1);
					auto work            = general_matrix { matrix_base { work_ptr, i, 1, 1, 0 } };

					// Compute x := tau * A * v, and store the result in work
					auto hemv_pctx = spec::problem_context {
						matmul::matmul3 {
							to_const(AMatrixType { PERFLIBS_LOWER, matrix_base { pctx.a.data(), i, i, pctx.a.strd_step(), 1 } }),
							to_const(householder_vec), work, tau, zero<value_type>
						},
						ArchitectureSpec{ }
					};
					compute(hemv_pctx);

					// w := x - 1/2 * tau * (x^H * v) * v
					auto dot_pctx = spec::problem_context {
						matmul::matmul3 {
							to_const(to_conj(work)),
							to_const(householder_vec),
							         general_matrix { matrix_base { &alpha, 1, 1, 0, 0 } }, one, zero<value_type>
						},
						pctx.architecture_spec
					};
					compute(dot_pctx);

					alpha *= -half * tau;

					// w := w + alpha*v
					auto axpy_pctx = spec::problem_context {
					matmul::matmul3 {
						to_const(householder_vec.transpose()),
						to_const(general_matrix { matrix_base { &one, 1, 1, 0, 0 } }),
						         work, alpha, one
						},
						pctx.architecture_spec
					};
					compute(axpy_pctx);

					// Apply the transformation as a rank-2 update
					// A := A - v * w**H - w * v**H
					auto her2_pctx = spec::problem_context{
						matmul::rank_update_2k {
							to_const(general_matrix             {              matrix_base { householder_vec.data(), 1, i, 0, 1                  }, false }),
							to_const(to_conj(work.transpose())),
							         AMatrixType { PERFLIBS_UPPER, matrix_base { pctx.a.data(),          i, i, 1, pctx.a.strd_step() }        },
							-one, one
						},
						pctx.architecture_spec
					};
					compute(her2_pctx);
				}
				else {
					a_gen(i-1,i-1, write) = real(a_gen(i-1,i-1));
				}
				a_gen(i-1,i, write)        = pctx.off_diagonal(i-1, 0);
				pctx.diagonal(i, 0, write) = real(a_gen(i,i));
			}
			pctx.diagonal(0, 0, write) = real(a_gen(0,0));
		}
		else {  // PERFLIBS_LOWER

			a_gen(0,0, write) = real(a_gen(0,0));

			for (auto i = 0_ki; i < pctx.a.cntg()-1; ++i) {
				// Generate elementary reflector to annihilate A(i+2:n,i)
				auto alpha      = a_gen(i+1, i);
				value_type& tau = pctx.tau(i, 0, write);

				auto reflector_pctx = spec::problem_context {
					generate_reflector{ alpha, a_gen.sub_matrix(min(pctx.a.cntg()-1, i+2), pctx.a.cntg()-i-2, i, 1), tau },
					ArchitectureSpec{ }
				};
				compute(reflector_pctx);

				pctx.off_diagonal(i, 0, write) = real(alpha);

				if (tau != zero<value_type>) {
					a_gen(i+1,i, write) = one;

					auto trailing_matrix = pctx.a.sub_matrix(i+1, pctx.a.cntg()-i-1, i+1, pctx.a.strd()-i-1);
					auto householder_vec = a_gen.sub_matrix(i+1, pctx.a.cntg()-i-1, i, 1);
					auto work            =  general_matrix { matrix_base { work_ptr, pctx.a.cntg()-i-1, 1, 1, 0 } };

					// Compute x := tau * A * v, and store the result in work
					auto hemv_pctx = spec::problem_context{
						matmul::matmul3 {
							to_const(AMatrixType { PERFLIBS_UPPER, trailing_matrix.transpose().get_matrix_base() }),
							to_const(householder_vec),
							work, tau, zero<value_type>
						},
						pctx.architecture_spec
					};
					compute(hemv_pctx);

					// w := x - 1/2 * tau * (x^H * v) * v
					auto dot_pctx = spec::problem_context {
						matmul::matmul3 {
							to_const(to_conj(work)),
							to_const(householder_vec),
							general_matrix { matrix_base { &alpha, 1, 1, 0, 0 } },
							one, zero<value_type>
						},
						pctx.architecture_spec
					};
					compute(dot_pctx);

					alpha *= -half * tau;

					// w := w + alpha*v
					auto axpy_pctx = spec::problem_context {
						matmul::matmul3 {
							to_const(householder_vec.transpose()),
							to_const(general_matrix { matrix_base { &one, 1, 1, 0, 0 }}),
							work, alpha, one
						},
						pctx.architecture_spec
					};
					compute(axpy_pctx);

					// Apply the transformation as a rank-2 update
					// A := A - v * w**H - w * v**H
					auto her2_pctx = spec::problem_context{
						matmul::rank_update_2k {
							to_const(general_matrix   { matrix_base { householder_vec.data(),  1, pctx.a.cntg()-i-1, 0, pctx.a.cntg_step() }, false }),
							to_const(to_conj(work.transpose())),
							trailing_matrix, -one, one
						},
						pctx.architecture_spec
					};
					compute(her2_pctx);
				}
				else {
					a_gen(i+1,i+1, write) = real(a_gen(i+1,i+1));
				}

				a_gen(i+1,i, write)        = pctx.off_diagonal(i, 0);
				pctx.diagonal(i, 0, write) = real(a_gen(i,i));
			}
			pctx.diagonal(pctx.a.cntg()-1, 0, write) = real(a_gen(pctx.a.cntg()-1,pctx.a.cntg()-1));
		}
		return true;
	}

	template<typename ProblemContext>
	constexpr bool operator()(const ProblemContext&) const {
		return false;
	}

	template<typename AMatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(
		const tridiag_pctx_t<AMatrixType, ArchitectureSpec>& pctx) const {
		return true;
	}
	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const {
		return false;
	}
}; // class tridiagonalize_unblocked
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_TRIDIAGONALIZE_UNBLOCKED_HPP

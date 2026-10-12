/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_BATCH_INTERLEAVE_PIVOTED_QR_FACTORIZATION_GENERIC_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_BATCH_INTERLEAVE_PIVOTED_QR_FACTORIZATION_GENERIC_HPP

#include "perflibs_util.hpp"
#include "perflibs_numeric_utils.hpp"
#include "dyn_array.hpp"
#include "pod_vector.hpp"

#include "framework/buffer_pool.hpp"
#include "framework/linalg_util.hpp"
#include "framework/parallel.hpp"
#include "matrix/adaptors.hpp"
#include "matrix/interleave_batch_matrix.hpp"
#include "spec/problem_context.hpp"
#include "spec/strategy_tag.hpp"

#include "packages/batch_interleave/factorization/fwd.hpp"
#include "packages/batch_interleave/factorization/helpers/find_threshold_absolute_maximum.hpp"
#include "packages/batch_interleave/factorization/helpers/generate_elementary_reflector.hpp"
#include "packages/batch_interleave/factorization/helpers/swap_columns.hpp"
#include "packages/batch_interleave/matmul/strategies.hpp"

#include <algorithm>
#include <limits>
#include <span>
#include <string_view>

namespace perflibs::linalg::batch_interleave::factorization {

/**
 * Generic (single-strategy) implementation of interleaved-batch QR factorization
 * with column pivoting.
 */
class batch_interleave_pivoted_qr_factorization_generic {
	template<typename T, typename IntType, typename ArchitectureSpec>
	using pivoted_qr_pctx_t = spec::problem_context<pivoted_qr_problem<T, IntType>, ArchitectureSpec>;

public:
	static constexpr std::string_view name() { return "batch_interleave_pivoted_qr_factorization_generic"; }

	template<typename T, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const pivoted_qr_pctx_t<T, IntType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		const auto& a_matrix = pctx.a.matrix();

		const auto ninter = pctx.interleave_count();
		const auto nbatch = pctx.batch_count();
		const auto m      = a_matrix.cntg();
		const auto n      = a_matrix.strd();

		const auto min_mn = std::min(m, n);
		const auto spec = get_spec(
			spec::strategy_tag<batch_interleave_pivoted_qr_factorization_generic>{}, pctx);
		const auto split = make_parallel_split(nbatch, 1, spec.max_threads);

		// Initialize default pivot and rank values
		parallel(split.threads, [&](kernel_inttype thread_num) {
			const auto [start, size] = work_distribution(thread_num, split);

			for (kernel_inttype nb = start; nb < start + size; nb++) {
				auto problem = pctx.problem(nb);

				for (kernel_inttype j = 0; j < n; j++) {
					problem.pivot(j, 0, write) = scalar<IntType> { static_cast<IntType>(j) };
				}

				// Assume rank is min(m,n) unless we discover otherwise
				problem.rank(0, 0, write) = scalar<IntType> { static_cast<IntType>(min_mn) };
			}
		});

		if (ninter == 0 || nbatch == 0 || m == 0 || n == 0) {
			return true;
		}

		// Tolerance: column norms are dot products here, so square the usual threshold
		const auto op_tol_mod = perflibs::is_complex_v<T> ? 4 : 1;
		const auto tol = std::numeric_limits<perflibs::remove_complex_t<T>>::epsilon() *
		           op_tol_mod * m * 15;

		// Process batches
		perflibs::pod_vector<T> nrm_storage(split.threads * ninter * n);
		buffer_pool nrm_pool { nrm_storage.data(), split.threads, ninter * n };
		parallel(split.threads, [&](kernel_inttype thread_num) {
			const auto [start, size] = work_distribution(thread_num, split);
			auto* nrm                = nrm_pool.get_buffer(thread_num);
			perflibs::dyn_array<T, max_interleave_size> max_val(ninter);
			perflibs::dyn_array<IntType, max_interleave_size> pos(ninter);
			auto nrm_batch = general_matrix {
				interleave_batch_matrix {
					nrm,
					ninter,
					n,
					1,
					ninter,
					0
				}
			};

			for (kernel_inttype nb = start;
			     nb < start + size;
			     nb++) {

				auto problem = pctx.problem(nb);
				auto a_batch = problem.a;

				auto rank_batch = problem.rank(0, 0, write);

				// Compute and store initial column norms (as dot products of each column with itself)
				auto first_column = a_batch.sub_matrix(0, m, 0, 1);
				auto first_norm   = nrm_batch.sub_matrix(0, 1, 0, 1);

				auto dot_pctx = spec::problem_context {
					matmul::matmul3_problem<T> {
						strided_batch_view { to_const(first_column), n, a_batch.strd_step() },
						strided_batch_view { to_const(first_column), n, a_batch.strd_step() },
						strided_batch_view { first_norm, n, nrm_batch.cntg_step() },
						one<T>,
						zero<T>,
						zero_mode::set
					},
					pctx.architecture_spec
				};
				matmul::compute(dot_pctx);
				for (kernel_inttype r = 0; r < min_mn; r++) {
					// Find column with maximum norm
					auto remaining_nrms = nrm_batch.sub_matrix(r, n - r, 0, 1);
					find_threshold_absolute_maximum(to_const(remaining_nrms), max_val, pos);

					// Indices are relative to the remaining columns; adjust to global column indices
					if (r > 0) {
						for (kernel_inttype nv = 0; nv < ninter; nv++) {
							pos[nv] += r;
						}
					}

					// Check rank threshold and find the largest remaining column norm
					T max_nrm = T(0.0);
					for (kernel_inttype nv = 0; nv < ninter; nv++) {
						if (rank_batch(nv) == min_mn &&
						    max_val[nv] < tol) {
							rank_batch(nv) = r;
						}
						if (max_val[nv] > max_nrm) {
							max_nrm = max_val[nv];
						}
					}

					if (max_nrm < tol) {
						break;
					}

					// Move the maximum-norm column into the current factorization position
					swap_columns(
						a_batch, problem.pivot, nrm_batch, std::span<const IntType> { pos }, r);

					// Form Householder reflectors
					const auto reflector = a_batch.sub_matrix(r, m - r, r, 1);
					const auto tau       = problem.tau.sub_matrix(r, 1, 0, 1);
					generate_elementary_reflector(reflector, tau, pctx.architecture_spec);

					// Apply Householder reflectors from the left.
					if (r < n - 1) {
						const auto trailing_matrix = a_batch.sub_matrix(r, m - r, r + 1, n - r - 1);
						const auto reflector_pctx = spec::problem_context {
							apply_elementary_reflector_problem<T> {
								to_const(reflector), to_const(tau), trailing_matrix
							},
							pctx.architecture_spec
						};
						compute(reflector_pctx);
					}

					// Update column norms
					for (kernel_inttype j = r + 1; j < n; j++) {
						const auto a_rj = a_batch(r, j, write);
						nrm_batch(j, 0, write) -= a_rj * a_rj;
					}
				}
			}
		});

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext&) const { return false; }

	template<typename T, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const pivoted_qr_pctx_t<T, IntType, ArchitectureSpec>&) const {
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }

}; // class batch_interleave_pivoted_qr_factorization_generic

} // namespace perflibs::linalg::batch_interleave::factorization

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_BATCH_INTERLEAVE_PIVOTED_QR_FACTORIZATION_GENERIC_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_BATCH_INTERLEAVE_CHOLESKY_FACTORIZATION_GENERIC_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_BATCH_INTERLEAVE_CHOLESKY_FACTORIZATION_GENERIC_HPP

#include "perflibs_util.hpp"
#include "framework/parallel.hpp"
#include "packages/batch_interleave/factorization/fwd.hpp"
#include "spec/strategy_tag.hpp"

#include <cmath>
#include <string_view>

namespace perflibs::linalg::batch_interleave::factorization {

/**
 * Generic Cholesky factorization of a batch of interleaved matrices.
 */
class batch_interleave_cholesky_factorization_generic {
	template<typename Problem>
	static PERFLIBS_LINALG_INLINE
	void factorize_batch(const Problem problem) {
		const auto a = to_general_matrix(problem.a.is_upper() ? problem.a.transpose() : problem.a);
		const scalar one { typename decltype(a)::data_type { 1.0 } };
		typename decltype(a)::value_type rdiag;

		for (kernel_inttype j = 0; j < a.strd(); j++) {
			if (j > 0) {
				for (kernel_inttype i = j; i < a.strd(); i++) {
					for (kernel_inttype k = 0; k < j; k++) {
						a(i, j, write) -= a(i, k, write) * a(j, k, write);
					}
				}
			}

			a(j, j, write) = sqrt(a(j, j, write));
			rdiag = one / a(j, j, write);
			for (kernel_inttype i = j + 1; i < a.strd(); i++) {
				a(i, j, write) *= rdiag;
			}
		}
	}

public:
	static constexpr std::string_view name() { return "batch_interleave_cholesky_factorization_generic"; }

	template<typename T, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const spec::problem_context<cholesky_problem<T>, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;
		if (pctx.interleave_count() == 0 || pctx.batch_count() == 0) return true;

		if constexpr (omp::is_mp) {
			const auto spec = get_spec(spec::strategy_tag<batch_interleave_cholesky_factorization_generic>{}, pctx);
			const auto batch_count = pctx.batch_count();
			const auto max_threads = spec.max_threads;
			if (max_threads > 1 && batch_count > 1) {
				const auto split = make_parallel_split(batch_count, 1, max_threads);

				parallel(split.threads, [&](const kernel_inttype thread_num) {
					const auto [start, size] = work_distribution(thread_num, split);
					for (kernel_inttype batch = start; batch < start + size; batch++) {
						factorize_batch(pctx.problem(batch));
					}
				});
				return true;
			}
		}

		for (kernel_inttype batch = 0; batch < pctx.batch_count(); batch++) {
			factorize_batch(pctx.problem(batch));
		}
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext&) const { return false; }

	template<typename T, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const spec::problem_context<cholesky_problem<T>, ArchitectureSpec>&) const {
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class batch_interleave_cholesky_factorization_generic

} // namespace perflibs::linalg::batch_interleave::factorization

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_BATCH_INTERLEAVE_CHOLESKY_FACTORIZATION_GENERIC_HPP

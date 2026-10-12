/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_BATCH_INTERLEAVE_GENERATE_Q_FROM_QR_GENERIC_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_BATCH_INTERLEAVE_GENERATE_Q_FROM_QR_GENERIC_HPP

#include "framework/parallel.hpp"
#include "matrix/operations.hpp"
#include "packages/batch_interleave/factorization/fwd.hpp"
#include "spec/problem_context.hpp"
#include "spec/strategy_tag.hpp"

#include <string_view>

namespace perflibs::linalg::batch_interleave::factorization {

/**
 * Form Q matrices from batches of interleaved Householder reflectors.
 */
class batch_interleave_generate_q_from_qr_generic {
private:
	template<typename Problem, typename ArchitectureSpec>
	static PERFLIBS_LINALG_INLINE
	void form_q(
		const Problem& problem, kernel_inttype reflector_count,
		const ArchitectureSpec& architecture_spec) {
		using data_type = typename decltype(problem.a)::data_type;
		const scalar zero { data_type { 0 } };
		const scalar one { data_type { 1 } };
		const scalar minus_one { data_type { -1 } };

		const auto m = problem.a.cntg();
		const auto n = problem.a.strd();

		for (kernel_inttype column = reflector_count; column < n; column++) {
			for (kernel_inttype row = 0; row < m; row++) {
				problem.a(row, column, write) = zero;
			}
			problem.a(column, column, write) = one;
		}

		for (kernel_inttype index = reflector_count - 1; index >= 0; index--) {
			const auto reflector = problem.a.sub_matrix(index, m - index, index, 1);
			const auto tau = problem.tau(index, 0, write);

			if (index < n - 1) {
				const auto trailing_matrix = problem.a.sub_matrix(index, m - index, index + 1, n - index - 1);
				const auto tau_matrix = problem.tau.sub_matrix(index, 1, 0, 1);
				const auto reflector_pctx = spec::problem_context {
					apply_elementary_reflector_problem<data_type> {
						to_const(reflector), to_const(tau_matrix), trailing_matrix
					},
					architecture_spec
				};
				compute(reflector_pctx);
			}

			for (kernel_inttype row = 1; row < reflector.cntg(); row++) {
				reflector(row, 0, write) *= minus_one * tau;
			}
			problem.a(index, index, write) = one - tau;

			for (kernel_inttype row = 0; row < index; row++) {
				problem.a(row, index, write) = zero;
			}
		}
	}

	template<typename Problem, typename ArchitectureSpec>
	static PERFLIBS_LINALG_INLINE
	void form_batch(const Problem& problem, const ArchitectureSpec& architecture_spec) {
		const auto interleave_count = problem.a.interleave_count();
		const auto reflector_counts = problem.reflector_count(0, 0, write);
		const auto counts_are_equal = reflector_counts.lanes(1, interleave_count - 1) == reflector_counts.lanes(0, interleave_count - 1);

		if (counts_are_equal) {
			form_q(problem, reflector_counts[0], architecture_spec);
			return;
		}

		for (kernel_inttype lane = 0; lane < interleave_count; lane++) {
			const auto lane_problem = Problem {
				problem.a.lanes(lane, 1),
				problem.reflector_count.lanes(lane, 1),
				problem.tau.lanes(lane, 1)
			};
			form_q(lane_problem, reflector_counts[lane], architecture_spec);
		}
	}

public:
	static constexpr std::string_view name() {
		return "batch_interleave_generate_q_from_qr_generic";
	}

	template<typename T, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(
		const spec::problem_context<generate_q_from_qr_problem<T, IntType>, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		const auto batch_count      = pctx.batch_count();
		const auto interleave_count = pctx.interleave_count();
		if (batch_count == 0 || interleave_count == 0
		 || pctx.a.matrix().cntg() == 0 || pctx.a.matrix().strd() == 0) {
			return true;
		}

		if constexpr (omp::is_mp) {
			const auto spec = get_spec(
				spec::strategy_tag<batch_interleave_generate_q_from_qr_generic>{}, pctx);
			const auto max_threads = spec.max_threads;
			if (max_threads > 1 && batch_count > 1) {
				const auto split = make_parallel_split(batch_count, 1, max_threads);

				parallel(split.threads, [&](kernel_inttype thread_num) {
					const auto [start, size] = work_distribution(thread_num, split);
					for (kernel_inttype batch = start; batch < start + size; batch++) {
						form_batch(pctx.problem(batch), pctx.architecture_spec);
					}
				});
				return true;
			}
		}

		for (kernel_inttype batch = 0; batch < batch_count; batch++) {
			form_batch(pctx.problem(batch), pctx.architecture_spec);
		}

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext&) const { return false; }

	template<typename T, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(
		const spec::problem_context<generate_q_from_qr_problem<T, IntType>, ArchitectureSpec>&) const {
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class batch_interleave_generate_q_from_qr_generic

} // namespace perflibs::linalg::batch_interleave::factorization

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_BATCH_INTERLEAVE_GENERATE_Q_FROM_QR_GENERIC_HPP

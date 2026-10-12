/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_BATCH_INTERLEAVE_THRESHOLD_PIVOTED_LU_FACTORIZATION_GENERIC_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_BATCH_INTERLEAVE_THRESHOLD_PIVOTED_LU_FACTORIZATION_GENERIC_HPP

#include "dyn_array.hpp"
#include "framework/parallel.hpp"
#include "packages/batch_interleave/factorization/fwd.hpp"
#include "packages/batch_interleave/factorization/helpers/find_threshold_absolute_maximum.hpp"
#include "packages/batch_interleave/matmul/strategies.hpp"
#include "spec/strategy_tag.hpp"

#include <string_view>
#include <utility>

namespace perflibs::linalg::batch_interleave::factorization {

/**
 * Generic LU factorization with threshold partial pivoting of interleaved
 * matrix batches.
 */
class batch_interleave_threshold_pivoted_lu_factorization_generic {
	template<typename T, typename IntType, typename ArchitectureSpec>
	using threshold_pivoted_lu_pctx_t =
		spec::problem_context<threshold_pivoted_lu_problem<T, IntType>, ArchitectureSpec>;

	template<typename Matrix, typename ArchitectureSpec>
	static PERFLIBS_LINALG_INLINE
	void update_trailing_matrix(const Matrix a, kernel_inttype column,
	                            const ArchitectureSpec& architecture_spec) {
		using data_type = typename Matrix::data_type;
		const scalar one_scalar { data_type { 1.0 } };

		const auto diagonal = a(column, column, write);
		typename Matrix::value_type reciprocal;
		reciprocal = one_scalar / diagonal;

		for (kernel_inttype row = column + 1; row < a.cntg(); row++) {
			a(row, column, write) *= reciprocal;
		}

		const auto trailing_size = a.strd() - column - 1;
		if (trailing_size == 0) return;

		const auto row_count = a.cntg() - column - 1;
		const auto lower = a.sub_matrix(column + 1, row_count, column, 1);
		const auto upper = a.sub_matrix(column, 1, column + 1, trailing_size);
		const auto trailing = a.sub_matrix(column + 1, row_count, column + 1, trailing_size);

		const auto ger_pctx = spec::problem_context {
			matmul::matmul3_problem<data_type> {
				strided_batch_view { to_const(lower.transpose()), 1, 0 },
				strided_batch_view { to_const(upper), 1, 0 },
				strided_batch_view { trailing, 1, 0 },
				-one<data_type>,
				one<data_type>,
				zero_mode::set
			},
			architecture_spec
		};
		matmul::compute(ger_pctx);
	}

	template<typename Problem, typename ArchitectureSpec>
	static PERFLIBS_LINALG_INLINE
	void factorize_batch(const Problem problem, const ArchitectureSpec& architecture_spec) {
		using data_type = typename decltype(problem.a)::data_type;
		using int_type  = typename decltype(problem.pivot)::data_type;

		const auto m                = problem.a.cntg();
		const auto n                = problem.a.strd();
		const auto min_mn           = min(m, n);
		const auto interleave_count = problem.a.interleave_count();
		const data_type zero { 0.0 };

		dyn_array<data_type, max_interleave_size> maximum(interleave_count);

		for (int_type column = 0; column < min_mn; column++) {
			auto pivot_values = problem.pivot(column, 0, write);
			if (problem.threshold > zero) {
				const auto pivot_column = to_const(problem.a.sub_matrix(column, m - column, column, 1));
				find_threshold_absolute_maximum(pivot_column, problem.threshold, maximum, pivot_values);

				for (kernel_inttype lane = 0; lane < interleave_count; lane++) {
					pivot_values[lane] += column;

					for (kernel_inttype target_column = 0; target_column < n; target_column++) {
						std::swap(problem.a(column, target_column, write)[lane],
						          problem.a(pivot_values[lane], target_column, write)[lane]);
					}
				}
			} else {
				pivot_values = scalar { column };
			}

			const auto diagonal = problem.a(column, column, write);
			bool all_diagonal_elements_are_nonzero = true;
			for (kernel_inttype lane = 0; lane < interleave_count; lane++) {
				all_diagonal_elements_are_nonzero &= diagonal[lane] != zero;
			}

			if (all_diagonal_elements_are_nonzero) {
				update_trailing_matrix(problem.a, column, architecture_spec);
				continue;
			}

			for (kernel_inttype lane = 0; lane < interleave_count; lane++) {
				if (diagonal[lane] == zero) continue;
				update_trailing_matrix(problem.a.lanes(lane, 1), column, architecture_spec);
			}
		}
	}

public:
	static constexpr std::string_view name() { return "batch_interleave_threshold_pivoted_lu_factorization_generic"; }

	template<typename T, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const threshold_pivoted_lu_pctx_t<T, IntType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;
		if (pctx.interleave_count() == 0 || pctx.batch_count() == 0) return true;

		if constexpr (omp::is_mp) {
			const auto spec = get_spec(
				spec::strategy_tag<batch_interleave_threshold_pivoted_lu_factorization_generic>{}, pctx);
			const auto batch_count = pctx.batch_count();
			const auto max_threads  = spec.max_threads;
			if (max_threads > 1 && batch_count > 1) {
				const auto split = make_parallel_split(batch_count, 1, max_threads);
				parallel(split.threads, [&](kernel_inttype thread_num) {
					const auto [start, size] = work_distribution(thread_num, split);
					for (kernel_inttype batch = start; batch < start + size; batch++) {
						factorize_batch(pctx.problem(batch), pctx.architecture_spec);
					}
				});
				return true;
			}
		}

		for (kernel_inttype batch = 0; batch < pctx.batch_count(); batch++) {
			factorize_batch(pctx.problem(batch), pctx.architecture_spec);
		}
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext&) const {
		return false;
	}

	template<typename T, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const threshold_pivoted_lu_pctx_t<T, IntType, ArchitectureSpec>&) const {
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const {
		return false;
	}
}; // class batch_interleave_threshold_pivoted_lu_factorization_generic

} // namespace perflibs::linalg::batch_interleave::factorization

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_BATCH_INTERLEAVE_THRESHOLD_PIVOTED_LU_FACTORIZATION_GENERIC_HPP

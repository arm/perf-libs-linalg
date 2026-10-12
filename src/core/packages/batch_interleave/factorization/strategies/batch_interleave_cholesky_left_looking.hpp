/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_BATCH_INTERLEAVE_CHOLESKY_LEFT_LOOKING_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_BATCH_INTERLEAVE_CHOLESKY_LEFT_LOOKING_HPP

#include "perflibs_util.hpp"
#include "packages/batch_interleave/factorization/fwd.hpp"
#include "packages/batch_interleave/matmul/fwd.hpp"

#include <cmath>
#include <string_view>

namespace perflibs::linalg::batch_interleave::factorization {

/**
 * Left-looking Cholesky factorization using interleaved matrix-vector updates across all batches.
 */
class batch_interleave_cholesky_left_looking {
	template<typename Matrix>
	static PERFLIBS_LINALG_INLINE
	void factorize_column(const Matrix a, const kernel_inttype j) {
		a(j, j, write) = sqrt(a(j, j, write));
		if (j + 1 == a.cntg()) return;

		typename Matrix::value_type rdiag;
		const scalar one { typename Matrix::data_type { 1 } };
		rdiag = one / a(j, j, write);
		for (kernel_inttype i = j + 1; i < a.cntg(); i++) {
			a(i, j, write) *= rdiag;
		}
	}

	template<typename T, typename ArchitectureSpec>
	static PERFLIBS_LINALG_INLINE
	void factorize(const spec::problem_context<cholesky_problem<T>, ArchitectureSpec>& pctx) {
		const auto& a_view = pctx.a.matrix();
		const auto a_matrix = to_general_matrix(a_view.is_upper() ? a_view.transpose() : a_view);
		const auto nbatch  = pctx.batch_count();
		const auto n       = a_matrix.cntg();
		const auto bstrd_a = pctx.a.batch_step();
		const T one { 1.0 };
		const T minus_one { -1.0 };
		const strided_batch_view matrices { a_matrix, nbatch, bstrd_a };
		for (kernel_inttype j = 0; j < n; j++) {
			if (j > 0) {
				const auto factorized_panel = a_matrix.sub_matrix(j, n - j, 0, j);
				const auto current_row      = factorized_panel.sub_matrix(0, 1, 0, j);
				const auto trailing_column  = a_matrix.sub_matrix(j, n - j, j, 1);
				const auto gemv_pctx = spec::problem_context {
					matmul::matmul3_problem<T> {
						strided_batch_view { to_const(factorized_panel.transpose()), nbatch, bstrd_a },
						strided_batch_view { to_const(current_row.transpose()), nbatch, bstrd_a },
						strided_batch_view { trailing_column, nbatch, bstrd_a },
						minus_one,
						one,
						zero_mode::set
					},
					pctx.architecture_spec
				};
				matmul::compute(gemv_pctx);
			}

			for (kernel_inttype batch = 0; batch < nbatch; batch++) {
				factorize_column(matrices[batch], j);
			}
		}
	}

public:
	static constexpr std::string_view name() { return "batch_interleave_cholesky_left_looking"; }

	template<typename T, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const spec::problem_context<cholesky_problem<T>, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;
		if (pctx.interleave_count() == 0 || pctx.batch_count() == 0) return true;

		factorize(pctx);
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
}; // class batch_interleave_cholesky_left_looking

} // namespace perflibs::linalg::batch_interleave::factorization

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_BATCH_INTERLEAVE_CHOLESKY_LEFT_LOOKING_HPP

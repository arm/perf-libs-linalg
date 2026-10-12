/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_STRATEGIES_BATCH_INTERLEAVE_MATMUL2_KERNEL_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_STRATEGIES_BATCH_INTERLEAVE_MATMUL2_KERNEL_HPP

#include "framework/parallel.hpp"
#include "matrix/interleave_batch.hpp"
#include "matrix/operations.hpp"
#include "spec/strategy_tag.hpp"

#include <string_view>

namespace perflibs::linalg::batch_interleave::matmul {

/** In-place triangular matrix multiplication using interleaved-batch kernels. */
class batch_interleave_matmul2_kernel {
	template<typename Matrix>
	static PERFLIBS_LINALG_INLINE
	void set_zero(const Matrix& b) {
		using data_type = typename Matrix::data_type;

		for (kernel_inttype j = 0; j < b.strd(); j++) {
			for (kernel_inttype i = 0; i < b.cntg(); i++) {
				b(i, j, write) = scalar { zero<data_type> };
			}
		}
	}

	template<typename Problem, typename Kernel>
	static PERFLIBS_LINALG_INLINE
	void compute_problem(const Problem& problem, const Kernel kernel) {
		if (problem.alpha == zero<typename Problem::scalar_type>) {
			set_zero(problem.b);
			return;
		}

		const auto& a = problem.a;
		const auto& b = problem.b;
		const auto m = b.cntg();
		const bool reverse = a.is_upper();
		const kernel_inttype direction = reverse ? -1 : 1;
		const auto a_data = reverse ? a.batch_data(m - 1, m - 1) : a.data();
		const auto b_data = reverse ? b.batch_data(m - 1, 0) : b.data();

		kernel(a.interleave_count(), a_data, direction * a.strd_step(), direction * a.cntg_step(),
		       b_data, direction * b.cntg_step(), b.strd_step(), m, b.strd(), !a.is_unit(), problem.alpha);
	}

	template<typename ProblemContext, typename Kernel>
	static PERFLIBS_LINALG_INLINE
	void compute_batches(const ProblemContext& pctx, const Kernel kernel, const kernel_inttype start_batch,
	                     const kernel_inttype batch_count) {
		for (kernel_inttype batch = start_batch; batch < start_batch + batch_count; batch++) {
			compute_problem(pctx.problem(batch), kernel);
		}
	}

public:
	static constexpr std::string_view name() { return "batch_interleave_matmul2_kernel"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<batch_interleave_matmul2_kernel>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator()(const ProblemContext& pctx) const {
		if (!can_compute(pctx)) {
			return false;
		}

		const auto batch_count = pctx.batch_count();
		if (pctx.interleave_count() == 0 || batch_count == 0 || empty(pctx.b.matrix())) return true;

		const auto strategy_tag = spec::strategy_tag<batch_interleave_matmul2_kernel>{};
		const auto spec = get_spec(strategy_tag, pctx);

		if constexpr (omp::is_mp) {
			const auto max_threads = spec.max_threads;
			if (max_threads > 1 && batch_count > 1) {
				const auto split = make_parallel_split(batch_count, 1, max_threads);

				parallel(split.threads, [&](const kernel_inttype thread_num) {
					const auto [start, size] = work_distribution(thread_num, split);
					compute_batches(pctx, spec.kernel, start, size);
				});
				return true;
			}
		}

		compute_batches(pctx, spec.kernel, 0, batch_count);
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<batch_interleave_matmul2_kernel>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		return pctx.beta_zero_mode == zero_mode::set
		    && pctx.beta == zero<typename ProblemContext::scalar_type>;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class batch_interleave_matmul2_kernel

} // namespace perflibs::linalg::batch_interleave::matmul

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_STRATEGIES_BATCH_INTERLEAVE_MATMUL2_KERNEL_HPP

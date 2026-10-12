/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_STRATEGIES_BATCH_INTERLEAVE_MATMUL3_KERNEL_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_STRATEGIES_BATCH_INTERLEAVE_MATMUL3_KERNEL_HPP

#include "framework/parallel.hpp"
#include "spec/strategy_tag.hpp"

#include <string_view>

namespace perflibs::linalg::batch_interleave::matmul {

/** Matrix multiplication using interleaved-batch kernels. */
class batch_interleave_matmul3_kernel {
	template<typename ProblemContext, typename Kernel>
	static PERFLIBS_LINALG_INLINE
	void compute_batches(const ProblemContext& pctx, const Kernel kernel, const kernel_inttype start_batch,
	                     const kernel_inttype batch_count) {
		const auto ninter = pctx.interleave_count();
		if (ninter == 0 || batch_count == 0) return;
		const auto a = pctx.a[start_batch];
		const auto b = pctx.b[start_batch];
		const auto c = pctx.c[start_batch];

		kernel(ninter, batch_count, c.cntg(), c.strd(), a.cntg(), pctx.alpha,
		       a.data(), pctx.a.batch_step(), a.cntg_step(), a.strd_step(),
		       b.data(), pctx.b.batch_step(), b.cntg_step(), b.strd_step(), pctx.beta,
		       c.data(), pctx.c.batch_step(), c.cntg_step(), c.strd_step());
	}

public:
	static constexpr std::string_view name() { return "batch_interleave_matmul3_kernel"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<batch_interleave_matmul3_kernel>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator()(const ProblemContext& pctx) const {
		if (!can_compute(pctx)) {
			return false;
		}

		const auto strategy_tag = spec::strategy_tag<batch_interleave_matmul3_kernel>{};
		const auto spec = get_spec(strategy_tag, pctx);
		const auto batch_count = pctx.batch_count();

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
	requires spec::has_get_spec<spec::strategy_tag<batch_interleave_matmul3_kernel>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		return pctx.beta_zero_mode == zero_mode::set || pctx.alpha == zero<typename ProblemContext::scalar_type>;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class batch_interleave_matmul3_kernel

} // namespace perflibs::linalg::batch_interleave::matmul

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_STRATEGIES_BATCH_INTERLEAVE_MATMUL3_KERNEL_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MISC_STRATEGIES_L2_NORM_HPP
#define PERFLIBS_LINALG_MISC_STRATEGIES_L2_NORM_HPP

#include "framework/l2_norm_kernels.hpp"
#include "framework/parallel.hpp"
#include "spec/strategy_tag.hpp"

#include "detect/omp.hpp"
#include "perflibs_util.hpp"

namespace perflibs::linalg::misc {

class l2_norm_strategy {
public:
	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<l2_norm_strategy>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator()(const ProblemContext& pctx) const {
		using scalar_type = typename ProblemContext::scalar_type;
		using real_type   = remove_complex_t<scalar_type>;

		const auto spec = get_spec(spec::strategy_tag<l2_norm_strategy>{}, pctx);

		const kernel_inttype  n    = pctx.x.cntg();
		const scalar_type    *x    = pctx.x.data();
		const kernel_inttype  incx = pctx.x.cntg_step();

		if constexpr (omp::is_mp) {
			if (spec.max_threads > 1 && incx != 1) {
				// For non-unit increments, we keep intermediate per-thread results in a separate
				// collection of buffers (l2_norm_accumulator).
				// We combine these results at the end using safe scaling.
				// This avoids the accuracy issues that arise from preemptively combining them.
				const auto split = make_parallel_split(n, 1, spec.max_threads, n);

				dyn_array<l2_norm_accumulator<scalar_type>, 224> acc(split.threads);

				parallel(split.threads, [&](kernel_inttype thread_num) {
					const auto [start, work] = work_distribution(thread_num, split, 1);
					auto thread_work = get_cntg_panel(pctx.x, start, work);
					for (kernel_inttype i = 0; i < work; ++i) {
						acc[thread_num].complex_add(thread_work(i, 0));
					}
				});

				l2_norm_accumulator<scalar_type> total_acc;
				for (kernel_inttype i = 0; i < split.threads; ++i) {
					total_acc.merge(acc[i]);
				}

				pctx.out = total_acc.finish();
				return true;

			} else if (spec.max_threads > 1) {
				// Store per-thread partial results as a scalar (the kernel returns the
				// partial norm of its chunk, which we then combine safely).
				const auto split = make_parallel_split(n, 1, spec.max_threads, n);
				dyn_array<real_type, 224> partial_results(split.threads);

				parallel(split.threads, [&](kernel_inttype thread_num) {
					const auto [start, work] = work_distribution(thread_num, split, 1);
					auto thread_work = get_cntg_panel(pctx.x, start, work);
					partial_results[thread_num] =
						spec.kernel(thread_work.cntg(), thread_work.data(), thread_work.cntg_step());
				});
				l2_norm_accumulator<real_type> acc;
				for (kernel_inttype i = 0; i < split.threads; ++i) {
					acc.add(partial_results[i]);
				}
				pctx.out = acc.finish();
				return true;
			} // if max_threads > 1
		} // if omp

		// Sequential - just call the kernel and return its result by setting pctx.out (no need for reduction)
		pctx.out = spec.kernel(n, x, incx);
		return true;
	} // bool operator()

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator()(const ProblemContext&) const { return false; }
}; // class l2_norm_strategy
} // namespace perflibs::linalg::misc

#endif // PERFLIBS_LINALG_MISC_STRATEGIES_L2_NORM_HPP

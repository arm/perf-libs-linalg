/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MISC_STRATEGIES_SWAP_HPP
#define PERFLIBS_LINALG_MISC_STRATEGIES_SWAP_HPP

#include "framework/swap_kernels.hpp"
#include "framework/parallel.hpp"
#include "spec/strategy_tag.hpp"

#include "detect/omp.hpp"
#include "perflibs_util.hpp"

namespace perflibs::linalg::misc {

class swap_strategy {
public:
	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<swap_strategy>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator()(const ProblemContext& pctx) const {
		using scalar_type = typename ProblemContext::scalar_type;

		const auto spec = get_spec(spec::strategy_tag<swap_strategy>{}, pctx);

		const kernel_inttype n    = pctx.x.cntg();
		scalar_type         *x    = pctx.x.data();
		const kernel_inttype incx = pctx.x.cntg_step();
		scalar_type         *y    = pctx.y.data();
		const kernel_inttype incy = pctx.y.cntg_step();

		if constexpr (omp::is_mp) {
			if (spec.max_threads > 1 && incx != 0 && incy != 0) {
				const auto split = make_parallel_split(n, 1, spec.max_threads);

				parallel(split.threads, [&](kernel_inttype thread_num) {
					const auto [start, work] = work_distribution(thread_num, split, 1);
					auto x_thread_work = get_cntg_panel(pctx.x, start, work);
					auto y_thread_work = get_cntg_panel(pctx.y, start, work);
					spec.kernel(work, x_thread_work.data(), y_thread_work.data(), x_thread_work.cntg_step(), y_thread_work.cntg_step());
				});
				return true;
			}
		}
		spec.kernel(n, x, y, incx, incy);
		return true;
	} // bool operator()

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator()(const ProblemContext&) const { return false; }
}; // class swap_strategy

} // namespace perflibs::linalg::misc

#endif // PERFLIBS_LINALG_MISC_STRATEGIES_SWAP_HPP

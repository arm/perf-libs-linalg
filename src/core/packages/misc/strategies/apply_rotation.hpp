/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MISC_STRATEGIES_APPLY_ROTATION_HPP
#define PERFLIBS_LINALG_MISC_STRATEGIES_APPLY_ROTATION_HPP

#include "framework/parallel.hpp"
#include "spec/strategy_tag.hpp"

namespace perflibs::linalg::misc {

class apply_rotation {
public:
	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<apply_rotation>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		const auto spec = spec::get_spec(spec::strategy_tag<apply_rotation>{}, pctx);

		if (pctx.n <= 0)
			return true;

		if constexpr (omp::is_mp) {
			if (spec.max_threads > 1 && pctx.incx != 0 && pctx.incy != 0) {
				const auto split = make_parallel_split(pctx.n, 1, spec.max_threads);
				parallel(split.threads, [=](kernel_inttype thread_num) {
					const auto [start, work] = work_distribution(thread_num, split, 1);
					spec.kernel(
						work,
						pctx.x + start * pctx.incx,
						pctx.y + start * pctx.incy,
						pctx.incx, pctx.incy,
						pctx.c, pctx.s);
				});
				return true;
			}
		}
		spec.kernel(pctx.n, pctx.x, pctx.y, pctx.incx, pctx.incy, pctx.c, pctx.s);
		return true;
	} // bool operator()

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator()(const ProblemContext&) const { return false; }
}; // class apply_rotation

} // namespace perflibs::linalg::misc

#endif // PERFLIBS_LINALG_MISC_STRATEGIES_APPLY_ROTATION_HPP
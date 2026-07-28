/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MISC_STRATEGIES_L1_NORM_HPP
#define PERFLIBS_LINALG_MISC_STRATEGIES_L1_NORM_HPP

#include "framework/l1_norm_kernels.hpp"
#include "framework/parallel.hpp"
#include "spec/strategy_tag.hpp"

#include "perflibs_util.hpp"

namespace perflibs::linalg::misc {

class l1_norm_strategy {
public:
	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<l1_norm_strategy>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator()(const ProblemContext& pctx) const {
		using real_type = remove_complex_t<typename ProblemContext::scalar_type>;

		const auto spec = get_spec(spec::strategy_tag<l1_norm_strategy>{}, pctx);

		const kernel_inttype n    = pctx.x.cntg();
		const auto          *x    = pctx.x.data();
		const kernel_inttype incx = pctx.x.cntg_step();

		if constexpr (omp::is_mp) {
			if (spec.max_threads > 1) {
				const auto split = make_parallel_split(n, 1, spec.max_threads, n);
				pctx.out = reduce_add_parallel<real_type>(split.threads, [&](kernel_inttype thread_num) {
					const auto [start, work] = work_distribution(thread_num, split, 1);
					const auto thread_work = get_cntg_panel(pctx.x, start, work);
					return spec.kernel(thread_work.cntg(), thread_work.data(), thread_work.cntg_step());
				});

				return true;
			}
		}

		pctx.out = spec.kernel(n, x, incx);
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator()(const ProblemContext&) const { return false; }
}; // class l1_norm_strategy

} // namespace perflibs::linalg::misc

#endif // PERFLIBS_LINALG_MISC_STRATEGIES_L1_NORM_HPP

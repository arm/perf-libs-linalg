/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_WAXPBY_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_WAXPBY_HPP

#include "framework/linalg_util.hpp"
#include "framework/parallel.hpp"
#include "matrix/matrix.hpp"
#include "matrix/adaptors.hpp"
#include "matrix/operations.hpp"

#include "spec/strategy_tag.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class matmul4_vector_scalar_strategy {
public:
	static constexpr std::string_view name() { return "matmul4_vector_scalar_strategy"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matmul4_vector_scalar_strategy>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if (! this->can_compute(pctx) ) return false;

		const auto spec = get_spec(spec::strategy_tag<matmul4_vector_scalar_strategy>{}, pctx);

		if constexpr (omp::is_mp) {
			if (spec.max_threads > 1 && pctx.d.cntg_step() != 0) {
				const auto split = make_parallel_split(pctx.d.cntg(), 1, spec.max_threads, pctx.d.cntg());

				parallel(split.threads, [=, &pctx](kernel_inttype thread_num) {
					const auto [start, work] = work_distribution(thread_num, split, 1);

					auto x_panel = get_cntg_panel(pctx.a, start, work);
					auto y_panel = get_cntg_panel(pctx.c, start, work);
					auto w_panel = get_cntg_panel(pctx.d, start, work);

					spec.kernel(work, pctx.alpha, x_panel.data(), pctx.beta,
						y_panel.data(), w_panel.data(), x_panel.cntg_step(),
						y_panel.cntg_step(), w_panel.cntg_step());
				});

				return true;
			}
		}

		spec.kernel(pctx.d.cntg(), pctx.alpha, pctx.a.data(), pctx.beta, pctx.c.data(), pctx.d.data(), pctx.a.strd_step(), pctx.c.cntg_step(), pctx.d.cntg_step());
		return true;
	} // bool operator()

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matmul4_vector_scalar_strategy>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		return pctx.a.cntg() == 1 && pctx.b.cntg() == 1
		    && pctx.b.strd() == 1;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class matmul4_vector_scalar_strategy
} // namespace perflibs::linalg::matmul

#endif //  PERFLIBS_LINALG_WAXPBY_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_STRATEGIES_NAIVE_INNER_PRODUCT_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_STRATEGIES_NAIVE_INNER_PRODUCT_HPP

#include "framework/compute_position.hpp"
#include "naive/dot.hpp"

#include <string_view>

namespace perflibs::linalg::batch_interleave::matmul {

class batch_interleave_matmul3_naive_inner_product {
public:
	static constexpr std::string_view name() { return "batch_interleave_matmul3_naive_inner_product"; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator()(const ProblemContext& pctx) const {
		if (!this->can_compute(pctx)) return false;

		for (kernel_inttype batch = 0; batch != pctx.batch_count(); ++batch) {
			auto problem = pctx.problem(batch);
			naive_dot{}(
				problem.a,
				problem.b,
				problem.c,
				compute_position { 0, 0, 0, 0 },
				problem.alpha,
				problem.beta
			);
		}

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		const auto& c = pctx.c.matrix();

		return pctx.beta_zero_mode == zero_mode::set
		    && c.cntg() == 1 && c.strd() == 1
		    && pctx.alpha == one<typename ProblemContext::scalar_type>
		    && pctx.beta == zero<typename ProblemContext::scalar_type>;
	}
}; // class batch_interleave_matmul3_naive_inner_product

} // namespace perflibs::linalg::batch_interleave::matmul

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_STRATEGIES_NAIVE_INNER_PRODUCT_HPP

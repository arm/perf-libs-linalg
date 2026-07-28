/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD3_NAIVE_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD3_NAIVE_HPP

#include "framework/linalg_util.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class matadd3_naive {
public:
	static constexpr std::string_view name() { return "matadd3_naive"; }

	template<typename... Ts, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator() (const spec::problem_context<matmul::matadd3<Ts...>, ArchitectureSpec>& pctx) const {
		if (!this->can_compute(pctx)) return false;

		for (kernel_inttype j = 0; j != pctx.c.strd(); ++j) {
			for (kernel_inttype i = 0; i != pctx.c.cntg(); ++i) {
				pctx.c(i, j, write) = pctx.alpha * pctx.a(i, j)
				                    + pctx.beta  * pctx.b(i, j);
			}
		}

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename... Ts, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const spec::problem_context<matmul::matadd3<Ts...>, ArchitectureSpec>& pctx) const {
		return pctx.alpha != zero<>
		    && pctx.beta  != zero<>;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
};

} // namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD3_NAIVE_HPP

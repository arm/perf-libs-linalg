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

		using c_data_type = typename decltype(pctx.c)::value_type;

		const bool use_cntg_loop =   pctx.c.cntg_step() == 1
		                         ||  pctx.c.strd_step() != 1
		                         || (pctx.a.strd_step() != 1 && pctx.b.strd_step() != 1);

		const auto compute_element = [&](kernel_inttype i, kernel_inttype j) {
			if (pctx.alpha != zero<>) {
				if (pctx.beta != zero<>) {
					pctx.c(i, j, write) = pctx.alpha * pctx.a(i, j)
					                    + pctx.beta  * pctx.b(i, j);
				}
				else {
					pctx.c(i, j, write) = pctx.alpha * pctx.a(i, j);
				}
			}
			else {
				if (pctx.beta != zero<>) {
					pctx.c(i, j, write) = pctx.beta * pctx.b(i, j);
				}
				else {
					pctx.c(i, j, write) = zero<c_data_type>;
				}
			}
		};

		if (use_cntg_loop) {
			for (kernel_inttype j = 0; j != pctx.c.strd(); ++j) {
				for (kernel_inttype i = 0; i != pctx.c.cntg(); ++i) {
					compute_element(i, j);
				}
			}
		}
		else {
			for (kernel_inttype i = 0; i != pctx.c.cntg(); ++i) {
				for (kernel_inttype j = 0; j != pctx.c.strd(); ++j) {
					compute_element(i, j);
				}
			}
		}

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename... Ts, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const spec::problem_context<matmul::matadd3<Ts...>, ArchitectureSpec>&) const {
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
};

} // namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD3_NAIVE_HPP

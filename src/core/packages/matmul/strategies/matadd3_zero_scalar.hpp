/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD3_ZERO_SCALAR_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD3_ZERO_SCALAR_HPP

#include "framework/linalg_util.hpp"
#include "matrix/operations.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class matadd3_zero_scalar {
	template<typename Scalar, typename SrcMatrix, typename DstMatrix>
	static PERFLIBS_LINALG_INLINE
	void copy_scaled(const Scalar scalar, const SrcMatrix& src, const DstMatrix& dst) {
		for (kernel_inttype j = 0; j != dst.strd(); ++j) {
			for (kernel_inttype i = 0; i != dst.cntg(); ++i) {
				dst(i, j, write) = scalar * src(i, j);
			}
		}
	}

public:
	static constexpr std::string_view name() { return "matadd3_zero_scalar"; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if (!this->can_compute(pctx)) return false;

		if (pctx.alpha == zero<>) {
			if (pctx.beta == zero<>) {
				set(zero<>, pctx.c);
			}
			else {
				copy_scaled(pctx.beta, pctx.b, pctx.c);
			}

			return true;
		}

		copy_scaled(pctx.alpha, pctx.a, pctx.c);

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		return pctx.alpha == zero<> || pctx.beta == zero<>;
	}
};

} // namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD3_ZERO_SCALAR_HPP

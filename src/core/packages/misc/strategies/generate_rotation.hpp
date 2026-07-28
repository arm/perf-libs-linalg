/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MISC_STRATEGIES_ROTG_HPP
#define PERFLIBS_LINALG_MISC_STRATEGIES_ROTG_HPP

#include "framework/parallel.hpp"
#include "spec/strategy_tag.hpp"

namespace perflibs::linalg::misc {

class generate_rotation {
public:
	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<generate_rotation>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		const auto spec = spec::get_spec(spec::strategy_tag<generate_rotation>{}, pctx);

		spec.kernel(pctx.a, pctx.b, pctx.c, pctx.s);
		return true;
	} // bool operator()

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator()(const ProblemContext&) const { return false; }
}; // class generate_rotation

} // namespace perflibs::linalg::misc

#endif // PERFLIBS_LINALG_MISC_STRATEGIES_ROTG_HPP

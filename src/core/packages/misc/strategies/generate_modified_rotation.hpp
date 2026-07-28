/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MISC_STRATEGIES_GENERATE_MODIFIED_ROTATION_HPP
#define PERFLIBS_LINALG_MISC_STRATEGIES_GENERATE_MODIFIED_ROTATION_HPP

#include "framework/parallel.hpp"
#include "spec/strategy_tag.hpp"

namespace perflibs::linalg::misc {

class generate_modified_rotation {
public:
	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<generate_modified_rotation>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		const auto spec = spec::get_spec(spec::strategy_tag<generate_modified_rotation>{}, pctx);

		spec.kernel(pctx.d1, pctx.d2, pctx.x, pctx.y,  pctx.param);
		return true;
	} // bool operator()

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator()(const ProblemContext&) const { return false; }
}; // class generate_modified_rotation

} // namespace perflibs::linalg::misc

#endif // PERFLIBS_LINALG_MISC_STRATEGIES_GENERATE_MODIFIED_ROTATION_HPP

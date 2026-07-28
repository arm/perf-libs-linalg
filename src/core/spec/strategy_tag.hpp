/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SPEC_STRATEGY_TAG_HPP
#define PERFLIBS_LINALG_SPEC_STRATEGY_TAG_HPP

#include <array>

namespace perflibs::linalg::spec {

template<typename StrategyType>
struct strategy_tag { };

template<typename StrategyTag, typename ProblemContext>
concept has_get_spec = requires(StrategyTag tag, const ProblemContext& pctx) {
	get_spec(tag, pctx);
};

struct strategy_selection_tag { };

template<std::size_t N>
struct strategy_selection_spec {
	std::array<std::size_t, N> strategy_preferences;
};

} //namespace perflibs::linalg::spec

#endif //PERFLIBS_LINALG_SPEC_STRATEGY_TAG_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_INTERFACES_GET_STRATEGY_NAME_HPP
#define PERFLIBS_LINALG_MATMUL_INTERFACES_GET_STRATEGY_NAME_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename ProblemContextBase, typename ArchitectureSpec>
std::string_view get_strategy_name_arch(const ProblemContextBase& pctx_base, std::size_t index) {
	using problem_context_type = spec::problem_context<ProblemContextBase, ArchitectureSpec>;

	return get_strategy_name(strategies<problem_context_type>, index);
}

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_MATMUL_INTERFACES_GET_STRATEGY_NAME_HPP

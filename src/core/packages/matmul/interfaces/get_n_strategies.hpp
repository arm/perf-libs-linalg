/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_INTERFACES_GET_N_STRATEGIES_HPP
#define PERFLIBS_LINALG_MATMUL_INTERFACES_GET_N_STRATEGIES_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"

namespace perflibs::linalg {

template<typename ProblemContextBase, typename ArchitectureSpec>
std::size_t get_n_strategies_arch(const ProblemContextBase& pctx_base) {
	using problem_context_type = spec::problem_context<ProblemContextBase, ArchitectureSpec>;

	return tuple_size( strategies<problem_context_type> );
}

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_MATMUL_INTERFACES_GET_N_STRATEGIES_HPP

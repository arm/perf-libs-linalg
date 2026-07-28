/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_INTERFACES_COMPUTE_INDEX_HPP
#define PERFLIBS_LINALG_MATMUL_INTERFACES_COMPUTE_INDEX_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"

namespace perflibs::linalg {

template<typename ProblemContextBase, typename ArchitectureSpec>
bool compute_index_arch(const ProblemContextBase& pctx_base, std::size_t index) {
	const auto pctx = spec::problem_context { pctx_base, ArchitectureSpec { machine::get_system_unsafe() } };

	return compute_index(strategies<spec::problem_context<ProblemContextBase, ArchitectureSpec>>, pctx, index);
}

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_MATMUL_INTERFACES_COMPUTE_INDEX_HPP

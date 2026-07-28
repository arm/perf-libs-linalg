/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_INTERFACES_COMPUTE_HPP
#define PERFLIBS_LINALG_MATMUL_INTERFACES_COMPUTE_HPP

#include "packages/matmul/strategies.hpp"
#include "framework/compute.hpp"
#include "spec/problem_context.hpp"

namespace perflibs::linalg {

template<typename ProblemContextBase, typename ArchitectureSpec>
void matmul::compute(const ProblemContextBase& pctx_base) {
	spec::problem_context pctx { pctx_base, ArchitectureSpec {} };

	matmul::compute(pctx);
}

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_MATMUL_INTERFACES_COMPUTE_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SPEC_PROBLEM_CONTEXT_HPP
#define PERFLIBS_LINALG_SPEC_PROBLEM_CONTEXT_HPP

#include <utility> //std::move

namespace perflibs::linalg::spec {

template<typename ProblemContextBase, typename ArchitectureSpec>
struct problem_context : ProblemContextBase {
	using architecture_spec_type = ArchitectureSpec;

	architecture_spec_type architecture_spec;
}; //struct problem_context

// struct to the precision of the output matrix
template<typename ProblemContext>
struct compute_precision { /* not supported yet */ };

template<typename ProblemContext>
using compute_precision_t = typename compute_precision<ProblemContext>::type;

} //namespace perflibs::linalg::spec

#endif //PERFLIBS_LINALG_SPEC_PROBLEM_CONTEXT_HPP

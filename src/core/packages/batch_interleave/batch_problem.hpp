/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_BATCH_PROBLEM_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_BATCH_PROBLEM_HPP

namespace perflibs::linalg::batch_interleave {

/**
 * Customization point for applying an outer batch dimension to an inner problem.
 *
 * Specializations provide a problem_type alias, batch_count(), interleave_count()
 * and problem(index).
 */
template<typename InnerProblem>
struct batch_problem;

} // namespace perflibs::linalg::batch_interleave

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_BATCH_PROBLEM_HPP

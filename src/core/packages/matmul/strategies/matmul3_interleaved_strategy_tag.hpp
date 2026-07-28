/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_L3_STRATEGY_TAG_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_L3_STRATEGY_TAG_HPP

#include "matrix/adaptors.hpp"
#include "matrix/matrix_base.hpp"

#include "packages/matmul/problem_context_bases.hpp"
#include "spec/problem_context.hpp"

namespace perflibs::linalg::matmul {

// TODO: rename l2 below (it is not level 2)
class l2_trmm_matmul;

class matmul3_interleaved_strategy_tag;

class matmul3_interleaved_strategy_tag_accelerator;

} //namespace perflibs::linalg::matmul

#endif //PERFLIBS_LINALG_MATMUL_STRATEGIES_L3_STRATEGY_TAG_HPP

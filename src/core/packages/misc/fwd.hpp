/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MISC_FWD_HPP
#define PERFLIBS_LINALG_MISC_FWD_HPP

#include "packages/misc/problem_context_bases.hpp"
#include "spec/problem_context.hpp"
#include "framework/compute.hpp"

//DO NOT include strategies.hpp or anything from the strategies dir

namespace perflibs::linalg::misc {

class apply_rotation;
class apply_modified_rotation;
class generate_rotation;
class generate_modified_rotation;
class find_index_strategy;
class l1_norm_strategy;
class l2_norm_strategy;
class swap_strategy;

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<rot<T...>, ArchitectureSpec>& pctx);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<rotg<T...>, ArchitectureSpec>& pctx);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<rotm<T...>, ArchitectureSpec>& pctx);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<rotmg<T...>, ArchitectureSpec>& pctx);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<find_index<T...>, ArchitectureSpec>& pctx);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<l1_norm<T...>, ArchitectureSpec>& pctx);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<l2_norm<T...>, ArchitectureSpec>& pctx);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<swap<T...>, ArchitectureSpec>& pctx);

} // namespace perflibs::linalg::misc

#endif //PERFLIBS_LINALG_MISC_FWD_HPP

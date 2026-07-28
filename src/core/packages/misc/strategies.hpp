/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_PACKAGE_MISC_STRATEGIES_HPP
#define PERFLIBS_LINALG_PACKAGE_MISC_STRATEGIES_HPP

#include "framework/compute.hpp"

#include "packages/misc/problem_context_bases.hpp"
#include "packages/misc/strategies/apply_rotation.hpp"
#include "packages/misc/strategies/generate_rotation.hpp"
#include "packages/misc/strategies/apply_modified_rotation.hpp"
#include "packages/misc/strategies/generate_modified_rotation.hpp"
#include "packages/misc/strategies/find_index.hpp"
#include "packages/misc/strategies/l1_norm.hpp"
#include "packages/misc/strategies/l2_norm.hpp"
#include "packages/misc/strategies/swap.hpp"

namespace perflibs::linalg {

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<misc::rot<T...>, ArchitectureSpec>> = std::tuple {
	misc::apply_rotation{ }
};


template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<misc::rotg<T...>, ArchitectureSpec>> = std::tuple {
	misc::generate_rotation{ }
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<misc::rotm<T...>, ArchitectureSpec>> = std::tuple {
	misc::apply_modified_rotation{ }
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<misc::rotmg<T...>, ArchitectureSpec>> = std::tuple {
	misc::generate_modified_rotation{ }
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<misc::find_index<T...>, ArchitectureSpec>> = std::tuple {
	misc::find_index_strategy{ }
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<misc::l1_norm<T...>, ArchitectureSpec>> = std::tuple {
	misc::l1_norm_strategy{ }
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<misc::l2_norm<T...>, ArchitectureSpec>> = std::tuple {
	misc::l2_norm_strategy{ }
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<misc::swap<T...>, ArchitectureSpec>> = std::tuple {
	misc::swap_strategy{ }
};


namespace misc {

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<rot<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<rotg<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<rotm<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<rotmg<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<find_index<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<l1_norm<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<l2_norm<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<swap<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

} // namespace misc

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_PACKAGE_MISC_STRATEGIES_HPP

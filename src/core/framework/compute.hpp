/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FRAMEWORK_COMPUTE_HPP
#define PERFLIBS_LINALG_FRAMEWORK_COMPUTE_HPP

#include "framework/linalg_util.hpp"
#include "spec/problem_context.hpp"
#include "spec/strategy_tag.hpp"

#include <array>
#include <tuple>

#include <numeric>

namespace perflibs::linalg {

template<typename ProblemContext>
constexpr auto strategies = std::tuple { };

template<typename ProblemContext>
using strategies_t = decltype(strategies<ProblemContext>);

template<typename Tuple>
PERFLIBS_LINALG_INLINE
constexpr auto tuple_size(const Tuple& t) {
	return std::tuple_size_v<Tuple>;
}

template<typename StrategyList, std::size_t... Is>
std::string_view strategy_names(const StrategyList& strategy_list, std::size_t index, std::index_sequence<Is...>) {
	constexpr auto names = std::array { std::tuple_element_t<Is, StrategyList>::name()...  };

	return names[ index ];
}

template<typename StrategyList>
std::string_view get_strategy_name(const StrategyList& strategy_list, std::size_t index) {
	return strategy_names(strategy_list, index, std::make_index_sequence<std::tuple_size_v<StrategyList>>{} );
}

template<std::size_t N, typename StrategyList, typename ProblemContext>
PERFLIBS_LINALG_INLINE
auto execute_strategy(const StrategyList& tuple, const ProblemContext& pctx) {
	return std::get<N>(tuple)(pctx);
}

template<typename StrategyList, typename ProblemContext, std::size_t... Is>
PERFLIBS_LINALG_INLINE
bool compute_index(const StrategyList& strategy_list, const ProblemContext& pctx, std::size_t index, std::index_sequence<Is...>) {
	constexpr auto execute = std::array { &execute_strategy<Is, StrategyList, ProblemContext>... };

	return execute[index](strategy_list, pctx);
}

template<typename StrategyList, typename ProblemContext>
PERFLIBS_LINALG_INLINE
bool compute_index(const StrategyList& strategy_list, const ProblemContext& pctx, std::size_t index) {
	return compute_index(strategy_list, pctx, index, std::make_index_sequence<std::tuple_size_v<StrategyList>>{});
}

template<typename ProblemContext, typename System>
PERFLIBS_LINALG_INLINE
auto get_spec_system(spec::strategy_selection_tag, const ProblemContext& pctx, System) {
	constexpr auto n_strategies = std::tuple_size_v<strategies_t<ProblemContext>>;

	spec::strategy_selection_spec<n_strategies> out;
	std::iota(begin(out.strategy_preferences), end(out.strategy_preferences), 0);

	return out;
}

template<typename ProblemContext>
auto estrange(ProblemContext pctx) {
	return pctx; //does nothing, can be override locally
}

template<typename ProblemContext>
//PERFLIBS_LINALG_INLINE
inline
void compute_impl(const ProblemContext& pctx) {
	static_assert(tuple_size(strategies<ProblemContext>) > 0, "no strategies for ProblemContext");

	const auto new_pctx = estrange(pctx);

	if constexpr( tuple_size(strategies<ProblemContext>) == 1)  {
		std::get<0>( strategies<ProblemContext> )(new_pctx);
	}
	else {
		const auto spec = get_spec(spec::strategy_selection_tag{}, pctx);

		for(const auto i : spec.strategy_preferences) {
			if( compute_index(strategies<ProblemContext>, new_pctx, i)) {
				return;
			}
		}
	}
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_FRAMEWORK_COMPUTE_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_TYPE_TRAITS_HPP
#define PERFLIBS_TYPE_TRAITS_HPP

#include <type_traits>

namespace perflibs {

template <typename T, typename U>
#ifdef __cpp_concepts
	requires std::is_trivial_v<T> && std::is_trivial_v<U>
#endif
struct trivial_pair {
  T first;
  U second;
};

template<typename T0, typename T1>
using is_same_remove_cv = std::is_same<std::remove_cv_t<T0>, std::remove_cv_t<T1>>;

template<typename T0, typename T1>
constexpr bool is_same_remove_cv_v = is_same_remove_cv<T0, T1>::value;

//Like is_same but works on a variadic list of types
template<typename T0, typename... Args>
constexpr bool all_same_v = (std::is_same_v<T0, Args> && ...);

//Like is_same but works on a variadic list of types
template<typename T0, typename... Args>
constexpr bool all_same_remove_cv_v = (is_same_remove_cv_v<T0, Args> && ...);

template<typename T, T V, T... Vs>
struct is_one_of_value : std::disjunction<std::bool_constant<V == Vs>...> { };

template<typename T, T V, T... Vs>
constexpr bool is_one_of_value_v = is_one_of_value<T, V, Vs...>::value;

} // namespace perflibs

#endif // PERFLIBS_TYPE_TRAITS_HPP

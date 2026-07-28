/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_NUMERIC_UTILS_HPP
#define PERFLIBS_NUMERIC_UTILS_HPP

#include <algorithm>
#include <type_traits>

namespace perflibs {

namespace consts {

///pi in type of FloatType as constexpr
///example usage   auto circum = pi<double>*2.0*r
template<typename FloatType>
constexpr FloatType pi = FloatType(3.1415926535897932385L);

}

template<typename T1, typename T2>
__attribute__((always_inline))
inline std::common_type_t<T1, T2> max(T1 a, T2 b) {
	return std::max<std::common_type_t<T1, T2>>(a, b);
}
template<typename T1, typename T2>
__attribute__((always_inline))
inline std::common_type_t<T1, T2> min(T1 a, T2 b) {
	return std::min<std::common_type_t<T1, T2>>(a, b);
}

} // namespace perflibs

/**
 * Rounds n UP to the nearest multiple of r, if n is not already a multiple
 */
template<typename T1, typename T2>
__attribute__((always_inline))
inline std::common_type_t<T1, T2> iround(T1 n_in, T2 r_in) {
	using type = std::common_type_t<T1, T2>;
	const type n = n_in;
	const type r = r_in;

	const type diff = n % r;

	if(diff)
		return n - diff + r;

	return n;
}

/**
 * Rounds n UP to the nearest multiple of r, if n is not already a multiple
 *
 * if r_in is 0, then we just return n, we also short circuit when r_in == 1
 * because that has the same effect
 */
template<typename T1, typename T2>
__attribute__((always_inline))
inline std::common_type_t<T1, T2> iround_if_not_zero(T1 n_in, T2 r_in) {
	return r_in <= 1 ? n_in : iround(n_in, r_in);
}

/**
 * Rounds n UP to the nearest multiple of r, then divides by r.
 */
template<typename IntType1, typename IntType2>
__attribute__((always_inline))
inline std::common_type_t<IntType1, IntType2> iround_div(IntType1 n, IntType2 r) {
	return iround(n, r) / r;
}

template<typename IntType1, typename IntType2>
__attribute__((always_inline))
inline std::common_type_t<IntType1, IntType2> iround_floor(IntType1 n, IntType2 r) {
	const std::common_type_t<IntType1, IntType2> diff = n % r;

	if(diff)
		return n - diff;

	return n;
}

#endif // PERFLIBS_NUMERIC_UTILS_HPP

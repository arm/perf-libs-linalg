/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FRAMEWORK_HAS_IF_HPP
#define PERFLIBS_LINALG_FRAMEWORK_HAS_IF_HPP

#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

template<bool Has, auto Default>
struct has_if {
	using value_type = decltype(Default);
	value_type  value;

#if PERFLIBS_ADAPTOR_AGGREGATE_INIT != 0
	PERFLIBS_LINALG_INLINE
	has_if(value_type value)
	:	value { value }
	{	}
#endif

	PERFLIBS_LINALG_INLINE
	operator value_type() const { return value; }
}; //struct has_if<true, Default>

template<auto Default>
struct has_if<false, Default>  {
	using value_type = decltype(Default);

	PERFLIBS_LINALG_INLINE
	has_if(value_type) { /* ignore arg */ }

	PERFLIBS_LINALG_INLINE
	constexpr operator value_type() const { return Default; }
}; //struct has_if<false, Default>

} //namespace perflibs::linalg {

#endif //PERFLIBS_LINALG_FRAMEWORK_HAS_IF_HPP

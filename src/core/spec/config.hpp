/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SPEC_CONFIG_HPP
#define PERFLIBS_LINALG_SPEC_CONFIG_HPP

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "perflibs_util.hpp"
#include "live_target.hpp"

namespace perflibs::linalg::spec {

constexpr kernel_inttype accelerator_count_not_set_value =
	std::numeric_limits<kernel_inttype>::lowest();

// inline: one shared cached PL_LINALG_THREAD_THROTTLE value across all TUs
inline const auto global_thread_throttle = [] {
	const auto value = std::getenv("PL_LINALG_THREAD_THROTTLE");
	return value == nullptr
	     ? true
	     : std::strcmp(value, "1") == 0;
}();

inline bool thread_throttle() {
	return global_thread_throttle;
}

// inline: one shared cached PL_LINALG_SME_UNITS value across all TUs
inline const auto global_accelerator_count = [] {
	const auto env_var = std::getenv("PL_LINALG_SME_UNITS");

	if(env_var == nullptr)
		return accelerator_count_not_set_value;

	kernel_inttype value;

	const auto env_var_len = std::strlen(env_var);
	const auto env_var_end = env_var + env_var_len;

	const auto [ptr, ec] = std::from_chars(env_var, env_var_end, value);

	if(ec == std::errc{} && ptr == env_var_end)
		return value;

	return accelerator_count_not_set_value;
}();

inline kernel_inttype get_accelerator_count() {
	return global_accelerator_count;
}

} // namespace perflibs::linalg::spec

#endif // PERFLIBS_LINALG_SPEC_CONFIG_HPP

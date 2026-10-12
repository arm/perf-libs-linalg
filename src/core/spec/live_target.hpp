/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LIVE_TARGET_HPP
#define PERFLIBS_LIVE_TARGET_HPP

#if defined(PL_LINALG_LIVE_TARGET)

#include "perflibs_util.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#endif

namespace perflibs::linalg::spec {

constexpr bool is_live() {
#if defined(PL_LINALG_LIVE_TARGET)
	return PL_LINALG_LIVE_TARGET != 0;
#else
	return false;
#endif
}

#if defined(PL_LINALG_LIVE_TARGET)
namespace live {

constexpr const char *live_target_dir_env_var = "PL_LINALG_LIVE_TARGET_JSON_DIR";
constexpr const char *verbosity_level_env_var = "PL_LINALG_LIVE_TARGET_VERBOSITY_LEVEL";

using verbosity_level_t = std::uint8_t;
constexpr verbosity_level_t default_verbosity_level = 3;

inline std::filesystem::path get_live_target_dir() {
	const auto live_target_dir = std::getenv(live_target_dir_env_var);
	return std::filesystem::path { !live_target_dir ? "." : live_target_dir };
}

inline verbosity_level_t get_verbosity_level() {
	if (const auto verbosity_level_str = std::getenv(verbosity_level_env_var)) {
		char *end = nullptr;
		return std::strtoul(verbosity_level_str, &end, 10);
	}
	return default_verbosity_level;
}

inline void print(verbosity_level_t current_verbosity_level, verbosity_level_t msg_verbosity_level,
                  const char *routine, const char *datatype, const char *msg) {
	if (current_verbosity_level >= msg_verbosity_level) {
		std::fprintf(stderr, "%s_%s: %s\n", routine, datatype, msg);
	}
}

//TODO: see whether it needs to be both anon and inline
namespace {
template<typename StrategyTag, typename ProblemContext>
extern inline std::nullopt_t live_tuned_routine_spec;
}

inline kernel_inttype parse_live_case_limit(const std::string& cond) {
	// extract e.g. 32 from "scale <= 32"
	// undefined behavior if `cond` doesn't end with an integer
	return std::stoi(cond.substr(cond.find_first_of("0123456789")));
}

template<typename T>
std::vector<std::pair<kernel_inttype, T>> parse_live_cases(const nlohmann::json& json) {
	std::vector<std::pair<kernel_inttype, T>> cases;

	for (const auto& entry : json) {
		const kernel_inttype condition = parse_live_case_limit( entry["condition"].get<std::string>() );
		const T              value     = entry["value"].get<T>();

		cases.emplace_back(condition, value);
	}

	return cases;
}

} // namespace live

#else


struct s {

	template<typename T>
	operator T() const { return T {}; }
};

template<typename StrategyTag, typename ProblemContext>
s *get_spec_live(StrategyTag, const ProblemContext& pctx);

#endif

} // namespace perflibs::linalg::spec

#endif // PERFLIBS_LIVE_TARGET_HPP

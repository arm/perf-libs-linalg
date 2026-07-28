/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#if defined(PL_LINALG_LIVE_TARGET)
#include "live_target.hpp"

#include <cstdio>
#include <cstdlib>

namespace perflibs::linalg::spec::live {

std::filesystem::path get_live_target_dir() {
	const auto live_target_dir = std::getenv(live_target_dir_env_var);
	return std::filesystem::path { !live_target_dir ? "." : live_target_dir };
}

verbosity_level_t get_verbosity_level() {
	if (const auto verbosity_level_str = std::getenv(verbosity_level_env_var)) {
		char *end = nullptr;
		return std::strtoul(verbosity_level_str, &end, 10);
	}
	return default_verbosity_level;
}

void print(verbosity_level_t current_verbosity_level, verbosity_level_t msg_verbosity_level,
           const char *routine, const char *datatype, const char *msg) {
	if (current_verbosity_level >= msg_verbosity_level) {
		std::fprintf(stderr, "%s_%s: %s\n", routine, datatype, msg);
	}
}

kernel_inttype parse_live_case_limit(const std::string& cond) {
	// extract e.g. 32 from "scale <= 32"
	// undefined behavior if `cond` doesn't end with an integer
	return std::stoi(cond.substr(cond.find_first_of("0123456789")));
}

} // namespace perflibs::linalg::spec::live
#endif

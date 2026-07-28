/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_COMMON_TIMER
#define PERFLIBS_COMMON_TIMER

#include <chrono>

namespace perflibs {

static inline std::chrono::time_point<std::chrono::steady_clock> timer_start() {
	return std::chrono::steady_clock::now();
}

static inline double timer_end(std::chrono::time_point<std::chrono::steady_clock> start_time) {
	auto end_time = std::chrono::steady_clock::now();
	std::chrono::duration<double> diff = end_time - start_time;
	return diff.count();
}

} // namespace perflibs

#endif

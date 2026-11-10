/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_FFT_JSON_STATISTICS_HPP
#define PERFLIBS_FFT_JSON_STATISTICS_HPP

#include "json_optional.hpp"
#include "statistics.hpp"

namespace perflibs {
	template<>
	struct json_impl<statistics::normal_distribution> {
		static inline nlohmann::json to_json(const statistics::normal_distribution &v) {
			auto j = nlohmann::json::object_t{};
			j["type"] = "normal_distribution";
			j["mean"] = perflibs::to_json(v.mean);
			j["stddev"] = perflibs::to_json(v.stddev);
			j["n"] = perflibs::to_json(v.n);
			return j;
		}
	};

}

#endif

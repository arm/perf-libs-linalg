/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "sysctl.hpp"

#include <optional>
#include <string>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace perflibs::machine {
std::optional<sysctl_info> get_sysctl_info() {
#ifdef __APPLE__
	sysctl_info info{};

	// Get string value
	auto get_value_str = [](const char *name) {
		size_t len = 0;
		auto err = sysctlbyname(name, 0, &len, NULL, 0);
		std::string value;
		if (err) {
			return value;
		}
		value.resize(len);
		sysctlbyname(name, value.data(), &len, NULL, 0);
		return value;
	};

	// Get integer value
	auto get_value = [](const char *name) {
		int32_t value;
		size_t value_len = sizeof(value);
		const int ret = sysctlbyname(name, &value, &value_len, NULL, 0);
		return ret ? -1 : value;
	};

	auto has_feature = [&get_value](const char *name) { return get_value(name) == 1; };

	info.brand = get_value_str("machdep.cpu.brand_string");
	info.fphp = has_feature("hw.optional.neon_hpfp");
	info.neon_bf16 = has_feature("hw.optional.arm.FEAT_BF16");

	return info;
#else
	return std::nullopt;
#endif
}
} // namespace perflibs::machine

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "registry.hpp"

#include <optional>

#ifdef _WIN32
#include <winsock.h>
#endif

namespace perflibs::machine {
std::optional<registry_info> get_registry_info() {
#ifdef _WIN32
	registry_info info{};
	auto get_value = [](const char *key, const char *value) {
		std::string ret;

		HKEY hkey;
		DWORD len = 0;
		auto err = RegOpenKeyExA(HKEY_LOCAL_MACHINE, key, 0, KEY_READ, &hkey);

		if (err) return ret;
		err = RegGetValueA(hkey, nullptr, value, RRF_RT_ANY, nullptr, nullptr, &len);
		if (err) return ret;
		ret.resize(len);
		RegGetValueA(hkey, nullptr, value, RRF_RT_ANY, nullptr, (void*)ret.c_str(), &len);
		return ret;
	};

	const auto cp0 = "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0";
	info.processor_name = get_value(cp0, "ProcessorNameString");
	return info;
#else
	return std::nullopt;
#endif
}
} // namespace perflibs::machine

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "dmi.hpp"

#include "os.hpp"

#include <fstream>
#include <optional>
#include <string>

namespace perflibs::machine {
std::optional<dmi_info> get_dmi_info() {
	if constexpr (!linalg::os::linux) {
		return std::nullopt;
	}

	dmi_info info{};

	if (std::ifstream stream{ "/sys/devices/virtual/dmi/id/sys_vendor" }) {
		std::getline(stream, info.sys_vendor);
	}

	return info;
}
} // namespace perflibs::machine

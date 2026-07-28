/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#pragma once

#include <optional>
#include <string>

namespace perflibs::machine {
struct dmi_info {
	std::string sys_vendor;
};

std::optional<dmi_info> get_dmi_info();
} // namespace perflibs::machine

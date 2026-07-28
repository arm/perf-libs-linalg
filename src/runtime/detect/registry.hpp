/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#pragma once

#include <optional>
#include <string>

namespace perflibs::machine {
struct registry_info {
	// Processor name
	std::string processor_name;
};

std::optional<registry_info> get_registry_info();
} // namespace perflibs::machine

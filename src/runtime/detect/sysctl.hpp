/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#pragma once

#include <optional>
#include <string>

namespace perflibs::machine {
struct sysctl_info {
	// Brand name
	std::string brand;

	// Is half-precision supported?
	bool fphp;

	// Is bfloat16 supported?
	bool neon_bf16;
};

std::optional<sysctl_info> get_sysctl_info();
} // namespace perflibs::machine

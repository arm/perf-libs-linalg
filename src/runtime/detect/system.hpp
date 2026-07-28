/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#pragma once

#include <string_view>

namespace perflibs::machine {
enum class system {
	unknown,

	// Cortex-A72
	taishan,

	// Neoverse N1
	n1_c6g,
	n1_c6g_arm64ec,
	n1_altra_x80,
	n1_altra_max_x128,

	// Neoverse V1
	v1_c7g,

	// Neoverse N2
	n2_y710,
	n2_cobalt100,

	// Neoverse N3
	n3_axion,

	// Neoverse V2
	v2_axion,
	v2_c8g,
	v2_grace,

	// Neoverse V3
	v3_c9g,
	v3_cobalt200,

	// Ampere One
	ampere_one_gcp,

	// Apple Silicon
	apple_m1,
	apple_m2,
	apple_m4,

	// Fujitsu
	fujitsu_monaka,
};

system get_system();
system get_system_unsafe();
std::string_view get_system_str();

} // namespace perflibs::machine

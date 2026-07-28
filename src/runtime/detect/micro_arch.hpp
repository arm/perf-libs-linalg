/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#pragma once

namespace perflibs::machine {

enum class micro_arch {
	generic,
	generic_sve,
	neoverse_n1,
	neoverse_v1,
	neoverse_n2,
	neoverse_n3,
	neoverse_v2,
	neoverse_v3,
	ampere_one,
	fujitsu_monaka,
};

/// Detect the broad micro-architecture family used by runtime dispatch.
micro_arch detect_micro_arch();

} // namespace perflibs::machine

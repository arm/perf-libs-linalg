/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include <cstddef>

namespace perflibs::linalg {

std::size_t vector_length_bytes_neon() {
	return 128zu / 8zu;
}

} // namespace perflibs::linalg

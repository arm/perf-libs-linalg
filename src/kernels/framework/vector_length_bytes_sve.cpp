/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include <cstddef>
#include <arm_sve.h>

namespace perflibs::linalg {

std::size_t vector_length_bytes_sve() {
	return svcntb();
}

} //namespace perflibs::linalg

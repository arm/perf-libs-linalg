/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#pragma once

#include <vector>

namespace perflibs::machine {

// Returns a vector representing number of CPUs per node
const std::vector<int>& get_numa_topology();


} // namespace perflibs::machine

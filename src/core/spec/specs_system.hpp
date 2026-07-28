/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SPEC_GET_TUNED_ROUTINE_SPEC_HPP
#define PERFLIBS_LINALG_SPEC_GET_TUNED_ROUTINE_SPEC_HPP

#include "detect/system.hpp"

namespace perflibs::linalg::spec {

template<machine::system WhichSystem>
using system_t = std::integral_constant<machine::system, WhichSystem>;

} // namespace perflibs::linalg::spec

#endif //PERFLIBS_LINALG_SPEC_GET_TUNED_ROUTINE_SPEC_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_UTIL_HPP
#define PERFLIBS_UTIL_HPP

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "perflibs_unused.hpp"
#include "pl_linalg_int.h"

// Make sure that int_type and pl_linalg_int_t agree.
typedef pl_linalg_int_t int_type;

typedef int64_t kernel_inttype;

#define restrict __restrict__

namespace perflibs {

#if (defined(__clang__) || __GNUC__ > 7) && ! defined(__APPLE__)
typedef size_t fortran_charlen_t;
#else
typedef int fortran_charlen_t;
#endif

__attribute__((always_inline))
inline
constexpr kernel_inttype operator""_ki(unsigned long long int i) {
	return static_cast<kernel_inttype>(i);
}

} //namespace perflibs

#endif //PERFLIBS_UTIL_HPP

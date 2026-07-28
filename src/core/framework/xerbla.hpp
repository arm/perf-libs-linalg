/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FRAMEWORK_XERBLA_HPP
#define PERFLIBS_LINALG_FRAMEWORK_XERBLA_HPP

#include <string_view>

#include "perflibs_util.hpp"
#include "perflibs_ftn_symb.hpp"

extern "C" {

void FTN_SYMB(xerbla)(const char *srname, pl_linalg_int_t *info, perflibs::fortran_charlen_t srname_len);

}

namespace perflibs::linalg {

inline void call_xerbla(std::string_view srname, pl_linalg_int_t info) {
	FTN_SYMB(xerbla)(srname.data(), &info, static_cast<perflibs::fortran_charlen_t>(srname.size()));
}

} //namespace perflibs::linalg

#endif // PERFLIBS_LINALG_FRAMEWORK_XERBLA_HPP

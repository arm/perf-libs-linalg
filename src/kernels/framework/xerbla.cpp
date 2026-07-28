/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "perflibs_ftn_symb.hpp"
#include "perflibs_util.hpp"
#include <type_traits>
#include <cstdint>
#include <cstdio>
#include <inttypes.h>

#ifdef __ELF__
// Convention is to weaken xerbla for testing/user override.
extern "C" void FTN_SYMB(xerbla)(const char *srname, const pl_linalg_int_t *info,
                                 perflibs::fortran_charlen_t SRNAME_LEN)
                                 __attribute__((weak));
#endif

extern "C" void FTN_SYMB(xerbla)(const char *srname, const pl_linalg_int_t *info,
                                 perflibs::fortran_charlen_t SRNAME_LEN) {

	printf(" ** On entry to %6.*s parameter number %2" PRId64 " had an illegal value\n",
	       (int) SRNAME_LEN, srname, (int64_t)*info);

}

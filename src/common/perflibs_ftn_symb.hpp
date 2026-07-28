/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_FTN_SYMB_HPP
#define PERFLIBS_FTN_SYMB_HPP

// Macro to produce platform-specific Fortran symbols
// Default to Linux-style single trailing underscore
#define FTN_SYMB(FN_NAME) FN_NAME##_

// Macro for variadic C/C++ prototypes, appending "..." on non-Arm64EC targets.
#if defined(_M_ARM64EC) || defined(__arm64ec__)
#define C_VARARGS_DECL
#define C_VARARGS_CALL(...)
#else
#define C_VARARGS_DECL , ...
#define C_VARARGS_CALL(...) , __VA_ARGS__
#endif

#endif

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#ifndef PL_LINALG_INT_T
#define PL_LINALG_INT_T

#ifdef INTEGER64
typedef int64_t  pl_linalg_int_t;
typedef uint64_t pl_linalg_uint_t;
#else
typedef int32_t  pl_linalg_int_t;
typedef uint32_t pl_linalg_uint_t;
#endif
typedef int32_t  pl_linalg_strlen_t;

#endif

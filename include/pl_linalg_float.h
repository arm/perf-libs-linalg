/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef _PERFLIBS_FLOAT_H
#define _PERFLIBS_FLOAT_H

// __bf16 type is only supported in GCC 10.1 and newer
#if !defined(__NVCOMPILER) && defined(__GNUC__) && !defined(__clang__) && (__GNUC__ > 10 || (__GNUC__ == 10 && __GNUC_MINOR__ >= 1))
#define _PERFLIBS_COMPILER_HAS_BF16_DATATYPE 1
// __bf16 type is only supported in Clang 11.0 and newer
#elif defined(__clang__) && (__clang_major__ > 11 || (__clang_major__ == 11 && __clang_minor__ >= 0))
#define _PERFLIBS_COMPILER_HAS_BF16_DATATYPE 1
#else
#define _PERFLIBS_COMPILER_HAS_BF16_DATATYPE 0
#endif

/*
  The BFloat BLAS and CBLAS extensions use `pl_linalg_bf16_t` which will be `__bf16` if there is compiler support
  for the data type. If there is no compiler support then `pl_linalg_bf16_t` will be void, meaning pointer
  parameters will be `void*`, routines which take BFloat16 scalars by value or return them will be hidden.

  The user may override logic to determine if compiler supports the `__bf16` datatype by defining `PL_LINALG_ALLOW_BF16_INTERFACE` to 1.
 */
#if ( defined(PL_LINALG_ALLOW_BF16_INTERFACE) && PL_LINALG_ALLOW_BF16_INTERFACE != 0 ) || _PERFLIBS_COMPILER_HAS_BF16_DATATYPE == 1 || ( defined(__ARM_FEATURE_BF16) && __ARM_FEATURE_BF16 != 0 ) || ( defined(__ARM_FEATURE_SVE_BF16) && __ARM_FEATURE_SVE_BF16 != 0)
typedef __bf16 pl_linalg_bf16_t;
#define _PERFLIBS_BF16_INTERFACE 1
#else
typedef void pl_linalg_bf16_t;
#define _PERFLIBS_BF16_INTERFACE 0
#endif


#if !defined(__NVCOMPILER) && ( !defined(_MSC_VER) || defined(__clang__) )
#define _PERFLIBS_COMPILER_HAS_FP16_DATATYPE 1
#else
#define _PERFLIBS_COMPILER_HAS_FP16_DATATYPE 0
#endif

/*
  The FP16 BLAS and CBLAS extensions use `pl_linalg_fp16_t` which will be `__fp16` if there is compiler support
  for the data type. If there is no compiler support then `pl_linalg_fp16_t` will be void, meaning pointer
  parameters will be `void*`, routines which take FP16 scalars by value or return them will be hidden.

  The user may override logic to determine if compiler supports the `__fp16` datatype by defining `PL_LINALG_ALLOW_FP16_INTERFACE` to 1.
 */
#if ( defined(PL_LINALG_ALLOW_FP16_INTERFACE) && PL_LINALG_ALLOW_FP16_INTERFACE != 0 ) || _PERFLIBS_COMPILER_HAS_FP16_DATATYPE == 1
#define _PERFLIBS_FP16_INTERFACE 1
typedef __fp16 pl_linalg_fp16_t;
#else
#define _PERFLIBS_FP16_INTERFACE 0
typedef void pl_linalg_fp16_t;
#endif

#endif  /* !defined(_PERFLIBS_FLOAT_H) */

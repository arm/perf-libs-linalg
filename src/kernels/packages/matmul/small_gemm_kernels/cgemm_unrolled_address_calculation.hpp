/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */


#if defined(A_TRANSPOSE) || defined(A_CONJUGATE)
#define A_ADDRESS_CALCULATION lda*(im + ium) + ik + iuk
#else
#define A_ADDRESS_CALCULATION lda*(ik + iuk) + im + ium
#endif

#if defined(B_TRANSPOSE) || defined(B_CONJUGATE)
#define B_ADDRESS_CALCULATION ldb*(ik + iuk) + in + iun
#else
#define B_ADDRESS_CALCULATION ldb*(in + iun) + ik + iuk
#endif


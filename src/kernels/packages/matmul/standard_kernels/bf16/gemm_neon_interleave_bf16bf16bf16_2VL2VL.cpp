/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "framework/linalg_util.hpp"

#include <cstdint>
#include <cstddef>

extern "C" {

void a64_interleaved_nomerge_bf16fp32bf16_mmla_4x8(
    const bfloat16_t *A,
    const bfloat16_t *B,
    bfloat16_t *C,

    kernel_inttype K,
    kernel_inttype N,
    kernel_inttype M,
    kernel_inttype ldc,
	bfloat16_t alpha,
	bfloat16_t beta)
{
  auto Apanel = B;
  auto Cpanel = C;

  struct KernelArgs {
    KernelArgs(
      const bfloat16_t *const B,
      const bfloat16_t *const A,
      bfloat16_t *const C,
      const int K_, const int M_, const int N_,
      const int ldc,
	float alpha, float beta)

      : Apanel(A),
        Bpanel(B),
        Cpanel(C),
        M(M_), N(N_), K(K_/4-1),
        ldcb(ldc * sizeof(bfloat16_t)),
		alpha(alpha), beta(beta),
        flags(0x0)
    {
    }

    const bfloat16_t *const Apanel;
    const bfloat16_t *const Bpanel;
    bfloat16_t *const Cpanel;
    const long M, N, K;
    const long ldcb;

    const float alpha;
    const float beta;

    std::uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs ka (
		A, B, C,
		K, M, N,
		ldc,
		(float)alpha, (float)beta);

  __asm__ __volatile__(
      "cbz %x[M], 19f\n"
      "1:"  // Height loop
      "add x28, %x[Cpanel], %x[ldc], LSL #1\n"
      "cmp %x[M], #0x4\n"
      "add x27, %x[Cpanel], %x[ldc], LSL #2\n"
      "add x26, %x[Cpanel], %x[ldc]\n"
      "add x25, x28, %x[ldc]\n"
      "bge 2f\n"
      "cmp %x[M], #0x2\n"
      "mov x25, %x[Cpanel]\n"
      "csel x26, x26, %x[Cpanel], GE\n"
      "csel x28, x28, %x[Cpanel], GT\n"
      "2:"  // all rows valid
      "ldr x24, [%x[args_ptr], %[offsetof_N]]\n"
      "ldr x23, [%x[args_ptr], %[offsetof_Bpanel]]\n"
      "mov x22, %x[Apanel]\n"
      "3:"  // Width loop
      "ldr q2, [x23, #0x0]\n"
      "ldr q3, [x23, #0x10]\n"
      "mov %x[Apanel], x22\n"
      "ldr q4, [x23, #0x20]\n"
      "ldr q5, [x23, #0x30]\n"
      "movi v24.16b, #0x0\n"
      "ldr x20, [%x[args_ptr], %[offsetof_K]]\n"
      "movi v25.16b, #0x0\n"
      "movi v26.16b, #0x0\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "movi v27.16b, #0x0\n"
      "movi v28.16b, #0x0\n"
      "movi v29.16b, #0x0\n"
      "prfm pldl1keep, [%x[Apanel], #0x0]\n"
      "cmp x20, #0x2\n"
      "movi v30.16b, #0x0\n"
      "movi v31.16b, #0x0\n"
      "prfm pldl1keep, [x23, #0x0]\n"
      "prfm pldl1keep, [x23, #0x40]\n"
      "prfm pldl1keep, [%x[Apanel], #0x40]\n"
      "prfm pldl1keep, [x23, #0x80]\n"
      "prfm pldl1keep, [x23, #0xc0]\n"
      "blt 5f\n"
      "4:"  // main loop head
      ".inst 0x6e42ec18  // bfmmla v24.4s, v0.8h, v2.8h\n"
      ".inst 0x6e42ec3c  // bfmmla v28.4s, v1.8h, v2.8h\n"
      "ldr q2, [x23, #0x40]\n"
      ".inst 0x6e43ec1a  // bfmmla v26.4s, v0.8h, v3.8h\n"
      ".inst 0x6e43ec3e  // bfmmla v30.4s, v1.8h, v3.8h\n"
      "ldr q3, [x23, #0x50]\n"
      ".inst 0x6e44ec19  // bfmmla v25.4s, v0.8h, v4.8h\n"
      ".inst 0x6e44ec3d  // bfmmla v29.4s, v1.8h, v4.8h\n"
      "ldr q4, [x23, #0x60]\n"
      ".inst 0x6e45ec1b  // bfmmla v27.4s, v0.8h, v5.8h\n"
      "ldr q0, [%x[Apanel], #0x20]\n"
      ".inst 0x6e45ec3f  // bfmmla v31.4s, v1.8h, v5.8h\n"
      "ldr q1, [%x[Apanel], #0x30]\n"
      "ldr q5, [x23, #0x70]\n"
      "sub x20, x20, #0x2\n"
      "prfm pldl1keep, [%x[Apanel], #0x80]\n"
      "prfm pldl1keep, [x23, #0x100]\n"
      "cmp x20, #0x2\n"
      "add %x[Apanel], %x[Apanel], #0x40\n"
      "prfm pldl1keep, [x23, #0x140]\n"
      "add x23, x23, #0x80\n"
      ".inst 0x6e42ec18  // bfmmla v24.4s, v0.8h, v2.8h\n"
      ".inst 0x6e43ec1a  // bfmmla v26.4s, v0.8h, v3.8h\n"
      ".inst 0x6e44ec19  // bfmmla v25.4s, v0.8h, v4.8h\n"
      ".inst 0x6e42ec3c  // bfmmla v28.4s, v1.8h, v2.8h\n"
      "ldr q2, [x23, #0x0]\n"
      ".inst 0x6e43ec3e  // bfmmla v30.4s, v1.8h, v3.8h\n"
      "ldr q3, [x23, #0x10]\n"
      ".inst 0x6e45ec1b  // bfmmla v27.4s, v0.8h, v5.8h\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      ".inst 0x6e44ec3d  // bfmmla v29.4s, v1.8h, v4.8h\n"
      "ldr q4, [x23, #0x20]\n"
      ".inst 0x6e45ec3f  // bfmmla v31.4s, v1.8h, v5.8h\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "ldr q5, [x23, #0x30]\n"
      "bge 4b\n"
      "5:"  // main loop skip
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "add x23, x23, #0x40\n"
      ".inst 0x6e42ec18  // bfmmla v24.4s, v0.8h, v2.8h\n"
      ".inst 0x6e42ec3c  // bfmmla v28.4s, v1.8h, v2.8h\n"
      ".inst 0x6e43ec1a  // bfmmla v26.4s, v0.8h, v3.8h\n"
      ".inst 0x6e43ec3e  // bfmmla v30.4s, v1.8h, v3.8h\n"
      ".inst 0x6e44ec19  // bfmmla v25.4s, v0.8h, v4.8h\n"
      ".inst 0x6e45ec1b  // bfmmla v27.4s, v0.8h, v5.8h\n"
      ".inst 0x6e44ec3d  // bfmmla v29.4s, v1.8h, v4.8h\n"
      ".inst 0x6e45ec3f  // bfmmla v31.4s, v1.8h, v5.8h\n"
      "cbz x20, 6f\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "ldr q6, [x23, #0x0]\n"
      "ldr q7, [x23, #0x10]\n"
      "ldr q8, [x23, #0x20]\n"
      "ldr q9, [x23, #0x30]\n"
      "add x23, x23, #0x40\n"
      ".inst 0x6e46ec18  // bfmmla v24.4s, v0.8h, v6.8h\n"
      ".inst 0x6e47ec1a  // bfmmla v26.4s, v0.8h, v7.8h\n"
      ".inst 0x6e46ec3c  // bfmmla v28.4s, v1.8h, v6.8h\n"
      ".inst 0x6e47ec3e  // bfmmla v30.4s, v1.8h, v7.8h\n"
      ".inst 0x6e48ec19  // bfmmla v25.4s, v0.8h, v8.8h\n"
      ".inst 0x6e49ec1b  // bfmmla v27.4s, v0.8h, v9.8h\n"
      ".inst 0x6e48ec3d  // bfmmla v29.4s, v1.8h, v8.8h\n"
      ".inst 0x6e49ec3f  // bfmmla v31.4s, v1.8h, v9.8h\n"
      "6:"  // multiply loop done
      "add x21, %x[args_ptr], %[offset_beta]\n"
      "add x20, %x[args_ptr], %[offset_alpha]\n"
      "uzp1 v10.2d, v24.2d, v26.2d\n"
      "ld1r { v1.4s }, [x21]\n"
      "ld1r { v0.4s }, [x20]\n"
      "cmp x24, #0x8\n"
      "uzp2 v24.2d, v24.2d, v26.2d\n"
      "uzp1 v26.2d, v25.2d, v27.2d\n"
      "uzp2 v25.2d, v25.2d, v27.2d\n"
      "uzp1 v27.2d, v28.2d, v30.2d\n"
      "uzp2 v28.2d, v28.2d, v30.2d\n"
      "uzp1 v30.2d, v29.2d, v31.2d\n"
      "uzp2 v29.2d, v29.2d, v31.2d\n"
      "blt 7f\n"
      "ldr q23, [%x[Cpanel], #0x0]\n"
      "ldr q22, [x26, #0x0]\n"
      "ldr q21, [x28, #0x0]\n"
      "ldr q20, [x25, #0x0]\n"
      "shll2 v19.4s, v23.8h, #0x10\n"
      "shll v23.4s, v23.4h, #0x10\n"
      "shll2 v18.4s, v22.8h, #0x10\n"
      "shll v22.4s, v22.4h, #0x10\n"
      "shll2 v17.4s, v21.8h, #0x10\n"
      "shll v21.4s, v21.4h, #0x10\n"
      "shll2 v16.4s, v20.8h, #0x10\n"
      "shll v20.4s, v20.4h, #0x10\n"
      "b 12f\n"
      "7:"  // partial result load for beta
      "tbz x24, #2, 9f\n"
      "ldr d20, [x25], #0x8\n"
      "ldr d21, [x28], #0x8\n"
      "ldr d22, [x26], #0x8\n"
      "ldr d23, [%x[Cpanel]], #0x8\n"
      "tbz x24, #1, 8f\n"
      "ld1 { v20.s }[2], [x25], #0x4\n"
      "ld1 { v21.s }[2], [x28], #0x4\n"
      "mov x20, #0xc\n"
      "ld1 { v22.s }[2], [x26], #0x4\n"
      "ld1 { v23.s }[2], [%x[Cpanel]], #0x4\n"
      "tbz x24, #0, 11f\n"
      "ld1 { v20.h }[6], [x25]\n"
      "ld1 { v21.h }[6], [x28]\n"
      "ld1 { v22.h }[6], [x26]\n"
      "ld1 { v23.h }[6], [%x[Cpanel]]\n"
      "b 11f\n"
      "8:"  // partial result load for beta: partial_1_4
      "mov x20, #0x8\n"
      "tbz x24, #0, 11f\n"
      "ld1 { v20.h }[4], [x25]\n"
      "ld1 { v21.h }[4], [x28]\n"
      "ld1 { v22.h }[4], [x26]\n"
      "ld1 { v23.h }[4], [%x[Cpanel]]\n"
      "b 11f\n"
      "9:"  // partial result load for beta: partial_2_0
      "tbz x24, #1, 10f\n"
      "ldr s20, [x25], #0x4\n"
      "ldr s21, [x28], #0x4\n"
      "mov x20, #0x4\n"
      "ldr s22, [x26], #0x4\n"
      "ldr s23, [%x[Cpanel]], #0x4\n"
      "tbz x24, #0, 11f\n"
      "ld1 { v20.h }[2], [x25]\n"
      "ld1 { v21.h }[2], [x28]\n"
      "ld1 { v22.h }[2], [x26]\n"
      "ld1 { v23.h }[2], [%x[Cpanel]]\n"
      "b 11f\n"
      "10:"  // partial result load for beta: partial_1_0
      "ldr h20, [x25, #0x0]\n"
      "ldr h21, [x28, #0x0]\n"
      "mov x20, #0x0\n"
      "ldr h22, [x26, #0x0]\n"
      "ldr h23, [%x[Cpanel], #0x0]\n"
      "11:"  // partial result load for beta: Done
      "sub %x[Cpanel], %x[Cpanel], x20\n"
      "sub x26, x26, x20\n"
      "shll2 v19.4s, v23.8h, #0x10\n"
      "sub x28, x28, x20\n"
      "sub x25, x25, x20\n"
      "shll v23.4s, v23.4h, #0x10\n"
      "shll2 v18.4s, v22.8h, #0x10\n"
      "shll v22.4s, v22.4h, #0x10\n"
      "shll2 v17.4s, v21.8h, #0x10\n"
      "shll v21.4s, v21.4h, #0x10\n"
      "shll2 v16.4s, v20.8h, #0x10\n"
      "shll v20.4s, v20.4h, #0x10\n"
      "12:"  // result load for beta done
      "fmul v28.4s, v28.4s, v0.4s\n"
      "fmul v27.4s, v27.4s, v0.4s\n"
      "cmp x24, #0x8\n"
      "fmul v24.4s, v24.4s, v0.4s\n"
      "fmul v10.4s, v10.4s, v0.4s\n"
      "fmul v29.4s, v29.4s, v0.4s\n"
      "fmul v30.4s, v30.4s, v0.4s\n"
      "fmla v28.4s, v20.4s, v1.4s\n"
      "fmla v27.4s, v21.4s, v1.4s\n"
      "fmul v25.4s, v25.4s, v0.4s\n"
      "fmla v24.4s, v22.4s, v1.4s\n"
      "fmla v10.4s, v23.4s, v1.4s\n"
      "fmul v26.4s, v26.4s, v0.4s\n"
      "fmla v29.4s, v16.4s, v1.4s\n"
      "fmla v30.4s, v17.4s, v1.4s\n"
      "fmla v25.4s, v18.4s, v1.4s\n"
      "fmla v26.4s, v19.4s, v1.4s\n"
      ".inst 0x0ea16b7b  // bfcvtn v27.4h, v27.4s\n"
      ".inst 0x0ea1694a  // bfcvtn v10.4h, v10.4s\n"
      ".inst 0x0ea16b18  // bfcvtn v24.4h, v24.4s\n"
      ".inst 0x0ea16b9c  // bfcvtn v28.4h, v28.4s\n"
      ".inst 0x4ea16bdb  // bfcvtn2 v27.8h, v30.4s\n"
      ".inst 0x4ea16b4a  // bfcvtn2 v10.8h, v26.4s\n"
      ".inst 0x4ea16b38  // bfcvtn2 v24.8h, v25.4s\n"
      ".inst 0x4ea16bbc  // bfcvtn2 v28.8h, v29.4s\n"
      "blt 13f\n"
      "str q28, [x25, #0x0]\n"
      "add x25, x25, #0x10\n"
      "str q27, [x28, #0x0]\n"
      "add x28, x28, #0x10\n"
      "str q24, [x26, #0x0]\n"
      "add x26, x26, #0x10\n"
      "str q10, [%x[Cpanel], #0x0]\n"
      "add %x[Cpanel], %x[Cpanel], #0x10\n"
      "b 18f\n"
      "13:"  // partial output
      "tbz x24, #2, 15f\n"
      "str d28, [x25], #0x8\n"
      "str d27, [x28], #0x8\n"
      "str d24, [x26], #0x8\n"
      "str d10, [%x[Cpanel]], #0x8\n"
      "tbz x24, #1, 14f\n"
      "st1 { v28.s }[2], [x25], #0x4\n"
      "st1 { v27.s }[2], [x28], #0x4\n"
      "st1 { v24.s }[2], [x26], #0x4\n"
      "st1 { v10.s }[2], [%x[Cpanel]], #0x4\n"
      "tbz x24, #0, 17f\n"
      "st1 { v28.h }[6], [x25]\n"
      "st1 { v27.h }[6], [x28]\n"
      "st1 { v24.h }[6], [x26]\n"
      "st1 { v10.h }[6], [%x[Cpanel]]\n"
      "b 17f\n"
      "14:"  // partial result store: partial_1_4
      "tbz x24, #0, 17f\n"
      "st1 { v28.h }[4], [x25]\n"
      "st1 { v27.h }[4], [x28]\n"
      "st1 { v24.h }[4], [x26]\n"
      "st1 { v10.h }[4], [%x[Cpanel]]\n"
      "b 17f\n"
      "15:"  // partial result store: partial_2_0
      "tbz x24, #1, 16f\n"
      "str s28, [x25], #0x4\n"
      "str s27, [x28], #0x4\n"
      "str s24, [x26], #0x4\n"
      "str s10, [%x[Cpanel]], #0x4\n"
      "tbz x24, #0, 17f\n"
      "st1 { v28.h }[2], [x25]\n"
      "st1 { v27.h }[2], [x28]\n"
      "st1 { v24.h }[2], [x26]\n"
      "st1 { v10.h }[2], [%x[Cpanel]]\n"
      "b 17f\n"
      "16:"  // partial result store: partial_1_0
      "str h28, [x25, #0x0]\n"
      "str h27, [x28, #0x0]\n"
      "str h24, [x26, #0x0]\n"
      "str h10, [%x[Cpanel], #0x0]\n"
      "17:"  // partial result store: Done
      "18:"  // store done
      "subs x24, x24, #0x8\n"
      "bgt 3b\n"
      "subs %x[M], %x[M], #0x4\n"
      "mov %x[Cpanel], x27\n"
      "bgt 1b\n"
      "19:"  // Exit
      : [Apanel] "+&r" (Apanel), [Cpanel] "+&r" (Cpanel), [M] "+&r" (M)
      : [args_ptr] "r" (&ka), [ldc] "r" (ldc * sizeof(bfloat16_t)), [offset_alpha] "I" (offsetof(KernelArgs, alpha)), [offset_beta] "I" (offsetof(KernelArgs, beta)), [offsetof_Bpanel] "I" (offsetof(KernelArgs, Bpanel)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_N] "I" (offsetof(KernelArgs, N))
      : "cc", "memory", "v0", "v1", "v10", "v16", "v17", "v18", "v19", "v2", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v3", "v30", "v31", "v4", "v5", "v6", "v7", "v8", "v9", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
    );
}

}  // extern "C"

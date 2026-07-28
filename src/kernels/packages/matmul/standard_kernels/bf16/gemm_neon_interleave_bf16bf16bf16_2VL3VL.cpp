/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "framework/linalg_util.hpp"

#include <cstdint>
#include <cstddef>

extern "C" {

void a64_interleaved_nomerge_bf16fp32bf16_mmla_4x12(
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
      "cbz %x[M], 23f\n"
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
      "movi v20.16b, #0x0\n"
      "ldr q6, [x23, #0x40]\n"
      "ldr q7, [x23, #0x50]\n"
      "movi v21.16b, #0x0\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "movi v22.16b, #0x0\n"
      "ldr x20, [%x[args_ptr], %[offsetof_K]]\n"
      "movi v23.16b, #0x0\n"
      "movi v24.16b, #0x0\n"
      "prfm pldl1keep, [%x[Apanel], #0x0]\n"
      "movi v25.16b, #0x0\n"
      "movi v26.16b, #0x0\n"
      "movi v27.16b, #0x0\n"
      "prfm pldl1keep, [x23, #0x0]\n"
      "movi v28.16b, #0x0\n"
      "cmp x20, #0x2\n"
      "movi v29.16b, #0x0\n"
      "prfm pldl1keep, [x23, #0x40]\n"
      "movi v30.16b, #0x0\n"
      "movi v31.16b, #0x0\n"
      "prfm pldl1keep, [x23, #0x80]\n"
      "prfm pldl1keep, [%x[Apanel], #0x40]\n"
      "prfm pldl1keep, [x23, #0xc0]\n"
      "prfm pldl1keep, [x23, #0x100]\n"
      "prfm pldl1keep, [x23, #0x140]\n"
      "blt 5f\n"
      "4:"  // main loop head
      ".inst 0x6e42ec14  // bfmmla v20.4s, v0.8h, v2.8h\n"
      ".inst 0x6e42ec3a  // bfmmla v26.4s, v1.8h, v2.8h\n"
      "ldr q2, [x23, #0x60]\n"
      ".inst 0x6e43ec17  // bfmmla v23.4s, v0.8h, v3.8h\n"
      ".inst 0x6e43ec3d  // bfmmla v29.4s, v1.8h, v3.8h\n"
      "ldr q3, [x23, #0x70]\n"
      ".inst 0x6e44ec15  // bfmmla v21.4s, v0.8h, v4.8h\n"
      ".inst 0x6e45ec18  // bfmmla v24.4s, v0.8h, v5.8h\n"
      "sub x20, x20, #0x2\n"
      ".inst 0x6e44ec3b  // bfmmla v27.4s, v1.8h, v4.8h\n"
      "ldr q4, [x23, #0x80]\n"
      ".inst 0x6e45ec3e  // bfmmla v30.4s, v1.8h, v5.8h\n"
      "ldr q5, [x23, #0x90]\n"
      ".inst 0x6e46ec16  // bfmmla v22.4s, v0.8h, v6.8h\n"
      ".inst 0x6e47ec19  // bfmmla v25.4s, v0.8h, v7.8h\n"
      "ldr q0, [%x[Apanel], #0x20]\n"
      ".inst 0x6e46ec3c  // bfmmla v28.4s, v1.8h, v6.8h\n"
      "ldr q6, [x23, #0xa0]\n"
      ".inst 0x6e47ec3f  // bfmmla v31.4s, v1.8h, v7.8h\n"
      "ldr q1, [%x[Apanel], #0x30]\n"
      "ldr q7, [x23, #0xb0]\n"
      "prfm pldl1keep, [%x[Apanel], #0x80]\n"
      "add %x[Apanel], %x[Apanel], #0x40\n"
      "cmp x20, #0x2\n"
      "prfm pldl1keep, [x23, #0x180]\n"
      "prfm pldl1keep, [x23, #0x1c0]\n"
      ".inst 0x6e42ec14  // bfmmla v20.4s, v0.8h, v2.8h\n"
      "prfm pldl1keep, [x23, #0x200]\n"
      "add x23, x23, #0xc0\n"
      ".inst 0x6e43ec17  // bfmmla v23.4s, v0.8h, v3.8h\n"
      ".inst 0x6e42ec3a  // bfmmla v26.4s, v1.8h, v2.8h\n"
      "ldr q2, [x23, #0x0]\n"
      ".inst 0x6e43ec3d  // bfmmla v29.4s, v1.8h, v3.8h\n"
      "ldr q3, [x23, #0x10]\n"
      ".inst 0x6e44ec15  // bfmmla v21.4s, v0.8h, v4.8h\n"
      ".inst 0x6e45ec18  // bfmmla v24.4s, v0.8h, v5.8h\n"
      ".inst 0x6e44ec3b  // bfmmla v27.4s, v1.8h, v4.8h\n"
      "ldr q4, [x23, #0x20]\n"
      ".inst 0x6e45ec3e  // bfmmla v30.4s, v1.8h, v5.8h\n"
      "ldr q5, [x23, #0x30]\n"
      ".inst 0x6e46ec16  // bfmmla v22.4s, v0.8h, v6.8h\n"
      ".inst 0x6e47ec19  // bfmmla v25.4s, v0.8h, v7.8h\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      ".inst 0x6e46ec3c  // bfmmla v28.4s, v1.8h, v6.8h\n"
      "ldr q6, [x23, #0x40]\n"
      ".inst 0x6e47ec3f  // bfmmla v31.4s, v1.8h, v7.8h\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "ldr q7, [x23, #0x50]\n"
      "bge 4b\n"
      "5:"  // main loop skip
      "add %x[Apanel], %x[Apanel], #0x20\n"
      ".inst 0x6e42ec14  // bfmmla v20.4s, v0.8h, v2.8h\n"
      "add x23, x23, #0x60\n"
      ".inst 0x6e42ec3a  // bfmmla v26.4s, v1.8h, v2.8h\n"
      ".inst 0x6e43ec17  // bfmmla v23.4s, v0.8h, v3.8h\n"
      ".inst 0x6e43ec3d  // bfmmla v29.4s, v1.8h, v3.8h\n"
      ".inst 0x6e44ec15  // bfmmla v21.4s, v0.8h, v4.8h\n"
      ".inst 0x6e45ec18  // bfmmla v24.4s, v0.8h, v5.8h\n"
      ".inst 0x6e44ec3b  // bfmmla v27.4s, v1.8h, v4.8h\n"
      ".inst 0x6e45ec3e  // bfmmla v30.4s, v1.8h, v5.8h\n"
      ".inst 0x6e46ec16  // bfmmla v22.4s, v0.8h, v6.8h\n"
      ".inst 0x6e47ec19  // bfmmla v25.4s, v0.8h, v7.8h\n"
      ".inst 0x6e46ec3c  // bfmmla v28.4s, v1.8h, v6.8h\n"
      ".inst 0x6e47ec3f  // bfmmla v31.4s, v1.8h, v7.8h\n"
      "cbz x20, 6f\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "ldr q8, [x23, #0x0]\n"
      "ldr q9, [x23, #0x10]\n"
      "ldr q10, [x23, #0x20]\n"
      "ldr q11, [x23, #0x30]\n"
      "ldr q12, [x23, #0x40]\n"
      "ldr q13, [x23, #0x50]\n"
      "add x23, x23, #0x60\n"
      ".inst 0x6e48ec14  // bfmmla v20.4s, v0.8h, v8.8h\n"
      ".inst 0x6e49ec17  // bfmmla v23.4s, v0.8h, v9.8h\n"
      ".inst 0x6e48ec3a  // bfmmla v26.4s, v1.8h, v8.8h\n"
      ".inst 0x6e49ec3d  // bfmmla v29.4s, v1.8h, v9.8h\n"
      ".inst 0x6e4aec15  // bfmmla v21.4s, v0.8h, v10.8h\n"
      ".inst 0x6e4bec18  // bfmmla v24.4s, v0.8h, v11.8h\n"
      ".inst 0x6e4aec3b  // bfmmla v27.4s, v1.8h, v10.8h\n"
      ".inst 0x6e4bec3e  // bfmmla v30.4s, v1.8h, v11.8h\n"
      ".inst 0x6e4cec16  // bfmmla v22.4s, v0.8h, v12.8h\n"
      ".inst 0x6e4dec19  // bfmmla v25.4s, v0.8h, v13.8h\n"
      ".inst 0x6e4cec3c  // bfmmla v28.4s, v1.8h, v12.8h\n"
      ".inst 0x6e4dec3f  // bfmmla v31.4s, v1.8h, v13.8h\n"
      "6:"  // multiply loop done
      "add x21, %x[args_ptr], %[offset_beta]\n"
      "add x20, %x[args_ptr], %[offset_alpha]\n"
      "uzp1 v14.2d, v20.2d, v23.2d\n"
      "ld1r { v8.4s }, [x21]\n"
      "ld1r { v7.4s }, [x20]\n"
      "cmp x24, #0xc\n"
      "uzp2 v20.2d, v20.2d, v23.2d\n"
      "uzp1 v23.2d, v21.2d, v24.2d\n"
      "uzp2 v21.2d, v21.2d, v24.2d\n"
      "uzp1 v24.2d, v22.2d, v25.2d\n"
      "uzp2 v22.2d, v22.2d, v25.2d\n"
      "uzp1 v25.2d, v26.2d, v29.2d\n"
      "uzp2 v26.2d, v26.2d, v29.2d\n"
      "uzp1 v29.2d, v27.2d, v30.2d\n"
      "uzp2 v27.2d, v27.2d, v30.2d\n"
      "uzp1 v30.2d, v28.2d, v31.2d\n"
      "uzp2 v28.2d, v28.2d, v31.2d\n"
      "blt 7f\n"
      "ldr q6, [%x[Cpanel], #0x0]\n"
      "ldr d19, [%x[Cpanel], #0x10]\n"
      "ldr q5, [x26, #0x0]\n"
      "ldr d18, [x26, #0x10]\n"
      "ldr q4, [x28, #0x0]\n"
      "ldr d17, [x28, #0x10]\n"
      "ldr q3, [x25, #0x0]\n"
      "ldr d16, [x25, #0x10]\n"
      "shll v2.4s, v19.4h, #0x10\n"
      "shll2 v1.4s, v6.8h, #0x10\n"
      "shll v6.4s, v6.4h, #0x10\n"
      "shll v0.4s, v18.4h, #0x10\n"
      "shll2 v31.4s, v5.8h, #0x10\n"
      "shll v5.4s, v5.4h, #0x10\n"
      "shll v19.4s, v17.4h, #0x10\n"
      "shll2 v18.4s, v4.8h, #0x10\n"
      "shll v4.4s, v4.4h, #0x10\n"
      "shll v17.4s, v16.4h, #0x10\n"
      "shll2 v16.4s, v3.8h, #0x10\n"
      "shll v3.4s, v3.4h, #0x10\n"
      "b 14f\n"
      "7:"  // partial result load for beta
      "movi v1.16b, #0x0\n"
      "movi v31.16b, #0x0\n"
      "movi v18.16b, #0x0\n"
      "movi v16.16b, #0x0\n"
      "tbz x24, #3, 9f\n"
      "ld1 { v3.8h }, [x25], #0x10\n"
      "ld1 { v4.8h }, [x28], #0x10\n"
      "ld1 { v5.8h }, [x26], #0x10\n"
      "ld1 { v6.8h }, [%x[Cpanel]], #0x10\n"
      "tbz x24, #1, 8f\n"
      "ldr s16, [x25], #0x4\n"
      "ldr s18, [x28], #0x4\n"
      "mov x20, #0x14\n"
      "ldr s31, [x26], #0x4\n"
      "ldr s1, [%x[Cpanel]], #0x4\n"
      "tbz x24, #0, 13f\n"
      "ld1 { v16.h }[2], [x25]\n"
      "ld1 { v18.h }[2], [x28]\n"
      "ld1 { v31.h }[2], [x26]\n"
      "ld1 { v1.h }[2], [%x[Cpanel]]\n"
      "b 13f\n"
      "8:"  // partial result load for beta: partial_1_8
      "mov x20, #0x10\n"
      "tbz x24, #0, 13f\n"
      "ldr h16, [x25, #0x0]\n"
      "ldr h18, [x28, #0x0]\n"
      "ldr h31, [x26, #0x0]\n"
      "ldr h1, [%x[Cpanel], #0x0]\n"
      "b 13f\n"
      "9:"  // partial result load for beta: partial_4_0
      "tbz x24, #2, 11f\n"
      "ldr d3, [x25], #0x8\n"
      "ldr d4, [x28], #0x8\n"
      "ldr d5, [x26], #0x8\n"
      "ldr d6, [%x[Cpanel]], #0x8\n"
      "tbz x24, #1, 10f\n"
      "ld1 { v3.s }[2], [x25], #0x4\n"
      "ld1 { v4.s }[2], [x28], #0x4\n"
      "mov x20, #0xc\n"
      "ld1 { v5.s }[2], [x26], #0x4\n"
      "ld1 { v6.s }[2], [%x[Cpanel]], #0x4\n"
      "tbz x24, #0, 13f\n"
      "ld1 { v3.h }[6], [x25]\n"
      "ld1 { v4.h }[6], [x28]\n"
      "ld1 { v5.h }[6], [x26]\n"
      "ld1 { v6.h }[6], [%x[Cpanel]]\n"
      "b 13f\n"
      "10:"  // partial result load for beta: partial_1_4
      "mov x20, #0x8\n"
      "tbz x24, #0, 13f\n"
      "ld1 { v3.h }[4], [x25]\n"
      "ld1 { v4.h }[4], [x28]\n"
      "ld1 { v5.h }[4], [x26]\n"
      "ld1 { v6.h }[4], [%x[Cpanel]]\n"
      "b 13f\n"
      "11:"  // partial result load for beta: partial_2_0
      "tbz x24, #1, 12f\n"
      "ldr s3, [x25], #0x4\n"
      "ldr s4, [x28], #0x4\n"
      "mov x20, #0x4\n"
      "ldr s5, [x26], #0x4\n"
      "ldr s6, [%x[Cpanel]], #0x4\n"
      "tbz x24, #0, 13f\n"
      "ld1 { v3.h }[2], [x25]\n"
      "ld1 { v4.h }[2], [x28]\n"
      "ld1 { v5.h }[2], [x26]\n"
      "ld1 { v6.h }[2], [%x[Cpanel]]\n"
      "b 13f\n"
      "12:"  // partial result load for beta: partial_1_0
      "ldr h3, [x25, #0x0]\n"
      "ldr h4, [x28, #0x0]\n"
      "mov x20, #0x0\n"
      "ldr h5, [x26, #0x0]\n"
      "ldr h6, [%x[Cpanel], #0x0]\n"
      "13:"  // partial result load for beta: Done
      "sub %x[Cpanel], %x[Cpanel], x20\n"
      "sub x26, x26, x20\n"
      "shll v2.4s, v1.4h, #0x10\n"
      "sub x28, x28, x20\n"
      "sub x25, x25, x20\n"
      "shll2 v1.4s, v6.8h, #0x10\n"
      "shll v6.4s, v6.4h, #0x10\n"
      "shll v0.4s, v31.4h, #0x10\n"
      "shll2 v31.4s, v5.8h, #0x10\n"
      "shll v5.4s, v5.4h, #0x10\n"
      "shll v19.4s, v18.4h, #0x10\n"
      "shll2 v18.4s, v4.8h, #0x10\n"
      "shll v4.4s, v4.4h, #0x10\n"
      "shll v17.4s, v16.4h, #0x10\n"
      "shll2 v16.4s, v3.8h, #0x10\n"
      "shll v3.4s, v3.4h, #0x10\n"
      "14:"  // result load for beta done
      "fmul v26.4s, v26.4s, v7.4s\n"
      "fmul v25.4s, v25.4s, v7.4s\n"
      "cmp x24, #0xc\n"
      "fmul v20.4s, v20.4s, v7.4s\n"
      "fmul v14.4s, v14.4s, v7.4s\n"
      "fmul v27.4s, v27.4s, v7.4s\n"
      "fmul v28.4s, v28.4s, v7.4s\n"
      "fmla v26.4s, v3.4s, v8.4s\n"
      "fmla v25.4s, v4.4s, v8.4s\n"
      "fmul v29.4s, v29.4s, v7.4s\n"
      "fmul v30.4s, v30.4s, v7.4s\n"
      "fmla v20.4s, v5.4s, v8.4s\n"
      "fmla v14.4s, v6.4s, v8.4s\n"
      "fmul v21.4s, v21.4s, v7.4s\n"
      "fmul v22.4s, v22.4s, v7.4s\n"
      "fmla v27.4s, v16.4s, v8.4s\n"
      "fmul v23.4s, v23.4s, v7.4s\n"
      "fmul v24.4s, v24.4s, v7.4s\n"
      "fmla v28.4s, v17.4s, v8.4s\n"
      "fmla v29.4s, v18.4s, v8.4s\n"
      "fmla v30.4s, v19.4s, v8.4s\n"
      ".inst 0x0ea16b39  // bfcvtn v25.4h, v25.4s\n"
      "fmla v21.4s, v31.4s, v8.4s\n"
      "fmla v22.4s, v0.4s, v8.4s\n"
      ".inst 0x0ea169ce  // bfcvtn v14.4h, v14.4s\n"
      "fmla v23.4s, v1.4s, v8.4s\n"
      "fmla v24.4s, v2.4s, v8.4s\n"
      ".inst 0x0ea16a94  // bfcvtn v20.4h, v20.4s\n"
      ".inst 0x0ea16b5a  // bfcvtn v26.4h, v26.4s\n"
      ".inst 0x4ea16bb9  // bfcvtn2 v25.8h, v29.4s\n"
      ".inst 0x0ea16bdd  // bfcvtn v29.4h, v30.4s\n"
      ".inst 0x4ea16ab4  // bfcvtn2 v20.8h, v21.4s\n"
      ".inst 0x0ea16ad5  // bfcvtn v21.4h, v22.4s\n"
      ".inst 0x4ea16aee  // bfcvtn2 v14.8h, v23.4s\n"
      ".inst 0x0ea16b17  // bfcvtn v23.4h, v24.4s\n"
      ".inst 0x4ea16b7a  // bfcvtn2 v26.8h, v27.4s\n"
      ".inst 0x0ea16b9b  // bfcvtn v27.4h, v28.4s\n"
      "blt 15f\n"
      "str q26, [x25, #0x0]\n"
      "str d27, [x25, #0x10]\n"
      "add x25, x25, #0x18\n"
      "str q25, [x28, #0x0]\n"
      "str d29, [x28, #0x10]\n"
      "add x28, x28, #0x18\n"
      "str q20, [x26, #0x0]\n"
      "str d21, [x26, #0x10]\n"
      "add x26, x26, #0x18\n"
      "str q14, [%x[Cpanel], #0x0]\n"
      "str d23, [%x[Cpanel], #0x10]\n"
      "add %x[Cpanel], %x[Cpanel], #0x18\n"
      "b 22f\n"
      "15:"  // partial output
      "tbz x24, #3, 17f\n"
      "st1 { v26.8h }, [x25], #0x10\n"
      "st1 { v25.8h }, [x28], #0x10\n"
      "st1 { v20.8h }, [x26], #0x10\n"
      "st1 { v14.8h }, [%x[Cpanel]], #0x10\n"
      "tbz x24, #1, 16f\n"
      "str s27, [x25], #0x4\n"
      "str s29, [x28], #0x4\n"
      "str s21, [x26], #0x4\n"
      "str s23, [%x[Cpanel]], #0x4\n"
      "tbz x24, #0, 21f\n"
      "st1 { v27.h }[2], [x25]\n"
      "st1 { v29.h }[2], [x28]\n"
      "st1 { v21.h }[2], [x26]\n"
      "st1 { v23.h }[2], [%x[Cpanel]]\n"
      "b 21f\n"
      "16:"  // partial result store: partial_1_8
      "tbz x24, #0, 21f\n"
      "str h27, [x25, #0x0]\n"
      "str h29, [x28, #0x0]\n"
      "str h21, [x26, #0x0]\n"
      "str h23, [%x[Cpanel], #0x0]\n"
      "b 21f\n"
      "17:"  // partial result store: partial_4_0
      "tbz x24, #2, 19f\n"
      "str d26, [x25], #0x8\n"
      "str d25, [x28], #0x8\n"
      "str d20, [x26], #0x8\n"
      "str d14, [%x[Cpanel]], #0x8\n"
      "tbz x24, #1, 18f\n"
      "st1 { v26.s }[2], [x25], #0x4\n"
      "st1 { v25.s }[2], [x28], #0x4\n"
      "st1 { v20.s }[2], [x26], #0x4\n"
      "st1 { v14.s }[2], [%x[Cpanel]], #0x4\n"
      "tbz x24, #0, 21f\n"
      "st1 { v26.h }[6], [x25]\n"
      "st1 { v25.h }[6], [x28]\n"
      "st1 { v20.h }[6], [x26]\n"
      "st1 { v14.h }[6], [%x[Cpanel]]\n"
      "b 21f\n"
      "18:"  // partial result store: partial_1_4
      "tbz x24, #0, 21f\n"
      "st1 { v26.h }[4], [x25]\n"
      "st1 { v25.h }[4], [x28]\n"
      "st1 { v20.h }[4], [x26]\n"
      "st1 { v14.h }[4], [%x[Cpanel]]\n"
      "b 21f\n"
      "19:"  // partial result store: partial_2_0
      "tbz x24, #1, 20f\n"
      "str s26, [x25], #0x4\n"
      "str s25, [x28], #0x4\n"
      "str s20, [x26], #0x4\n"
      "str s14, [%x[Cpanel]], #0x4\n"
      "tbz x24, #0, 21f\n"
      "st1 { v26.h }[2], [x25]\n"
      "st1 { v25.h }[2], [x28]\n"
      "st1 { v20.h }[2], [x26]\n"
      "st1 { v14.h }[2], [%x[Cpanel]]\n"
      "b 21f\n"
      "20:"  // partial result store: partial_1_0
      "str h26, [x25, #0x0]\n"
      "str h25, [x28, #0x0]\n"
      "str h20, [x26, #0x0]\n"
      "str h14, [%x[Cpanel], #0x0]\n"
      "21:"  // partial result store: Done
      "22:"  // store done
      "subs x24, x24, #0xc\n"
      "bgt 3b\n"
      "subs %x[M], %x[M], #0x4\n"
      "mov %x[Cpanel], x27\n"
      "bgt 1b\n"
      "23:"  // Exit
      : [Apanel] "+&r" (Apanel), [Cpanel] "+&r" (Cpanel), [M] "+&r" (M)
      : [args_ptr] "r" (&ka), [ldc] "r" (ldc * sizeof(bfloat16_t)), [offset_alpha] "I" (offsetof(KernelArgs, alpha)), [offset_beta] "I" (offsetof(KernelArgs, beta)), [offsetof_Bpanel] "I" (offsetof(KernelArgs, Bpanel)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_N] "I" (offsetof(KernelArgs, N))
      : "cc", "memory", "v0", "v1", "v10", "v11", "v12", "v13", "v14", "v16", "v17", "v18", "v19", "v2", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v3", "v30", "v31", "v4", "v5", "v6", "v7", "v8", "v9", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
    );
}

}  // extern "C"

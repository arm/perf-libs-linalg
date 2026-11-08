/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "framework/linalg_util.hpp"
#include <bit>
#include <cstdint>

extern "C" void a64_interleaved_nomerge_bf16fp32bf16_mmla_6x8(
    const bfloat16_t *, const bfloat16_t *, bfloat16_t *, kernel_inttype,
    kernel_inttype, kernel_inttype, kernel_inttype, bfloat16_t, bfloat16_t);

static inline __attribute__((always_inline)) void bgemm_mmla_12x8_tile(
    const bfloat16_t *A, const bfloat16_t *B, bfloat16_t *C,
    kernel_inttype K, kernel_inttype ldc, bfloat16_t alpha, bfloat16_t beta,
    unsigned alpha_mode, unsigned beta_mode)
{
    const float alpha_fp32 = alpha;
    const float beta_fp32 = beta;

    const bfloat16_t *B0_5 = B;
    const bfloat16_t *B6_11 = B + K * 6;
    bfloat16_t *C_even = C;
    bfloat16_t *C_odd = C + ldc;
    const kernel_inttype ldc2 = ldc * (2 * sizeof(bfloat16_t));

    __asm__ __volatile__(
      " ldp    q0, q1, [%[B0_5]]\n"
      " ldr    q2, [%[B0_5], #32]\n"
      " ldp    q3, q4, [%[B6_11]]\n"
      " ldr    q5, [%[B6_11], #32]\n"
      " ldp    q6, q7, [%[A]]\n"

      " movi   v8.16b,  #0\n"
      " movi   v9.16b,  #0\n"
      " movi   v10.16b, #0\n"
      " movi   v11.16b, #0\n"
      " movi   v12.16b, #0\n"
      " movi   v13.16b, #0\n"
      " movi   v14.16b, #0\n"
      " movi   v15.16b, #0\n"
      " movi   v16.16b, #0\n"
      " movi   v17.16b, #0\n"
      " movi   v18.16b, #0\n"
      " movi   v19.16b, #0\n"
      " movi   v20.16b, #0\n"
      " movi   v21.16b, #0\n"
      " movi   v22.16b, #0\n"
      " movi   v23.16b, #0\n"
      " movi   v24.16b, #0\n"
      " movi   v25.16b, #0\n"
      " movi   v26.16b, #0\n"
      " movi   v27.16b, #0\n"
      " movi   v28.16b, #0\n"
      " movi   v29.16b, #0\n"
      " movi   v30.16b, #0\n"
      " movi   v31.16b, #0\n"

      " sub    %[K], %[K], #4\n"
      " cbz    %[K], 4f\n"
      " b      2f\n"

      "1:\n"
      " ldp    q6, q7, [%[A]]\n"
      "2:\n"
      " bfmmla v8.4s,   v0.8h,   v6.8h\n"
      " bfmmla v9.4s,   v0.8h,   v7.8h\n"
      " bfmmla v12.4s,  v1.8h,   v6.8h\n"
      " bfmmla v13.4s,  v1.8h,   v7.8h\n"
      " bfmmla v16.4s,  v2.8h,   v6.8h\n"
      " bfmmla v17.4s,  v2.8h,   v7.8h\n"
      " bfmmla v20.4s,  v3.8h,   v6.8h\n"
      " bfmmla v21.4s,  v3.8h,   v7.8h\n"
      " bfmmla v24.4s,  v4.8h,   v6.8h\n"
      " bfmmla v25.4s,  v4.8h,   v7.8h\n"
      " bfmmla v28.4s,  v5.8h,   v6.8h\n"
      " bfmmla v29.4s,  v5.8h,   v7.8h\n"

      " ldp    q6, q7, [%[A], #32]\n"
      " add    %[A], %[A], #64\n"
      " subs   %[K], %[K], #4\n"
      " bfmmla v10.4s,  v0.8h,   v6.8h\n"
      " bfmmla v11.4s,  v0.8h,   v7.8h\n"
      " bfmmla v14.4s,  v1.8h,   v6.8h\n"
      " bfmmla v15.4s,  v1.8h,   v7.8h\n"
      " ldp    q0, q1, [%[B0_5], #48]\n"
      " bfmmla v18.4s,  v2.8h,   v6.8h\n"
      " bfmmla v19.4s,  v2.8h,   v7.8h\n"
      " ldr    q2, [%[B0_5], #80]\n"
      " bfmmla v22.4s,  v3.8h,   v6.8h\n"
      " bfmmla v23.4s,  v3.8h,   v7.8h\n"
      " bfmmla v26.4s,  v4.8h,   v6.8h\n"
      " bfmmla v27.4s,  v4.8h,   v7.8h\n"
      " ldp    q3, q4, [%[B6_11], #48]\n"
      " bfmmla v30.4s,  v5.8h,   v6.8h\n"
      " bfmmla v31.4s,  v5.8h,   v7.8h\n"
      " ldr    q5, [%[B6_11], #80]\n"
      " add    %[B0_5], %[B0_5], #48\n"
      " add    %[B6_11], %[B6_11], #48\n"
      " b.ne   1b\n"
      " ldp    q6, q7, [%[A]]\n"

      "4:\n"
      " bfmmla v8.4s,   v0.8h,   v6.8h\n"
      " bfmmla v9.4s,   v0.8h,   v7.8h\n"
      " bfmmla v12.4s,  v1.8h,   v6.8h\n"
      " bfmmla v13.4s,  v1.8h,   v7.8h\n"
      " bfmmla v16.4s,  v2.8h,   v6.8h\n"
      " bfmmla v17.4s,  v2.8h,   v7.8h\n"
      " bfmmla v20.4s,  v3.8h,   v6.8h\n"
      " bfmmla v21.4s,  v3.8h,   v7.8h\n"
      " bfmmla v24.4s,  v4.8h,   v6.8h\n"
      " bfmmla v25.4s,  v4.8h,   v7.8h\n"
      " bfmmla v28.4s,  v5.8h,   v6.8h\n"
      " bfmmla v29.4s,  v5.8h,   v7.8h\n"

      " ldp    q6, q7, [%[A], #32]\n"
      " bfmmla v10.4s,  v0.8h,   v6.8h\n"
      " bfmmla v11.4s,  v0.8h,   v7.8h\n"
      " bfmmla v14.4s,  v1.8h,   v6.8h\n"
      " bfmmla v15.4s,  v1.8h,   v7.8h\n"
      " bfmmla v18.4s,  v2.8h,   v6.8h\n"
      " bfmmla v19.4s,  v2.8h,   v7.8h\n"
      " bfmmla v22.4s,  v3.8h,   v6.8h\n"
      " bfmmla v23.4s,  v3.8h,   v7.8h\n"
      " bfmmla v26.4s,  v4.8h,   v6.8h\n"
      " bfmmla v27.4s,  v4.8h,   v7.8h\n"
      " bfmmla v30.4s,  v5.8h,   v6.8h\n"
      " bfmmla v31.4s,  v5.8h,   v7.8h\n"

      " fmov   s6,  %w[alpha]\n"
      " fmov   s7,  %w[beta]\n"
      " cbz    %w[beta_mode], 200f\n"
      " ldr    q4, [%[C_even]]\n"
      "200:\n"
      " uzp1   v0.2d,   v8.2d,   v9.2d\n"
      " uzp2   v1.2d,   v8.2d,   v9.2d\n"
      " cbz    %w[beta_mode], 201f\n"
      " ldr    q9, [%[C_odd]]\n"
      "201:\n"
      " uzp1   v2.2d,   v10.2d,  v11.2d\n"
      " uzp2   v3.2d,   v10.2d,  v11.2d\n"

      " movi   v8.16b,  #0\n"

      " cbz    %w[alpha_mode], 202f\n"
      " fmul   v0.4s,   v0.4s,   v6.s[0]\n"
      " fmul   v1.4s,   v1.4s,   v6.s[0]\n"
      " fmul   v2.4s,   v2.4s,   v6.s[0]\n"
      " fmul   v3.4s,   v3.4s,   v6.s[0]\n"
      "202:\n"

      " cbz    %w[beta_mode], 204f\n"
      " zip2   v5.8h,   v8.8h,   v4.8h\n"
      " zip1   v4.8h,   v8.8h,   v4.8h\n"
      " zip2   v10.8h,  v8.8h,   v9.8h\n"
      " zip1   v9.8h,   v8.8h,   v9.8h\n"
      " tbz    %w[beta_mode], #1, 203f\n"
      " fmla   v0.4s,   v4.4s,   v7.s[0]\n"
      " fmla   v2.4s,   v5.4s,   v7.s[0]\n"
      " fmla   v1.4s,   v9.4s,   v7.s[0]\n"
      " fmla   v3.4s,   v10.4s,  v7.s[0]\n"
      " b      204f\n"
      "203:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v2.4s,   v2.4s,   v5.4s\n"
      " fadd   v1.4s,   v1.4s,   v9.4s\n"
      " fadd   v3.4s,   v3.4s,   v10.4s\n"

      "204:\n"
      " bfcvtn v4.4h,   v0.4s\n"
      " bfcvtn v5.4h,   v1.4s\n"
      " bfcvtn2 v4.8h,   v2.4s\n"
      " bfcvtn2 v5.8h,   v3.4s\n"
      " st1    {v4.8h}, [%[C_even]], %[ldc2]\n"
      " st1    {v5.8h}, [%[C_odd]], %[ldc2]\n"

      " cbz    %w[beta_mode], 210f\n"
      " ldr    q4, [%[C_even]]\n"
      "210:\n"
      " uzp1   v0.2d,   v12.2d,  v13.2d\n"
      " uzp2   v1.2d,   v12.2d,  v13.2d\n"
      " cbz    %w[beta_mode], 211f\n"
      " ldr    q9, [%[C_odd]]\n"
      "211:\n"
      " uzp1   v2.2d,   v14.2d,  v15.2d\n"
      " uzp2   v3.2d,   v14.2d,  v15.2d\n"

      " cbz    %w[alpha_mode], 212f\n"
      " fmul   v0.4s,   v0.4s,   v6.s[0]\n"
      " fmul   v1.4s,   v1.4s,   v6.s[0]\n"
      " fmul   v2.4s,   v2.4s,   v6.s[0]\n"
      " fmul   v3.4s,   v3.4s,   v6.s[0]\n"
      "212:\n"

      " cbz    %w[beta_mode], 214f\n"
      " zip2   v5.8h,   v8.8h,   v4.8h\n"
      " zip1   v4.8h,   v8.8h,   v4.8h\n"
      " zip2   v10.8h,  v8.8h,   v9.8h\n"
      " zip1   v9.8h,   v8.8h,   v9.8h\n"
      " tbz    %w[beta_mode], #1, 213f\n"
      " fmla   v0.4s,   v4.4s,   v7.s[0]\n"
      " fmla   v2.4s,   v5.4s,   v7.s[0]\n"
      " fmla   v1.4s,   v9.4s,   v7.s[0]\n"
      " fmla   v3.4s,   v10.4s,  v7.s[0]\n"
      " b      214f\n"
      "213:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v2.4s,   v2.4s,   v5.4s\n"
      " fadd   v1.4s,   v1.4s,   v9.4s\n"
      " fadd   v3.4s,   v3.4s,   v10.4s\n"

      "214:\n"
      " bfcvtn v4.4h,   v0.4s\n"
      " bfcvtn v5.4h,   v1.4s\n"
      " bfcvtn2 v4.8h,   v2.4s\n"
      " bfcvtn2 v5.8h,   v3.4s\n"
      " st1    {v4.8h}, [%[C_even]], %[ldc2]\n"
      " st1    {v5.8h}, [%[C_odd]], %[ldc2]\n"

      " cbz    %w[beta_mode], 220f\n"
      " ldr    q4, [%[C_even]]\n"
      "220:\n"
      " uzp1   v0.2d,   v16.2d,  v17.2d\n"
      " uzp2   v1.2d,   v16.2d,  v17.2d\n"
      " cbz    %w[beta_mode], 221f\n"
      " ldr    q9, [%[C_odd]]\n"
      "221:\n"
      " uzp1   v2.2d,   v18.2d,  v19.2d\n"
      " uzp2   v3.2d,   v18.2d,  v19.2d\n"

      " cbz    %w[alpha_mode], 222f\n"
      " fmul   v0.4s,   v0.4s,   v6.s[0]\n"
      " fmul   v1.4s,   v1.4s,   v6.s[0]\n"
      " fmul   v2.4s,   v2.4s,   v6.s[0]\n"
      " fmul   v3.4s,   v3.4s,   v6.s[0]\n"
      "222:\n"

      " cbz    %w[beta_mode], 224f\n"
      " zip2   v5.8h,   v8.8h,   v4.8h\n"
      " zip1   v4.8h,   v8.8h,   v4.8h\n"
      " zip2   v10.8h,  v8.8h,   v9.8h\n"
      " zip1   v9.8h,   v8.8h,   v9.8h\n"
      " tbz    %w[beta_mode], #1, 223f\n"
      " fmla   v0.4s,   v4.4s,   v7.s[0]\n"
      " fmla   v2.4s,   v5.4s,   v7.s[0]\n"
      " fmla   v1.4s,   v9.4s,   v7.s[0]\n"
      " fmla   v3.4s,   v10.4s,  v7.s[0]\n"
      " b      224f\n"
      "223:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v2.4s,   v2.4s,   v5.4s\n"
      " fadd   v1.4s,   v1.4s,   v9.4s\n"
      " fadd   v3.4s,   v3.4s,   v10.4s\n"

      "224:\n"
      " bfcvtn v4.4h,   v0.4s\n"
      " bfcvtn v5.4h,   v1.4s\n"
      " bfcvtn2 v4.8h,   v2.4s\n"
      " bfcvtn2 v5.8h,   v3.4s\n"
      " st1    {v4.8h}, [%[C_even]], %[ldc2]\n"
      " st1    {v5.8h}, [%[C_odd]], %[ldc2]\n"

      " cbz    %w[beta_mode], 230f\n"
      " ldr    q4, [%[C_even]]\n"
      "230:\n"
      " uzp1   v0.2d,   v20.2d,  v21.2d\n"
      " uzp2   v1.2d,   v20.2d,  v21.2d\n"
      " cbz    %w[beta_mode], 231f\n"
      " ldr    q9, [%[C_odd]]\n"
      "231:\n"
      " uzp1   v2.2d,   v22.2d,  v23.2d\n"
      " uzp2   v3.2d,   v22.2d,  v23.2d\n"

      " cbz    %w[alpha_mode], 232f\n"
      " fmul   v0.4s,   v0.4s,   v6.s[0]\n"
      " fmul   v1.4s,   v1.4s,   v6.s[0]\n"
      " fmul   v2.4s,   v2.4s,   v6.s[0]\n"
      " fmul   v3.4s,   v3.4s,   v6.s[0]\n"
      "232:\n"

      " cbz    %w[beta_mode], 234f\n"
      " zip2   v5.8h,   v8.8h,   v4.8h\n"
      " zip1   v4.8h,   v8.8h,   v4.8h\n"
      " zip2   v10.8h,  v8.8h,   v9.8h\n"
      " zip1   v9.8h,   v8.8h,   v9.8h\n"
      " tbz    %w[beta_mode], #1, 233f\n"
      " fmla   v0.4s,   v4.4s,   v7.s[0]\n"
      " fmla   v2.4s,   v5.4s,   v7.s[0]\n"
      " fmla   v1.4s,   v9.4s,   v7.s[0]\n"
      " fmla   v3.4s,   v10.4s,  v7.s[0]\n"
      " b      234f\n"
      "233:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v2.4s,   v2.4s,   v5.4s\n"
      " fadd   v1.4s,   v1.4s,   v9.4s\n"
      " fadd   v3.4s,   v3.4s,   v10.4s\n"

      "234:\n"
      " bfcvtn v4.4h,   v0.4s\n"
      " bfcvtn v5.4h,   v1.4s\n"
      " bfcvtn2 v4.8h,   v2.4s\n"
      " bfcvtn2 v5.8h,   v3.4s\n"
      " st1    {v4.8h}, [%[C_even]], %[ldc2]\n"
      " st1    {v5.8h}, [%[C_odd]], %[ldc2]\n"

      " cbz    %w[beta_mode], 240f\n"
      " ldr    q4, [%[C_even]]\n"
      "240:\n"
      " uzp1   v0.2d,   v24.2d,  v25.2d\n"
      " uzp2   v1.2d,   v24.2d,  v25.2d\n"
      " cbz    %w[beta_mode], 241f\n"
      " ldr    q9, [%[C_odd]]\n"
      "241:\n"
      " uzp1   v2.2d,   v26.2d,  v27.2d\n"
      " uzp2   v3.2d,   v26.2d,  v27.2d\n"

      " cbz    %w[alpha_mode], 242f\n"
      " fmul   v0.4s,   v0.4s,   v6.s[0]\n"
      " fmul   v1.4s,   v1.4s,   v6.s[0]\n"
      " fmul   v2.4s,   v2.4s,   v6.s[0]\n"
      " fmul   v3.4s,   v3.4s,   v6.s[0]\n"
      "242:\n"

      " cbz    %w[beta_mode], 244f\n"
      " zip2   v5.8h,   v8.8h,   v4.8h\n"
      " zip1   v4.8h,   v8.8h,   v4.8h\n"
      " zip2   v10.8h,  v8.8h,   v9.8h\n"
      " zip1   v9.8h,   v8.8h,   v9.8h\n"
      " tbz    %w[beta_mode], #1, 243f\n"
      " fmla   v0.4s,   v4.4s,   v7.s[0]\n"
      " fmla   v2.4s,   v5.4s,   v7.s[0]\n"
      " fmla   v1.4s,   v9.4s,   v7.s[0]\n"
      " fmla   v3.4s,   v10.4s,  v7.s[0]\n"
      " b      244f\n"
      "243:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v2.4s,   v2.4s,   v5.4s\n"
      " fadd   v1.4s,   v1.4s,   v9.4s\n"
      " fadd   v3.4s,   v3.4s,   v10.4s\n"

      "244:\n"
      " bfcvtn v4.4h,   v0.4s\n"
      " bfcvtn v5.4h,   v1.4s\n"
      " bfcvtn2 v4.8h,   v2.4s\n"
      " bfcvtn2 v5.8h,   v3.4s\n"
      " st1    {v4.8h}, [%[C_even]], %[ldc2]\n"
      " st1    {v5.8h}, [%[C_odd]], %[ldc2]\n"

      " cbz    %w[beta_mode], 250f\n"
      " ldr    q4, [%[C_even]]\n"
      "250:\n"
      " uzp1   v0.2d,   v28.2d,  v29.2d\n"
      " uzp2   v1.2d,   v28.2d,  v29.2d\n"
      " cbz    %w[beta_mode], 251f\n"
      " ldr    q9, [%[C_odd]]\n"
      "251:\n"
      " uzp1   v2.2d,   v30.2d,  v31.2d\n"
      " uzp2   v3.2d,   v30.2d,  v31.2d\n"

      " cbz    %w[alpha_mode], 252f\n"
      " fmul   v0.4s,   v0.4s,   v6.s[0]\n"
      " fmul   v1.4s,   v1.4s,   v6.s[0]\n"
      " fmul   v2.4s,   v2.4s,   v6.s[0]\n"
      " fmul   v3.4s,   v3.4s,   v6.s[0]\n"
      "252:\n"

      " cbz    %w[beta_mode], 254f\n"
      " zip2   v5.8h,   v8.8h,   v4.8h\n"
      " zip1   v4.8h,   v8.8h,   v4.8h\n"
      " zip2   v10.8h,  v8.8h,   v9.8h\n"
      " zip1   v9.8h,   v8.8h,   v9.8h\n"
      " tbz    %w[beta_mode], #1, 253f\n"
      " fmla   v0.4s,   v4.4s,   v7.s[0]\n"
      " fmla   v2.4s,   v5.4s,   v7.s[0]\n"
      " fmla   v1.4s,   v9.4s,   v7.s[0]\n"
      " fmla   v3.4s,   v10.4s,  v7.s[0]\n"
      " b      254f\n"
      "253:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v2.4s,   v2.4s,   v5.4s\n"
      " fadd   v1.4s,   v1.4s,   v9.4s\n"
      " fadd   v3.4s,   v3.4s,   v10.4s\n"

      "254:\n"
      " bfcvtn v4.4h,   v0.4s\n"
      " bfcvtn v5.4h,   v1.4s\n"
      " bfcvtn2 v4.8h,   v2.4s\n"
      " bfcvtn2 v5.8h,   v3.4s\n"
      " str    q4, [%[C_even]]\n"
      " str    q5, [%[C_odd]]\n"
      : [A] "+&r"(A), [B0_5] "+&r"(B0_5),
        [B6_11] "+&r"(B6_11), [K] "+&r"(K),
        [C_even] "+&r"(C_even), [C_odd] "+&r"(C_odd)
      : [ldc2] "r"(ldc2),
        [alpha] "r"(std::bit_cast<std::uint32_t>(alpha_fp32)),
        [beta] "r"(std::bit_cast<std::uint32_t>(beta_fp32)),
        [alpha_mode] "r"(alpha_mode), [beta_mode] "r"(beta_mode)
      : "memory", "cc",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

// Twelve columns span two complete B6 panels.
// A remains packed in groups of eight rows; both B panels advance by K=4.
extern "C" void a64_interleaved_nomerge_bf16fp32bf16_mmla_12x8(
    const bfloat16_t *A, const bfloat16_t *B, bfloat16_t *C,
    kernel_inttype K, kernel_inttype N, kernel_inttype M,
    kernel_inttype ldc, bfloat16_t alpha, bfloat16_t beta)
{
    if (K <= 0 || N <= 0 || M <= 0) return;

    const unsigned alpha_mode = float(alpha) != 1.0f;
    const unsigned beta_mode = float(beta) == 0.0f ? 0 : (float(beta) == 1.0f ? 1 : 2);

    if (N < 8 || M < 12) {
        a64_interleaved_nomerge_bf16fp32bf16_mmla_6x8(
            A, B, C, K, N, M, ldc, alpha, beta);
        return;
    }

    kernel_inttype col = 0;
    const kernel_inttype full_rows = N - N % 8;

    for (; col + 12 <= M; col += 12) {
        for (kernel_inttype row = 0; row < full_rows; row += 8) {
            bgemm_mmla_12x8_tile(
                A + row * K, B + col * K, C + col * ldc + row,
                K, ldc, alpha, beta, alpha_mode, beta_mode);
        }
    }

    if (col != M) {
        a64_interleaved_nomerge_bf16fp32bf16_mmla_6x8(
            A, B + col * K, C + col * ldc,
            K, full_rows, M - col, ldc, alpha, beta);
    }

    // Handle remaining rows across all columns once.
    if (full_rows != N) {
        a64_interleaved_nomerge_bf16fp32bf16_mmla_6x8(
            A + full_rows * K, B, C + full_rows,
            K, N - full_rows, M, ldc, alpha, beta);
    }
}

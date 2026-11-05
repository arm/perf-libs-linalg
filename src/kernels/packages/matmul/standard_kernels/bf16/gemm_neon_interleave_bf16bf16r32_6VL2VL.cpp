/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "framework/linalg_util.hpp"
#include <algorithm>

static void sbgemm_mmla_6x8_tile(
    const bfloat16_t *A, const bfloat16_t *B, float *C,
    kernel_inttype K, kernel_inttype N, kernel_inttype M,
    kernel_inttype ldc, float alpha, float beta,
    unsigned alpha_mode, unsigned beta_mode)
{
    // Alias absent columns to C0; the epilogue stores C0 last.
    float *C0 = C;
    float *C1 = M > 1 ? C + ldc : C;
    float *C2 = M > 2 ? C + 2 * ldc : C;
    float *C3 = M > 3 ? C + 3 * ldc : C;
    float *C4 = M > 4 ? C + 4 * ldc : C;
    float *C5 = M > 5 ? C + 5 * ldc : C;
    kernel_inttype chunks = K / 4 - 1;

    __asm__ __volatile__(
      " cmp    %[N], #4\n"
      " b.le   500f\n"
      // Start the first panel loads before clearing the accumulators.
      " ldp    q0, q1, [%[B]]\n"
      " ldp    q6, q7, [%[A]]\n"
      " ldr    q2, [%[B], #32]\n"
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

      // Only the main K loop reads the next panel. K=4 uses the final chunk.
      " cbz    %[chunks], 102f\n"
      " b      101f\n"
      "100:\n"
      " ldp    q6, q7, [%[A]]\n"
      "101:\n"
      " bfmmla v20.4s,  v0.8h,   v6.8h\n"
      " bfmmla v21.4s,  v0.8h,   v7.8h\n"
      " bfmmla v24.4s,  v1.8h,   v6.8h\n"
      " bfmmla v25.4s,  v1.8h,   v7.8h\n"
      " bfmmla v28.4s,  v2.8h,   v6.8h\n"
      " bfmmla v29.4s,  v2.8h,   v7.8h\n"

      " ldp    q6, q7, [%[A], #32]\n"
      " bfmmla v22.4s,  v0.8h,   v6.8h\n"
      " bfmmla v23.4s,  v0.8h,   v7.8h\n"
      " bfmmla v26.4s,  v1.8h,   v6.8h\n"
      " bfmmla v27.4s,  v1.8h,   v7.8h\n"
      " ldp    q0, q1, [%[B], #48]\n"
      " bfmmla v30.4s,  v2.8h,   v6.8h\n"
      " bfmmla v31.4s,  v2.8h,   v7.8h\n"
      " ldr    q2, [%[B], #80]\n"

      " add    %[B], %[B], #48\n"
      " add    %[A], %[A], #64\n"
      " subs   %[chunks], %[chunks], #1\n"
      " b.ne   100b\n"
      " ldp    q6, q7, [%[A]]\n"
      "102:\n"
      " bfmmla v20.4s,  v0.8h,   v6.8h\n"
      " bfmmla v21.4s,  v0.8h,   v7.8h\n"
      " bfmmla v24.4s,  v1.8h,   v6.8h\n"
      " bfmmla v25.4s,  v1.8h,   v7.8h\n"
      " bfmmla v28.4s,  v2.8h,   v6.8h\n"
      " bfmmla v29.4s,  v2.8h,   v7.8h\n"

      " ldp    q6, q7, [%[A], #32]\n"
      " bfmmla v22.4s,  v0.8h,   v6.8h\n"
      " bfmmla v23.4s,  v0.8h,   v7.8h\n"
      " bfmmla v26.4s,  v1.8h,   v6.8h\n"
      " bfmmla v27.4s,  v1.8h,   v7.8h\n"
      " bfmmla v30.4s,  v2.8h,   v6.8h\n"
      " bfmmla v31.4s,  v2.8h,   v7.8h\n"

      " b      502f\n"
      // Up to four rows need only the first A pair per K=4 chunk.
      "500:\n"
      " ldp    q0, q1, [%[B]]\n"
      " ldr    q2, [%[B], #32]\n"
      " ldp    q6, q7, [%[A]]\n"
      " movi   v20.16b, #0\n"
      " movi   v21.16b, #0\n"
      " movi   v24.16b, #0\n"
      " movi   v25.16b, #0\n"
      " movi   v28.16b, #0\n"
      " movi   v29.16b, #0\n"
      " cbz    %[chunks], 503f\n"
      "501:\n"
      " bfmmla v20.4s,  v0.8h,   v6.8h\n"
      " bfmmla v21.4s,  v0.8h,   v7.8h\n"
      " bfmmla v24.4s,  v1.8h,   v6.8h\n"
      " bfmmla v25.4s,  v1.8h,   v7.8h\n"
      " bfmmla v28.4s,  v2.8h,   v6.8h\n"
      " bfmmla v29.4s,  v2.8h,   v7.8h\n"
      " ldp    q0, q1, [%[B], #48]\n"
      " ldr    q2, [%[B], #80]\n"
      " add    %[B], %[B], #48\n"
      " add    %[A], %[A], #64\n"
      " ldp    q6, q7, [%[A]]\n"
      " subs   %[chunks], %[chunks], #1\n"
      " b.ne   501b\n"
      "503:\n"
      " bfmmla v20.4s,  v0.8h,   v6.8h\n"
      " bfmmla v21.4s,  v0.8h,   v7.8h\n"
      " bfmmla v24.4s,  v1.8h,   v6.8h\n"
      " bfmmla v25.4s,  v1.8h,   v7.8h\n"
      " bfmmla v28.4s,  v2.8h,   v6.8h\n"
      " bfmmla v29.4s,  v2.8h,   v7.8h\n"
      "502:\n"
      " cmp    %[N], #8\n"
      " b.lt   106f\n"
      // Unpack columns 0-1, then reuse their registers for C.
      " uzp1   v0.2d,   v20.2d,  v21.2d\n"
      " uzp1   v1.2d,   v22.2d,  v23.2d\n"
      " uzp2   v2.2d,   v20.2d,  v21.2d\n"
      " uzp2   v3.2d,   v22.2d,  v23.2d\n"
      " cbz    %w[beta_mode], 103f\n"
      " ldp    q20, q21, [ %[C0] ]\n"
      " ldp    q22, q23, [ %[C1] ]\n"
      "103:\n"

      // Unpack columns 2-3, then reuse their registers for C.
      " uzp1   v4.2d,   v24.2d,  v25.2d\n"
      " uzp1   v5.2d,   v26.2d,  v27.2d\n"
      " uzp2   v6.2d,   v24.2d,  v25.2d\n"
      " uzp2   v7.2d,   v26.2d,  v27.2d\n"
      " cbz    %w[beta_mode], 104f\n"
      " ldp    q24, q25, [ %[C2] ]\n"
      " ldp    q26, q27, [ %[C3] ]\n"
      "104:\n"

      // Unpack columns 4-5, then reuse their registers for C.
      " uzp1   v18.2d,  v28.2d,  v29.2d\n"
      " uzp1   v19.2d,  v30.2d,  v31.2d\n"
      " uzp2   v10.2d,  v28.2d,  v29.2d\n"
      " uzp2   v11.2d,  v30.2d,  v31.2d\n"
      " cbz    %w[beta_mode], 105f\n"
      " ldp    q28, q29, [ %[C4] ]\n"
      " ldp    q30, q31, [ %[C5] ]\n"
      "105:\n"

      " b      110f\n"
      "106:\n"
      " cmp    %[N], #4\n"
      " b.le   400f\n"
      // Unpack columns 0-1, then reuse their registers for C.
      " uzp1   v0.2d,   v20.2d,  v21.2d\n"
      " uzp1   v1.2d,   v22.2d,  v23.2d\n"
      " uzp2   v2.2d,   v20.2d,  v21.2d\n"
      " uzp2   v3.2d,   v22.2d,  v23.2d\n"
      " cbz    %w[beta_mode], 423f\n"
      " ldr    q20, [%[C0]]\n"
      " ldr    q22, [%[C1]]\n"
      " tbz    %[N], #1, 420f\n"
      " ldr    d21, [%[C0], #16]\n"
      " ldr    d23, [%[C1], #16]\n"
      " tbz    %[N], #0, 423f\n"
      " add    x16, %[C0], #24\n"
      " ld1    {v21.s}[2], [x16]\n"
      " add    x16, %[C1], #24\n"
      " ld1    {v23.s}[2], [x16]\n"
      " b      423f\n"
      "420:\n"
      " ldr    s21, [%[C0], #16]\n"
      " ldr    s23, [%[C1], #16]\n"
      "423:\n"
      // Unpack columns 2-3, then reuse their registers for C.
      " uzp1   v4.2d,   v24.2d,  v25.2d\n"
      " uzp1   v5.2d,   v26.2d,  v27.2d\n"
      " uzp2   v6.2d,   v24.2d,  v25.2d\n"
      " uzp2   v7.2d,   v26.2d,  v27.2d\n"
      " cbz    %w[beta_mode], 427f\n"
      " ldr    q24, [%[C2]]\n"
      " ldr    q26, [%[C3]]\n"
      " tbz    %[N], #1, 424f\n"
      " ldr    d25, [%[C2], #16]\n"
      " ldr    d27, [%[C3], #16]\n"
      " tbz    %[N], #0, 427f\n"
      " add    x16, %[C2], #24\n"
      " ld1    {v25.s}[2], [x16]\n"
      " add    x16, %[C3], #24\n"
      " ld1    {v27.s}[2], [x16]\n"
      " b      427f\n"
      "424:\n"
      " ldr    s25, [%[C2], #16]\n"
      " ldr    s27, [%[C3], #16]\n"
      "427:\n"
      // Unpack columns 4-5, then reuse their registers for C.
      " uzp1   v18.2d,  v28.2d,  v29.2d\n"
      " uzp1   v19.2d,  v30.2d,  v31.2d\n"
      " uzp2   v10.2d,  v28.2d,  v29.2d\n"
      " uzp2   v11.2d,  v30.2d,  v31.2d\n"
      " cbz    %w[beta_mode], 431f\n"
      " ldr    q28, [%[C4]]\n"
      " ldr    q30, [%[C5]]\n"
      " tbz    %[N], #1, 428f\n"
      " ldr    d29, [%[C4], #16]\n"
      " ldr    d31, [%[C5], #16]\n"
      " tbz    %[N], #0, 431f\n"
      " add    x16, %[C4], #24\n"
      " ld1    {v29.s}[2], [x16]\n"
      " add    x16, %[C5], #24\n"
      " ld1    {v31.s}[2], [x16]\n"
      " b      431f\n"
      "428:\n"
      " ldr    s29, [%[C4], #16]\n"
      " ldr    s31, [%[C5], #16]\n"
      "431:\n"
      " b      110f\n"
      // Up to four rows: unpack, load and update only the low halves.
      "400:\n"
      " cbz    %w[beta_mode], 410f\n"
      " cmp    %[N], #4\n"
      " b.eq   411f\n"
      " tbz    %[N], #1, 412f\n"
      " uzp1   v0.2d,   v20.2d,  v21.2d\n"
      " uzp2   v2.2d,   v20.2d,  v21.2d\n"
      " ldr    d20, [%[C0]]\n"
      " ldr    d22, [%[C1]]\n"
      " uzp1   v4.2d,   v24.2d,  v25.2d\n"
      " uzp2   v6.2d,   v24.2d,  v25.2d\n"
      " ldr    d24, [%[C2]]\n"
      " ldr    d26, [%[C3]]\n"
      " uzp1   v18.2d,  v28.2d,  v29.2d\n"
      " uzp2   v10.2d,  v28.2d,  v29.2d\n"
      " ldr    d28, [%[C4]]\n"
      " ldr    d30, [%[C5]]\n"
      " tbz    %[N], #0, 413f\n"
      " add    x16, %[C0], #8\n"
      " ld1    {v20.s}[2], [x16]\n"
      " add    x16, %[C1], #8\n"
      " ld1    {v22.s}[2], [x16]\n"
      " add    x16, %[C2], #8\n"
      " ld1    {v24.s}[2], [x16]\n"
      " add    x16, %[C3], #8\n"
      " ld1    {v26.s}[2], [x16]\n"
      " add    x16, %[C4], #8\n"
      " ld1    {v28.s}[2], [x16]\n"
      " add    x16, %[C5], #8\n"
      " ld1    {v30.s}[2], [x16]\n"
      " b      413f\n"
      "411:\n"
      " uzp1   v0.2d,   v20.2d,  v21.2d\n"
      " uzp2   v2.2d,   v20.2d,  v21.2d\n"
      " ldr    q20, [%[C0]]\n"
      " ldr    q22, [%[C1]]\n"
      " uzp1   v4.2d,   v24.2d,  v25.2d\n"
      " uzp2   v6.2d,   v24.2d,  v25.2d\n"
      " ldr    q24, [%[C2]]\n"
      " ldr    q26, [%[C3]]\n"
      " uzp1   v18.2d,  v28.2d,  v29.2d\n"
      " uzp2   v10.2d,  v28.2d,  v29.2d\n"
      " ldr    q28, [%[C4]]\n"
      " ldr    q30, [%[C5]]\n"
      " b      413f\n"
      "412:\n"
      " uzp1   v0.2d,   v20.2d,  v21.2d\n"
      " uzp2   v2.2d,   v20.2d,  v21.2d\n"
      " ldr    s20, [%[C0]]\n"
      " ldr    s22, [%[C1]]\n"
      " uzp1   v4.2d,   v24.2d,  v25.2d\n"
      " uzp2   v6.2d,   v24.2d,  v25.2d\n"
      " ldr    s24, [%[C2]]\n"
      " ldr    s26, [%[C3]]\n"
      " uzp1   v18.2d,  v28.2d,  v29.2d\n"
      " uzp2   v10.2d,  v28.2d,  v29.2d\n"
      " ldr    s28, [%[C4]]\n"
      " ldr    s30, [%[C5]]\n"
      " b      413f\n"
      "410:\n"
      " uzp1   v0.2d,   v20.2d,  v21.2d\n"
      " uzp2   v2.2d,   v20.2d,  v21.2d\n"
      " uzp1   v4.2d,   v24.2d,  v25.2d\n"
      " uzp2   v6.2d,   v24.2d,  v25.2d\n"
      " uzp1   v18.2d,  v28.2d,  v29.2d\n"
      " uzp2   v10.2d,  v28.2d,  v29.2d\n"
      "413:\n"
      " cbz    %w[alpha_mode], 414f\n"
      " fmul   v0.4s,   v0.4s,   %[alpha].s[0]\n"
      " fmul   v2.4s,   v2.4s,   %[alpha].s[0]\n"
      " fmul   v4.4s,   v4.4s,   %[alpha].s[0]\n"
      " fmul   v6.4s,   v6.4s,   %[alpha].s[0]\n"
      " fmul   v18.4s,  v18.4s,  %[alpha].s[0]\n"
      " fmul   v10.4s,  v10.4s,  %[alpha].s[0]\n"
      "414:\n"
      " cbz    %w[beta_mode], 113f\n"
      " tbz    %w[beta_mode], #1, 415f\n"
      " fmla   v0.4s,   v20.4s,  %[beta].s[0]\n"
      " fmla   v2.4s,   v22.4s,  %[beta].s[0]\n"
      " fmla   v4.4s,   v24.4s,  %[beta].s[0]\n"
      " fmla   v6.4s,   v26.4s,  %[beta].s[0]\n"
      " fmla   v18.4s,  v28.4s,  %[beta].s[0]\n"
      " fmla   v10.4s,  v30.4s,  %[beta].s[0]\n"
      " b      113f\n"
      "415:\n"
      " fadd   v0.4s,   v0.4s,   v20.4s\n"
      " fadd   v2.4s,   v2.4s,   v22.4s\n"
      " fadd   v4.4s,   v4.4s,   v24.4s\n"
      " fadd   v6.4s,   v6.4s,   v26.4s\n"
      " fadd   v18.4s,  v18.4s,  v28.4s\n"
      " fadd   v10.4s,  v10.4s,  v30.4s\n"
      " b      113f\n"
      "110:\n"
      " cbz    %w[alpha_mode], 111f\n"
      " fmul   v0.4s,   v0.4s,   %[alpha].s[0]\n"
      " fmul   v1.4s,   v1.4s,   %[alpha].s[0]\n"
      " fmul   v2.4s,   v2.4s,   %[alpha].s[0]\n"
      " fmul   v3.4s,   v3.4s,   %[alpha].s[0]\n"
      " fmul   v4.4s,   v4.4s,   %[alpha].s[0]\n"
      " fmul   v5.4s,   v5.4s,   %[alpha].s[0]\n"
      " fmul   v6.4s,   v6.4s,   %[alpha].s[0]\n"
      " fmul   v7.4s,   v7.4s,   %[alpha].s[0]\n"
      " fmul   v18.4s,  v18.4s,  %[alpha].s[0]\n"
      " fmul   v19.4s,  v19.4s,  %[alpha].s[0]\n"
      " fmul   v10.4s,  v10.4s,  %[alpha].s[0]\n"
      " fmul   v11.4s,  v11.4s,  %[alpha].s[0]\n"
      "111:\n"
      " cbz    %w[beta_mode], 113f\n"
      " tbz    %w[beta_mode], #1, 112f\n"
      " fmla   v0.4s,   v20.4s,  %[beta].s[0]\n"
      " fmla   v1.4s,   v21.4s,  %[beta].s[0]\n"
      " fmla   v2.4s,   v22.4s,  %[beta].s[0]\n"
      " fmla   v3.4s,   v23.4s,  %[beta].s[0]\n"
      " fmla   v4.4s,   v24.4s,  %[beta].s[0]\n"
      " fmla   v5.4s,   v25.4s,  %[beta].s[0]\n"
      " fmla   v6.4s,   v26.4s,  %[beta].s[0]\n"
      " fmla   v7.4s,   v27.4s,  %[beta].s[0]\n"
      " fmla   v18.4s,  v28.4s,  %[beta].s[0]\n"
      " fmla   v19.4s,  v29.4s,  %[beta].s[0]\n"
      " fmla   v10.4s,  v30.4s,  %[beta].s[0]\n"
      " fmla   v11.4s,  v31.4s,  %[beta].s[0]\n"
      " b      113f\n"
      "112:\n"
      " fadd   v0.4s,   v0.4s,   v20.4s\n"
      " fadd   v1.4s,   v1.4s,   v21.4s\n"
      " fadd   v2.4s,   v2.4s,   v22.4s\n"
      " fadd   v3.4s,   v3.4s,   v23.4s\n"
      " fadd   v4.4s,   v4.4s,   v24.4s\n"
      " fadd   v5.4s,   v5.4s,   v25.4s\n"
      " fadd   v6.4s,   v6.4s,   v26.4s\n"
      " fadd   v7.4s,   v7.4s,   v27.4s\n"
      " fadd   v18.4s,  v18.4s,  v28.4s\n"
      " fadd   v19.4s,  v19.4s,  v29.4s\n"
      " fadd   v10.4s,  v10.4s,  v30.4s\n"
      " fadd   v11.4s,  v11.4s,  v31.4s\n"

      // All C loads are complete. Store column zero last for aliased tails.
      "113:\n"
      " cmp    %[N], #8\n"
      " b.lt   114f\n"
      " stp    q10, q11, [ %[C5] ]\n"
      " stp    q18, q19, [ %[C4] ]\n"
      " stp    q6, q7, [ %[C3] ]\n"
      " stp    q4, q5, [ %[C2] ]\n"
      " stp    q2, q3, [ %[C1] ]\n"
      " stp    q0, q1, [ %[C0] ]\n"
      " b      118f\n"
      "114:\n"
      " tbz    %[N], #2, 116f\n"
      " str    q10, [ %[C5] ], #16\n"
      " str    q18, [ %[C4] ], #16\n"
      " str    q6, [ %[C3] ], #16\n"
      " str    q4, [ %[C2] ], #16\n"
      " str    q2, [ %[C1] ], #16\n"
      " str    q0, [ %[C0] ], #16\n"
      " tbz    %[N], #1, 115f\n"
      " str    d11, [ %[C5] ], #8\n"
      " str    d19, [ %[C4] ], #8\n"
      " str    d7, [ %[C3] ], #8\n"
      " str    d5, [ %[C2] ], #8\n"
      " str    d3, [ %[C1] ], #8\n"
      " str    d1, [ %[C0] ], #8\n"
      " tbz    %[N], #0, 118f\n"
      " st1    { v11.s }[2], [ %[C5] ]\n"
      " st1    { v19.s }[2], [ %[C4] ]\n"
      " st1    { v7.s }[2], [ %[C3] ]\n"
      " st1    { v5.s }[2], [ %[C2] ]\n"
      " st1    { v3.s }[2], [ %[C1] ]\n"
      " st1    { v1.s }[2], [ %[C0] ]\n"
      " b      118f\n"
      "115:\n"
      " tbz    %[N], #0, 118f\n"
      " str    s11, [ %[C5] ]\n"
      " str    s19, [ %[C4] ]\n"
      " str    s7, [ %[C3] ]\n"
      " str    s5, [ %[C2] ]\n"
      " str    s3, [ %[C1] ]\n"
      " str    s1, [ %[C0] ]\n"
      " b      118f\n"
      "116:\n"
      " tbz    %[N], #1, 117f\n"
      " str    d10, [ %[C5] ], #8\n"
      " str    d18, [ %[C4] ], #8\n"
      " str    d6, [ %[C3] ], #8\n"
      " str    d4, [ %[C2] ], #8\n"
      " str    d2, [ %[C1] ], #8\n"
      " str    d0, [ %[C0] ], #8\n"
      " tbz    %[N], #0, 118f\n"
      " st1    { v10.s }[2], [ %[C5] ]\n"
      " st1    { v18.s }[2], [ %[C4] ]\n"
      " st1    { v6.s }[2], [ %[C3] ]\n"
      " st1    { v4.s }[2], [ %[C2] ]\n"
      " st1    { v2.s }[2], [ %[C1] ]\n"
      " st1    { v0.s }[2], [ %[C0] ]\n"
      " b      118f\n"
      "117:\n"
      " tbz    %[N], #0, 118f\n"
      " str    s10, [ %[C5] ]\n"
      " str    s18, [ %[C4] ]\n"
      " str    s6, [ %[C3] ]\n"
      " str    s4, [ %[C2] ]\n"
      " str    s2, [ %[C1] ]\n"
      " str    s0, [ %[C0] ]\n"
      " b      118f\n"
      "118:\n"
      : [A] "+&r"(A), [B] "+&r"(B), [chunks] "+&r"(chunks),
        [C0] "+&r"(C0), [C1] "+&r"(C1), [C2] "+&r"(C2),
        [C3] "+&r"(C3), [C4] "+&r"(C4), [C5] "+&r"(C5),
        [alpha] "+&w"(alpha), [beta] "+&w"(beta)
      : [N] "r"(N),
        [alpha_mode] "r"(alpha_mode), [beta_mode] "r"(beta_mode)
      : "cc", "memory", "x16",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v10", "v11", "v18", "v19",
        "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31"
    );
}

static inline __attribute__((always_inline)) void sbgemm_mmla_8x8_tile(
    const bfloat16_t *A, const bfloat16_t *B, float *C,
    kernel_inttype K, kernel_inttype ldc, float alpha, float beta,
    unsigned alpha_mode, unsigned beta_mode)
{
    const bfloat16_t *B0_5 = B;
    const bfloat16_t *B6_11 = B + K * 6;
    float *C_even = C;
    float *C_odd = C + ldc;
    const kernel_inttype ldc2 = ldc * (2 * sizeof(float));

    __asm__ volatile(
      " sub    %[K], %[K], #4\n"

      " ldp    q0, q1, [%[B0_5]]\n"
      " ldp    q6, q7, [%[A]]\n"
      " ldr    q2, [%[B0_5], #32]\n"
      " ldr    q3, [%[B6_11]]\n"

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

      // Process the final K=4 chunk separately, without panel lookahead.
      " cbz    %[K], 4f\n"
      // The first A pair is already loaded; later iterations reload it here.
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
      " ldr    q3, [%[B6_11], #48]\n"
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
      " ldp    q6, q7, [%[A], #32]\n"
      " bfmmla v10.4s,  v0.8h,   v6.8h\n"
      " bfmmla v11.4s,  v0.8h,   v7.8h\n"
      " bfmmla v14.4s,  v1.8h,   v6.8h\n"
      " bfmmla v15.4s,  v1.8h,   v7.8h\n"
      " bfmmla v18.4s,  v2.8h,   v6.8h\n"
      " bfmmla v19.4s,  v2.8h,   v7.8h\n"
      " bfmmla v22.4s,  v3.8h,   v6.8h\n"
      " bfmmla v23.4s,  v3.8h,   v7.8h\n"

      // Scalar modes were classified once by the wrapper.
      " ldr    s6, [%[alpha]]\n"
      " ldr    s7, [%[beta]]\n"

      // Share each output group; skip only the operations its scalars do not need.

      // Columns 0 and 1.
      " cbz    %w[beta_mode], 9f\n"
      " ldp    q4, q5, [%[C_even]]\n"
      "9:\n"
      " uzp1   v0.2d,   v8.2d,   v9.2d\n"
      " uzp1   v1.2d,   v10.2d,  v11.2d\n"
      " uzp2   v2.2d,   v8.2d,   v9.2d\n"
      " cbz    %w[beta_mode], 5f\n"
      " ldp    q8, q9, [%[C_odd]]\n"
      "5:\n"
      " uzp2   v3.2d,   v10.2d,  v11.2d\n"

      " cbz    %w[alpha_mode], 6f\n"
      " fmul   v0.4s,   v0.4s,   v6.s[0]\n"
      " fmul   v1.4s,   v1.4s,   v6.s[0]\n"
      " fmul   v2.4s,   v2.4s,   v6.s[0]\n"
      " fmul   v3.4s,   v3.4s,   v6.s[0]\n"
      "6:\n"

      " cbz    %w[beta_mode], 8f\n"
      " tbz    %w[beta_mode], #1, 7f\n"
      " fmla   v0.4s,   v4.4s,   v7.s[0]\n"
      " fmla   v1.4s,   v5.4s,   v7.s[0]\n"
      " fmla   v2.4s,   v8.4s,   v7.s[0]\n"
      " fmla   v3.4s,   v9.4s,   v7.s[0]\n"
      " b      8f\n"
      "7:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v1.4s,   v1.4s,   v5.4s\n"
      " fadd   v2.4s,   v2.4s,   v8.4s\n"
      " fadd   v3.4s,   v3.4s,   v9.4s\n"
      "8:\n"

      " st1    {v0.4s, v1.4s}, [%[C_even]], %[ldc2]\n"
      " st1    {v2.4s, v3.4s}, [%[C_odd]], %[ldc2]\n"

      // Columns 2 and 3.
      " cbz    %w[beta_mode], 5f\n"
      " ldp    q4, q5, [%[C_even]]\n"
      " ldp    q8, q9, [%[C_odd]]\n"
      "5:\n"
      " uzp1   v0.2d,   v12.2d,  v13.2d\n"
      " uzp1   v1.2d,   v14.2d,  v15.2d\n"
      " uzp2   v2.2d,   v12.2d,  v13.2d\n"
      " uzp2   v3.2d,   v14.2d,  v15.2d\n"

      " cbz    %w[alpha_mode], 6f\n"
      " fmul   v0.4s,   v0.4s,   v6.s[0]\n"
      " fmul   v1.4s,   v1.4s,   v6.s[0]\n"
      " fmul   v2.4s,   v2.4s,   v6.s[0]\n"
      " fmul   v3.4s,   v3.4s,   v6.s[0]\n"
      "6:\n"

      " cbz    %w[beta_mode], 8f\n"
      " tbz    %w[beta_mode], #1, 7f\n"
      " fmla   v0.4s,   v4.4s,   v7.s[0]\n"
      " fmla   v1.4s,   v5.4s,   v7.s[0]\n"
      " fmla   v2.4s,   v8.4s,   v7.s[0]\n"
      " fmla   v3.4s,   v9.4s,   v7.s[0]\n"
      " b      8f\n"
      "7:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v1.4s,   v1.4s,   v5.4s\n"
      " fadd   v2.4s,   v2.4s,   v8.4s\n"
      " fadd   v3.4s,   v3.4s,   v9.4s\n"
      "8:\n"

      " st1    {v0.4s, v1.4s}, [%[C_even]], %[ldc2]\n"
      " st1    {v2.4s, v3.4s}, [%[C_odd]], %[ldc2]\n"

      // Columns 4 and 5.
      " cbz    %w[beta_mode], 5f\n"
      " ldp    q4, q5, [%[C_even]]\n"
      " ldp    q8, q9, [%[C_odd]]\n"
      "5:\n"
      " uzp1   v0.2d,   v16.2d,  v17.2d\n"
      " uzp1   v1.2d,   v18.2d,  v19.2d\n"
      " uzp2   v2.2d,   v16.2d,  v17.2d\n"
      " uzp2   v3.2d,   v18.2d,  v19.2d\n"

      " cbz    %w[alpha_mode], 6f\n"
      " fmul   v0.4s,   v0.4s,   v6.s[0]\n"
      " fmul   v1.4s,   v1.4s,   v6.s[0]\n"
      " fmul   v2.4s,   v2.4s,   v6.s[0]\n"
      " fmul   v3.4s,   v3.4s,   v6.s[0]\n"
      "6:\n"

      " cbz    %w[beta_mode], 8f\n"
      " tbz    %w[beta_mode], #1, 7f\n"
      " fmla   v0.4s,   v4.4s,   v7.s[0]\n"
      " fmla   v1.4s,   v5.4s,   v7.s[0]\n"
      " fmla   v2.4s,   v8.4s,   v7.s[0]\n"
      " fmla   v3.4s,   v9.4s,   v7.s[0]\n"
      " b      8f\n"
      "7:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v1.4s,   v1.4s,   v5.4s\n"
      " fadd   v2.4s,   v2.4s,   v8.4s\n"
      " fadd   v3.4s,   v3.4s,   v9.4s\n"
      "8:\n"

      " st1    {v0.4s, v1.4s}, [%[C_even]], %[ldc2]\n"
      " st1    {v2.4s, v3.4s}, [%[C_odd]], %[ldc2]\n"

      // Columns 6 and 7.
      " cbz    %w[beta_mode], 5f\n"
      " ldp    q4, q5, [%[C_even]]\n"
      " ldp    q8, q9, [%[C_odd]]\n"
      "5:\n"
      " uzp1   v0.2d,   v20.2d,  v21.2d\n"
      " uzp1   v1.2d,   v22.2d,  v23.2d\n"
      " uzp2   v2.2d,   v20.2d,  v21.2d\n"
      " uzp2   v3.2d,   v22.2d,  v23.2d\n"

      " cbz    %w[alpha_mode], 6f\n"
      " fmul   v0.4s,   v0.4s,   v6.s[0]\n"
      " fmul   v1.4s,   v1.4s,   v6.s[0]\n"
      " fmul   v2.4s,   v2.4s,   v6.s[0]\n"
      " fmul   v3.4s,   v3.4s,   v6.s[0]\n"
      "6:\n"

      " cbz    %w[beta_mode], 8f\n"
      " tbz    %w[beta_mode], #1, 7f\n"
      " fmla   v0.4s,   v4.4s,   v7.s[0]\n"
      " fmla   v1.4s,   v5.4s,   v7.s[0]\n"
      " fmla   v2.4s,   v8.4s,   v7.s[0]\n"
      " fmla   v3.4s,   v9.4s,   v7.s[0]\n"
      " b      8f\n"
      "7:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v1.4s,   v1.4s,   v5.4s\n"
      " fadd   v2.4s,   v2.4s,   v8.4s\n"
      " fadd   v3.4s,   v3.4s,   v9.4s\n"
      "8:\n"

      " st1    {v0.4s, v1.4s}, [%[C_even]]\n"
      " st1    {v2.4s, v3.4s}, [%[C_odd]]\n"

      : [A] "+&r"(A), [B0_5] "+&r"(B0_5),
        [B6_11] "+&r"(B6_11),
        [C_even] "+&r"(C_even), [C_odd] "+&r"(C_odd), [K] "+&r"(K)
      : [ldc2] "r"(ldc2), [alpha] "r"(&alpha), [beta] "r"(&beta),
        [alpha_mode] "r"(alpha_mode), [beta_mode] "r"(beta_mode)
      : "memory", "cc",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
}

static inline __attribute__((always_inline)) void sbgemm_mmla_12x8_tile(
    const bfloat16_t *A, const bfloat16_t *B, float *C,
    kernel_inttype K, kernel_inttype ldc, float alpha, float beta,
    unsigned alpha_mode, unsigned beta_mode)
{
    const bfloat16_t *B0_5 = B;
    const bfloat16_t *B6_11 = B + K * 6;
    float *C_even = C;
    float *C_odd = C + ldc;
    const kernel_inttype ldc2 = ldc * (2 * sizeof(float));

    __asm__ volatile(
      " sub    %[K], %[K], #4\n"

      " ldp    q0, q1, [%[B0_5]]\n"
      " ldp    q6, q7, [%[A]]\n"
      " ldr    q2, [%[B0_5], #32]\n"
      " ldp    q3, q4, [%[B6_11]]\n"
      " ldr    q5, [%[B6_11], #32]\n"

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

      // Process the final K=4 chunk separately, without panel lookahead.
      " cbz    %[K], 4f\n"
      // The first A pair is already loaded; later iterations reload it here.
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

      // Scalar modes were classified once by the wrapper.
      " ldr    s6, [%[alpha]]\n"
      " ldr    s7, [%[beta]]\n"

      // Share each output group; skip only the operations its scalars do not need.

      // Columns 0 and 1.
      " cbz    %w[beta_mode], 9f\n"
      " ldp    q4, q5, [%[C_even]]\n"
      "9:\n"
      " uzp1   v0.2d,   v8.2d,   v9.2d\n"
      " uzp1   v1.2d,   v10.2d,  v11.2d\n"
      " uzp2   v2.2d,   v8.2d,   v9.2d\n"
      " cbz    %w[beta_mode], 5f\n"
      " ldp    q8, q9, [%[C_odd]]\n"
      "5:\n"
      " uzp2   v3.2d,   v10.2d,  v11.2d\n"

      " cbz    %w[alpha_mode], 6f\n"
      " fmul   v0.4s,   v0.4s,   v6.s[0]\n"
      " fmul   v1.4s,   v1.4s,   v6.s[0]\n"
      " fmul   v2.4s,   v2.4s,   v6.s[0]\n"
      " fmul   v3.4s,   v3.4s,   v6.s[0]\n"
      "6:\n"

      " cbz    %w[beta_mode], 8f\n"
      " tbz    %w[beta_mode], #1, 7f\n"
      " fmla   v0.4s,   v4.4s,   v7.s[0]\n"
      " fmla   v1.4s,   v5.4s,   v7.s[0]\n"
      " fmla   v2.4s,   v8.4s,   v7.s[0]\n"
      " fmla   v3.4s,   v9.4s,   v7.s[0]\n"
      " b      8f\n"
      "7:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v1.4s,   v1.4s,   v5.4s\n"
      " fadd   v2.4s,   v2.4s,   v8.4s\n"
      " fadd   v3.4s,   v3.4s,   v9.4s\n"
      "8:\n"

      " st1    {v0.4s, v1.4s}, [%[C_even]], %[ldc2]\n"
      " st1    {v2.4s, v3.4s}, [%[C_odd]], %[ldc2]\n"

      // Columns 2 and 3.
      " cbz    %w[beta_mode], 5f\n"
      " ldp    q4, q5, [%[C_even]]\n"
      " ldp    q8, q9, [%[C_odd]]\n"
      "5:\n"
      " uzp1   v0.2d,   v12.2d,  v13.2d\n"
      " uzp1   v1.2d,   v14.2d,  v15.2d\n"
      " uzp2   v2.2d,   v12.2d,  v13.2d\n"
      " uzp2   v3.2d,   v14.2d,  v15.2d\n"

      " cbz    %w[alpha_mode], 6f\n"
      " fmul   v0.4s,   v0.4s,   v6.s[0]\n"
      " fmul   v1.4s,   v1.4s,   v6.s[0]\n"
      " fmul   v2.4s,   v2.4s,   v6.s[0]\n"
      " fmul   v3.4s,   v3.4s,   v6.s[0]\n"
      "6:\n"

      " cbz    %w[beta_mode], 8f\n"
      " tbz    %w[beta_mode], #1, 7f\n"
      " fmla   v0.4s,   v4.4s,   v7.s[0]\n"
      " fmla   v1.4s,   v5.4s,   v7.s[0]\n"
      " fmla   v2.4s,   v8.4s,   v7.s[0]\n"
      " fmla   v3.4s,   v9.4s,   v7.s[0]\n"
      " b      8f\n"
      "7:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v1.4s,   v1.4s,   v5.4s\n"
      " fadd   v2.4s,   v2.4s,   v8.4s\n"
      " fadd   v3.4s,   v3.4s,   v9.4s\n"
      "8:\n"

      " st1    {v0.4s, v1.4s}, [%[C_even]], %[ldc2]\n"
      " st1    {v2.4s, v3.4s}, [%[C_odd]], %[ldc2]\n"

      // Columns 4 and 5.
      " cbz    %w[beta_mode], 5f\n"
      " ldp    q4, q5, [%[C_even]]\n"
      " ldp    q8, q9, [%[C_odd]]\n"
      "5:\n"
      " uzp1   v0.2d,   v16.2d,  v17.2d\n"
      " uzp1   v1.2d,   v18.2d,  v19.2d\n"
      " uzp2   v2.2d,   v16.2d,  v17.2d\n"
      " uzp2   v3.2d,   v18.2d,  v19.2d\n"

      " cbz    %w[alpha_mode], 6f\n"
      " fmul   v0.4s,   v0.4s,   v6.s[0]\n"
      " fmul   v1.4s,   v1.4s,   v6.s[0]\n"
      " fmul   v2.4s,   v2.4s,   v6.s[0]\n"
      " fmul   v3.4s,   v3.4s,   v6.s[0]\n"
      "6:\n"

      " cbz    %w[beta_mode], 8f\n"
      " tbz    %w[beta_mode], #1, 7f\n"
      " fmla   v0.4s,   v4.4s,   v7.s[0]\n"
      " fmla   v1.4s,   v5.4s,   v7.s[0]\n"
      " fmla   v2.4s,   v8.4s,   v7.s[0]\n"
      " fmla   v3.4s,   v9.4s,   v7.s[0]\n"
      " b      8f\n"
      "7:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v1.4s,   v1.4s,   v5.4s\n"
      " fadd   v2.4s,   v2.4s,   v8.4s\n"
      " fadd   v3.4s,   v3.4s,   v9.4s\n"
      "8:\n"

      " st1    {v0.4s, v1.4s}, [%[C_even]], %[ldc2]\n"
      " st1    {v2.4s, v3.4s}, [%[C_odd]], %[ldc2]\n"

      // Columns 6 and 7.
      " cbz    %w[beta_mode], 5f\n"
      " ldp    q4, q5, [%[C_even]]\n"
      " ldp    q8, q9, [%[C_odd]]\n"
      "5:\n"
      " uzp1   v0.2d,   v20.2d,  v21.2d\n"
      " uzp1   v1.2d,   v22.2d,  v23.2d\n"
      " uzp2   v2.2d,   v20.2d,  v21.2d\n"
      " uzp2   v3.2d,   v22.2d,  v23.2d\n"

      " cbz    %w[alpha_mode], 6f\n"
      " fmul   v0.4s,   v0.4s,   v6.s[0]\n"
      " fmul   v1.4s,   v1.4s,   v6.s[0]\n"
      " fmul   v2.4s,   v2.4s,   v6.s[0]\n"
      " fmul   v3.4s,   v3.4s,   v6.s[0]\n"
      "6:\n"

      " cbz    %w[beta_mode], 8f\n"
      " tbz    %w[beta_mode], #1, 7f\n"
      " fmla   v0.4s,   v4.4s,   v7.s[0]\n"
      " fmla   v1.4s,   v5.4s,   v7.s[0]\n"
      " fmla   v2.4s,   v8.4s,   v7.s[0]\n"
      " fmla   v3.4s,   v9.4s,   v7.s[0]\n"
      " b      8f\n"
      "7:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v1.4s,   v1.4s,   v5.4s\n"
      " fadd   v2.4s,   v2.4s,   v8.4s\n"
      " fadd   v3.4s,   v3.4s,   v9.4s\n"
      "8:\n"

      " st1    {v0.4s, v1.4s}, [%[C_even]], %[ldc2]\n"
      " st1    {v2.4s, v3.4s}, [%[C_odd]], %[ldc2]\n"

      // Columns 8 and 9.
      " cbz    %w[beta_mode], 5f\n"
      " ldp    q4, q5, [%[C_even]]\n"
      " ldp    q8, q9, [%[C_odd]]\n"
      "5:\n"
      " uzp1   v0.2d,   v24.2d,  v25.2d\n"
      " uzp1   v1.2d,   v26.2d,  v27.2d\n"
      " uzp2   v2.2d,   v24.2d,  v25.2d\n"
      " uzp2   v3.2d,   v26.2d,  v27.2d\n"

      " cbz    %w[alpha_mode], 6f\n"
      " fmul   v0.4s,   v0.4s,   v6.s[0]\n"
      " fmul   v1.4s,   v1.4s,   v6.s[0]\n"
      " fmul   v2.4s,   v2.4s,   v6.s[0]\n"
      " fmul   v3.4s,   v3.4s,   v6.s[0]\n"
      "6:\n"

      " cbz    %w[beta_mode], 8f\n"
      " tbz    %w[beta_mode], #1, 7f\n"
      " fmla   v0.4s,   v4.4s,   v7.s[0]\n"
      " fmla   v1.4s,   v5.4s,   v7.s[0]\n"
      " fmla   v2.4s,   v8.4s,   v7.s[0]\n"
      " fmla   v3.4s,   v9.4s,   v7.s[0]\n"
      " b      8f\n"
      "7:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v1.4s,   v1.4s,   v5.4s\n"
      " fadd   v2.4s,   v2.4s,   v8.4s\n"
      " fadd   v3.4s,   v3.4s,   v9.4s\n"
      "8:\n"

      " st1    {v0.4s, v1.4s}, [%[C_even]], %[ldc2]\n"
      " st1    {v2.4s, v3.4s}, [%[C_odd]], %[ldc2]\n"

      // Columns 10 and 11.
      " cbz    %w[beta_mode], 5f\n"
      " ldp    q4, q5, [%[C_even]]\n"
      " ldp    q8, q9, [%[C_odd]]\n"
      "5:\n"
      " uzp1   v0.2d,   v28.2d,  v29.2d\n"
      " uzp1   v1.2d,   v30.2d,  v31.2d\n"
      " uzp2   v2.2d,   v28.2d,  v29.2d\n"
      " uzp2   v3.2d,   v30.2d,  v31.2d\n"

      " cbz    %w[alpha_mode], 6f\n"
      " fmul   v0.4s,   v0.4s,   v6.s[0]\n"
      " fmul   v1.4s,   v1.4s,   v6.s[0]\n"
      " fmul   v2.4s,   v2.4s,   v6.s[0]\n"
      " fmul   v3.4s,   v3.4s,   v6.s[0]\n"
      "6:\n"

      " cbz    %w[beta_mode], 8f\n"
      " tbz    %w[beta_mode], #1, 7f\n"
      " fmla   v0.4s,   v4.4s,   v7.s[0]\n"
      " fmla   v1.4s,   v5.4s,   v7.s[0]\n"
      " fmla   v2.4s,   v8.4s,   v7.s[0]\n"
      " fmla   v3.4s,   v9.4s,   v7.s[0]\n"
      " b      8f\n"
      "7:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v1.4s,   v1.4s,   v5.4s\n"
      " fadd   v2.4s,   v2.4s,   v8.4s\n"
      " fadd   v3.4s,   v3.4s,   v9.4s\n"
      "8:\n"

      " st1    {v0.4s, v1.4s}, [%[C_even]]\n"
      " st1    {v2.4s, v3.4s}, [%[C_odd]]\n"
      : [A] "+&r"(A), [B0_5] "+&r"(B0_5),
        [B6_11] "+&r"(B6_11),
        [C_even] "+&r"(C_even), [C_odd] "+&r"(C_odd), [K] "+&r"(K)
      : [ldc2] "r"(ldc2), [alpha] "r"(&alpha), [beta] "r"(&beta),
        [alpha_mode] "r"(alpha_mode), [beta_mode] "r"(beta_mode)
      : "memory", "cc",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

// Compute the output in blocks of at most six columns and eight rows.
static inline void a64_interleaved_nomerge_bf16fp32_mmla_6x8(
    const bfloat16_t *A, const bfloat16_t *B, float *C,
    kernel_inttype K, kernel_inttype N, kernel_inttype M,
    kernel_inttype ldc, float alpha, float beta)
{
    const unsigned alpha_mode = alpha != 1.0f;
    const unsigned beta_mode = beta == 0.0f ? 0 : (beta == 1.0f ? 1 : 2);

    for (kernel_inttype col = 0; col < M; col += 6) {
        for (kernel_inttype row = 0; row < N; row += 8) {
            sbgemm_mmla_6x8_tile(
                A + row * K, B + col * K, C + col * ldc + row,
                K, std::min<kernel_inttype>(8, N - row),
                std::min<kernel_inttype>(6, M - col), ldc,
                alpha, beta, alpha_mode, beta_mode);
        }
    }
}

// Handle full eight-row blocks across exactly eight columns.
// Eight columns span one complete B6 panel and the first pair of the next.
// A remains packed in groups of eight rows; both B panels advance by K=4.
void a64_interleaved_nomerge_bf16fp32_mmla_8x8(
    const bfloat16_t *A, const bfloat16_t *B, float *C,
    kernel_inttype K, kernel_inttype N, kernel_inttype ldc,
    float alpha, float beta)
{
    const unsigned alpha_mode = alpha != 1.0f;
    const unsigned beta_mode = beta == 0.0f ? 0 : (beta == 1.0f ? 1 : 2);

    for (kernel_inttype row = 0; row < N; row += 8) {
        sbgemm_mmla_8x8_tile(
            A + row * K, B, C + row,
            K, ldc, alpha, beta, alpha_mode, beta_mode);
    }
}

// Twelve columns span two complete B6 panels.
// A remains packed in groups of eight rows; both B panels advance by K=4.
extern "C" void a64_interleaved_nomerge_bf16fp32_mmla_12x8(
    const bfloat16_t *A, const bfloat16_t *B, float *C,
    kernel_inttype K, kernel_inttype N, kernel_inttype M,
    kernel_inttype ldc, float alpha, float beta)
{
    if (K <= 0 || N <= 0 || M <= 0) return;

    if (N < 8) {
        a64_interleaved_nomerge_bf16fp32_mmla_6x8(
            A, B, C, K, N, M, ldc, alpha, beta);
        return;
    }

    const unsigned alpha_mode = alpha != 1.0f;
    const unsigned beta_mode = beta == 0.0f ? 0 : (beta == 1.0f ? 1 : 2);

    kernel_inttype col = 0;
    const kernel_inttype full_rows = N - N % 8;

    for (; col + 12 <= M; col += 12) {
        for (kernel_inttype row = 0; row < full_rows; row += 8) {
            sbgemm_mmla_12x8_tile(
                A + row * K, B + col * K, C + col * ldc + row,
                K, ldc, alpha, beta, alpha_mode, beta_mode);
        }
    }

    if (col != M) {
        if (M - col == 8) {
            a64_interleaved_nomerge_bf16fp32_mmla_8x8(
                A, B + col * K, C + col * ldc,
                K, full_rows, ldc, alpha, beta);
        } else {
            a64_interleaved_nomerge_bf16fp32_mmla_6x8(
                A, B + col * K, C + col * ldc,
                K, full_rows, M - col, ldc, alpha, beta);
        }
    }

    // Handle remaining rows across all columns once.
    if (full_rows != N) {
        a64_interleaved_nomerge_bf16fp32_mmla_6x8(
            A + full_rows * K, B, C + full_rows,
            K, N - full_rows, M, ldc, alpha, beta);
    }
}

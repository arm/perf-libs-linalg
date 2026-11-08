/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "framework/linalg_util.hpp"
#include <algorithm>
#include <bit>
#include <cstdint>

static void bgemm_mmla_6x8_tile(
    const bfloat16_t *A, const bfloat16_t *B, bfloat16_t *C,
    kernel_inttype K, kernel_inttype N, kernel_inttype M,
    kernel_inttype ldc, bfloat16_t alpha, bfloat16_t beta,
    unsigned alpha_mode, unsigned beta_mode)
{
    float alpha_fp32 = alpha;
    const std::uint32_t beta_bits = std::bit_cast<std::uint16_t>(beta);

    // Alias absent columns to C0; the epilogue stores C0 last.
    bfloat16_t *C0 = C;
    bfloat16_t *C1 = M > 1 ? C + ldc : C;
    bfloat16_t *C2 = M > 2 ? C + 2 * ldc : C;
    bfloat16_t *C3 = M > 3 ? C + 3 * ldc : C;
    bfloat16_t *C4 = M > 4 ? C + 4 * ldc : C;
    bfloat16_t *C5 = M > 5 ? C + 5 * ldc : C;
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

      " uzp1   v0.2d,   v20.2d,  v21.2d\n"
      " uzp2   v2.2d,   v20.2d,  v21.2d\n"
      " uzp1   v4.2d,   v24.2d,  v25.2d\n"
      " uzp2   v6.2d,   v24.2d,  v25.2d\n"
      " uzp1   v18.2d,  v28.2d,  v29.2d\n"
      " uzp2   v10.2d,  v28.2d,  v29.2d\n"

      " cmp    %[N], #4\n"
      " b.le   301f\n"
      " uzp1   v1.2d,   v22.2d,  v23.2d\n"
      " uzp2   v3.2d,   v22.2d,  v23.2d\n"
      " uzp1   v5.2d,   v26.2d,  v27.2d\n"
      " uzp2   v7.2d,   v26.2d,  v27.2d\n"
      " uzp1   v19.2d,  v30.2d,  v31.2d\n"
      " uzp2   v11.2d,  v30.2d,  v31.2d\n"

      // Load only valid BF16 rows; all C reads precede aliased stores.

      "301:\n"
      " cbz    %w[beta_mode], 320f\n"
      " cmp    %[N], #8\n"
      " b.ne   302f\n"
      " ldr    q20, [%[C0]]\n"
      " ldr    q22, [%[C1]]\n"
      " ldr    q24, [%[C2]]\n"
      " ldr    q26, [%[C3]]\n"
      " ldr    q28, [%[C4]]\n"
      " ldr    q30, [%[C5]]\n"
      " b      310f\n"

      "302:\n"
      " tbz    %[N], #2, 305f\n"
      " ldr    d20, [%[C0]]\n"
      " ldr    d22, [%[C1]]\n"
      " ldr    d24, [%[C2]]\n"
      " ldr    d26, [%[C3]]\n"
      " ldr    d28, [%[C4]]\n"
      " ldr    d30, [%[C5]]\n"
      " tbz    %[N], #1, 304f\n"
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
      " tbz    %[N], #0, 310f\n"
      " add    x16, %[C0], #12\n"
      " ld1    {v20.h}[6], [x16]\n"
      " add    x16, %[C1], #12\n"
      " ld1    {v22.h}[6], [x16]\n"
      " add    x16, %[C2], #12\n"
      " ld1    {v24.h}[6], [x16]\n"
      " add    x16, %[C3], #12\n"
      " ld1    {v26.h}[6], [x16]\n"
      " add    x16, %[C4], #12\n"
      " ld1    {v28.h}[6], [x16]\n"
      " add    x16, %[C5], #12\n"
      " ld1    {v30.h}[6], [x16]\n"
      " b      310f\n"

      "304:\n"
      " tbz    %[N], #0, 310f\n"
      " add    x16, %[C0], #8\n"
      " ld1    {v20.h}[4], [x16]\n"
      " add    x16, %[C1], #8\n"
      " ld1    {v22.h}[4], [x16]\n"
      " add    x16, %[C2], #8\n"
      " ld1    {v24.h}[4], [x16]\n"
      " add    x16, %[C3], #8\n"
      " ld1    {v26.h}[4], [x16]\n"
      " add    x16, %[C4], #8\n"
      " ld1    {v28.h}[4], [x16]\n"
      " add    x16, %[C5], #8\n"
      " ld1    {v30.h}[4], [x16]\n"
      " b      310f\n"

      "305:\n"
      " tbz    %[N], #1, 306f\n"
      " ldr    s20, [%[C0]]\n"
      " ldr    s22, [%[C1]]\n"
      " ldr    s24, [%[C2]]\n"
      " ldr    s26, [%[C3]]\n"
      " ldr    s28, [%[C4]]\n"
      " ldr    s30, [%[C5]]\n"
      " tbz    %[N], #0, 310f\n"
      " add    x16, %[C0], #4\n"
      " ld1    {v20.h}[2], [x16]\n"
      " add    x16, %[C1], #4\n"
      " ld1    {v22.h}[2], [x16]\n"
      " add    x16, %[C2], #4\n"
      " ld1    {v24.h}[2], [x16]\n"
      " add    x16, %[C3], #4\n"
      " ld1    {v26.h}[2], [x16]\n"
      " add    x16, %[C4], #4\n"
      " ld1    {v28.h}[2], [x16]\n"
      " add    x16, %[C5], #4\n"
      " ld1    {v30.h}[2], [x16]\n"
      " b      310f\n"

      "306:\n"
      " ldr    h20, [%[C0]]\n"
      " ldr    h22, [%[C1]]\n"
      " ldr    h24, [%[C2]]\n"
      " ldr    h26, [%[C3]]\n"
      " ldr    h28, [%[C4]]\n"
      " ldr    h30, [%[C5]]\n"
      // Interleave BF16 columns for bottom/top widening multiply-add.

      "310:\n"
      " dup    v16.8h,  %w[beta_bits]\n"
      " cmp    %[N], #4\n"
      " b.le   311f\n"
      " zip2   v21.8h,  v20.8h,  v22.8h\n"
      " zip2   v25.8h,  v24.8h,  v26.8h\n"
      " zip2   v29.8h,  v28.8h,  v30.8h\n"

      "311:\n"
      " zip1   v20.8h,  v20.8h,  v22.8h\n"
      " zip1   v24.8h,  v24.8h,  v26.8h\n"
      " zip1   v28.8h,  v28.8h,  v30.8h\n"
      // At most four rows need only the low result vectors.

      "320:\n"
      " cmp    %[N], #4\n"
      " b.le   340f\n"
      " cbz    %w[alpha_mode], 330f\n"
      " fmul   v0.4s,   v0.4s,   %[alpha].s[0]\n"
      " fmul   v1.4s,   v1.4s,   %[alpha].s[0]\n"
      " fmul   v2.4s,   v2.4s,   %[alpha].s[0]\n"
      " fmul   v3.4s,   v3.4s,   %[alpha].s[0]\n"
      " fmul   v4.4s,   v4.4s,   %[alpha].s[0]\n"
      " fmul   v5.4s,   v5.4s,   %[alpha].s[0]\n"
      " fmul   v6.4s,   v6.4s,   %[alpha].s[0]\n"
      " fmul   v7.4s,   v7.4s,   %[alpha].s[0]\n"
      " fmul   v18.4s,   v18.4s,   %[alpha].s[0]\n"
      " fmul   v19.4s,   v19.4s,   %[alpha].s[0]\n"
      " fmul   v10.4s,  v10.4s,  %[alpha].s[0]\n"
      " fmul   v11.4s,  v11.4s,  %[alpha].s[0]\n"

      "330:\n"
      " cbz    %w[beta_mode], 332f\n"
      " bfmlalb v0.4s,   v20.8h,  v16.8h\n"
      " bfmlalt v2.4s,   v20.8h,  v16.8h\n"
      " bfmlalb v1.4s,   v21.8h,  v16.8h\n"
      " bfmlalt v3.4s,   v21.8h,  v16.8h\n"
      " bfmlalb v4.4s,   v24.8h,  v16.8h\n"
      " bfmlalt v6.4s,   v24.8h,  v16.8h\n"
      " bfmlalb v5.4s,   v25.8h,  v16.8h\n"
      " bfmlalt v7.4s,   v25.8h,  v16.8h\n"
      " bfmlalb v18.4s,  v28.8h,  v16.8h\n"
      " bfmlalt v10.4s,  v28.8h,  v16.8h\n"
      " bfmlalb v19.4s,  v29.8h,  v16.8h\n"
      " bfmlalt v11.4s,  v29.8h,  v16.8h\n"

      "332:\n"
      " bfcvtn v0.4h,   v0.4s\n"
      " bfcvtn2 v0.8h,   v1.4s\n"

      " bfcvtn v2.4h,   v2.4s\n"
      " bfcvtn2 v2.8h,   v3.4s\n"

      " bfcvtn v4.4h,   v4.4s\n"
      " bfcvtn2 v4.8h,   v5.4s\n"

      " bfcvtn v6.4h,   v6.4s\n"
      " bfcvtn2 v6.8h,   v7.4s\n"

      " bfcvtn v18.4h,  v18.4s\n"
      " bfcvtn2 v18.8h,  v19.4s\n"

      " bfcvtn v10.4h,  v10.4s\n"
      " bfcvtn2 v10.8h,  v11.4s\n"
      " b      350f\n"

      "340:\n"
      " cbz    %w[alpha_mode], 341f\n"
      " fmul   v0.4s,   v0.4s,   %[alpha].s[0]\n"
      " fmul   v2.4s,   v2.4s,   %[alpha].s[0]\n"
      " fmul   v4.4s,   v4.4s,   %[alpha].s[0]\n"
      " fmul   v6.4s,   v6.4s,   %[alpha].s[0]\n"
      " fmul   v18.4s,   v18.4s,   %[alpha].s[0]\n"
      " fmul   v10.4s,  v10.4s,  %[alpha].s[0]\n"

      "341:\n"
      " cbz    %w[beta_mode], 343f\n"
      " bfmlalb v0.4s,   v20.8h,  v16.8h\n"
      " bfmlalt v2.4s,   v20.8h,  v16.8h\n"
      " bfmlalb v4.4s,   v24.8h,  v16.8h\n"
      " bfmlalt v6.4s,   v24.8h,  v16.8h\n"
      " bfmlalb v18.4s,  v28.8h,  v16.8h\n"
      " bfmlalt v10.4s,  v28.8h,  v16.8h\n"

      "343:\n"
      " bfcvtn v0.4h,   v0.4s\n"
      " bfcvtn v2.4h,   v2.4s\n"
      " bfcvtn v4.4h,   v4.4s\n"
      " bfcvtn v6.4h,   v6.4s\n"
      " bfcvtn v18.4h,  v18.4s\n"
      " bfcvtn v10.4h,  v10.4s\n"
      " b      351f\n"

      // Round once to BF16; store aliased column zero last.

      "350:\n"
      " cmp    %[N], #8\n"
      " b.ne   351f\n"
      " str    q10, [%[C5]]\n"
      " str    q18, [%[C4]]\n"
      " str    q6, [%[C3]]\n"
      " str    q4, [%[C2]]\n"
      " str    q2, [%[C1]]\n"
      " str    q0, [%[C0]]\n"
      " b      360f\n"

      "351:\n"
      " tbz    %[N], #2, 354f\n"
      " str    d10, [%[C5]], #8\n"
      " str    d18, [%[C4]], #8\n"
      " str    d6, [%[C3]], #8\n"
      " str    d4, [%[C2]], #8\n"
      " str    d2, [%[C1]], #8\n"
      " str    d0, [%[C0]], #8\n"
      " tbz    %[N], #1, 353f\n"
      " st1    {v10.s}[2], [%[C5]]\n"
      " st1    {v18.s}[2], [%[C4]]\n"
      " st1    {v6.s}[2], [%[C3]]\n"
      " st1    {v4.s}[2], [%[C2]]\n"
      " st1    {v2.s}[2], [%[C1]]\n"
      " st1    {v0.s}[2], [%[C0]]\n"
      " tbz    %[N], #0, 360f\n"
      " add    x16, %[C5], #4\n"
      " st1    {v10.h}[6], [x16]\n"
      " add    x16, %[C4], #4\n"
      " st1    {v18.h}[6], [x16]\n"
      " add    x16, %[C3], #4\n"
      " st1    {v6.h}[6], [x16]\n"
      " add    x16, %[C2], #4\n"
      " st1    {v4.h}[6], [x16]\n"
      " add    x16, %[C1], #4\n"
      " st1    {v2.h}[6], [x16]\n"
      " add    x16, %[C0], #4\n"
      " st1    {v0.h}[6], [x16]\n"
      " b      360f\n"

      "353:\n"
      " tbz    %[N], #0, 360f\n"
      " st1    {v10.h}[4], [%[C5]]\n"
      " st1    {v18.h}[4], [%[C4]]\n"
      " st1    {v6.h}[4], [%[C3]]\n"
      " st1    {v4.h}[4], [%[C2]]\n"
      " st1    {v2.h}[4], [%[C1]]\n"
      " st1    {v0.h}[4], [%[C0]]\n"
      " b      360f\n"

      "354:\n"
      " tbz    %[N], #1, 355f\n"
      " str    s10, [%[C5]], #4\n"
      " str    s18, [%[C4]], #4\n"
      " str    s6, [%[C3]], #4\n"
      " str    s4, [%[C2]], #4\n"
      " str    s2, [%[C1]], #4\n"
      " str    s0, [%[C0]], #4\n"
      " tbz    %[N], #0, 360f\n"
      " st1    {v10.h}[2], [%[C5]]\n"
      " st1    {v18.h}[2], [%[C4]]\n"
      " st1    {v6.h}[2], [%[C3]]\n"
      " st1    {v4.h}[2], [%[C2]]\n"
      " st1    {v2.h}[2], [%[C1]]\n"
      " st1    {v0.h}[2], [%[C0]]\n"
      " b      360f\n"

      "355:\n"
      " str    h10, [%[C5]]\n"
      " str    h18, [%[C4]]\n"
      " str    h6, [%[C3]]\n"
      " str    h4, [%[C2]]\n"
      " str    h2, [%[C1]]\n"
      " str    h0, [%[C0]]\n"

      "360:\n"
      : [A] "+&r"(A), [B] "+&r"(B), [chunks] "+&r"(chunks),
        [C0] "+&r"(C0), [C1] "+&r"(C1), [C2] "+&r"(C2),
        [C3] "+&r"(C3), [C4] "+&r"(C4), [C5] "+&r"(C5),
        [alpha] "+&w"(alpha_fp32)
      : [N] "r"(N), [beta_bits] "r"(beta_bits),
        [alpha_mode] "r"(alpha_mode), [beta_mode] "r"(beta_mode)
      : "cc", "memory", "x16",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v10", "v11", "v16", "v18", "v19",
        "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31"
    );
}

static inline __attribute__((always_inline)) void bgemm_mmla_8x8_tile(
    const bfloat16_t *A, const bfloat16_t *B, bfloat16_t *C,
    kernel_inttype K, kernel_inttype ldc, bfloat16_t alpha, bfloat16_t beta,
    unsigned alpha_mode, unsigned beta_mode)
{
    float alpha_fp32 = alpha;
    float beta_fp32 = beta;
    const bfloat16_t *B0_5 = B;
    const bfloat16_t *B6_11 = B + K * 6;
    bfloat16_t *C_even = C;
    bfloat16_t *C_odd = C + ldc;
    const kernel_inttype ldc2 = ldc * (2 * sizeof(bfloat16_t));

    __asm__ __volatile__(
      " ldp    q0, q1, [%[B0_5]]\n"
      " ldr    q2, [%[B0_5], #32]\n"
      " ldr    q3, [%[B6_11]]\n"
      " ldp    q6, q7, [%[A]]\n"

      " movi   v18.16b, #0\n"
      " movi   v19.16b, #0\n"
      " movi   v16.16b, #0\n"
      " movi   v17.16b, #0\n"
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
      " bfmmla v18.4s,  v0.8h,   v6.8h\n"
      " bfmmla v19.4s,  v0.8h,   v7.8h\n"
      " bfmmla v20.4s,  v1.8h,   v6.8h\n"
      " bfmmla v21.4s,  v1.8h,   v7.8h\n"
      " bfmmla v24.4s,  v2.8h,   v6.8h\n"
      " bfmmla v25.4s,  v2.8h,   v7.8h\n"
      " bfmmla v28.4s,  v3.8h,   v6.8h\n"
      " bfmmla v29.4s,  v3.8h,   v7.8h\n"
      " ldp    q6, q7, [%[A], #32]\n"
      " add    %[A], %[A], #64\n"
      " subs   %[K], %[K], #4\n"
      " bfmmla v16.4s,  v0.8h,   v6.8h\n"
      " bfmmla v17.4s,  v0.8h,   v7.8h\n"
      " bfmmla v22.4s,  v1.8h,   v6.8h\n"
      " bfmmla v23.4s,  v1.8h,   v7.8h\n"
      " ldp    q0, q1, [%[B0_5], #48]\n"
      " bfmmla v26.4s,  v2.8h,   v6.8h\n"
      " bfmmla v27.4s,  v2.8h,   v7.8h\n"
      " ldr    q2, [%[B0_5], #80]\n"
      " bfmmla v30.4s,  v3.8h,   v6.8h\n"
      " bfmmla v31.4s,  v3.8h,   v7.8h\n"
      " ldr    q3, [%[B6_11], #48]\n"
      " add    %[B0_5], %[B0_5], #48\n"
      " add    %[B6_11], %[B6_11], #48\n"
      " b.ne   1b\n"
      " ldp    q6, q7, [%[A]]\n"

      "4:\n"
      " bfmmla v18.4s,  v0.8h,   v6.8h\n"
      " bfmmla v19.4s,  v0.8h,   v7.8h\n"
      " bfmmla v20.4s,  v1.8h,   v6.8h\n"
      " bfmmla v21.4s,  v1.8h,   v7.8h\n"
      " bfmmla v24.4s,  v2.8h,   v6.8h\n"
      " bfmmla v25.4s,  v2.8h,   v7.8h\n"
      " bfmmla v28.4s,  v3.8h,   v6.8h\n"
      " bfmmla v29.4s,  v3.8h,   v7.8h\n"
      " ldp    q6, q7, [%[A], #32]\n"
      " bfmmla v16.4s,  v0.8h,   v6.8h\n"
      " bfmmla v17.4s,  v0.8h,   v7.8h\n"
      " bfmmla v22.4s,  v1.8h,   v6.8h\n"
      " bfmmla v23.4s,  v1.8h,   v7.8h\n"
      " bfmmla v26.4s,  v2.8h,   v6.8h\n"
      " bfmmla v27.4s,  v2.8h,   v7.8h\n"
      " bfmmla v30.4s,  v3.8h,   v6.8h\n"
      " bfmmla v31.4s,  v3.8h,   v7.8h\n"

      " cbz    %w[beta_mode], 200f\n"
      " ldr    q4, [%[C_even]]\n"

      "200:\n"
      " uzp1   v0.2d,   v18.2d,  v19.2d\n"
      " uzp2   v1.2d,   v18.2d,  v19.2d\n"
      " cbz    %w[beta_mode], 201f\n"
      " ldr    q19, [%[C_odd]]\n"

      "201:\n"
      " uzp1   v2.2d,   v16.2d,  v17.2d\n"
      " uzp2   v3.2d,   v16.2d,  v17.2d\n"
      " movi   v18.16b, #0\n"
      " cbz    %w[alpha_mode], 202f\n"
      " fmul   v0.4s,   v0.4s,   %[alpha].s[0]\n"
      " fmul   v1.4s,   v1.4s,   %[alpha].s[0]\n"
      " fmul   v2.4s,   v2.4s,   %[alpha].s[0]\n"
      " fmul   v3.4s,   v3.4s,   %[alpha].s[0]\n"

      "202:\n"
      " cbz    %w[beta_mode], 204f\n"
      " zip2   v5.8h,   v18.8h,  v4.8h\n"
      " zip1   v4.8h,   v18.8h,  v4.8h\n"
      " zip2   v16.8h,  v18.8h,  v19.8h\n"
      " zip1   v19.8h,  v18.8h,  v19.8h\n"
      " tbz    %w[beta_mode], #1, 203f\n"
      " fmla   v0.4s,   v4.4s,   %[beta].s[0]\n"
      " fmla   v2.4s,   v5.4s,   %[beta].s[0]\n"
      " fmla   v1.4s,   v19.4s,   %[beta].s[0]\n"
      " fmla   v3.4s,   v16.4s,  %[beta].s[0]\n"
      " b      204f\n"

      "203:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v2.4s,   v2.4s,   v5.4s\n"
      " fadd   v1.4s,   v1.4s,   v19.4s\n"
      " fadd   v3.4s,   v3.4s,   v16.4s\n"

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
      " uzp1   v0.2d,   v20.2d,  v21.2d\n"
      " uzp2   v1.2d,   v20.2d,  v21.2d\n"
      " cbz    %w[beta_mode], 211f\n"
      " ldr    q19, [%[C_odd]]\n"

      "211:\n"
      " uzp1   v2.2d,   v22.2d,  v23.2d\n"
      " uzp2   v3.2d,   v22.2d,  v23.2d\n"
      " cbz    %w[alpha_mode], 212f\n"
      " fmul   v0.4s,   v0.4s,   %[alpha].s[0]\n"
      " fmul   v1.4s,   v1.4s,   %[alpha].s[0]\n"
      " fmul   v2.4s,   v2.4s,   %[alpha].s[0]\n"
      " fmul   v3.4s,   v3.4s,   %[alpha].s[0]\n"

      "212:\n"
      " cbz    %w[beta_mode], 214f\n"
      " zip2   v5.8h,   v18.8h,  v4.8h\n"
      " zip1   v4.8h,   v18.8h,  v4.8h\n"
      " zip2   v16.8h,  v18.8h,  v19.8h\n"
      " zip1   v19.8h,  v18.8h,  v19.8h\n"
      " tbz    %w[beta_mode], #1, 213f\n"
      " fmla   v0.4s,   v4.4s,   %[beta].s[0]\n"
      " fmla   v2.4s,   v5.4s,   %[beta].s[0]\n"
      " fmla   v1.4s,   v19.4s,   %[beta].s[0]\n"
      " fmla   v3.4s,   v16.4s,  %[beta].s[0]\n"
      " b      214f\n"

      "213:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v2.4s,   v2.4s,   v5.4s\n"
      " fadd   v1.4s,   v1.4s,   v19.4s\n"
      " fadd   v3.4s,   v3.4s,   v16.4s\n"

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
      " uzp1   v0.2d,   v24.2d,  v25.2d\n"
      " uzp2   v1.2d,   v24.2d,  v25.2d\n"
      " cbz    %w[beta_mode], 221f\n"
      " ldr    q19, [%[C_odd]]\n"

      "221:\n"
      " uzp1   v2.2d,   v26.2d,  v27.2d\n"
      " uzp2   v3.2d,   v26.2d,  v27.2d\n"
      " cbz    %w[alpha_mode], 222f\n"
      " fmul   v0.4s,   v0.4s,   %[alpha].s[0]\n"
      " fmul   v1.4s,   v1.4s,   %[alpha].s[0]\n"
      " fmul   v2.4s,   v2.4s,   %[alpha].s[0]\n"
      " fmul   v3.4s,   v3.4s,   %[alpha].s[0]\n"

      "222:\n"
      " cbz    %w[beta_mode], 224f\n"
      " zip2   v5.8h,   v18.8h,  v4.8h\n"
      " zip1   v4.8h,   v18.8h,  v4.8h\n"
      " zip2   v16.8h,  v18.8h,  v19.8h\n"
      " zip1   v19.8h,  v18.8h,  v19.8h\n"
      " tbz    %w[beta_mode], #1, 223f\n"
      " fmla   v0.4s,   v4.4s,   %[beta].s[0]\n"
      " fmla   v2.4s,   v5.4s,   %[beta].s[0]\n"
      " fmla   v1.4s,   v19.4s,   %[beta].s[0]\n"
      " fmla   v3.4s,   v16.4s,  %[beta].s[0]\n"
      " b      224f\n"

      "223:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v2.4s,   v2.4s,   v5.4s\n"
      " fadd   v1.4s,   v1.4s,   v19.4s\n"
      " fadd   v3.4s,   v3.4s,   v16.4s\n"

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
      " uzp1   v0.2d,   v28.2d,  v29.2d\n"
      " uzp2   v1.2d,   v28.2d,  v29.2d\n"
      " cbz    %w[beta_mode], 231f\n"
      " ldr    q19, [%[C_odd]]\n"

      "231:\n"
      " uzp1   v2.2d,   v30.2d,  v31.2d\n"
      " uzp2   v3.2d,   v30.2d,  v31.2d\n"
      " cbz    %w[alpha_mode], 232f\n"
      " fmul   v0.4s,   v0.4s,   %[alpha].s[0]\n"
      " fmul   v1.4s,   v1.4s,   %[alpha].s[0]\n"
      " fmul   v2.4s,   v2.4s,   %[alpha].s[0]\n"
      " fmul   v3.4s,   v3.4s,   %[alpha].s[0]\n"

      "232:\n"
      " cbz    %w[beta_mode], 234f\n"
      " zip2   v5.8h,   v18.8h,  v4.8h\n"
      " zip1   v4.8h,   v18.8h,  v4.8h\n"
      " zip2   v16.8h,  v18.8h,  v19.8h\n"
      " zip1   v19.8h,  v18.8h,  v19.8h\n"
      " tbz    %w[beta_mode], #1, 233f\n"
      " fmla   v0.4s,   v4.4s,   %[beta].s[0]\n"
      " fmla   v2.4s,   v5.4s,   %[beta].s[0]\n"
      " fmla   v1.4s,   v19.4s,   %[beta].s[0]\n"
      " fmla   v3.4s,   v16.4s,  %[beta].s[0]\n"
      " b      234f\n"

      "233:\n"
      " fadd   v0.4s,   v0.4s,   v4.4s\n"
      " fadd   v2.4s,   v2.4s,   v5.4s\n"
      " fadd   v1.4s,   v1.4s,   v19.4s\n"
      " fadd   v3.4s,   v3.4s,   v16.4s\n"

      "234:\n"
      " bfcvtn v4.4h,   v0.4s\n"
      " bfcvtn v5.4h,   v1.4s\n"
      " bfcvtn2 v4.8h,   v2.4s\n"
      " bfcvtn2 v5.8h,   v3.4s\n"
      " str    q4, [%[C_even]]\n"
      " str    q5, [%[C_odd]]\n"
      : [A] "+&r"(A), [B0_5] "+&r"(B0_5),
        [B6_11] "+&r"(B6_11), [K] "+&r"(K),
        [C_even] "+&r"(C_even), [C_odd] "+&r"(C_odd),
        [alpha] "+&w"(alpha_fp32), [beta] "+&w"(beta_fp32)
      : [ldc2] "r"(ldc2),
        [alpha_mode] "r"(alpha_mode), [beta_mode] "r"(beta_mode)
      : "memory", "cc",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

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

// Compute blocks of at most six columns and eight rows using A8/B6 packing.
static inline void a64_interleaved_nomerge_bf16fp32bf16_mmla_6x8(
    const bfloat16_t *A, const bfloat16_t *B, bfloat16_t *C,
    kernel_inttype K, kernel_inttype N, kernel_inttype M,
    kernel_inttype ldc, bfloat16_t alpha, bfloat16_t beta,
    unsigned alpha_mode, unsigned beta_mode)
{
    for (kernel_inttype col = 0; col < M; col += 6) {
        for (kernel_inttype row = 0; row < N; row += 8) {
            bgemm_mmla_6x8_tile(
                A + row * K, B + col * K, C + col * ldc + row,
                K, std::min<kernel_inttype>(8, N - row),
                std::min<kernel_inttype>(6, M - col), ldc, alpha, beta,
                alpha_mode, beta_mode);
        }
    }
}

// Handle full eight-row blocks across exactly eight columns.
// Eight columns span one complete B6 panel and the first pair of the next.
// A remains packed in groups of eight rows; both B panels advance by K=4.
void a64_interleaved_nomerge_bf16fp32bf16_mmla_8x8(
    const bfloat16_t *A, const bfloat16_t *B, bfloat16_t *C,
    kernel_inttype K, kernel_inttype N, kernel_inttype ldc,
    bfloat16_t alpha, bfloat16_t beta,
    unsigned alpha_mode, unsigned beta_mode)
{
    for (kernel_inttype row = 0; row < N; row += 8) {
        bgemm_mmla_8x8_tile(
            A + row * K, B, C + row,
            K, ldc, alpha, beta, alpha_mode, beta_mode);
    }
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

    if (N < 8 || (M < 12 && M != 8)) {
        a64_interleaved_nomerge_bf16fp32bf16_mmla_6x8(
            A, B, C, K, N, M, ldc, alpha, beta, alpha_mode, beta_mode);
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
        if (M - col == 8) {
            a64_interleaved_nomerge_bf16fp32bf16_mmla_8x8(
                A, B + col * K, C + col * ldc,
                K, full_rows, ldc, alpha, beta, alpha_mode, beta_mode);
        } else {
            a64_interleaved_nomerge_bf16fp32bf16_mmla_6x8(
                A, B + col * K, C + col * ldc,
                K, full_rows, M - col, ldc, alpha, beta, alpha_mode, beta_mode);
        }
    }

    // Handle remaining rows across all columns once.
    if (full_rows != N) {
        a64_interleaved_nomerge_bf16fp32bf16_mmla_6x8(
            A + full_rows * K, B, C + full_rows,
            K, N - full_rows, M, ldc, alpha, beta, alpha_mode, beta_mode);
    }
}

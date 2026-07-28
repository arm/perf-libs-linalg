/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "framework/linalg_util.hpp"

#include <cstdint>
#include <cstddef>

extern "C" {

void sme2_interleaved_nomerge_fp32_mopa_2VLx2VL(const float *const A, const float *const B, float *const C, const int M, const int N, const int K, int ldc, float alpha, float beta) {

  struct KernelArgs {
    KernelArgs(
      const float *const B,
      const float *const A,
      float *const C,
      const int K, const int N, const int M,
      const int ldc,
	float alpha, float beta)

      : A(A),
        B(B),
        C(C),
        M(M), N(N), K(K),
        ldcb(ldc * sizeof(float)),
		alpha(alpha), beta(beta),
        flags(0x0)
    {
      if (beta != 0.0)
      {
        flags |= 1 << 0;  // SCALE_OUTPUT_MATRIX_BY_BETA
      }
    }

    const float *const A;
    const float *const B;
    float *const C;
    const long M, N, K;
    const long ldcb;

    const float alpha;
    const float beta;

    std::uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(
		A, B, C,
		M, N, K,
		ldc,
		alpha, beta);

  __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ldr x16, [%x[args], %[offsetof_flags]]\n"
      "mov x15, #0x0\n"
      "mov x14, #0x0\n"
      "ptrue p0.b\n"
      ".inst 0x25207811  // ptrue pn9.b\n"
      "ldr x13, [%x[args], %[offsetof_K]]\n"
      "ldr w11, [%x[args], %[offsetof_M]]\n"
      "ldr w10, [%x[args], %[offsetof_N]]\n"
      "ldr x9, [%x[args], %[offsetof_A]]\n"
      "1:"  // M loop
      "ldr x28, [%x[args], %[offsetof_B]]\n"
      "2:"  // N loop
      "lsr x21, x13, #0x2\n"
      "mov x27, x9\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"  // Zero accumulator registers
      ".inst 0x25aa45d0  // whilelt pn8.s, x14, x10, VLx2\n"
      "and x20, x13, #0x3\n"
      "cbz x21, 5f\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa140c772  // ld1w { z18.s, z22.s, z26.s, z30.s }, pn9.b/Z, [x27]\n"
      ".inst 0xa141c773  // ld1w { z19.s, z23.s, z27.s, z31.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      ".inst 0xa040c788  // ld1w { z8.s-z11.s }, pn9.b/Z, [x28]\n"
      ".inst 0xa141c790  // ld1w { z16.s, z20.s, z24.s, z28.s }, pn9.b/Z, [x28, #0x4, MUL VL]\n"
      "addvl x28, x28, #8\n"
      "ble 4f\n"
      "3:"  // K loop
      ".inst 0x80880240  // fmopa za0.s, p0/M, p0/M, z18.s, z8.s\n"
      "subs x21, x21, #0x1\n"
      ".inst 0x80890241  // fmopa za1.s, p0/M, p0/M, z18.s, z9.s\n"
      ".inst 0x808802c2  // fmopa za2.s, p0/M, p0/M, z22.s, z8.s\n"
      ".inst 0x808902c3  // fmopa za3.s, p0/M, p0/M, z22.s, z9.s\n"
      ".inst 0x808a0340  // fmopa za0.s, p0/M, p0/M, z26.s, z10.s\n"
      ".inst 0x808b0341  // fmopa za1.s, p0/M, p0/M, z26.s, z11.s\n"
      ".inst 0x808a03c2  // fmopa za2.s, p0/M, p0/M, z30.s, z10.s\n"
      ".inst 0x808b03c3  // fmopa za3.s, p0/M, p0/M, z30.s, z11.s\n"
      ".inst 0xa140c772  // ld1w { z18.s, z22.s, z26.s, z30.s }, pn9.b/Z, [x27]\n"
      ".inst 0x80900260  // fmopa za0.s, p0/M, p0/M, z19.s, z16.s\n"
      ".inst 0xa040c788  // ld1w { z8.s-z11.s }, pn9.b/Z, [x28]\n"
      ".inst 0x80940261  // fmopa za1.s, p0/M, p0/M, z19.s, z20.s\n"
      ".inst 0x809002e2  // fmopa za2.s, p0/M, p0/M, z23.s, z16.s\n"
      ".inst 0x809402e3  // fmopa za3.s, p0/M, p0/M, z23.s, z20.s\n"
      ".inst 0x80980360  // fmopa za0.s, p0/M, p0/M, z27.s, z24.s\n"
      ".inst 0x809c0361  // fmopa za1.s, p0/M, p0/M, z27.s, z28.s\n"
      ".inst 0x809803e2  // fmopa za2.s, p0/M, p0/M, z31.s, z24.s\n"
      ".inst 0x809c03e3  // fmopa za3.s, p0/M, p0/M, z31.s, z28.s\n"
      ".inst 0xa141c773  // ld1w { z19.s, z23.s, z27.s, z31.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      ".inst 0xa141c790  // ld1w { z16.s, z20.s, z24.s, z28.s }, pn9.b/Z, [x28, #0x4, MUL VL]\n"
      "addvl x28, x28, #8\n"
      "bgt 3b\n"
      "4:"  // K loop tail
      ".inst 0x80880240  // fmopa za0.s, p0/M, p0/M, z18.s, z8.s\n"
      ".inst 0x80890241  // fmopa za1.s, p0/M, p0/M, z18.s, z9.s\n"
      ".inst 0x808802c2  // fmopa za2.s, p0/M, p0/M, z22.s, z8.s\n"
      ".inst 0x808902c3  // fmopa za3.s, p0/M, p0/M, z22.s, z9.s\n"
      ".inst 0x808a0340  // fmopa za0.s, p0/M, p0/M, z26.s, z10.s\n"
      ".inst 0x808b0341  // fmopa za1.s, p0/M, p0/M, z26.s, z11.s\n"
      ".inst 0x808a03c2  // fmopa za2.s, p0/M, p0/M, z30.s, z10.s\n"
      ".inst 0x808b03c3  // fmopa za3.s, p0/M, p0/M, z30.s, z11.s\n"
      ".inst 0x80900260  // fmopa za0.s, p0/M, p0/M, z19.s, z16.s\n"
      ".inst 0x80940261  // fmopa za1.s, p0/M, p0/M, z19.s, z20.s\n"
      ".inst 0x809002e2  // fmopa za2.s, p0/M, p0/M, z23.s, z16.s\n"
      ".inst 0x809402e3  // fmopa za3.s, p0/M, p0/M, z23.s, z20.s\n"
      ".inst 0x80980360  // fmopa za0.s, p0/M, p0/M, z27.s, z24.s\n"
      ".inst 0x809c0361  // fmopa za1.s, p0/M, p0/M, z27.s, z28.s\n"
      ".inst 0x809803e2  // fmopa za2.s, p0/M, p0/M, z31.s, z24.s\n"
      ".inst 0x809c03e3  // fmopa za3.s, p0/M, p0/M, z31.s, z28.s\n"
      "5:"  // K oddments
      "cbz x20, 7f\n"
      "6:"  // K oddments: Loop
      ".inst 0xa040477c  // ld1w { z28.s-z29.s }, pn9.b/Z, [x27]\n"
      "subs x20, x20, #0x1\n"
      "addvl x27, x27, #2\n"
      ".inst 0xa1404791  // ld1w { z17.s, z25.s }, pn9.b/Z, [x28]\n"
      "addvl x28, x28, #2\n"
      ".inst 0x80910380  // fmopa za0.s, p0/M, p0/M, z28.s, z17.s\n"
      ".inst 0x80990381  // fmopa za1.s, p0/M, p0/M, z28.s, z25.s\n"
      ".inst 0x809103a2  // fmopa za2.s, p0/M, p0/M, z29.s, z17.s\n"
      ".inst 0x809903a3  // fmopa za3.s, p0/M, p0/M, z29.s, z25.s\n"
      "bgt 6b\n"
      "7:"  // K oddments: End
      "ldr x26, [%x[args], %[offsetof_C]]\n"
      "add x20, %x[args], %[offset_alpha]\n"
      "sub x25, x11, x15\n"
      "ldr x24, [%x[args], %[offsetof_ldcb]]\n"
      "cntw x23\n"
      "ld1rw { z24.s }, p0/Z, [x20]\n"  // Load {name} from KernelArgs
      "add x26, x26, x14, LSL #2\n"  // C += n
      "madd x26, x15, x24, x26\n"  // C += m * ldc
      "tbnz x16, #0, 16f\n"
      "cmp x25, x23\n"
      "mov x12, #0x0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 11f\n"
      "10:"  // Store to output array: Set Zero: Accumulator row 0 loop
      ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      "fmul z4.s, z4.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z12.s, z12.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "add x12, x12, #0x4\n"
      "fmul z5.s, z5.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z13.s, z13.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "cmp x12, x21, LSL #2\n"
      "fmul z6.s, z6.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z14.s, z14.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z7.s, z7.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z15.s, z15.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1604344  // st1w { z4.s, z12.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1604345  // st1w { z5.s, z13.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1604346  // st1w { z6.s, z14.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1604347  // st1w { z7.s, z15.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "blt 10b\n"
      "11:"  // Store to output array: Set Zero: Accumulator row 0 oddments
      "cbz x20, 12f\n"
      ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      "fmul z4.s, z4.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z12.s, z12.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1604344  // st1w { z4.s, z12.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 12f\n"  // bah!
      "subs x20, x20, #0x1\n"
      "fmul z5.s, z5.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z13.s, z13.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1604345  // st1w { z5.s, z13.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 12f\n"  // bah!
      "fmul z6.s, z6.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z14.s, z14.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1604346  // st1w { z6.s, z14.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "12:"  // Store to output array: Set Zero: Accumulator row 0 oddments: End
      "subs x25, x25, x22\n"
      "beq 24f\n"
      "cmp x25, x23\n"
      "mov x12, #0x0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 14f\n"
      "13:"  // Store to output array: Set Zero: Accumulator row 1 loop
      ".inst 0xc0860444  // mova { z4.s-z7.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      "fmul z4.s, z4.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z12.s, z12.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "add x12, x12, #0x4\n"
      "fmul z5.s, z5.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z13.s, z13.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "cmp x12, x21, LSL #2\n"
      "fmul z6.s, z6.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z14.s, z14.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z7.s, z7.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z15.s, z15.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1604344  // st1w { z4.s, z12.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1604345  // st1w { z5.s, z13.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1604346  // st1w { z6.s, z14.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1604347  // st1w { z7.s, z15.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "blt 13b\n"
      "14:"  // Store to output array: Set Zero: Accumulator row 1 oddments
      "cbz x20, 15f\n"
      ".inst 0xc0860444  // mova { z4.s-z7.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      "fmul z4.s, z4.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z12.s, z12.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1604344  // st1w { z4.s, z12.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 15f\n"  // bah!
      "subs x20, x20, #0x1\n"
      "fmul z5.s, z5.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z13.s, z13.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1604345  // st1w { z5.s, z13.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 15f\n"  // bah!
      "fmul z6.s, z6.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z14.s, z14.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1604346  // st1w { z6.s, z14.s }, p8, [x26]\n"
      "15:"  // Store to output array: Set Zero: Accumulator row 1 oddments: End
      "subs x25, x25, x22\n"
      "beq 24f\n"
      "b 23f\n"
      "16:"  // Store to output array: Scale Beta
      "cmp x25, x23\n"
      "add x20, %x[args], %[offset_beta]\n"
      "csel x22, x25, x23, LT\n"
      "ld1rw { z9.s }, p0/Z, [x20]\n"  // Load {name} from KernelArgs
      "mov x12, #0x0\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 18f\n"
      "17:"  // Store to output array: Scale Beta: Accumulator row 0 loop
      ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      ".inst 0xa1404342  // ld1w { z2.s, z10.s }, p8/Z, [x26]\n"
      "fmul z4.s, z4.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z12.s, z12.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "add x12, x12, #0x4\n"
      "fmul z5.s, z5.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z13.s, z13.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "cmp x12, x21, LSL #2\n"
      "fmul z6.s, z6.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z14.s, z14.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z7.s, z7.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z15.s, z15.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmla z4.s, p0/M, z2.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z12.s, p0/M, z10.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1604344  // st1w { z4.s, z12.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1404351  // ld1w { z17.s, z25.s }, p8/Z, [x26]\n"
      "fmla z5.s, p0/M, z17.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z13.s, p0/M, z25.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1604345  // st1w { z5.s, z13.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1404351  // ld1w { z17.s, z25.s }, p8/Z, [x26]\n"
      "fmla z6.s, p0/M, z17.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z14.s, p0/M, z25.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1604346  // st1w { z6.s, z14.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1404351  // ld1w { z17.s, z25.s }, p8/Z, [x26]\n"
      "fmla z7.s, p0/M, z17.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z15.s, p0/M, z25.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1604347  // st1w { z7.s, z15.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "blt 17b\n"
      "18:"  // Store to output array: Scale Beta: Accumulator row 0 oddments
      "cbz x20, 19f\n"
      ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      ".inst 0xa1404351  // ld1w { z17.s, z25.s }, p8/Z, [x26]\n"
      "fmul z4.s, z4.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z12.s, z12.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "subs x20, x20, #0x1\n"
      "fmla z4.s, p0/M, z17.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z12.s, p0/M, z25.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1604344  // st1w { z4.s, z12.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 19f\n"  // bah!
      "fmul z5.s, z5.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z13.s, z13.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1404351  // ld1w { z17.s, z25.s }, p8/Z, [x26]\n"
      "subs x20, x20, #0x1\n"
      "fmla z5.s, p0/M, z17.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z13.s, p0/M, z25.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1604345  // st1w { z5.s, z13.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 19f\n"  // bah!
      "fmul z6.s, z6.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z14.s, z14.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa0404352  // ld1w { z18.s-z19.s }, p8/Z, [x26]\n"
      "fmla z6.s, p0/M, z18.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z14.s, p0/M, z19.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1604346  // st1w { z6.s, z14.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "19:"  // Store to output array: Scale Beta: Accumulator row 0 oddments: End
      "subs x25, x25, x22\n"
      "beq 24f\n"
      "cmp x25, x23\n"
      "mov x12, #0x0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 21f\n"
      "20:"  // Store to output array: Scale Beta: Accumulator row 1 loop
      ".inst 0xc0860444  // mova { z4.s-z7.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa1404351  // ld1w { z17.s, z25.s }, p8/Z, [x26]\n"
      "fmul z4.s, z4.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z12.s, z12.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "add x12, x12, #0x4\n"
      "fmul z5.s, z5.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z13.s, z13.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "cmp x12, x21, LSL #2\n"
      "fmul z6.s, z6.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z14.s, z14.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z7.s, z7.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z15.s, z15.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmla z4.s, p0/M, z17.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z12.s, p0/M, z25.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1604344  // st1w { z4.s, z12.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1404351  // ld1w { z17.s, z25.s }, p8/Z, [x26]\n"
      "fmla z5.s, p0/M, z17.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z13.s, p0/M, z25.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1604345  // st1w { z5.s, z13.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa0404350  // ld1w { z16.s-z17.s }, p8/Z, [x26]\n"
      "fmla z6.s, p0/M, z16.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z14.s, p0/M, z17.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1604346  // st1w { z6.s, z14.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa0404352  // ld1w { z18.s-z19.s }, p8/Z, [x26]\n"
      "fmla z7.s, p0/M, z18.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z15.s, p0/M, z19.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1604347  // st1w { z7.s, z15.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "blt 20b\n"
      "21:"  // Store to output array: Scale Beta: Accumulator row 1 oddments
      "cbz x20, 22f\n"
      ".inst 0xc0860444  // mova { z4.s-z7.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa1404347  // ld1w { z7.s, z15.s }, p8/Z, [x26]\n"
      "fmul z4.s, z4.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z12.s, z12.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "subs x20, x20, #0x1\n"
      "fmla z4.s, p0/M, z7.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z12.s, p0/M, z15.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1604344  // st1w { z4.s, z12.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 22f\n"  // bah!
      "fmul z5.s, z5.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z13.s, z13.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1404347  // ld1w { z7.s, z15.s }, p8/Z, [x26]\n"
      "subs x20, x20, #0x1\n"
      "fmla z5.s, p0/M, z7.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z13.s, p0/M, z15.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1604345  // st1w { z5.s, z13.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 22f\n"  // bah!
      "fmul z6.s, z6.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      "fmul z14.s, z14.s, z24.s\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa0404356  // ld1w { z22.s-z23.s }, p8/Z, [x26]\n"
      "fmla z6.s, p0/M, z22.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z14.s, p0/M, z23.s, z9.s\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1604346  // st1w { z6.s, z14.s }, p8, [x26]\n"
      "22:"  // Store to output array: Scale Beta: Accumulator row 1 oddments: End
      "subs x25, x25, x22\n"
      "beq 24f\n"
      "23:"  // Store to output array: Write End
      "24:"  // Store to output array: End
      "incw x14, ALL, MUL #2\n"
      "cmp x14, x10\n"
      "blt 2b\n"
      "incw x15, ALL, MUL #2\n"
      "mov x14, #0x0\n"
      "cmp x15, x11\n"
      "mov x9, x27\n"
      "blt 1b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [offset_alpha] "I" (offsetof(KernelArgs, alpha)), [offset_beta] "I" (offsetof(KernelArgs, beta)), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p10", "p11", "p12", "p13", "p14", "p15", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x9", "z0", "z1", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z2", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z3", "z30", "z31", "z4", "z5", "z6", "z7", "z8", "z9"
    );
}

}  // extern "C"

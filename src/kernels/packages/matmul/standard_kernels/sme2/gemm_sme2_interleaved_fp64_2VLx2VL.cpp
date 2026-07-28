/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "framework/linalg_util.hpp"

#include <cstdint>
#include <cstddef>

extern "C" {

void sme2_interleaved_nomerge_fp64_mopa_2VLx2VL(const double *const A, const double *const B, double *const C, const int M, const int N, const int K, int ldc, double alpha, double beta) {

  struct KernelArgs {
    KernelArgs(
      const double *const B,
      const double *const A,
      double *const C,
      const int K, const int N, const int M,
      const int ldc,
	double alpha, double beta)

      : A(A),
        B(B),
        C(C),
        M(M), N(N), K(K),
        ldcb(ldc * sizeof(double)),
		alpha(alpha), beta(beta),
        flags(0x0)
    {
      if (beta != 0.0)
      {
        flags |= 1 << 0;  // SCALE_OUTPUT_MATRIX_BY_BETA
      }
    }

    const double *const A;
    const double *const B;
    double *const C;
    const long M, N, K;
    const long ldcb;

    const double alpha;
    const double beta;

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
      ".inst 0xc008000f  // zero { zad0, zad1, zad2, zad3 }\n"  // Zero accumulator registers
      ".inst 0x25ea45d0  // whilelt pn8.d, x14, x10, VLx2\n"
      "and x20, x13, #0x3\n"
      "cbz x21, 5f\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa140e772  // ld1d { z18.d, z22.d, z26.d, z30.d }, pn9.b/Z, [x27]\n"
      ".inst 0xa141e773  // ld1d { z19.d, z23.d, z27.d, z31.d }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      ".inst 0xa040e788  // ld1d { z8.d-z11.d }, pn9.b/Z, [x28]\n"
      ".inst 0xa141e790  // ld1d { z16.d, z20.d, z24.d, z28.d }, pn9.b/Z, [x28, #0x4, MUL VL]\n"
      "addvl x28, x28, #8\n"
      "ble 4f\n"
      "3:"  // K loop
      ".inst 0x80c80240  // fmopa za0.d, p0/M, p0/M, z18.d, z8.d\n"
      "subs x21, x21, #0x1\n"
      ".inst 0x80c90241  // fmopa za1.d, p0/M, p0/M, z18.d, z9.d\n"
      ".inst 0x80c802c2  // fmopa za2.d, p0/M, p0/M, z22.d, z8.d\n"
      ".inst 0x80c902c3  // fmopa za3.d, p0/M, p0/M, z22.d, z9.d\n"
      ".inst 0x80ca0340  // fmopa za0.d, p0/M, p0/M, z26.d, z10.d\n"
      ".inst 0x80cb0341  // fmopa za1.d, p0/M, p0/M, z26.d, z11.d\n"
      ".inst 0x80ca03c2  // fmopa za2.d, p0/M, p0/M, z30.d, z10.d\n"
      ".inst 0x80cb03c3  // fmopa za3.d, p0/M, p0/M, z30.d, z11.d\n"
      ".inst 0xa140e772  // ld1d { z18.d, z22.d, z26.d, z30.d }, pn9.b/Z, [x27]\n"
      ".inst 0x80d00260  // fmopa za0.d, p0/M, p0/M, z19.d, z16.d\n"
      ".inst 0xa040e788  // ld1d { z8.d-z11.d }, pn9.b/Z, [x28]\n"
      ".inst 0x80d40261  // fmopa za1.d, p0/M, p0/M, z19.d, z20.d\n"
      ".inst 0x80d002e2  // fmopa za2.d, p0/M, p0/M, z23.d, z16.d\n"
      ".inst 0x80d402e3  // fmopa za3.d, p0/M, p0/M, z23.d, z20.d\n"
      ".inst 0x80d80360  // fmopa za0.d, p0/M, p0/M, z27.d, z24.d\n"
      ".inst 0x80dc0361  // fmopa za1.d, p0/M, p0/M, z27.d, z28.d\n"
      ".inst 0x80d803e2  // fmopa za2.d, p0/M, p0/M, z31.d, z24.d\n"
      ".inst 0x80dc03e3  // fmopa za3.d, p0/M, p0/M, z31.d, z28.d\n"
      ".inst 0xa141e773  // ld1d { z19.d, z23.d, z27.d, z31.d }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      ".inst 0xa141e790  // ld1d { z16.d, z20.d, z24.d, z28.d }, pn9.b/Z, [x28, #0x4, MUL VL]\n"
      "addvl x28, x28, #8\n"
      "bgt 3b\n"
      "4:"  // K loop tail
      ".inst 0x80c80240  // fmopa za0.d, p0/M, p0/M, z18.d, z8.d\n"
      ".inst 0x80c90241  // fmopa za1.d, p0/M, p0/M, z18.d, z9.d\n"
      ".inst 0x80c802c2  // fmopa za2.d, p0/M, p0/M, z22.d, z8.d\n"
      ".inst 0x80c902c3  // fmopa za3.d, p0/M, p0/M, z22.d, z9.d\n"
      ".inst 0x80ca0340  // fmopa za0.d, p0/M, p0/M, z26.d, z10.d\n"
      ".inst 0x80cb0341  // fmopa za1.d, p0/M, p0/M, z26.d, z11.d\n"
      ".inst 0x80ca03c2  // fmopa za2.d, p0/M, p0/M, z30.d, z10.d\n"
      ".inst 0x80cb03c3  // fmopa za3.d, p0/M, p0/M, z30.d, z11.d\n"
      ".inst 0x80d00260  // fmopa za0.d, p0/M, p0/M, z19.d, z16.d\n"
      ".inst 0x80d40261  // fmopa za1.d, p0/M, p0/M, z19.d, z20.d\n"
      ".inst 0x80d002e2  // fmopa za2.d, p0/M, p0/M, z23.d, z16.d\n"
      ".inst 0x80d402e3  // fmopa za3.d, p0/M, p0/M, z23.d, z20.d\n"
      ".inst 0x80d80360  // fmopa za0.d, p0/M, p0/M, z27.d, z24.d\n"
      ".inst 0x80dc0361  // fmopa za1.d, p0/M, p0/M, z27.d, z28.d\n"
      ".inst 0x80d803e2  // fmopa za2.d, p0/M, p0/M, z31.d, z24.d\n"
      ".inst 0x80dc03e3  // fmopa za3.d, p0/M, p0/M, z31.d, z28.d\n"
      "5:"  // K oddments
      "cbz x20, 7f\n"
      "6:"  // K oddments: Loop
      ".inst 0xa040677c  // ld1d { z28.d-z29.d }, pn9.b/Z, [x27]\n"
      "subs x20, x20, #0x1\n"
      "addvl x27, x27, #2\n"
      ".inst 0xa1406791  // ld1d { z17.d, z25.d }, pn9.b/Z, [x28]\n"
      "addvl x28, x28, #2\n"
      ".inst 0x80d10380  // fmopa za0.d, p0/M, p0/M, z28.d, z17.d\n"
      ".inst 0x80d90381  // fmopa za1.d, p0/M, p0/M, z28.d, z25.d\n"
      ".inst 0x80d103a2  // fmopa za2.d, p0/M, p0/M, z29.d, z17.d\n"
      ".inst 0x80d903a3  // fmopa za3.d, p0/M, p0/M, z29.d, z25.d\n"
      "bgt 6b\n"
      "7:"  // K oddments: End
      "ldr x26, [%x[args], %[offsetof_C]]\n"
      "add x20, %x[args], %[offset_alpha]\n"
      "sub x25, x11, x15\n"
      "ldr x24, [%x[args], %[offsetof_ldcb]]\n"
      "cntd x23\n"
      "ld1rd { z24.d }, p0/Z, [x20]\n"  // Load {name} from KernelArgs
      "add x26, x26, x14, LSL #3\n"  // C += n
      "madd x26, x15, x24, x26\n"  // C += m * ldc
      "tbnz x16, #0, 16f\n"
      "cmp x25, x23\n"
      "mov x12, #0x0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 11f\n"
      "10:"  // Store to output array: Set Zero: Accumulator row 0 loop
      ".inst 0xc0c60404  // mova { z4.d-z7.d }, za0h.d[x12]\n"
      ".inst 0xc0c6042c  // mova { z12.d-z15.d }, za1h.d[x12]\n"
      "fmul z4.d, z4.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z12.d, z12.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "add x12, x12, #0x4\n"
      "fmul z5.d, z5.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z13.d, z13.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "cmp x12, x21, LSL #2\n"
      "fmul z6.d, z6.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z14.d, z14.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z7.d, z7.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z15.d, z15.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1606344  // st1d { z4.d, z12.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1606345  // st1d { z5.d, z13.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1606346  // st1d { z6.d, z14.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1606347  // st1d { z7.d, z15.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "blt 10b\n"
      "11:"  // Store to output array: Set Zero: Accumulator row 0 oddments
      "cbz x20, 12f\n"
      ".inst 0xc0c60404  // mova { z4.d-z7.d }, za0h.d[x12]\n"
      ".inst 0xc0c6042c  // mova { z12.d-z15.d }, za1h.d[x12]\n"
      "subs x20, x20, #0x1\n"
      "fmul z4.d, z4.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z12.d, z12.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1606344  // st1d { z4.d, z12.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 12f\n"  // bah!
      "subs x20, x20, #0x1\n"
      "fmul z5.d, z5.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z13.d, z13.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1606345  // st1d { z5.d, z13.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 12f\n"  // bah!
      "fmul z6.d, z6.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z14.d, z14.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1606346  // st1d { z6.d, z14.d }, p8, [x26]\n"
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
      ".inst 0xc0c60444  // mova { z4.d-z7.d }, za2h.d[x12]\n"
      ".inst 0xc0c6046c  // mova { z12.d-z15.d }, za3h.d[x12]\n"
      "fmul z4.d, z4.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z12.d, z12.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "add x12, x12, #0x4\n"
      "fmul z5.d, z5.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z13.d, z13.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "cmp x12, x21, LSL #2\n"
      "fmul z6.d, z6.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z14.d, z14.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z7.d, z7.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z15.d, z15.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1606344  // st1d { z4.d, z12.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1606345  // st1d { z5.d, z13.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1606346  // st1d { z6.d, z14.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1606347  // st1d { z7.d, z15.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "blt 13b\n"
      "14:"  // Store to output array: Set Zero: Accumulator row 1 oddments
      "cbz x20, 15f\n"
      ".inst 0xc0c60444  // mova { z4.d-z7.d }, za2h.d[x12]\n"
      ".inst 0xc0c6046c  // mova { z12.d-z15.d }, za3h.d[x12]\n"
      "subs x20, x20, #0x1\n"
      "fmul z4.d, z4.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z12.d, z12.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1606344  // st1d { z4.d, z12.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 15f\n"  // bah!
      "subs x20, x20, #0x1\n"
      "fmul z5.d, z5.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z13.d, z13.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1606345  // st1d { z5.d, z13.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 15f\n"  // bah!
      "fmul z6.d, z6.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z14.d, z14.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1606346  // st1d { z6.d, z14.d }, p8, [x26]\n"
      "15:"  // Store to output array: Set Zero: Accumulator row 1 oddments: End
      "subs x25, x25, x22\n"
      "beq 24f\n"
      "b 23f\n"
      "16:"  // Store to output array: Scale Beta
      "cmp x25, x23\n"
      "add x20, %x[args], %[offset_beta]\n"
      "csel x22, x25, x23, LT\n"
      "ld1rd { z9.d }, p0/Z, [x20]\n"  // Load {name} from KernelArgs
      "mov x12, #0x0\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 18f\n"
      "17:"  // Store to output array: Scale Beta: Accumulator row 0 loop
      ".inst 0xc0c60404  // mova { z4.d-z7.d }, za0h.d[x12]\n"
      ".inst 0xc0c6042c  // mova { z12.d-z15.d }, za1h.d[x12]\n"
      ".inst 0xa1406342  // ld1d { z2.d, z10.d }, p8/Z, [x26]\n"
      "fmul z4.d, z4.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z12.d, z12.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "add x12, x12, #0x4\n"
      "fmul z5.d, z5.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z13.d, z13.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "cmp x12, x21, LSL #2\n"
      "fmul z6.d, z6.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z14.d, z14.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z7.d, z7.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z15.d, z15.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmla z4.d, p0/M, z2.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z12.d, p0/M, z10.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1606344  // st1d { z4.d, z12.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1406351  // ld1d { z17.d, z25.d }, p8/Z, [x26]\n"
      "fmla z5.d, p0/M, z17.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z13.d, p0/M, z25.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1606345  // st1d { z5.d, z13.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1406351  // ld1d { z17.d, z25.d }, p8/Z, [x26]\n"
      "fmla z6.d, p0/M, z17.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z14.d, p0/M, z25.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1606346  // st1d { z6.d, z14.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1406351  // ld1d { z17.d, z25.d }, p8/Z, [x26]\n"
      "fmla z7.d, p0/M, z17.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z15.d, p0/M, z25.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1606347  // st1d { z7.d, z15.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "blt 17b\n"
      "18:"  // Store to output array: Scale Beta: Accumulator row 0 oddments
      "cbz x20, 19f\n"
      ".inst 0xc0c60404  // mova { z4.d-z7.d }, za0h.d[x12]\n"
      ".inst 0xc0c6042c  // mova { z12.d-z15.d }, za1h.d[x12]\n"
      ".inst 0xa1406351  // ld1d { z17.d, z25.d }, p8/Z, [x26]\n"
      "fmul z4.d, z4.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z12.d, z12.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "subs x20, x20, #0x1\n"
      "fmla z4.d, p0/M, z17.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z12.d, p0/M, z25.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1606344  // st1d { z4.d, z12.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 19f\n"  // bah!
      "fmul z5.d, z5.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z13.d, z13.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1406351  // ld1d { z17.d, z25.d }, p8/Z, [x26]\n"
      "subs x20, x20, #0x1\n"
      "fmla z5.d, p0/M, z17.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z13.d, p0/M, z25.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1606345  // st1d { z5.d, z13.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 19f\n"  // bah!
      "fmul z6.d, z6.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z14.d, z14.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa0406352  // ld1d { z18.d-z19.d }, p8/Z, [x26]\n"
      "fmla z6.d, p0/M, z18.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z14.d, p0/M, z19.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1606346  // st1d { z6.d, z14.d }, p8, [x26]\n"
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
      ".inst 0xc0c60444  // mova { z4.d-z7.d }, za2h.d[x12]\n"
      ".inst 0xc0c6046c  // mova { z12.d-z15.d }, za3h.d[x12]\n"
      ".inst 0xa1406351  // ld1d { z17.d, z25.d }, p8/Z, [x26]\n"
      "fmul z4.d, z4.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z12.d, z12.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "add x12, x12, #0x4\n"
      "fmul z5.d, z5.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z13.d, z13.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "cmp x12, x21, LSL #2\n"
      "fmul z6.d, z6.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z14.d, z14.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z7.d, z7.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z15.d, z15.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmla z4.d, p0/M, z17.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z12.d, p0/M, z25.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1606344  // st1d { z4.d, z12.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1406351  // ld1d { z17.d, z25.d }, p8/Z, [x26]\n"
      "fmla z5.d, p0/M, z17.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z13.d, p0/M, z25.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1606345  // st1d { z5.d, z13.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa0406350  // ld1d { z16.d-z17.d }, p8/Z, [x26]\n"
      "fmla z6.d, p0/M, z16.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z14.d, p0/M, z17.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1606346  // st1d { z6.d, z14.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa0406352  // ld1d { z18.d-z19.d }, p8/Z, [x26]\n"
      "fmla z7.d, p0/M, z18.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z15.d, p0/M, z19.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1606347  // st1d { z7.d, z15.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "blt 20b\n"
      "21:"  // Store to output array: Scale Beta: Accumulator row 1 oddments
      "cbz x20, 22f\n"
      ".inst 0xc0c60444  // mova { z4.d-z7.d }, za2h.d[x12]\n"
      ".inst 0xc0c6046c  // mova { z12.d-z15.d }, za3h.d[x12]\n"
      ".inst 0xa1406347  // ld1d { z7.d, z15.d }, p8/Z, [x26]\n"
      "fmul z4.d, z4.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z12.d, z12.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "subs x20, x20, #0x1\n"
      "fmla z4.d, p0/M, z7.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z12.d, p0/M, z15.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1606344  // st1d { z4.d, z12.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 22f\n"  // bah!
      "fmul z5.d, z5.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z13.d, z13.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa1406347  // ld1d { z7.d, z15.d }, p8/Z, [x26]\n"
      "subs x20, x20, #0x1\n"
      "fmla z5.d, p0/M, z7.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z13.d, p0/M, z15.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1606345  // st1d { z5.d, z13.d }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 22f\n"  // bah!
      "fmul z6.d, z6.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      "fmul z14.d, z14.d, z24.d\n"  // Apply Alpha scalar to accumulator slice
      ".inst 0xa0406356  // ld1d { z22.d-z23.d }, p8/Z, [x26]\n"
      "fmla z6.d, p0/M, z22.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      "fmla z14.d, p0/M, z23.d, z9.d\n"  // Apply Beta to tmp loaded from C and add alphaAB
      ".inst 0xa1606346  // st1d { z6.d, z14.d }, p8, [x26]\n"
      "22:"  // Store to output array: Scale Beta: Accumulator row 1 oddments: End
      "subs x25, x25, x22\n"
      "beq 24f\n"
      "23:"  // Store to output array: Write End
      "24:"  // Store to output array: End
      "incd x14, ALL, MUL #2\n"
      "cmp x14, x10\n"
      "blt 2b\n"
      "incd x15, ALL, MUL #2\n"
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

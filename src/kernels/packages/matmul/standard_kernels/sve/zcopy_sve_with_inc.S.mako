## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

// x0 : size of vectors
// x1 : Pointer to vec0
// x2 : vec0 Stride
// x3 : Pointer to vec1
// x4 : vec1 Stride

  <% func_name = "zcopy_sve_kernel_with_inc" %>
  ${prologue(func_name)}
  stp x29, x30, [sp, #-16]!
  mov x29, sp

  ptrue p0.d             // ignore predication
  cntb x6                // vector len in bytes

  // Save pointer increments in bytes
  mul x10, x2, x6
  mul x11, x4, x6

  // After this x5 is the loop counter, x7 is the number of elements
  // number of bytes by which we move the x pointer in the unrolled loop
  // (the number of elements we process in the two ld1d instructions is
  // equal to the number of double lanes)
  cntd x7
  sub x5, x0, x7

  // Setup the gather/scatter address offsets
  lsl x2, x2, #1
  lsl x4, x4, #1

  index z6.d, #0, x2
  index z7.d, #1, x2
  zip1 z2.d, z6.d, z7.d
  zip2 z3.d, z6.d, z7.d
  index z6.d, #0, x4
  index z7.d, #1, x4
  zip1 z4.d, z6.d, z7.d
  zip2 z5.d, z6.d, z7.d

  lsl x0, x0, #3         // work in bytes
  mul x0, x0, x2
  add x9, x1, x0         // end pointer

  b .Lloop_cond_z

.Lloop_z:
  // Copy the element
  ld1d z0.d,  p0/z,  [x1, z2.d, lsl #3]
  ld1d z1.d,  p0/z,  [x1, z3.d, lsl #3]
  st1d z0.d,  p0,    [x3, z4.d, lsl #3]
  st1d z1.d,  p0,    [x3, z5.d, lsl #3]
  // Increment the counter & offsets
  sub x5, x5, x7
  add x1, x1, x10
  add x3, x3, x11

.Lloop_cond_z:
  cmp x5, #0
  bge .Lloop_z

.Lnoncontiguous_loop_cleanup_z:
  cmp x1, x9
  beq .Lcleanup_z

  ldr q0, [x1]
  add x1, x1, x2, lsl 3
  str q0, [x3]
  add x3, x3, x4, lsl 3

  b .Lnoncontiguous_loop_cleanup_z

.Lcleanup_z:
  ldp x29, x30, [sp], #16
  ret
  ${epilogue(func_name)}

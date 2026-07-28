## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

	<% func_name = "sdot_sve_sg_kernel" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	size  .req x0  // n, the length of the two vectors
	x_ptr .req x1  // pointer to x, the first operand vector
	y_ptr .req x2  // pointer to y, the second operand vector
	incx  .req x3  // storage spacing between elements of x
	incy  .req x4  // storage spacing between elements of y

	// per-element dot product accumulators
	fmov  z0.s, #0.0

	// build the offset indices for elements of x and y
	index z1.s, #0, w3
	index z2.s, #0, w4

	// calculate the VL increment for the base-pointers to x and y
	rdvl x5, #1
	mul  incx, x5, incx // calculate VL * incx
	mul  incy, x5, incy // calculate VL * incy

	// setup loop counter variables
	mov x6, #0
	b   .L_loop_vecdot_entry

.L_loop_vecdot:
	// load active/remaining vector elements
	ld1w {z3.s}, p0/z, [x_ptr, z1.s, sxtw #2]
	ld1w {z4.s}, p0/z, [y_ptr, z2.s, sxtw #2]
	// multiply elements of x and y, accumulate in z0
	fmla  z0.s,  p0/m, z3.s, z4.s
	// increment pointers and loop counter
	add  x_ptr, x_ptr, incx
	add  y_ptr, y_ptr, incy
	incw x6

.L_loop_vecdot_entry:
	whilelt p0.s, x6, size
	b.first .L_loop_vecdot

	// reduce elements of accumulator vector, store in s0
	ptrue p1.s
	faddv s0, p1, z0.s

	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}

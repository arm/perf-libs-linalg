## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

	<% func_name = "sdot_sve_kernel" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	size  .req x0  // n, the length of the two vectors
	x_ptr .req x1  // pointer to x, the first operand vector
	y_ptr .req x2  // pointer to y, the second operand vector

	// per-element dot product accumulators
	fmov  z0.s, #0.0
	fmov  z3.s, #0.0
	ptrue p0.s

	// setup loop counter variables
	add      size, x_ptr, size, lsl #2
	incb     x_ptr
	whilelt  p1.b, x_ptr, size
	b.nfrst .L_tail_vecdot

.L_unrolled_loop_vecdot:
	// load two vectors each of x and y elements
	ld1w {z1.s}, p0/z, [x_ptr, #-1, mul vl]
	ld1w {z4.s}, p1/z, [x_ptr]
	ld1w {z2.s}, p0/z, [y_ptr]
	ld1w {z5.s}, p1/z, [y_ptr, #1,  mul vl]

	// multiply the elements of x and y, accumulate in z0 and z3
	fmla z0.s, p0/m, z1.s, z2.s
	fmla z3.s, p0/m, z4.s, z5.s

	// increment pointers and loop
	incb    x_ptr, all, mul #2
	incb    y_ptr, all, mul #2
	whilelt p1.b,  x_ptr, size
	b.first .L_unrolled_loop_vecdot

.L_tail_vecdot:
	decb     x_ptr
	whilelt  p1.b, x_ptr, size
	b.nfrst .L_return_vecdot

	ld1w {z1.s}, p1/z, [x_ptr]
	ld1w {z2.s}, p1/z, [y_ptr]

	fmla z0.s, p1/m, z1.s, z2.s

.L_return_vecdot:
	// reduce elements of accumulator vector, store in s0
	fadd  z0.s, z0.s, z3.s
	faddv s0, p0, z0.s

	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}

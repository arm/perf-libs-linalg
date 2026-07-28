## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

% for T in [ 'd', 'ds' ]:
<%
name = T + "dot"
%>

	<% func_name = name + "_sve_sg_kernel" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	size  .req x0  // n, the length of the two vectors
	x_ptr .req x1  // pointer to x, the first operand vector
	y_ptr .req x2  // pointer to y, the second operand vector
	incx  .req x3  // storage spacing between elements of x
	incy  .req x4  // storage spacing between elements of y

	// per-element dot product accumulators
	fmov  z0.d, #0.0

	// build the offset indices for elements of x and y
	index z1.d, #0, x3
	index z2.d, #0, x4

	// calculate the VL increment for the base-pointers to x and y
	rdvl x5, #1
	%if T=='ds':
	lsr x5, x5, #1
	%endif
	mul  incx, x5, incx // calculate VL * incx
	mul  incy, x5, incy // calculate VL * incy

	// setup loop counter variables
	mov x6, #0
	b   .L_${name}_loop_vecdot_entry

.L_${name}_loop_vecdot:
	// load active/remaining vector elements
	%if T=='d':
	ld1d {z3.d}, p0/z, [x_ptr, z1.d, lsl #3]
	ld1d {z4.d}, p0/z, [y_ptr, z2.d, lsl #3]
	%else:
	ld1w {z3.d}, p0/z, [x_ptr, z1.d, lsl #2]
	ld1w {z4.d}, p0/z, [y_ptr, z2.d, lsl #2]
	// convert to double
	fcvt z3.d, p0/m, z3.s
	fcvt z4.d, p0/m, z4.s
	%endif
	// multiply elements of x and y, accumulate in z0
	fmla  z0.d,  p0/m, z3.d, z4.d
	// increment pointers and loop counter
	add  x_ptr, x_ptr, incx
	add  y_ptr, y_ptr, incy
	incd x6

.L_${name}_loop_vecdot_entry:
	whilelt p0.d, x6, size
	b.first .L_${name}_loop_vecdot

	// reduce elements of accumulator vector, store in d0
	ptrue p1.d
	faddv d0, p1, z0.d

	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}
%endfor

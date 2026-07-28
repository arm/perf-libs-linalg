## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

% for T in [ 'd', 'ds' ]:
<%
name = T + "dot"
%>
	<% func_name = name + "_sve_kernel" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	size  .req x0 // n, the length of the two vectors
	x_ptr .req x1 // pointer to x, the first operand vector
	y_ptr .req x2 // pointer to y, the second operand vector
	ctr .req x3   // loop counter variable
	veclb .req x4 // vector length of 2 vectors
	vecld .req x5 // vector length of number of doubles loaded from array

	// per-element dot product accumulators
	fmov  z0.d, #0.0
	fmov  z3.d, #0.0
	cntd ctr
	cntd vecld, all, mul #2
	%if T=='d':
	cntb veclb, all, mul #2
	%else:
	cntb veclb
	%endif
	ptrue p0.d

	//setup loop counter variables
	whilelt  p1.d, ctr, size
	b.nfrst .L_${name}_tail_vecdot

.L_${name}_unrolled_loop_vecdot:
	// load two vectors each of x and y elements
	%if T=='d':
	ld1d {z1.d}, p0/z, [x_ptr]
	ld1d {z4.d}, p1/z, [x_ptr, #1,  mul vl]
	ld1d {z2.d}, p0/z, [y_ptr]
	ld1d {z5.d}, p1/z, [y_ptr, #1,  mul vl]
	%else:
	ld1w {z1.d}, p0/z, [x_ptr]
	ld1w {z4.d}, p1/z, [x_ptr, #1, mul vl]
	ld1w {z2.d}, p0/z, [y_ptr]
	ld1w {z5.d}, p1/z, [y_ptr, #1,  mul vl]
	fcvt    z1.d, p0/m, z1.s
	fcvt    z4.d, p0/m, z4.s
	fcvt    z2.d, p0/m, z2.s
	fcvt    z5.d, p0/m, z5.s
	%endif
	// multiply the elements of x and y, accumulate in z0 and z3
	fmla z0.d, p0/m, z1.d, z2.d
	fmla z3.d, p0/m, z4.d, z5.d

	// increment pointers and loop
	add x_ptr, x_ptr, veclb
	add y_ptr, y_ptr, veclb
	add ctr, ctr, vecld // number of doubles (we load singles into double lanes)
	whilelt p1.d,  ctr, size
	b.first .L_${name}_unrolled_loop_vecdot

.L_${name}_tail_vecdot:
	decd ctr
	whilelt  p1.d, ctr, size
	b.nfrst .L_${name}_return_vecdot
	%if T=='d':
	ld1d {z1.d}, p1/z, [x_ptr]
	ld1d {z2.d}, p1/z, [y_ptr]
	%else:
	ld1w {z1.d}, p1/z, [x_ptr]
	ld1w {z2.d}, p1/z, [y_ptr]
	fcvt    z1.d, p0/m, z1.s
	fcvt    z2.d, p0/m, z2.s
	%endif
	fmla z0.d, p1/m, z1.d, z2.d

.L_${name}_return_vecdot:
	// reduce elements of accumulator vector, store in d0
	fadd  z0.d, z0.d, z3.d
	faddv d0, p0, z0.d

	ldp x29, x30, [sp], #16
	ret

	${epilogue(func_name)}
%endfor

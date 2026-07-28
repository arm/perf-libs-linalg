## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

%for conj in [ "c", "u" ]:
	<% func_name = "cdot" + conj + "_sve_sg_kernel" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	size  .req x0  // n, the length of the two vectors
	x_ptr .req x1  // pointer to x, the first operand vector
	y_ptr .req x2  // pointer to y, the second operand vector
	incx  .req x3  // storage spacing between elements of x
	incy  .req x4  // storage spacing between elements of y


	// per-element dot product accumulator
	fmov z0.s, #0.0 // Accumulator for real
	fmov z1.s, #0.0 // Accumulator for imaginary

	ptrue p2.s
	pfalse p1.b
	zip1 p1.s, p1.s, p2.s // Mask for imaginary components

	// build the offset indices for elements of x and y
	index z2.d, #0, incx
	index z3.d, #0, incy

	// calculate the VL increment for the base-pointers to x and y
	rdvl x5, #1
	mul  incx, x5, incx // calculate VL * incx
	mul  incy, x5, incy // calculate VL * incy

	// setup loop counter variables
	mov x6, #0
	b   .L_loop_vecdot_entry_${conj}

.L_loop_vecdot_${conj}:
	// Load pairs of (real, imag) components
	ld1d {z4.d}, p0/z, [x_ptr, z2.d, lsl #3]
	ld1d {z5.d}, p0/z, [y_ptr, z3.d, lsl #3]

%if conj=="u":
	// Negate the imaginary component of x
	fneg z4.s, p1/m, z4.s
%endif

	fmla z0.s, p2/m, z4.s, z5.s // real * real, imag * imag
	// Negate the imaginary component of x and rotate
	fneg z4.s, p1/m, z4.s
	trn1 z6.s, z4.s, z4.s
	trn2 z4.s, z4.s, z6.s
	fmla z1.s, p2/m, z4.s, z5.s

	// increment pointers and loop counter
	add  x_ptr, x_ptr, incx
	add  y_ptr, y_ptr, incy
	incd x6

.L_loop_vecdot_entry_${conj}:
	whilelt p0.d, x6, size
	b.first .L_loop_vecdot_${conj}

	// reduce to scalar, storing real part in s0, imag in s1
	faddv s0, p2, z0.s
	faddv s1, p2, z1.s

	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}
%endfor

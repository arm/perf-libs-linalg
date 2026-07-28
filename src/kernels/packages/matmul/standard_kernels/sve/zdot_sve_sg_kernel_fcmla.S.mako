## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

%for conj, imm in [ ("c", "#270"), ("u", "#90") ]:
	<% func_name = "zdot" + conj + "_sve_sg_kernel_fcmla" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	size  .req x0  // n, the length of the two vectors
	x_ptr .req x1  // pointer to x, the first operand vector
	y_ptr .req x2  // pointer to y, the second operand vector
	incx  .req x3  // storage spacing between elements of x
	incy  .req x4  // storage spacing between elements of y

	// complex numbers have two elements each, so we process 2n elements
	lsl size, size, #1

	// per-element dot product accumulator
	fmov z0.d, #0.0

	// build the offset indices for real parts of elements of x
	lsl x5, incx, #1
	index z1.d, #0, x5
	// the imaginary parts are adjacent to the real parts
	index z2.d, #1, x5
	// zip the real and imaginary indices together
	zip1 z1.d, z1.d, z2.d

	// build the real and imaginary offset indices for elements of y
	lsl x5, incy, #1
	index z2.d, #0, x5
	index z3.d, #1, x5
	zip1  z2.d, z2.d, z3.d

	// calculate the VL increment for the base-pointers to x and y
	rdvl x5, #1
	mul  incx, x5, incx // calculate VL * incx
	mul  incy, x5, incy // calculate VL * incy

	// construct alternating predicates to manipulate real and imag parts
	ptrue p1.d
	pfalse p2.b
	zip1 p3.d, p1.d, p2.d
	zip1 p4.d, p2.d, p1.d

	// setup loop counter variables
	mov x6, #0
	b   .L_loop_vecdot_entry_fcmla_${conj}

.L_loop_vecdot_fcmla_${conj}:
	// load active/remaining vector elements
	ld1d {z3.d}, p0/z, [x_ptr, z1.d, lsl #3]
	ld1d {z4.d}, p0/z, [y_ptr, z2.d, lsl #3]
	// multiply complex elements of x and y, accumulate in z0
	fcmla z0.d,  p0/m, z3.d, z4.d, #0
	fcmla z0.d,  p0/m, z3.d, z4.d, ${imm}
	// increment pointers and loop counter
	add  x_ptr, x_ptr, incx
	add  y_ptr, y_ptr, incy
	incd x6

.L_loop_vecdot_entry_fcmla_${conj}:
	whilelt p0.d, x6, size
	b.first .L_loop_vecdot_fcmla_${conj}

	// use predication to reduce real and imag parts in accumulator
	// store real and imaginary results in d0 and d1 respectively
	faddv  d1, p4, z0.d
	faddv  d0, p3, z0.d

	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}
%endfor

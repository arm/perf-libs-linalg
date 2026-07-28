## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

%for conj, imm in [ ("c", "#270"), ("u", "#90") ]:
	<% func_name = "cdot" + conj + "_sve_sg_kernel_fcmla" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	size  .req x0  // n, the length of the two vectors
	x_ptr .req x1  // pointer to x, the first operand vector
	y_ptr .req x2  // pointer to y, the second operand vector
	incx  .req x3  // storage spacing between elements of x
	incy  .req x4  // storage spacing between elements of y

	// per-element dot product accumulators
	// CDOTU uses one accumulator
	fmov z0.s, #0.0
%if conj=="c":
	// CDOTC uses two accumulators: one for real, one for imaginary
	fmov z1.s, #0.0
%endif

	// build the offset indices for elements of x and y
	index z1.d, #0, x3
	index z2.d, #0, x4

	// calculate the VL increment for the base-pointers to x and y
	rdvl x5, #1
	mul  incx, x5, incx // calculate VL * incx
	mul  incy, x5, incy // calculate VL * incy

	// needed to multiply both real and imag components within loop
	// p0.d would only have the bit for the real component set
	ptrue p1.s

	// construct alternating predicates to manipulate real and imag parts
	pfalse p2.b
	zip1 p3.s, p1.s, p2.s
	zip1 p4.s, p2.s, p1.s

	// setup loop counter variables
	mov x6, #0
	b   .L_loop_vecdot_entry_fcmla_${conj}

.L_loop_vecdot_fcmla_${conj}:
	// load active/remaining vector elements
	ld1d {z3.d}, p0/z, [x_ptr, z1.d, lsl #3]
	ld1d {z4.d}, p0/z, [y_ptr, z2.d, lsl #3]
	// multiply complex elements of x and y, accumulate in z0
	fcmla z0.s,  p1/m, z3.s, z4.s, #0
	fcmla z0.s,  p1/m, z3.s, z4.s, ${imm}
	// increment pointers and loop counter
	add  x_ptr, x_ptr, incx
	add  y_ptr, y_ptr, incy
	incd x6

.L_loop_vecdot_entry_fcmla_${conj}:
	whilelt p0.d, x6, size
	b.first .L_loop_vecdot_fcmla_${conj}

	// use predication to reduce real and imag parts in accumulator
	// store real and imaginary results in s0 and s1 respectively
	faddv s1, p4, z0.s
	faddv s0, p3, z0.s

	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}
%endfor


## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

%for conj, imm in [ ("c", "#270"), ("u", "#90") ]:
	<% func_name = "zdot" + conj + "_sve_kernel_fcmla" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	size  .req x0  // n, the length of the two vectors
	x_ptr .req x1  // pointer to x, the first operand vector
	y_ptr .req x2  // pointer to y, the second operand vector

	// per-element dot product accumulators
	fmov  z0.d, #0.0
	fmov  z3.d, #0.0
	ptrue p0.d

	// setup loop counter variables
	// lsl #4 because complex elements have two elements each
	add	 size, x_ptr, size, lsl #4
	incb	 x_ptr
	whilelt	 p1.b, x_ptr, size
	b.nfrst .L_tail_vecdot_fcmla_${conj}

.L_unrolled_loop_vecdot_fcmla_${conj}:
	// load two vectors each of x and y elements
	ld1d {z1.d}, p0/z, [x_ptr, #-1, mul vl]
	ld1d {z4.d}, p1/z, [x_ptr]
	ld1d {z2.d}, p0/z, [y_ptr]
	ld1d {z5.d}, p1/z, [y_ptr, #1,	mul vl]

	// multiply the elements of x and y, accumulate in z0 and z3
	fcmla z0.d, p0/m, z1.d, z2.d, #0
	fcmla z0.d, p0/m, z1.d, z2.d, ${imm}
	fcmla z3.d, p0/m, z4.d, z5.d, #0
	fcmla z3.d, p0/m, z4.d, z5.d, ${imm}

	// increment pointers and loop
	incb	x_ptr, all, mul #2
	incb	y_ptr, all, mul #2
	whilelt p1.b,  x_ptr, size
	b.first .L_unrolled_loop_vecdot_fcmla_${conj}

.L_tail_vecdot_fcmla_${conj}:
	decb	 x_ptr
	whilelt	 p1.b, x_ptr, size
	b.nfrst .L_return_vecdot_fcmla_${conj}

	ld1d {z1.d}, p1/z, [x_ptr]
	ld1d {z2.d}, p1/z, [y_ptr]

	fcmla z0.d, p1/m, z1.d, z2.d, #0
	fcmla z0.d, p1/m, z1.d, z2.d, ${imm}

.L_return_vecdot_fcmla_${conj}:
	// deinterleave real and imaginary components
	uzp1 z1.d, z0.d, z3.d
	uzp2 z2.d, z0.d, z3.d

	// reduce to scalar, storing real part in d0, imag in d1
	faddv d0, p0, z1.d
	faddv d1, p0, z2.d

	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}
%endfor

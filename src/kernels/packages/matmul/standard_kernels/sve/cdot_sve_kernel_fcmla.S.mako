## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

%for conj, imm in [ ("c", "#270"), ("u", "#90") ]:
	<% func_name = "cdot" + conj + "_sve_kernel_fcmla" %>
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
	// lsl #3 because complex elements have two elements each
	add      size, x_ptr, size, lsl #3
	incb     x_ptr
	whilelt  p1.b, x_ptr, size
	b.nfrst .L_tail_vecdot_fcmla_${conj}

.L_unrolled_loop_vecdot_fcmla_${conj}:
	// load two vectors each of x and y elements
	ld1w {z1.s}, p0/z, [x_ptr, #-1, mul vl]
	ld1w {z4.s}, p1/z, [x_ptr]
	ld1w {z2.s}, p0/z, [y_ptr]
	ld1w {z5.s}, p1/z, [y_ptr, #1,  mul vl]

	// multiply the elements of x and y, accumulate in z0 and z3
	fcmla z0.s, p0/m, z1.s, z2.s, #0
	fcmla z0.s, p0/m, z1.s, z2.s, ${imm}
	fcmla z3.s, p0/m, z4.s, z5.s, #0
	fcmla z3.s, p0/m, z4.s, z5.s, ${imm}

	// increment pointers and loop
	incb    x_ptr, all, mul #2
	incb    y_ptr, all, mul #2
	whilelt p1.b,  x_ptr, size
	b.first .L_unrolled_loop_vecdot_fcmla_${conj}

.L_tail_vecdot_fcmla_${conj}:
	decb     x_ptr
	whilelt  p1.b, x_ptr, size
	b.nfrst .L_return_vecdot_fcmla_${conj}

	ld1w {z1.s}, p1/z, [x_ptr]
	ld1w {z2.s}, p1/z, [y_ptr]

	fcmla z0.s, p1/m, z1.s, z2.s, #0
	fcmla z0.s, p1/m, z1.s, z2.s, ${imm}

.L_return_vecdot_fcmla_${conj}:
	// deinterleave real and imaginary components
	uzp1 z1.s, z0.s, z3.s
	uzp2 z2.s, z0.s, z3.s

	// reduce to scalar, storing real part in s0, imag in s1
	faddv s0, p0, z1.s
	faddv s1, p0, z2.s

	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}
%endfor


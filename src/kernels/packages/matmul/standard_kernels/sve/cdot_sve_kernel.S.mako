## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

%for conj, fml1, fml2 in [ ("c", "fmla", "fmls"), ("u", "fmls", "fmla") ]:
	<% func_name = "cdot" + conj + "_sve_kernel" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	size  .req x0  // n, the length of the two vectors
	x_ptr .req x1  // pointer to x, the first operand vector
	y_ptr .req x2  // pointer to y, the second operand vector

	// per-element dot product accumulators
	fmov  z0.s, #0.0 // Sum of real components
	fmov  z1.s, #0.0 // Sum of imaginary components

	// Set up loop counter
	mov x3, #0
	b .L_loop_cond_${conj}

.L_loop_vecdot_${conj}:
	// load two vectors each of x and y elements
	ld2w {z2.s, z3.s}, p0/z, [x_ptr]
	ld2w {z4.s, z5.s}, p0/z, [y_ptr]

	fmla z0.s, p0/m, z2.s, z4.s // real * real
	${fml1} z0.s, p0/m, z3.s, z5.s // +/-imag * imag
	fmla z1.s, p0/m, z2.s, z5.s // real * imag
	${fml2} z1.s, p0/m, z3.s, z4.s // +/-imag * real

	// increment pointers and loop
	incb x_ptr, all, mul #2
	incb y_ptr, all, mul #2
	incw x3

.L_loop_cond_${conj}:
	whilelt p0.s, x3, x0
	b.first .L_loop_vecdot_${conj}

	// reduce to scalar, storing real part in s0, imag in s1
	ptrue p0.s
	faddv s0, p0, z0.s
	faddv s1, p0, z1.s

	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}
%endfor


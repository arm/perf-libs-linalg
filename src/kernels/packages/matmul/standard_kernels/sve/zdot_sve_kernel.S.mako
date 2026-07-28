## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

%for conj, fml1, fml2 in [ ("c", "fmla", "fmls"), ("u", "fmls", "fmla") ]:
	<% func_name = "zdot" + conj + "_sve_kernel" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	size	.req x0	 // n, the length of the two vectors
	x_ptr	.req x1	 // pointer to x, the first operand vector
	y_ptr	.req x2	 // pointer to y, the second operand vector

	// per-element dot product accumulators
	fmov  z0.d, #0.0 // Stores real components
	fmov  z1.d, #0.0 // Stores imaginary components
	mov x6, #0 // Loop counter

	cntd x5
	cmp x5, size, lsl #1 // If there is only enough to fill a single vector
	b.ge .L_small_vecdot_${conj}

	b .L_loop_vecdot_entry_${conj}

.L_loop_vecdot_${conj}:
	// load two vectors each of x and y elements
	ld2d {z2.d, z3.d}, p0/z, [x_ptr]
	ld2d {z4.d, z5.d}, p0/z, [y_ptr]

	// Multiply the real and imaginary components of the x and y
	// vectors, and accumulate into z0 and z1
	fmla z0.d, p0/m, z2.d, z4.d // Real x real
	${fml1} z0.d, p0/m, z3.d, z5.d // +/-imag x imag
	fmla z1.d, p0/m, z2.d, z5.d // real x imag
	${fml2} z1.d, p0/m, z3.d, z4.d // imag x real

	// increment pointers and loop
	incb x_ptr, all, mul #2
	incb y_ptr, all, mul #2
	incd x6

.L_loop_vecdot_entry_${conj}:
	whilelt	 p0.d, x6, size
	b.first .L_loop_vecdot_${conj}
	b .L_return_vecdot_${conj}

.L_small_vecdot_${conj}:
	lsl size, size, #1
	whilelt	 p3.d, x6, size
	b.nfrst .L_return_vecdot_${conj}
	ld1d {z2.d}, p3/z, [x_ptr]
	ld1d {z3.d}, p3/z, [y_ptr]

	trn1 z4.d, z2.d, z2.d // real components of x
	trn2 z5.d, z2.d, z4.d // rotated x
	ptrue p1.d
	pfalse p2.b

%if conj == "c":
	zip1 p1.d, p1.d, p2.d // Mask for imaginary components in z5
	fneg z5.d, p1/m, z5.d // Negate imaginary components of x in z5
%else:  # conj == "u"
	zip1 p1.d, p2.d, p1.d // Mask for imaginary components in z2
	fneg z2.d, p1/m, z2.d // Negate imaginary components of x in z2
%endif

	fmla z0.d, p3/m, z2.d, z3.d // All real components of sum
	fmla z1.d, p3/m, z5.d, z3.d // All imaginary components of sum

.L_return_vecdot_${conj}:
	// reduce to scalar, storing real part in d0, imag in d1
	ptrue p4.d
	faddv d0, p4, z0.d
	faddv d1, p4, z1.d

	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}
%endfor

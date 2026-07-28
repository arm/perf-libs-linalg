## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

%for conj, fml1, fml2 in [ ("c", "fmla", "fmls"), ("u", "fmls", "fmla") ]:
	<% func_name = "zdot" + conj + "_sve_sg_kernel" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	size	   .req x0  // n, the length of the two vectors
	x_ptr	   .req x1  // pointer to x, the first operand vector
	y_ptr	   .req x2  // pointer to y, the second operand vector
	incx	   .req x3  // storage spacing between elements of x
	incy	   .req x4  // storage spacing between elements of y
	ind_real_x .req z1  // Indices of real components of x to read in
	ind_imag_x .req z2  // Indices of imaginary components of x to read in
	ind_real_y .req z3  // Indices of real components of y to read in
	ind_imag_y .req z4  // Indices of imaginary components of y to read in
	real_x	   .req z5  // Real part of x
	imag_x	   .req z6  // Imaginary part of x
	real_y	   .req z7  // Real part of y
	imag_y	   .req z16  // Imaginary part of y
	real_r	   .req z17  // Real part of the result
	imag_r	   .req z18 // imaginary part of the result

	// per-element dot product accumulator
	fmov real_r.d, #0.0 // real parts
	fmov imag_r.d, #0.0 // imaginary parts

	// build the offset indices for real parts of elements of x
	lsl x5, incx, #1
	index ind_real_x.d, #0, x5
	// the imaginary parts are adjacent to the real parts
	index ind_imag_x.d, #1, x5

	// build the real and imaginary offset indices for elements of y
	lsl x5, incy, #1
	index ind_real_y.d, #0, x5
	index ind_imag_y.d, #1, x5

	// calculate the VL increment for the base-pointers to x and y
	// double the vector length because each complex number is 2 components
	rdvl x5, #2
	mul  incx, x5, incx // calculate VL * incx
	mul  incy, x5, incy // calculate VL * incy

	// an always true predicate for storing results
	ptrue p1.d

	// setup loop counter variables
	mov x5, #0
	b .L_loop_vecdot_entry_${conj}

.L_loop_vecdot_${conj}:
	// load active/remaining vector elements
	ld1d {real_x.d}, p0/z, [x_ptr, ind_real_x.d, lsl #3]
	ld1d {imag_x.d}, p0/z, [x_ptr, ind_imag_x.d, lsl #3]
	ld1d {real_y.d}, p0/z, [y_ptr, ind_real_y.d, lsl #3]
	ld1d {imag_y.d}, p0/z, [y_ptr, ind_imag_y.d, lsl #3]

	fmla real_r.d, p0/m, real_x.d, real_y.d // real x real
	${fml1} real_r.d, p0/m, imag_x.d, imag_y.d // +/-imag x imag
	fmla imag_r.d, p0/m, real_x.d, imag_y.d // real x imag
	${fml2} imag_r.d, p0/m, imag_x.d, real_y.d // +/-imag x real

	// increment pointers and loop counter
	add  x_ptr, x_ptr, incx
	add  y_ptr, y_ptr, incy

	incd x5

.L_loop_vecdot_entry_${conj}:
	whilelt p0.d, x5, size
	b.first .L_loop_vecdot_${conj}

	faddv  d0, p1, real_r.d
	faddv  d1, p1, imag_r.d

	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}
%endfor

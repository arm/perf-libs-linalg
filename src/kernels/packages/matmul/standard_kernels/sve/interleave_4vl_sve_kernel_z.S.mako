## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

// 4x vector length interleave kernel for 16-byte (e.g. complex<double>) elements
	<% func_name = "interleave_4vl_sve_kernel_z" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	src_cntg .req x0 // number of columns in source matrix
	src_strd .req x1 // number of rows in source matrix
	src_ptr  .req x2 // pointer to source matrix
	src_ld   .req x3 // stride between start of each contiguous row
	dst_cntg .req x4 // number of columns in destination matrix
	dst_strd .req x5 // number of rows in destination matrix
	dst_ptr  .req x6 // pointer to destination matrix
	dst_ld   .req x7 // stride between start of each contiguous row
	// interleave_rows assumed to be 4*VL

	// always true predicate for first load and both stores
        ptrue p1.b
	ptrue p2.b
	ptrue p3.b
	ptrue p4.b
	ptrue p5.b
	// setup a zero vector for padding
	mov   z0.b, #0

	// calculate the VL increment (in bytes) for the src base-pointer
	rdvl x9, #1             // length of vector in bytes
	mul  x9, x9, src_ld

	// set src_ld, dst_ld and dst_cntg as byte sizes
	lsl src_ld, src_ld, #4
	lsl dst_ld, dst_ld, #4
	lsl dst_cntg, dst_cntg, #4
	// set dst_cntg as a pointer
	add dst_cntg, dst_cntg, dst_ptr

	// build offset indices (in bytes) for matrix elements
	index z1.d, #0, src_ld  // indexes of real parts of each element
	index z2.d, #1, src_ld	// indexes of imag parts of each element
	zip1  z1.d, z1.d, z2.d	// zip together.
	// z1 now contains the offset indices for the first interleave vector.
	// For the second, third and fourth interleave vectors, add multiples of
	// VL offset (1, 2 and 3 respectively).
	lsl   x10,  x9, #1     // calculate 2*VL byte offset
	add   x11,  x9, x10    // calculate 3*VL byte offset
	mov   z2.d, x9
	mov   z3.d, x10
	mov   z4.d, x11
	add   z2.d, z2.d, z1.d // offset indices for second interleave vector
	add   z3.d, z3.d, z1.d // offset indices for third interleave vector
	add   z4.d, z4.d, z1.d // offset indices for fourth interleave vector

	// calculate panel size (4*VL increment) in bytes
	lsl x9, x9, #2
	panel_size .req x9

	// each 16-byte complex element contains two 8-byte elements. We can't
	// increment a src row counter easily by VL/16 so double src_strd
	// instead.
	lsl src_strd, src_strd, #1
	// use x13 as a counter from 0 up to src_strd
	src_row_count .req x13
	mov src_row_count, xzr

	b   .L_interleave_loop_entry_first_panel

.L_interleave_panel_loop:
	ld1w {z5.d}, p2/z, [x10, z1.d]
	ld1w {z6.d}, p3/z, [x10, z2.d]
	ld1w {z7.d}, p4/z, [x10, z3.d]
	ld1w {z8.d}, p5/z, [x10, z4.d]
	st1w {z5.d}, p1,   [x11]
	st1w {z6.d}, p1,   [x11, #1, mul vl]
	st1w {z7.d}, p1,   [x11, #2, mul vl]
	st1w {z8.d}, p1,   [x11, #3, mul vl]
	// increment src and dst pointers
	add  x10, x10, #16
	incb x11, all, mul #4
	// decrement panel column count and loop if more columns
	subs x12, x12, #1
	b.ne .L_interleave_panel_loop

	// set predicate to zero remaining elements
	whilelt p0.b, x11, dst_cntg
	b.nfrst .L_interleave_loop_entry

.L_zero_pad_remainder_loop:
	// zero-pad the rest of the row in the dst matrix
	st1b    {z0.b}, p0, [x11]
	incb    x11
	whilelt p0.b, x11, dst_cntg
	b.first .L_zero_pad_remainder_loop

.L_interleave_loop_entry:
	add     dst_ptr, dst_ptr, dst_ld
	add     src_ptr, src_ptr, panel_size
	add     dst_cntg, dst_cntg, dst_ld
	sub     dst_strd, dst_strd, #1
.L_interleave_loop_entry_first_panel:
	mov     x10, src_ptr
	mov     x11, dst_ptr
	mov     x12, src_cntg
	incd    src_row_count, all, mul #4
	cmp     src_row_count, src_strd
	b.lo    .L_interleave_panel_loop
	// last panel tail case; set load predicates
	decd    src_row_count
	whilelt p5.d, src_row_count, src_strd
	decd    src_row_count
	whilelt p4.d, src_row_count, src_strd
	decd    src_row_count
	whilelt p3.d, src_row_count, src_strd
	decd    src_row_count
	whilelt p2.d, src_row_count, src_strd
	incd    src_row_count, all, mul #4
	b.first .L_interleave_panel_loop

	// test if dst rows remaining to zero-pad: if not, return
	cmp     dst_strd, #0
	b.eq    .L_return
	whilelt p0.b, x11, dst_cntg

.L_zero_pad_row_loop:
	// zero-pad the remaining rows in the dst matrix
	st1b    {z0.b}, p0, [x11]
	incb    x11
	whilelt p0.b, x11, dst_cntg
	b.first .L_zero_pad_row_loop

.L_zero_row_entry:
	add     dst_ptr, dst_ptr, dst_ld
	add     dst_cntg, dst_cntg, dst_ld
	mov     x11, dst_ptr
	whilelt p0.b, x11, dst_cntg
	subs    dst_strd, dst_strd, #1
	b.ne    .L_zero_pad_row_loop

.L_return:
	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}
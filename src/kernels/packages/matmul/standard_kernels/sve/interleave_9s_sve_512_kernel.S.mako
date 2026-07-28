## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

// interleave kernel for 9x 4-byte (e.g. float) elements
	<% func_name = "interleave_9s_sve_512_kernel" %>
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

	// set the interleave width to 36 (9x4) bytes
	interleave_width .req x8
	mov  interleave_width, xzr
	incb interleave_width, vl4, mul #9
	// predicates to control loads and stores
	whilelt p1.b, xzr, interleave_width
	mov  p2.b, p1.b
	// calculate panel size (9x 4-byte element increment) in bytes
	panel_size .req x13
	mul  panel_size, interleave_width, src_ld
	// setup a zero vector for padding
	mov  z0.b, #0

	// calculate the interleave increment for the src base-pointer
	rdvl x9, #1             // length of vector in bytes
	mul  x9, x9, src_ld

	// set src_ld, dst_ld and dst_cntg as byte sizes
	lsl src_ld, src_ld, #2
	lsl dst_ld, dst_ld, #2
	lsl dst_cntg, dst_cntg, #2
	// set dst_cntg as a pointer
	add dst_cntg, dst_cntg, dst_ptr

	// build offset indices (in bytes) for matrix elements
	index z1.s, #0, w3  // first vector of interleave

	// use x14 as a counter from 0 up to src_strd
	src_row_count .req x14
	mov src_row_count, xzr

	b   .L_interleave_loop_entry_first_panel

.L_interleave_panel_loop:
	ld1w {z2.s}, p2/z, [x10, z1.s, sxtw]
	st1w {z2.s}, p1,   [x11]
	// increment src and dst pointers
	add  x10, x10, #4
	incb x11, vl4, mul #9
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
	add 	src_row_count, src_row_count, #9
	cmp     src_row_count, src_strd
	b.lt    .L_interleave_panel_loop
	// last panel tail case; set load predicates
	sub	x15, src_row_count, #9
	whilelt p2.s, x15, src_strd
	b.first .L_interleave_panel_loop

	// test if dst rows remaining to zero-pad: if not, return
	cbz	dst_strd, .L_return
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

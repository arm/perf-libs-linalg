## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

// interleave kernel for 9x 8-byte (e.g. double or complex<float>) elements
	<% func_name = "interleave_9d_sve_512_kernel" %>
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
	// interleave_rows is 9x 8-byte elements

	// predicates to control loads and stores
	ptrue p1.b
	ptrue p2.b
	// setup a zero vector for padding
	mov   z0.b, #0

	// calculate the interleave increment for the src base-pointer
	rdvl x9, #1             // length of vector in bytes
	mul  x9, x9, src_ld

	// calculate panel size (9x 8-byte element increment) in bytes
	mov x13, #72
	mul x13, x13, src_ld
	panel_size .req x13

	// set src_ld, dst_ld and dst_cntg as byte sizes
	lsl src_ld, src_ld, #3
	lsl dst_ld, dst_ld, #3
	lsl dst_cntg, dst_cntg, #3
	// set dst_cntg as a pointer
	add dst_cntg, dst_cntg, dst_ptr

	// build offset indices (in bytes) for matrix elements
	index z1.d, #0, src_ld  // first vector of interleave
	// 9th element of interleave will be handled as a scalar

	// use x14 as a counter from 0 up to src_strd
	src_row_count .req x14
	mov src_row_count, xzr

	b   .L_interleave_loop_entry_first_panel

.L_interleave_panel_loop_8:
	ld1d {z2.d}, p2/z, [x10, z1.d]
	st1d {z2.d}, p1,   [x11]
	// increment src and dst pointers
	add  x10, x10, #8
	incb x11, vl8, mul #9
	// decrement panel column count and loop if more columns
	subs x12, x12, #1
	b.ne .L_interleave_panel_loop_8
	b    .L_interleave_panel_loop_end

.L_interleave_panel_loop_9:
	ld1d {z2.d}, p2/z, [x10, z1.d]
	ldr  d3,           [x10, x9]
	st1d {z2.d}, p1,   [x11]
	str  d3,           [x11, #64]
	// increment src and dst pointers
	add  x10, x10, #8
	incb x11, vl8, mul #9
	// decrement panel column count and loop if more columns
	subs x12, x12, #1
	b.ne .L_interleave_panel_loop_9

.L_interleave_panel_loop_end:
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
	b.le    .L_interleave_panel_loop_9
	// last panel tail case (<9 rows); set load predicates
	// in this case we only need to consider the first 8 elements
	sub	x15, src_row_count, #9
	whilelt p2.d, x15, src_strd
	b.first .L_interleave_panel_loop_8

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

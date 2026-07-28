## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

	<% func_name = "tran_interleave_5z_sve_256_kernel" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	// function arguments
	src_cntg .req x0 // number of columns in source matrix
	src_strd .req x1 // number of rows in source matrix
	src_ptr  .req x2 // pointer to source matrix
	src_ld   .req x3 // stride between state of each contiguous row
	dst_cntg .req x4 // number of columns in destination matrix
	dst_strd .req x5 // number of rows in destination matrix
	dst_ptr  .req x6 // pointer to destination matrix
	dst_ld   .req x7 // stride between state of each contiguous row

	cbz src_strd, .L_return

	// setup interleave_width predicate. Size 80 bytes (5*16-byte elements)
	interleave_width .req x9
	mov  interleave_width, xzr
	incb interleave_width, vl16, mul #5
	ptrue p1.b
	ptrue p2.b
	ptrue p3.b, vl16
	ptrue p4.b
	ptrue p5.b
	ptrue p6.b, vl16
	// setup a zero vector for padding
	mov   z0.b, #0

	// make src_ld, dst_ld src_cntg and dst_cntg byte sizes
	lsl src_ld,   src_ld,   #4
	lsl dst_ld,   dst_ld,   #4
	lsl src_cntg, src_cntg, #4
	lsl dst_cntg, dst_cntg, #4
	// set src_cntg and dst_cntg as pointers
	add src_cntg, src_cntg, src_ptr
	add dst_cntg, dst_cntg, dst_ptr

	b   .L_trn_int_loop_entry_first_panel

.L_trn_int_panel_loop:
	ld1b {z1.b}, p4/z, [x10]
	ld1b {z2.b}, p5/z, [x10, #1, mul vl]
	ld1b {z3.b}, p6/z, [x10, #2, mul vl]
	st1b {z1.b}, p1,   [x11]
	st1b {z2.b}, p2,   [x11, #1, mul vl]
	st1b {z3.b}, p3,   [x11, #2, mul vl]
	// increment src and dst pointers
	add  x10, x10,   src_ld
	incb x11, vl16, mul #5
	// decrement panel row count and loop if more rows
	subs x12, x12, #1
	b.ne .L_trn_int_panel_loop

	// set predicate to zero remaining elements
	whilelt p0.b, x11, dst_cntg
	b.nfrst .L_trn_int_loop_entry

.L_zero_pad_remainder_loop:
	// zero-pad the rest of the row in the dst matrix
	st1b    {z0.b}, p0, [x11]
	incb    x11
	whilelt p0.b, x11, dst_cntg
	b.first .L_zero_pad_remainder_loop

.L_trn_int_loop_entry:
	add     dst_ptr,  dst_ptr,  dst_ld
	add     dst_cntg, dst_cntg, dst_ld
	sub     dst_strd, dst_strd, #1
	incb    src_ptr, vl16, mul #5
.L_trn_int_loop_entry_first_panel:
	mov     x10, src_ptr
	mov     x11, dst_ptr
	mov     x12, src_strd
	// if more than interleave_width bytes remaining, loop
	sub 	x13, src_cntg, src_ptr
	cmp 	x13, interleave_width
	b.gt    .L_trn_int_panel_loop
	// else last panel tail case:
	// set predicate governing number of elements remaining and loop
	incb    src_ptr, vl16, mul #4
        whilelt p6.b, src_ptr, src_cntg
	decb    src_ptr
	whilelt p5.b, src_ptr, src_cntg
	decb    src_ptr
	whilelt p4.b, src_ptr, src_cntg
	b.first .L_trn_int_panel_loop

	// if dst rows remaining, zero-pad; else return
	cbz     dst_strd, .L_return
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

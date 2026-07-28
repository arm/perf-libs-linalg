## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

// 4x vector length transpose-interleave kernel for 16-byte (e.g. complex<double>) elements
	<% func_name = "tran_interleave_4vl_sve_kernel_z" %>
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

	cbz src_strd, .L_return

	// initialize predicates for loads and stores
	ptrue p1.b
	ptrue p2.b
	ptrue p3.b
	ptrue p4.b
	ptrue p5.b
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

	b .L_trn_int_loop_entry_first_panel

.L_trn_int_panel_loop:
	ld1b {z1.b}, p2/z, [x9, #-4, mul vl]
	ld1b {z2.b}, p3/z, [x9, #-3, mul vl]
	ld1b {z3.b}, p4/z, [x9, #-2, mul vl]
	ld1b {z4.b}, p5/z, [x9, #-1, mul vl]
	st1b {z1.b}, p1,   [x10]
	st1b {z2.b}, p1,   [x10, #1, mul vl]
	st1b {z3.b}, p1,   [x10, #2, mul vl]
	st1b {z4.b}, p1,   [x10, #3, mul vl]
	// increment src and dst pointers
	add  x9,  x9,  src_ld
	incb x10, all, mul #4
	// decrement panel row count and loop if more rows
	subs x11, x11, #1
	b.ne .L_trn_int_panel_loop

	// set predicate to zero remaining elements
	whilelt p0.b, x10, dst_cntg
	b.nfrst .L_trn_int_loop_entry

.L_zero_pad_remainder_loop:
	// zero-pad the rest of the row in the dst matrix
	st1b    {z0.b}, p0, [x10]
	incb    x10
	whilelt p0.b, x10, dst_cntg
	b.first .L_zero_pad_remainder_loop

.L_trn_int_loop_entry:
	add     dst_ptr, dst_ptr, dst_ld
	add     dst_cntg, dst_cntg, dst_ld
	sub     dst_strd, dst_strd, #1
.L_trn_int_loop_entry_first_panel:
	incb    src_ptr, all, mul #4
	mov     x9, src_ptr
	mov     x10, dst_ptr
	mov     x11, src_strd
	cmp     src_ptr, src_cntg
	b.lo    .L_trn_int_panel_loop
	// last panel tail case; set load predicates and loop
	decb    x9
	whilelt p5.b, x9, src_cntg
	decb    x9
	whilelt p4.b, x9, src_cntg
	decb    x9
	whilelt p3.b, x9, src_cntg
	decb    x9
	whilelt p2.b, x9, src_cntg
	incb    x9, all, mul #4
	b.first .L_trn_int_panel_loop

	// test if dst rows remaining to zero-pad: if not, return
	cmp     dst_strd, #0
	b.eq    .L_return
	whilelt p0.b, x10, dst_cntg

.L_zero_pad_row_loop:
	// zero-pad the remaining rows in the dst matrix
	st1b    {z0.b}, p0, [x10]
	incb    x10
	whilelt p0.b, x10, dst_cntg
	b.first .L_zero_pad_row_loop

.L_zero_row_entry:
	add     dst_ptr, dst_ptr, dst_ld
	add     dst_cntg, dst_cntg, dst_ld
	mov     x10, dst_ptr
	whilelt p0.b, x10, dst_cntg
	subs    dst_strd, dst_strd, #1
	b.ne    .L_zero_pad_row_loop

.L_return:
	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}

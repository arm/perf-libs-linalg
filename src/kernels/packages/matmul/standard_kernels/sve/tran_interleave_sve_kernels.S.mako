## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

%for elem, prec, vl, shift, pattern in [ ("5", "z", "512", 4, ", vl16"), ("8", "d", "256", 3, ""), ("12", "s", "256", 2, ", vl16") ]:
	<% sfx = "_" + elem + prec + "_" + vl %>
	<% func_name = "tran_interleave_" + elem + prec + "_sve_"+ vl + "_kernel" %>
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

	cbz src_strd, .L_return${sfx}

	// setup interleave_width predicate
	interleave_width .req x9
	mov  interleave_width, xzr
	incb interleave_width, vl16, mul #${shift+1}
	ptrue p1.b
	ptrue p2.b ${pattern}
	ptrue p3.b
	ptrue p4.b ${pattern}
	// setup a zero vector for padding
	mov   z0.b, #0

	// make src_ld, dst_ld src_cntg and dst_cntg byte sizes
	lsl src_ld,   src_ld,   #${shift}
	lsl dst_ld,   dst_ld,   #${shift}
	lsl src_cntg, src_cntg, #${shift}
	lsl dst_cntg, dst_cntg, #${shift}
	// set src_cntg and dst_cntg as pointers
	add src_cntg, src_cntg, src_ptr
	add dst_cntg, dst_cntg, dst_ptr

	b   .L_trn_int_loop_entry_first_panel${sfx}

.L_trn_int_panel_loop${sfx}:
	ld1b {z1.b}, p3/z, [x10]
	ld1b {z2.b}, p4/z, [x10, #1, mul vl]
	st1b {z1.b}, p1,   [x11]
	st1b {z2.b}, p2,   [x11, #1, mul vl]
	// increment src and dst pointers
	add  x10, x10,   src_ld
	incb x11, vl16, mul #${shift+1}
	// decrement panel row count and loop if more rows
	subs x12, x12, #1
	b.ne .L_trn_int_panel_loop${sfx}

	// set predicate to zero remaining elements
	whilelt p0.b, x11, dst_cntg
	b.nfrst .L_trn_int_loop_entry${sfx}

.L_zero_pad_remainder_loop${sfx}:
	// zero-pad the rest of the row in the dst matrix
	st1b    {z0.b}, p0, [x11]
	incb    x11
	whilelt p0.b, x11, dst_cntg
	b.first .L_zero_pad_remainder_loop${sfx}

.L_trn_int_loop_entry${sfx}:
	add     dst_ptr,  dst_ptr,  dst_ld
	add     dst_cntg, dst_cntg, dst_ld
	sub     dst_strd, dst_strd, #1
	incb    src_ptr, vl16, mul #${shift+1}
.L_trn_int_loop_entry_first_panel${sfx}:
	mov     x10, src_ptr
	mov     x11, dst_ptr
	mov     x12, src_strd
	// if more than interleave_width bytes remaining, loop
	sub 	x13, src_cntg, src_ptr
	cmp 	x13, interleave_width
	b.gt    .L_trn_int_panel_loop${sfx}
	// else last panel tail case:
	// set predicate governing number of elements remaining and loop
	incb    src_ptr
	whilelt p4.b, src_ptr, src_cntg
	decb    src_ptr
	whilelt p3.b, src_ptr, src_cntg
	b.first .L_trn_int_panel_loop${sfx}

	// if dst rows remaining, zero-pad; else return
	cbz     dst_strd, .L_return${sfx}
	whilelt p0.b, x11, dst_cntg

.L_zero_pad_row_loop${sfx}:
	// zero-pad the remaining rows in the dst matrix
	st1b    {z0.b}, p0, [x11]
	incb    x11
	whilelt p0.b, x11, dst_cntg
	b.first .L_zero_pad_row_loop${sfx}

.L_zero_row_entry${sfx}:
	add     dst_ptr, dst_ptr, dst_ld
	add     dst_cntg, dst_cntg, dst_ld
	mov     x11, dst_ptr
	whilelt p0.b, x11, dst_cntg
	subs    dst_strd, dst_strd, #1
	b.ne    .L_zero_pad_row_loop${sfx}

.L_return${sfx}:
	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}
%endfor


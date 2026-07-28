## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

// 2x vector length interleave kernel for 4-byte (e.g. float) or
// 2x vector length interleave kernel for 8-byte (e.g. double or complex<float>) elements
%for prec, mem, x, bytes, offset, incr in [ ("s", "w", "w", "#2", ", sxtw", "4"), ("d", "d", "x", "#3", "", "8") ]:
	<% func_name = "interleave_2vl_sve_kernel_" + prec %>
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
	// interleave_rows assumed to be 2*VL

	// always true predicate for first load and both stores
        ptrue p1.b
	ptrue p2.b
	ptrue p3.b
	// setup a zero vector for padding
	mov   z0.b, #0

	// calculate the VL increment (in bytes) for the src base-pointer
	rdvl x9, #1             // length of vector in bytes
	mul  x9, x9, src_ld

	// set src_ld, dst_ld and dst_cntg as byte sizes
	lsl src_ld, src_ld, ${bytes}
	lsl dst_ld, dst_ld, ${bytes}
	lsl dst_cntg, dst_cntg, ${bytes}
	// set dst_cntg as a pointer
	add dst_cntg, dst_cntg, dst_ptr

	// build offset indices (in bytes) for matrix elements
	index z1.${prec}, #0, ${x}3     // first vector of interleave
	index z2.${prec}, ${x}9, ${x}3  // second vector of interleave

	// calculate panel size (2*VL increment) in bytes
	lsl x9, x9, #1
	panel_size .req x9

	// use x13 as a counter from 0 up to src_strd
	src_row_count .req x13
	mov src_row_count, xzr

	b   .L_interleave_loop_entry_first_panel_${prec}

.L_interleave_panel_loop_${prec}:
	ld1${mem} {z3.${prec}}, p2/z, [x10, z1.${prec} ${offset}]
	ld1${mem} {z4.${prec}}, p3/z, [x10, z2.${prec} ${offset}]
	st1${mem} {z3.${prec}}, p1,   [x11]
	st1${mem} {z4.${prec}}, p1,   [x11, #1, mul vl]
	// increment src and dst pointers
	add  x10, x10, #${incr}
	incb x11, all, mul #2
	// decrement panel column count and loop if more columns
	subs x12, x12, #1
	b.ne .L_interleave_panel_loop_${prec}

	// set predicate to zero remaining elements
	whilelt p0.b, x11, dst_cntg
	b.nfrst .L_interleave_loop_entry_${prec}

.L_zero_pad_remainder_loop_${prec}:
	// zero-pad the rest of the row in the dst matrix
	st1b    {z0.b}, p0, [x11]
	incb    x11
	whilelt p0.b, x11, dst_cntg
	b.first .L_zero_pad_remainder_loop_${prec}

.L_interleave_loop_entry_${prec}:
	add     dst_ptr, dst_ptr, dst_ld
	add     src_ptr, src_ptr, panel_size
	add     dst_cntg, dst_cntg, dst_ld
	sub     dst_strd, dst_strd, #1
.L_interleave_loop_entry_first_panel_${prec}:
	mov     x10, src_ptr
	mov     x11, dst_ptr
	mov     x12, src_cntg
	inc${mem}    src_row_count, all, mul #2
	cmp     src_row_count, src_strd
	b.lo    .L_interleave_panel_loop_${prec}
	// last panel tail case; set load predicates
	dec${mem}    src_row_count
	whilelt p3.${prec}, src_row_count, src_strd
	dec${mem}    src_row_count
	whilelt p2.${prec}, src_row_count, src_strd
	inc${mem}    src_row_count, all, mul #2
	b.first .L_interleave_panel_loop_${prec}

	// test if dst rows remaining to zero-pad: if not, return
	cmp     dst_strd, #0
	b.eq    .L_return_${prec}
	whilelt p0.b, x11, dst_cntg

.L_zero_pad_row_loop_${prec}:
	// zero-pad the remaining rows in the dst matrix
	st1b    {z0.b}, p0, [x11]
	incb    x11
	whilelt p0.b, x11, dst_cntg
	b.first .L_zero_pad_row_loop_${prec}

.L_zero_row_entry_${prec}:
	add     dst_ptr, dst_ptr, dst_ld
	add     dst_cntg, dst_cntg, dst_ld
	mov     x11, dst_ptr
	whilelt p0.b, x11, dst_cntg
	subs    dst_strd, dst_strd, #1
	b.ne    .L_zero_pad_row_loop_${prec}

.L_return_${prec}:
	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}
%endfor

## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>
<%namespace file="hgemm_8_2_kernel_save.s.inc" import="hgemm_8_2_kernel_save"/>

//pointers to the first element of the input matrices
a_ptr .req x0
b_ptr .req x1
c_ptr .req x2

cntg .req x3
a_strd .req x4
b_strd .req x5

//the distance in contiguous elements between adjacent rows of C,
c_stride .req x6

//alpha & beta scalars, soon unreq'ed
alpha_in .req v0
beta_in .req v1

/**
 * This kernel uses a 8*3VLen register block, which in user data is 8*24=192
 * Registers:
 *  v[v0-v3] is used for loading from A
 *  v[v4-v5] is used for loading from B
 *  v[7-v30] are used for the reg block
 */
a_0   .req v0
a_1   .req v1
a_2   .req v2
a_3   .req v3
a_0q  .req q0
a_1q  .req q1
a_2q  .req q2
a_3q  .req q3
a_0h  .req h0
a_1h  .req h1
a_2h  .req h2
a_3h  .req h3


b_0   .req v4
b_1   .req v5
b_0q   .req q4
b_1q  .req q5
b_0h   .req h4
b_1h  .req h5

//register holding alpha & beta
al_be .req v6
al_beh .req h6

//reg block mapping
c_0_0 .req v7
c_0_1 .req v8
c_0_2 .req v9
c_0_3 .req v10
c_0_4 .req v11
c_0_5 .req v12
c_0_6 .req v13
c_0_7 .req v14

c_1_0 .req v15
c_1_1 .req v16
c_1_2 .req v17
c_1_3 .req v18
c_1_4 .req v19
c_1_5 .req v20
c_1_6 .req v21
c_1_7 .req v22

c_2_0 .req v23
c_2_1 .req v24
c_2_2 .req v25
c_2_3 .req v26
c_2_4 .req v27
c_2_5 .req v28
c_2_6 .req v29
c_2_7 .req v30

c_0_0q .req q7
c_0_1q .req q8
c_0_2q .req q9
c_0_3q .req q10
c_0_4q .req q11
c_0_5q .req q12
c_0_6q .req q13
c_0_7q .req q14

c_1_0q .req q15
c_1_1q .req q16
c_1_2q .req q17
c_1_3q .req q18
c_1_4q .req q19
c_1_5q .req q20
c_1_6q .req q21
c_1_7q .req q22

c_2_0q .req q23
c_2_1q .req q24
c_2_2q .req q25
c_2_3q .req q26
c_2_4q .req q27
c_2_5q .req q28
c_2_6q .req q29
c_2_7q .req q30

c_0_0h .req h7
c_0_1h .req h8
c_0_2h .req h9
c_0_3h .req h10
c_0_4h .req h11
c_0_5h .req h12
c_0_6h .req h13
c_0_7h .req h14

c_1_0h .req h15
c_1_1h .req h16
c_1_2h .req h17
c_1_3h .req h18
c_1_4h .req h19
c_1_5h .req h20
c_1_6h .req h21
c_1_7h .req h22

c_2_0h .req h23
c_2_1h .req h24
c_2_2h .req h25
c_2_3h .req h26
c_2_4h .req h27
c_2_5h .req h28
c_2_6h .req h29
c_2_7h .req h30

//a pointer to the start of the current interleave row of A & B
a_row       .req x7
b_row       .req x8

//the distance between interleaved rows of A & B
a_row_inc   .req x9
b_row_inc   .req x10

a_elem      .req x11
b_elem      .req x12


//the loop counters from 0 too [a_strd, b_strd, cntg] respectively
a_strd_idx  .req x13
b_strd_idx  .req x14
cntg_idx    .req x15

//pointers to rows of C
c_0_ptr     .req x16
c_1_ptr     .req x17
c_2_ptr     .req x19
c_3_ptr     .req x20
c_4_ptr     .req x21
c_5_ptr     .req x22
c_6_ptr     .req x23
c_7_ptr     .req x24

c_cntg_rem  .req x25
c_cntg_remw .req w25
c_strd_rem  .req x26

tmpx0       .req x27
tmpx1       .req x28
tmpx1w      .req w28


//void hgemm_8_2_kernel(__fp16* a, __fp16* b, __fp16* c, int64_t a_strd, int64_t b_strd, int64_t cntg, c_strd, __fp16 alpha, __fp16 beta)
	<% func_name = "hgemm_8_2_kernel" %>
	${prologue(func_name)}
	sub sp, sp, #384
	stp   x29, x30, [sp]
	mov   x29, sp        // This should point to the address we just stored to.

	stp  x8,  x9, [ sp, #64 ]
	stp x10, x11, [ sp, #80 ]
	stp x12, x13, [ sp, #96 ]
	stp x14, x15, [ sp, #112 ]
	stp x16, x17, [ sp, #128 ]
	stp x19, x20, [ sp, #144 ]
	stp x21, x22, [ sp, #160 ]
	stp x23, x24, [ sp, #176 ]
	stp x25, x26, [ sp, #192 ]
	stp x27, x28, [ sp, #208 ]

	stp  d8,  d9, [ sp, #304 ]
	stp d10, d11, [ sp, #320 ]
	stp d12, d13, [ sp, #336 ]
	stp d14, d15, [ sp, #352 ]

	//set up the constants
	add a_row_inc, cntg, cntg, lsl 1
	lsl a_row_inc, a_row_inc, 4
	lsl b_row_inc, cntg, 4

	//multiply c_stride by sizeof(__fp16)
	lsl c_stride, c_stride, 1

	//move alpha and beta into a single register
	ins al_be.h[0], beta_in.h[0]
	ins al_be.h[1], alpha_in.h[0]

	mov b_strd_idx, 0
	//move the pointer to the current b_row back to the beginning of B
	mov b_row, b_ptr
//LOOP over the rows of B
.L_B_LOOP:
	//this param is used in the save c code to ensure the right amount of the reg block is save out
	sub c_strd_rem, b_strd, b_strd_idx

	add b_strd_idx, b_strd_idx, 8

	//reset the a loop and its associated pointer
	mov a_strd_idx, 0
	mov a_row, a_ptr

	//We need to increment which row our c_ptrs are on
	//We also store the location of next set of 8 rows in c_ptr
	mov c_0_ptr, c_ptr
	add c_1_ptr, c_0_ptr, c_stride
	add c_2_ptr, c_1_ptr, c_stride
	add c_3_ptr, c_2_ptr, c_stride
	add c_4_ptr, c_3_ptr, c_stride
	add c_5_ptr, c_4_ptr, c_stride
	add c_6_ptr, c_5_ptr, c_stride
	add c_7_ptr, c_6_ptr, c_stride
	add c_ptr, c_7_ptr, c_stride
//LOOP over the rows of A
.L_A_LOOP:
	sub c_cntg_rem, a_strd, a_strd_idx
	add a_strd_idx, a_strd_idx, 24
	mov a_elem, a_row

	//zero the register block
	movi c_0_0.8h, 0
	movi c_0_1.8h, 0
	movi c_0_2.8h, 0
	movi c_0_3.8h, 0
	movi c_0_4.8h, 0
	movi c_0_5.8h, 0
	movi c_0_6.8h, 0
	movi c_0_7.8h, 0

	movi c_1_0.8h, 0
	movi c_1_1.8h, 0
	movi c_1_2.8h, 0
	movi c_1_3.8h, 0
	movi c_1_4.8h, 0
	movi c_1_5.8h, 0
	movi c_1_6.8h, 0
	movi c_1_7.8h, 0

	movi c_2_0.8h, 0
	movi c_2_1.8h, 0
	movi c_2_2.8h, 0
	movi c_2_3.8h, 0
	movi c_2_4.8h, 0
	movi c_2_5.8h, 0
	movi c_2_6.8h, 0
	movi c_2_7.8h, 0

	mov cntg_idx, 0


	//set b_elem back to the begging of the row, a_elem is reset at the end of every loop
	mov b_elem, b_row
.L_HOTLOOP_0:
	//the loop has now begun, so increment the hostloop counter by the unroll amount
	add cntg_idx, cntg_idx, 4
	cmp cntg, cntg_idx


	ldr b_0q, [ b_elem ]
	ldr a_0q, [ a_elem ]
	ldr a_1q, [ a_elem, 16 ]
	ldr a_2q, [ a_elem, 32 ]
	add b_elem, b_elem, 16
	add a_elem, a_elem, 48

	fmla c_0_0.8h, a_0.8h, b_0.h[0]
	fmla c_0_1.8h, a_0.8h, b_0.h[1]
	fmla c_0_2.8h, a_0.8h, b_0.h[2]
	fmla c_0_3.8h, a_0.8h, b_0.h[3]
	fmla c_0_4.8h, a_0.8h, b_0.h[4]
	fmla c_0_5.8h, a_0.8h, b_0.h[5]
	fmla c_0_6.8h, a_0.8h, b_0.h[6]
	fmla c_0_7.8h, a_0.8h, b_0.h[7]

	fmla c_1_0.8h, a_1.8h, b_0.h[0]
	fmla c_1_1.8h, a_1.8h, b_0.h[1]
	fmla c_1_2.8h, a_1.8h, b_0.h[2]
	fmla c_1_3.8h, a_1.8h, b_0.h[3]
	fmla c_1_4.8h, a_1.8h, b_0.h[4]
	fmla c_1_5.8h, a_1.8h, b_0.h[5]
	fmla c_1_6.8h, a_1.8h, b_0.h[6]
	fmla c_1_7.8h, a_1.8h, b_0.h[7]

	fmla c_2_0.8h, a_2.8h, b_0.h[0]
	fmla c_2_1.8h, a_2.8h, b_0.h[1]
	fmla c_2_2.8h, a_2.8h, b_0.h[2]
	fmla c_2_3.8h, a_2.8h, b_0.h[3]
	fmla c_2_4.8h, a_2.8h, b_0.h[4]
	fmla c_2_5.8h, a_2.8h, b_0.h[5]
	fmla c_2_6.8h, a_2.8h, b_0.h[6]
	fmla c_2_7.8h, a_2.8h, b_0.h[7]


.L_HOTLOOP_1:
	ldr b_1q, [ b_elem ]
	ldr a_3q, [ a_elem ]
	ldr a_0q, [ a_elem, 16 ]
	ldr a_1q, [ a_elem, 32 ]
	add b_elem, b_elem, 16
	add a_elem, a_elem, 48

	fmla c_0_0.8h, a_3.8h, b_1.h[0]
	fmla c_0_1.8h, a_3.8h, b_1.h[1]
	fmla c_0_2.8h, a_3.8h, b_1.h[2]
	fmla c_0_3.8h, a_3.8h, b_1.h[3]
	fmla c_0_4.8h, a_3.8h, b_1.h[4]
	fmla c_0_5.8h, a_3.8h, b_1.h[5]
	fmla c_0_6.8h, a_3.8h, b_1.h[6]
	fmla c_0_7.8h, a_3.8h, b_1.h[7]

	fmla c_1_0.8h, a_0.8h, b_1.h[0]
	fmla c_1_1.8h, a_0.8h, b_1.h[1]
	fmla c_1_2.8h, a_0.8h, b_1.h[2]
	fmla c_1_3.8h, a_0.8h, b_1.h[3]
	fmla c_1_4.8h, a_0.8h, b_1.h[4]
	fmla c_1_5.8h, a_0.8h, b_1.h[5]
	fmla c_1_6.8h, a_0.8h, b_1.h[6]
	fmla c_1_7.8h, a_0.8h, b_1.h[7]

	fmla c_2_0.8h, a_1.8h, b_1.h[0]
	fmla c_2_1.8h, a_1.8h, b_1.h[1]
	fmla c_2_2.8h, a_1.8h, b_1.h[2]
	fmla c_2_3.8h, a_1.8h, b_1.h[3]
	fmla c_2_4.8h, a_1.8h, b_1.h[4]
	fmla c_2_5.8h, a_1.8h, b_1.h[5]
	fmla c_2_6.8h, a_1.8h, b_1.h[6]
	fmla c_2_7.8h, a_1.8h, b_1.h[7]


.L_HOTLOOP_2:
	ldr b_0q, [ b_elem ]
	ldr a_2q, [ a_elem ]
	ldr a_3q, [ a_elem, 16 ]
	ldr a_0q, [ a_elem, 32 ]
	add b_elem, b_elem, 16
	add a_elem, a_elem, 48

	fmla c_0_0.8h, a_2.8h, b_0.h[0]
	fmla c_0_1.8h, a_2.8h, b_0.h[1]
	fmla c_0_2.8h, a_2.8h, b_0.h[2]
	fmla c_0_3.8h, a_2.8h, b_0.h[3]
	fmla c_0_4.8h, a_2.8h, b_0.h[4]
	fmla c_0_5.8h, a_2.8h, b_0.h[5]
	fmla c_0_6.8h, a_2.8h, b_0.h[6]
	fmla c_0_7.8h, a_2.8h, b_0.h[7]

	fmla c_1_0.8h, a_3.8h, b_0.h[0]
	fmla c_1_1.8h, a_3.8h, b_0.h[1]
	fmla c_1_2.8h, a_3.8h, b_0.h[2]
	fmla c_1_3.8h, a_3.8h, b_0.h[3]
	fmla c_1_4.8h, a_3.8h, b_0.h[4]
	fmla c_1_5.8h, a_3.8h, b_0.h[5]
	fmla c_1_6.8h, a_3.8h, b_0.h[6]
	fmla c_1_7.8h, a_3.8h, b_0.h[7]

	fmla c_2_0.8h, a_0.8h, b_0.h[0]
	fmla c_2_1.8h, a_0.8h, b_0.h[1]
	fmla c_2_2.8h, a_0.8h, b_0.h[2]
	fmla c_2_3.8h, a_0.8h, b_0.h[3]
	fmla c_2_4.8h, a_0.8h, b_0.h[4]
	fmla c_2_5.8h, a_0.8h, b_0.h[5]
	fmla c_2_6.8h, a_0.8h, b_0.h[6]
	fmla c_2_7.8h, a_0.8h, b_0.h[7]


.L_HOTLOOP_3:
	ldr b_1q, [ b_elem ]
	ldr a_1q, [ a_elem ]
	ldr a_2q, [ a_elem, 16 ]
	ldr a_3q, [ a_elem, 32 ]
	add b_elem, b_elem, 16
	add a_elem, a_elem, 48

	fmla c_0_0.8h, a_1.8h, b_1.h[0]
	fmla c_0_1.8h, a_1.8h, b_1.h[1]
	fmla c_0_2.8h, a_1.8h, b_1.h[2]
	fmla c_0_3.8h, a_1.8h, b_1.h[3]
	fmla c_0_4.8h, a_1.8h, b_1.h[4]
	fmla c_0_5.8h, a_1.8h, b_1.h[5]
	fmla c_0_6.8h, a_1.8h, b_1.h[6]
	fmla c_0_7.8h, a_1.8h, b_1.h[7]

	fmla c_1_0.8h, a_2.8h, b_1.h[0]
	fmla c_1_1.8h, a_2.8h, b_1.h[1]
	fmla c_1_2.8h, a_2.8h, b_1.h[2]
	fmla c_1_3.8h, a_2.8h, b_1.h[3]
	fmla c_1_4.8h, a_2.8h, b_1.h[4]
	fmla c_1_5.8h, a_2.8h, b_1.h[5]
	fmla c_1_6.8h, a_2.8h, b_1.h[6]
	fmla c_1_7.8h, a_2.8h, b_1.h[7]

	fmla c_2_0.8h, a_3.8h, b_1.h[0]
	fmla c_2_1.8h, a_3.8h, b_1.h[1]
	fmla c_2_2.8h, a_3.8h, b_1.h[2]
	fmla c_2_3.8h, a_3.8h, b_1.h[3]
	fmla c_2_4.8h, a_3.8h, b_1.h[4]
	fmla c_2_5.8h, a_3.8h, b_1.h[5]
	fmla c_2_6.8h, a_3.8h, b_1.h[6]
	fmla c_2_7.8h, a_3.8h, b_1.h[7]

	bhi .L_HOTLOOP_0

.L_HOTLOOP_END:
//SAVE_C
//IF ALPHA is not 0.0 and BETA is not 1.0
	b .L_SAVE_C_REG_BLOCK

///BETA is 0.0 and ALPHA is 1.0
	stp c_0_0q, c_1_0q, [ c_0_ptr ]
	str c_2_0q,         [ c_0_ptr, #32 ]
	stp c_0_1q, c_1_1q, [ c_1_ptr ]
	str c_2_1q,         [ c_1_ptr, #32 ]
	stp c_0_2q, c_1_2q, [ c_2_ptr ]
	str c_2_2q,         [ c_2_ptr, #32 ]
	stp c_0_3q, c_1_3q, [ c_3_ptr ]
	str c_2_3q,         [ c_3_ptr, #32 ]
	stp c_0_4q, c_1_4q, [ c_4_ptr ]
	str c_2_4q,         [ c_4_ptr, #32 ]
	stp c_0_5q, c_1_5q, [ c_5_ptr ]
	str c_2_5q,         [ c_5_ptr, #32 ]
	stp c_0_6q, c_1_6q, [ c_6_ptr ]
	str c_2_6q,         [ c_6_ptr, #32 ]
	stp c_0_7q, c_1_7q, [ c_7_ptr ]
	str c_2_7q,         [ c_7_ptr, #32 ]
//on the return of any of save paths,
//inside of the A loop we need to progress c_ptrs on 24 elements
.L_SAVE_RETURN:
	add c_0_ptr, c_0_ptr, 48
	add c_1_ptr, c_1_ptr, 48
	add c_2_ptr, c_2_ptr, 48
	add c_3_ptr, c_3_ptr, 48
	add c_4_ptr, c_4_ptr, 48
	add c_5_ptr, c_5_ptr, 48
	add c_6_ptr, c_6_ptr, 48
	add c_7_ptr, c_7_ptr, 48

.L_A_LOOP_END:
	//increment the a_row pointer, for the next iteration
	add a_row, a_row, a_row_inc
	cmp a_strd, a_strd_idx
	bhi .L_A_LOOP

.L_B_LOOP_END:
	//increment the b_row pointer, for the next iteration
	add b_row, b_row, b_row_inc
	cmp b_strd, b_strd_idx
	bhi .L_B_LOOP

.L_END:
	ldp  x8,  x9, [ sp, #64 ]
	ldp x10, x11, [ sp, #80 ]
	ldp x12, x13, [ sp, #96 ]
	ldp x14, x15, [ sp, #112 ]
	ldp x16, x17, [ sp, #128 ]
	ldp x19, x20, [ sp, #144 ]
	ldp x21, x22, [ sp, #160 ]
	ldp x23, x24, [ sp, #176 ]
	ldp x25, x26, [ sp, #192 ]
	ldp x27, x28, [ sp, #208 ]

	ldp  d8,  d9, [ sp, #304 ]
	ldp d10, d11, [ sp, #320 ]
	ldp d12, d13, [ sp, #336 ]
	ldp d14, d15, [ sp, #352 ]

	ldp x29, x30, [sp]
	add sp, sp, #384

	ret

	${hgemm_8_2_kernel_save()}
	${epilogue(func_name)}

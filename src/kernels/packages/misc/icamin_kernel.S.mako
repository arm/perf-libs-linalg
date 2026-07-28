## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

#define ELEM_SIZE 8
#define N x0
#define X x1
#define incx x2
#define index x3
#define iterations x4
#define Z x5
#define X_copy x7

#define min s0
#define temp1_s s1
#define temp1_v v1.2s
#define temp2_d d4
#define temp2_v v4.2s

	<% func_name = "icamin_kernel" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	mov index, xzr
	cmp N, xzr
	ble .Lexit
	cmp incx, xzr
	ble .Lexit

	mov X_copy, X

	ldr temp2_d, [X]
	add X, X, ELEM_SIZE
	fabs temp2_v, temp2_v
	faddp temp1_v, temp2_v, temp2_v
	fmov min , temp1_s
	mov Z, #1
	mov index, Z
	fabs min, min

	subs N, N, 1
	ble .Lexit


.Loop32_check:
	asr iterations, N, 5    //compute the number of iterations for Loop32
	cmp iterations, xzr
	beq .Lscalar_check
	add Z, Z, 1

.Loop32:
	ldp q2, q3, [X]
	ldp q4, q5, [X, 4*ELEM_SIZE]
	ldp q6, q7, [X, 8*ELEM_SIZE]
	ldp q16, q17, [X, 12*ELEM_SIZE]
	ldp q18, q19, [X, 16*ELEM_SIZE]
	ldp q20, q21, [X, 20*ELEM_SIZE]
	ldp q22, q23, [X, 24*ELEM_SIZE]
	ldp q24, q25, [X, 28*ELEM_SIZE]

	add X, X, 32*ELEM_SIZE

	fabs v2.4s, v2.4s
	fabs v3.4s, v3.4s
	fabs v4.4s, v4.4s
	fabs v5.4s, v5.4s
	fabs v6.4s, v6.4s
	fabs v7.4s, v7.4s
	fabs v16.4s, v16.4s
	fabs v17.4s, v17.4s
	fabs v18.4s, v18.4s
	fabs v19.4s, v19.4s
	fabs v20.4s, v20.4s
	fabs v21.4s, v21.4s
	fabs v22.4s, v22.4s
	fabs v23.4s, v23.4s
	fabs v24.4s, v24.4s
	fabs v25.4s, v25.4s

	faddp v2.4s, v2.4s, v3.4s
	faddp v4.4s, v4.4s, v5.4s
	faddp v6.4s, v6.4s, v7.4s
	faddp v16.4s, v16.4s, v17.4s
	faddp v18.4s, v18.4s, v19.4s
	faddp v20.4s, v20.4s, v21.4s
	faddp v22.4s, v22.4s, v23.4s
	faddp v24.4s, v24.4s, v25.4s

	fmin v2.4s, v2.4s, v4.4s
	fmin v6.4s, v6.4s, v16.4s
	fmin v18.4s, v18.4s, v20.4s
	fmin v22.4s, v22.4s, v24.4s

	PRFM PLDL1KEEP, [X, #1024]
	PRFM PLDL1KEEP, [X, #1024+64]
	PRFM PLDL1KEEP, [X, #1024+128]
	PRFM PLDL1KEEP, [X, #1024+192]

	fmin v2.4s, v2.4s, v6.4s
	fmin v18.4s, v18.4s, v22.4s
	fmin v2.4s, v2.4s, v18.4s
	fminv temp1_s, v2.4s
	fcmp min, temp1_s
	fcsel min, min, temp1_s, le
	csel index, index, Z, le

	add Z, Z, #32

	subs iterations, iterations, 1
	bne .Loop32

.Loop32_reduction:
	sub x6, index, 1
	lsl x6, x6, 3         //(index-1)*ELEM_SIZE
	add X_copy, X_copy, x6
	mov x6, 0
.Look_and_look:
	add x6, x6, #1
	cmp x6, #32
	bge .Lstop
	ldr temp2_d, [x7]
	fabs temp2_v, temp2_v
	faddp temp1_v, temp2_v, temp2_v
	fcmp min, temp1_s
	add x7, x7, #ELEM_SIZE
	bne .Look_and_look
.Lstop:
	sub x6, x6, #1
	add index, index, x6

	sub Z, Z, 1


.Lscalar_check:
	ands iterations, N, 31
	ble .Lexit
.Lscalar:
	ldr temp2_d, [X]
	add X, X, ELEM_SIZE
	add Z, Z, #1
	fabs temp2_v, temp2_v
	faddp temp1_v, temp2_v, temp2_v
	fcmp min, temp1_s
	fcsel min, min, temp1_s, le
	csel index, index, Z, le

	subs iterations, iterations, 1
	bne .Lscalar

.Lexit:
	fmov w1, min
	mov x0, index
	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}

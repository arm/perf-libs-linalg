## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

#define ELEM_SIZE 16
#define N x0
#define X x1
#define incx x2
#define index x3
#define iterations x4
#define Z x5
#define X_copy x7

#define max  d0
#define temp1 d1
#define temp1_v v1.2d
#define temp2 q4
#define temp2_v v4.2d

	<% func_name = "izamax_kernel" %>
	${prologue(func_name)}
	stp x29, x30, [sp, -16]!
	mov x29, sp

	mov index, xzr
	cmp N, xzr
	ble .Lexit
	cmp incx, xzr
	ble .Lexit


	mov X_copy, X

	ldr temp2, [X]
	fabs temp2_v, temp2_v
	faddp temp1_v, temp2_v, temp2_v
	fmov max , temp1
	add X, X, ELEM_SIZE
	mov Z, 1
	mov index, Z
	fabs max, max

	subs N, N, 1
	ble .Lexit


.Loop16_check:
	asr iterations, N, 4    //compute the number of iterations for Loop16
	cmp iterations, xzr
	beq .Lscalar_check
	add Z, Z, 1

.Loop16:
	ldp q2, q3, [X]
	ldp q4, q5, [X, 2*ELEM_SIZE]
	ldp q6, q7, [X, 4*ELEM_SIZE]
	ldp q16, q17, [X, 6*ELEM_SIZE]
	ldp q18, q19, [X, 8*ELEM_SIZE]
	ldp q20, q21, [X, 10*ELEM_SIZE]
	ldp q22, q23, [X, 12*ELEM_SIZE]
	ldp q24, q25, [X, 14*ELEM_SIZE]

	add X, X, 16*ELEM_SIZE

	fabs v2.2d, v2.2d
	fabs v3.2d, v3.2d
	fabs v4.2d, v4.2d
	fabs v5.2d, v5.2d
	fabs v6.2d, v6.2d
	fabs v7.2d, v7.2d
	fabs v16.2d, v16.2d
	fabs v17.2d, v17.2d
	fabs v18.2d, v18.2d
	fabs v19.2d, v19.2d
	fabs v20.2d, v20.2d
	fabs v21.2d, v21.2d
	fabs v22.2d, v22.2d
	fabs v23.2d, v23.2d
	fabs v24.2d, v24.2d
	fabs v25.2d, v25.2d

	faddp v2.2d, v2.2d, v3.2d
	faddp v4.2d, v4.2d, v5.2d
	faddp v6.2d, v6.2d, v7.2d
	faddp v16.2d, v16.2d, v17.2d
	faddp v18.2d, v18.2d, v19.2d
	faddp v20.2d, v20.2d, v21.2d
	faddp v22.2d, v22.2d, v23.2d
	faddp v24.2d, v24.2d, v25.2d

	fmax v2.2d, v2.2d, v4.2d
	fmax v6.2d, v6.2d, v16.2d
	fmax v18.2d, v18.2d, v20.2d
	fmax v22.2d, v22.2d, v24.2d

	PRFM PLDL1KEEP, [X, 1024]
	PRFM PLDL1KEEP, [X, 1024+64]
	PRFM PLDL1KEEP, [X, 1024+128]
	PRFM PLDL1KEEP, [X, 1024+192]

	fmax v2.2d, v2.2d, v6.2d
	fmax v18.2d, v18.2d, v22.2d
	fmax v2.2d, v2.2d, v18.2d

	ins v3.d[0], v2.d[1]
	fmax temp1, d3, d2
	fcmp max, temp1
	fcsel max, max, temp1, ge
	csel index, index, Z, ge
	add Z, Z, 16

	subs iterations, iterations, 1
	bne .Loop16

.Loop16_reduction:
	sub x6, index, 1
	lsl x6, x6, 4
	add x7, x7, x6
	mov x6, 0

.Look_and_look:
	add x6, x6, 1
	cmp x6, 16
	bge .Lstop
	ldr temp2, [x7]
	fabs temp2_v, temp2_v
	faddp temp1_v, temp2_v, temp2_v
	fcmp max, temp1
	add x7, x7, ELEM_SIZE
	bne .Look_and_look
.Lstop:
	sub x6, x6, 1
	add index, index, x6

	sub Z, Z, 1


.Lscalar_check:
	ands iterations, N, 15
	ble .Lexit
.Lscalar:
	ldr temp2, [X]
	add X, X, ELEM_SIZE
	add Z, Z, 1
	fabs temp2_v, temp2_v
	faddp temp1_v, temp2_v, temp2_v
	fcmp max, temp1
	fcsel max, max, temp1, ge
	csel index, index, Z, ge

	subs iterations, iterations, 1
	bne .Lscalar


.Lexit:
	fmov x1, max
	mov x0, index
	ldp x29, x30, [sp], 16
	ret
	${epilogue(func_name)}

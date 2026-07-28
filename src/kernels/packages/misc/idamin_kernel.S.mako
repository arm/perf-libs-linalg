## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

#define ELEM_SIZE 8
#define N x0
#define X x1
#define incx x2
#define index x3

#define iterations x4	      // The number of iterations that will be carried out
#define Z x5
#define X_copy x7

#define min d0
#define temp1 d1    //General temporary variable storing current element
#define temp2 d2    //Used to iterate through the vector in L32 reduction

	<% func_name = "idamin_kernel" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	mov index, xzr
	cmp N, xzr
	ble .Lexit
	cmp incx, xzr
	ble .Lexit

	mov X_copy, X

	ldr min, [X]
	add X, X, ELEM_SIZE
	mov Z, 1
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

	fmin v2.2d, v2.2d, v3.2d
	fmin v4.2d, v4.2d, v5.2d
	fmin v6.2d, v6.2d, v7.2d
	fmin v16.2d, v16.2d, v17.2d
	fmin v18.2d, v18.2d, v19.2d
	fmin v20.2d, v20.2d, v21.2d
	fmin v22.2d, v22.2d, v23.2d
	fmin v24.2d, v24.2d, v25.2d

	PRFM PLDL1KEEP, [X, #1024]
	PRFM PLDL1KEEP, [X, #1024+64]
	PRFM PLDL1KEEP, [X, #1024+128]
	PRFM PLDL1KEEP, [X, #1024+192]

	fmin v2.2d, v2.2d, v4.2d
	fmin v6.2d, v6.2d, v16.2d
	fmin v18.2d, v18.2d, v20.2d
	fmin v22.2d, v22.2d, v24.2d
	fmin v2.2d, v2.2d, v6.2d
	fmin v18.2d, v18.2d, v22.2d
	fmin v2.2d, v2.2d, v18.2d

	ins v3.d[0], v2.d[1]
	fmin temp1, d3, d2
	fcmp min, temp1
	fcsel min, min, temp1, le
	csel index, index, Z, le
	add Z, Z, 32

	subs iterations, iterations, 1
	bne .Loop32

.Loop32_reduction:
	sub x6, index, 1
	lsl x6, x6, 3         //(index-1)*ELEM_SIZE
	add X_copy, X_copy, x6
	mov x6, 0
.Look_and_look:
	add x6, x6, 1
	cmp x6, 32
	bge .Lstop
	ldr temp2, [X_copy]
	fabs temp2, temp2
	fcmp min, temp2
	add X_copy, X_copy, ELEM_SIZE
	bne .Look_and_look
.Lstop:
	sub x6, x6, 1
	add index, index, x6
	sub Z, Z, 1


.Lscalar_check:
	ands iterations, N, 31
	ble .Lexit
.Lscalar:
	ldr temp1, [X]
	add X, X, ELEM_SIZE
	add Z, Z, 1
	fabs temp1, temp1
	fcmp min, temp1
	fcsel min, min, temp1, le
	csel index, index, Z, le

	subs iterations, iterations, 1
	bne .Lscalar

.Lexit:
	fmov x1, min
	mov x0, index
	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}

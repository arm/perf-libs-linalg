## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

#define ELEM_SIZE 8
#define N x0
#define X x1
#define NT x3

	<% func_name = "dasum_kernel" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	cmp N, xzr
	ble .Lexit

	mov NT, N
	fmov d0, xzr
	fmov d1, xzr
	fmov d2, xzr
	fmov d3, xzr
	fmov d4, xzr
	fmov d5, xzr
	fmov d6, xzr
	fmov d7, xzr

	cmp NT, #32
	blt .Loop4_check

.Loop32:
	ldp q16, q17, [X]
	ldp q18, q19, [X, 4*ELEM_SIZE]

	fabs v16.2d, v16.2d
	fabs v17.2d, v17.2d
	fabs v18.2d, v18.2d
	fabs v19.2d, v19.2d

	ldp q20, q21, [X, 8*ELEM_SIZE]
	ldp q22, q23, [X, 12*ELEM_SIZE]

	fadd v16.2d, v16.2d, v17.2d
	fadd v18.2d, v18.2d, v19.2d

	fabs v20.2d, v20.2d
	fabs v21.2d, v21.2d
	fabs v22.2d, v22.2d
	fabs v23.2d, v23.2d

	ldp q24, q25, [X, 16*ELEM_SIZE]
	ldp q26, q27, [X, 20*ELEM_SIZE]

	fadd v20.2d, v20.2d, v21.2d
	fadd v22.2d, v22.2d, v23.2d

	fabs v24.2d, v24.2d
	fabs v25.2d, v25.2d
	fabs v26.2d, v26.2d
	fabs v27.2d, v27.2d

	ldp q28, q29, [X, 24*ELEM_SIZE]
	ldp q30, q31, [X, 28*ELEM_SIZE]

	fadd v24.2d, v24.2d, v25.2d
	fadd v26.2d, v26.2d, v27.2d

	add X, X, 32*ELEM_SIZE

	fabs v28.2d, v28.2d
	fabs v29.2d, v29.2d
	fabs v30.2d, v30.2d
	fabs v31.2d, v31.2d

	PRFM PLDL1KEEP, [X, #1024]
	PRFM PLDL1KEEP, [X, #1024+64]

	fadd v28.2d, v28.2d, v29.2d
	fadd v30.2d, v30.2d, v31.2d

	fadd v0.2d, v0.2d, v16.2d
	fadd v1.2d, v1.2d, v18.2d
	fadd v2.2d, v2.2d, v20.2d
	fadd v3.2d, v3.2d, v22.2d

	PRFM PLDL1KEEP, [X, #1024+128]
	PRFM PLDL1KEEP, [X, #1024+192]

	fadd v4.2d, v4.2d, v24.2d
	fadd v5.2d, v5.2d, v26.2d
	fadd v6.2d, v6.2d, v28.2d
	fadd v7.2d, v7.2d, v30.2d

	sub NT, NT, #32
	cmp NT, #32
	blt .Loop32_reduction
	b .Loop32

.Loop32_reduction:
	fadd v0.2d, v0.2d, v1.2d
	fadd v2.2d, v2.2d, v3.2d
	fadd v4.2d, v4.2d, v5.2d
	fadd v6.2d, v6.2d, v7.2d
	fadd v0.2d, v0.2d, v2.2d
	fadd v4.2d, v4.2d, v6.2d
	fadd v0.2d, v0.2d, v4.2d


.Loop4_check:
	cmp NT, #4
	blt .Loop4_reduction
.Loop4:
	ldp q16, q17, [X]
	fabs v16.2d, v16.2d
	fabs v17.2d, v17.2d
	fadd v16.2d, v16.2d, v17.2d
	fadd v0.2d, v0.2d, v16.2d

	sub NT, NT, #4
	add X, X, 4*ELEM_SIZE
	cmp NT, #4
	blt .Loop4_reduction
	b .Loop4

.Loop4_reduction:
	faddp d0,  v0.2d


.Lscalar:
	cmp NT, #1
	blt .Lexit
	sub NT, NT, #1
	ldr d1, [X]
	add X, X, ELEM_SIZE
	fabs d1, d1
	fadd d0, d0, d1
	b .Lscalar

.Lexit:
	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}

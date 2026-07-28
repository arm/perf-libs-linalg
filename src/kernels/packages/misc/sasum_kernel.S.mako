## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

#define ELEM_SIZE 4
#define N x0
#define X x1
#define NT x3

	<% func_name = "sasum_kernel" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	cmp N, xzr
	ble .Lexit

	mov NT, N
	dup v0.4s, wzr
	dup v1.4s, wzr
	dup v2.4s, wzr
	dup v3.4s, wzr
	dup v4.4s, wzr
	dup v5.4s, wzr
	dup v6.4s, wzr
	dup v7.4s, wzr

	cmp NT, #64
	blt .Loop32_check

.Loop64:
	ldp q16, q17, [X]
	ldp q18, q19, [X, 8*ELEM_SIZE]

	fabs v16.4s, v16.4s
	fabs v17.4s, v17.4s
	fabs v18.4s, v18.4s
	fabs v19.4s, v19.4s

	ldp q20, q21, [X, 16*ELEM_SIZE]
	ldp q22, q23, [X, 24*ELEM_SIZE]

	fadd v16.4s, v16.4s, v17.4s
	fadd v18.4s, v18.4s, v19.4s

	fabs v20.4s, v20.4s
	fabs v21.4s, v21.4s
	fabs v22.4s, v22.4s
	fabs v23.4s, v23.4s

	ldp q24, q25, [X, 32*ELEM_SIZE]
	ldp q26, q27, [X, 40*ELEM_SIZE]

	fadd v20.4s, v20.4s, v21.4s
	fadd v22.4s, v22.4s, v23.4s

	fabs v24.4s, v24.4s
	fabs v25.4s, v25.4s
	fabs v26.4s, v26.4s
	fabs v27.4s, v27.4s

	ldp q28, q29, [X, 48*ELEM_SIZE]
	ldp q30, q31, [X, 56*ELEM_SIZE]

	fadd v24.4s, v24.4s, v25.4s
	fadd v26.4s, v26.4s, v27.4s

	add X, X, 64*ELEM_SIZE

	fabs v28.4s, v28.4s
	fabs v29.4s, v29.4s
	fabs v30.4s, v30.4s
	fabs v31.4s, v31.4s

	PRFM PLDL1KEEP, [X, #1024]
	PRFM PLDL1KEEP, [X, #1024+64]

	fadd v28.4s, v28.4s, v29.4s
	fadd v30.4s, v30.4s, v31.4s

	fadd v0.4s, v0.4s, v16.4s
	fadd v1.4s, v1.4s, v18.4s
	fadd v2.4s, v2.4s, v20.4s
	fadd v3.4s, v3.4s, v22.4s

	PRFM PLDL1KEEP, [X, #1024+128]
	PRFM PLDL1KEEP, [X, #1024+192]

	fadd v4.4s, v4.4s, v24.4s
	fadd v5.4s, v5.4s, v26.4s
	fadd v6.4s, v6.4s, v28.4s
	fadd v7.4s, v7.4s, v30.4s

	sub NT, NT, #64
	cmp NT, #64
	bge .Loop64

.Loop64_reduction:
	fadd v0.4s, v0.4s, v1.4s
	fadd v2.4s, v2.4s, v3.4s
	fadd v4.4s, v4.4s, v5.4s
	fadd v6.4s, v6.4s, v7.4s
	fadd v0.4s, v0.4s, v2.4s
	fadd v4.4s, v4.4s, v6.4s
	fadd v0.4s, v0.4s, v4.4s

.Loop32_check:
	cmp NT, #32
	blt .Loop16_check
	dup v1.4s, wzr
	dup v2.4s, wzr
	dup v3.4s, wzr
.Loop32:
	ldp q16, q17, [X]
	ldp q18, q19, [X, 8*ELEM_SIZE]

	fabs v16.4s, v16.4s
	fabs v17.4s, v17.4s
	fabs v18.4s, v18.4s
	fabs v19.4s, v19.4s

	ldp q20, q21, [X, 16*ELEM_SIZE]
	ldp q22, q23, [X, 24*ELEM_SIZE]

	fadd v16.4s, v16.4s, v17.4s
	fadd v18.4s, v18.4s, v19.4s

	fabs v20.4s, v20.4s
	fabs v21.4s, v21.4s
	fabs v22.4s, v22.4s
	fabs v23.4s, v23.4s

	add X, X, 32*ELEM_SIZE
	sub NT, NT, #32

	fadd v20.4s, v20.4s, v21.4s
	fadd v22.4s, v22.4s, v23.4s

	PRFM PLDL1KEEP, [X, #1024]
	PRFM PLDL1KEEP, [X, #1024+64]

	fadd v0.4s, v0.4s, v16.4s
	fadd v1.4s, v1.4s, v18.4s
	fadd v2.4s, v2.4s, v20.4s
	fadd v3.4s, v3.4s, v22.4s

	cmp NT, #32
	bge .Loop32

.Loop32_reduction:
	fadd v0.4s, v0.4s, v1.4s
	fadd v2.4s, v2.4s, v3.4s
	fadd v0.4s, v0.4s, v2.4s

.Loop16_check:
	cmp NT, #16
	blt .Loop8_check
	dup v1.4s, wzr
.Loop16:
	ldp q16, q17, [X]
	ldp q18, q19, [X, 8*ELEM_SIZE]

	add X, X, 16*ELEM_SIZE
	sub NT, NT, #16

	fabs v16.4s, v16.4s
	fabs v17.4s, v17.4s
	fabs v18.4s, v18.4s
	fabs v19.4s, v19.4s

	fadd v16.4s, v16.4s, v17.4s
	fadd v18.4s, v18.4s, v19.4s

	PRFM PLDL1KEEP, [X, #1024]
	PRFM PLDL1KEEP, [X, #1024+64]

	fadd v0.4s, v0.4s, v16.4s
	fadd v1.4s, v1.4s, v18.4s

	cmp NT, #16
	bge .Loop16

.Loop16_reduction:
	fadd v0.4s, v0.4s, v1.4s

.Loop8_check:
	cmp NT, #8
	blt .Loop4_check
.Loop8:
	ldp q16, q17, [X]
	add X, X, 8*ELEM_SIZE

	fabs v16.4s, v16.4s
	fabs v17.4s, v17.4s

	fadd v16.4s, v16.4s, v17.4s
	sub NT, NT, #8

	fadd v0.4s, v0.4s, v16.4s

	cmp NT, #8
	bge .Loop8

.Loop4_check:
	cmp NT, #4
	blt .Loop4_reduction
.Loop4:
	ldr q16, [X]

	sub NT, NT, #4
	add X, X, 4*ELEM_SIZE

	fabs v16.4s, v16.4s
	fadd v0.4s, v0.4s, v16.4s

	cmp NT, #4
	bge .Loop4

.Loop4_reduction:
	faddp v0.4s, v0.4s, v0.4s
	faddp s0, v0.2s


.Lscalar:
	cmp NT, #1
	blt .Lexit
	sub NT, NT, #1
	ldr s1, [X]
	add X, X, ELEM_SIZE
	fabs s1, s1
	fadd s0, s0, s1
	b .Lscalar

.Lexit:
	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}

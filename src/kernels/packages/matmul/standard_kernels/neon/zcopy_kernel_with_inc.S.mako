## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

//Input Parameter:
#define N x0
#define DX x1
#define INCX x2
#define DY x3
#define INCY x4

<% kernel = "zcopy_kernel_with_inc" %>
	<%
		tmp1_d = "d2"
		elem_size = 16
	%>
	${prologue(kernel)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	cmp N, 0
	beq .L${kernel}_end
	mov x6, ${elem_size}
	mul INCX, INCX, x6
	mul INCY, INCY, x6

.L${kernel}_loop:
	ldp d2, d3, [DX]    // temp <- x[i]
	ADD DX, DX, INCX
	sub N, N, 1         // n--
	stp d2, d3, [DY]    // y[i] <- temp
	ADD DY, DY, INCY
	cmp N, 0
	beq .L${kernel}_end
	b .L${kernel}_loop

.L${kernel}_end:
	ldp x29, x30, [sp], #16
	ret
	${epilogue(kernel)}
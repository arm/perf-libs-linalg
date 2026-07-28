## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

#define LOOP_MULTIPLE 4
#define ELEM_SIZE 16

	<% func_name = "zswap_kernel" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	// x0 stores n, the length of the vector
	// x1 stores dx, a pointer to the first vector
	// x2 stores dy, a pointer to the second vector
	// incx and incy are assumed to be 1

	// calculate the limit when the computation should terminate
	// stored in x3 (x3 = x1 + x0*sizeof(elem))
	mov x3, ELEM_SIZE
	madd x3, x0, x3, x1


.Loop1:
	// do single increments until x0 is a multiple of 4
	tst x0, LOOP_MULTIPLE-1
	beq .Loop4

	// swap the current element
	ldr q17, [ x1 ]
	ldr q18, [ x2 ]
	str q18, [ x1 ]
	str q17, [ x2 ]

	// increment and repeat
	add x2, x2, ELEM_SIZE
	add x1, x1, ELEM_SIZE
	sub x0, x0, 1

	b .Loop1

.Loop4:
	// check if we've reached the end yet
	cmp x1, x3
	beq .Lcleanup

	// swap the current 4 elements
	ldr q17, [ x1 ]
	ldr q18, [ x1, ELEM_SIZE ]
	ldr q21, [ x2 ]
	ldr q22, [ x2, ELEM_SIZE ]

	ldr q19, [ x1, 2*ELEM_SIZE ]
	ldr q20, [ x1, 3*ELEM_SIZE ]
	ldr q23, [ x2, 2*ELEM_SIZE ]
	ldr q24, [ x2, 3*ELEM_SIZE ]

	str q17, [ x2 ]
	str q18, [ x2, ELEM_SIZE ]
	str q21, [ x1 ]
	str q22, [ x1, ELEM_SIZE ]

	str q19, [ x2, 2*ELEM_SIZE ]
	str q20, [ x2, 3*ELEM_SIZE ]
	str q23, [ x1, 2*ELEM_SIZE ]
	str q24, [ x1, 3*ELEM_SIZE ]

	prfm pldl1keep, [ x1, #512 ]
	prfm pldl1keep, [ x2, #512 ]

	// increment and repeat
	add x1, x1, 4*ELEM_SIZE
	add x2, x2, 4*ELEM_SIZE
	b .Loop4

.Lcleanup:
	ldp x29, x30, [sp], #16
	ret
	${epilogue(func_name)}

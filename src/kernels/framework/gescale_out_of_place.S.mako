## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

// This is similar to the scal kernel, with an additional loop over strd.

cntg .req x0
strd .req x1
a_ptr .req x2
lda .req x3
b_ptr .req x4
ldb .req x5

a_row_end .req x8

% for T in [ 'd', 's', 'h' ]:


<%
if T=='d':
	dbits = 64
	sfx = 'd'
	shft = 3
elif T=='s':
	dbits = 32
	sfx = 's'
	shft = 2
elif T=='h':
	dbits = 16
	sfx = 'h'
	shft = 1
%>
<%
cache_line_size = 64
dbytes = int(dbits / 8)
vwidth = 128
vbytes = int(vwidth / 8)
velems = int(vwidth / dbits)
loop_multiple = int(cache_line_size / dbytes)
name = T + "gescal_out_of_place"
L= "L_"+ name
%>
	//scalar val
	sval .req ${sfx}0
	treg .req ${sfx}17

	${prologue(name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	mov x6, cntg
	mov x9, a_ptr
	mov x7, b_ptr

	// broadcast the scale factor in d0 to all of v16
	dup v16.${velems}${sfx}, v0.${sfx}[0]

	b .${L}strd_loop_cond
.${L}strd_loop:

	mov a_ptr, x9
	mov b_ptr, x7
	mov cntg, x6

	// calculate the limit when the computation should terminate
	// stored in x4 (x4 = x1 + x0*sizeof(elem))
	mov a_row_end, ${dbytes}
	madd a_row_end, cntg, a_row_end, a_ptr

.${L}oop1:
	// do single increments until x0 is a multiple of 8
	tst cntg, ${loop_multiple}-1
	beq .${L}oop4



	// scale the current element
	ldr treg, [ a_ptr ]
	add a_ptr, a_ptr, ${dbytes}
	fmul treg, treg, sval
	str treg, [ b_ptr ]
	add b_ptr, b_ptr, ${dbytes}

	// increment and repeat
	sub cntg, cntg, 1

	b .${L}oop1

.${L}oop4:
	// check if we've reached the end yet
	cmp a_ptr, a_row_end
	beq .${L}done_loop_4

	// scale the current 8 elements
	ldp q17, q18, [ a_ptr ]
	ldp q19, q20, [ a_ptr, ${vbytes*2}]
	add a_ptr, a_ptr, ${vbytes*4}

	fmul v17.${velems}${sfx}, v17.${velems}${sfx}, v16.${velems}${sfx}
	fmul v18.${velems}${sfx}, v18.${velems}${sfx}, v16.${velems}${sfx}
	fmul v19.${velems}${sfx}, v19.${velems}${sfx}, v16.${velems}${sfx}
	fmul v20.${velems}${sfx}, v20.${velems}${sfx}, v16.${velems}${sfx}

	stp q17, q18, [ b_ptr ]
	stp q19, q20, [ b_ptr, ${vbytes*2} ]

	prfm pldl1keep, [ a_ptr, #512 ]

	// increment and repeat
	add b_ptr, b_ptr, ${vbytes*4}
	b .${L}oop4

.${L}done_loop_4:
	sub strd, strd, #1
	add x9, x9, lda, lsl #${shft}
	add x7, x7, ldb, lsl #${shft}

.${L}strd_loop_cond:
	cmp strd, #0
	bgt .${L}strd_loop

	ldp x29, x30, [sp], #16
	ret


	.unreq sval
	.unreq treg

	${epilogue(name)}

% endfor

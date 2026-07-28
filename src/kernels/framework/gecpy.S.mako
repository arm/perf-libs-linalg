## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

cntg .req x0
strd .req x1
a_ptr .req x2
lda .req x3
b_ptr .req x4
ldb .req x5

cntg_t .req x6
a_cur .req x7
b_cur .req x8

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
name = T + "gecpy"
L= "L_"+ name
%>

	${prologue(name)}
	cmp strd, 0
	b .${L}_strd_loop_cond
.${L}_strd_loop:
	mov cntg_t, cntg
	mov a_cur, a_ptr
	mov b_cur, b_ptr
	b .${L}_cntg_loop_cond

.${L}_cntg_loop:
	sub cntg_t, cntg_t, 1
	ldr ${sfx}1, [a_cur]
	add a_cur, a_cur, ${dbytes}
	str ${sfx}1, [b_cur]
	add b_cur, b_cur, ${dbytes}

.${L}_cntg_loop_cond:
	tst cntg_t, ${loop_multiple}-1
	bne .${L}_cntg_loop

	cmp cntg_t, 0
	b .${L}_cntg_loop_fast_cond

.${L}_cntg_loop_fast:
	subs cntg_t, cntg_t, ${4*velems}
	ldp q1, q2, [ a_cur ]
	ldp q3, q4, [ a_cur, ${2*vbytes} ]
	add a_cur, a_cur, ${4*vbytes}

	stp q1, q2, [ b_cur ]
	stp q3, q4, [ b_cur, ${2*vbytes} ]
	add b_cur, b_cur, ${4*vbytes}

.${L}_cntg_loop_fast_cond:
	bne .${L}_cntg_loop_fast

	subs strd, strd, 1
	//increment the row ptrs
	add a_ptr, a_ptr, lda, lsl ${shft}
	add b_ptr, b_ptr, ldb, lsl ${shft}

.${L}_strd_loop_cond:
	bne .${L}_strd_loop

.${L}_end:
	ret

	${epilogue(name)}

%endfor

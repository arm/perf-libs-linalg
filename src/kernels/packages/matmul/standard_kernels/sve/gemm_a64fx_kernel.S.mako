## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

<% import math %>
##produces the start of the BCMS loop used to iterate over A & B from a given start idx
<%def name="loop_start(n)">
	mov ${n}_idx, ${n}_start
	mov ${n}_end, ${n}
	b .${L}_${n}_loop_cond
.${L}_${n}_loop:
</%def>

<%def name="loop_end(n, add_instr)">
	${add_instr.format(n + "_idx")}
.${L}_${n}_loop_cond:
	cmp ${n}_idx, ${n}_end
	blt .${L}_${n}_loop

	cbz ${n}_start, .${L}_${n}_loop_end

	mov ${n}_start, #0
	mov ${n}_idx, #0
	mov ${n}_end, ${n}_start
.${L}_${n}_loop_end:
	mov ${n}_start, #0
	mov ${n}_end, ${n}
</%def>
##We create two versions of the kernel,
##with different loop orders for A and B

A_ptr .req x0
B_ptr .req x1
C_ptr .req x2

k .req x3
m .req x4
n .req x5

lda .req x6
ldb .req x7
ldc .req x8

m_start .req x9
n_start .req x10
n_idx .req x11
n_end .req x12
n_tmp .req x13
m_idx .req x14
m_end .req x15
m_tmp .req x16
k_idx .req x17
A_cur_ptr .req x19
B_cur_ptr .req x20
A_cur_ptr_k .req x21
B_cur_ptr_k .req x22
C_cur_ptr .req x23
## Note that if you add more x registers here you will need to
## change the values req'd for alpha_x and beta_x on lines 106-7

%for dtype in [ "s", "d" ]:
<%
base_name="gemm_a64fx_big"

Creg_N = 9
Creg_M = 2

if dtype == "d":
	bits=64
	sfx = "d"
	word = "d"
	x = "x"
	incvl = "incd"
	inc2vl = "incw"
if dtype == "s":
	bits=32
	sfx = "s"
	word = "w"
	x = "w"
	incvl = "incw"
	inc2vl = "inch"
if dtype == "h":
	bits=16
	sfx = "h"
	word = "h"
	x = "r"
	incvl = "inch"
	inc2vl = "incb"

a_iters=2

inc2vl_format = inc2vl + " {}"

ld1 = "ld1{}".format(word)
ld1r = "ld1r{}".format(word)
st1 = "st1{}".format(word)
inc = "inc{}".format(word)
lanes = int(512 / bits)
bytes = int(bits / 8)
shft = int(math.log(bytes, 2))
##the following code is used to 'allocate' registers to varying uses
##all register use in this code should done using aliases
%>

alpha_in .req ${sfx}0
beta_in .req ${sfx}1

## These 2 numbers depend on the x register req's above
// registers used to hold the values of Alpha & Beta when there are no free vector regs
alpha_x .req ${x}24
beta_x .req ${x}25

all_true_prd .req p0

<% preg = 1 %>\
<% zreg = 0 %>\

%for crm in range(0, Creg_M):
%for it in range(0, a_iters):
	A_it${it}_m${crm} .req z${zreg}
	<% zreg += 1 %>\
%endfor
	C_m${crm} .req A_it0_m${crm}
	m_prd_${crm} .req p${preg}
	<% preg += 1 %>\
%endfor
%for crn in range(0, Creg_N):
	B_n${crn} .req z${zreg}
	<% zreg += 1 %>\
%endfor

alpha_z .req A_it1_m0
beta_z .req B_n0

//reserve the register block
%for crn in range(0, Creg_N):
%for crm in range(0, Creg_M):
	C_m${crm}_n${crn} .req z${zreg}
	<% zreg += 1 %>\
%endfor
%endfor

%for ab in { 'A', 'B' }:
<% name="{}{}_{}".format(dtype, base_name, ab) %>
	${prologue(name)}
${name}_prologue:
	##args 9 and 10 are passed on stack
	ldp x8, x9, [ sp ]
	ldr x10, [sp, 16]

	stp x29, x30, [sp, #-320]!
	mov x29, sp
	stp x19, x20, [ sp, #224 ]
	stp x21, x22, [ sp, #208 ]
	stp x23, x24, [ sp, #192 ]
	stp x25, x26, [ sp, #176 ]
	stp x27, x28, [ sp, #160 ]
	stp d8, d9, [ sp, #80 ]
	stp d10, d11, [ sp, #64 ]
	stp d12, d13, [ sp, #48 ]
	stp d14, d15, [ sp, #32 ]

	//shift now so we don't have to do so later
	lsl lda, lda, ${shft}
	lsl ldb, ldb, ${shft}

	ptrue all_true_prd.${sfx}

	fmov alpha_x, alpha_in
	fmov beta_x, beta_in


//if beta == 0
	fcmp beta_in, 0.0e+0
	beq .L_${name}_b0
//if beta == 1
	fmov alpha_in, 1.0e+0
	fcmp beta_in, alpha_in
	beq .L_${name}_b1
//else
	b .L_${name}_b2


%for beta in { 0, 1, 2 }:
<% L = "L_{}_b{}".format(name, beta) %>
.${L}:
//zero all of the C reg block, this is done in subsequent iterations in the save code
%for crn in range(0, Creg_N):
%for crm in range(0, Creg_M):
	## the non-predicated SVE eor only exists with a .d suffix
	eor C_m${crm}_n${crn}.d, C_m${crm}_n${crn}.d, C_m${crm}_n${crn}.d
%endfor
%endfor
% if ab == "A":
	mov B_cur_ptr, B_ptr
	${loop_start('n')}
	mov A_cur_ptr, A_ptr
	${loop_start('m')}
% else:
	mov A_cur_ptr, A_ptr
	${loop_start('m')}
	mov B_cur_ptr, B_ptr
	${loop_start('n')}
%endif ##if ab
	mov A_cur_ptr_k, A_cur_ptr
	mov B_cur_ptr_k, B_cur_ptr

	//madd A_cur_ptr_k, m_idx, lda, A_ptr
	//madd B_cur_ptr_k, n_idx, ldb, B_ptr
.${L}_k_loop:
	//recent k and subs to account for initial load
	subs k_idx, k, 1
	${ld1} { A_it0_m0.${sfx} }, all_true_prd/z, [ A_cur_ptr_k, 0, mul vl]
	${ld1} { A_it0_m1.${sfx} }, all_true_prd/z, [ A_cur_ptr_k, 1, mul vl]
	//${inc} A_cur_ptr_k, all, mul #2
	addvl A_cur_ptr_k, A_cur_ptr_k, 2

	${ld1r} { B_n0.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${0*bytes} ]
	${ld1r} { B_n1.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${1*bytes} ]
	${ld1r} { B_n2.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${2*bytes} ]
	${ld1r} { B_n3.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${3*bytes} ]
	${ld1r} { B_n4.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${4*bytes} ]
	${ld1r} { B_n5.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${5*bytes} ]
	${ld1r} { B_n6.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${6*bytes} ]
	${ld1r} { B_n7.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${7*bytes} ]
	${ld1r} { B_n8.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${8*bytes} ]
	add B_cur_ptr_k, B_cur_ptr_k , ${9*bytes}

	//we only have one iterations worth of work
	beq .${L}_k_loop_it0_1
	subs k_idx, k_idx, 1

	//else we have at least two iterations
	${ld1} { A_it1_m0.${sfx} }, all_true_prd/z, [ A_cur_ptr_k, 0, mul vl]
	beq .${L}_k_loop_it0_out

.${L}_k_loop_it0:
#CYCLE 1
	fmla C_m0_n0.${sfx}, all_true_prd/m, B_n0.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n0.${sfx}, all_true_prd/m, B_n0.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n0.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${0*bytes} ]
	${ld1} { A_it1_m1.${sfx} }, all_true_prd/z, [ A_cur_ptr_k, 1, mul vl]
#CYCLE 2
	fmla C_m0_n1.${sfx}, all_true_prd/m, B_n1.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n1.${sfx}, all_true_prd/m, B_n1.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n1.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${1*bytes} ]
	//${inc} A_cur_ptr_k, all, mul #2
	addvl A_cur_ptr_k, A_cur_ptr_k, 2
#CYCLE 3
	fmla C_m0_n2.${sfx}, all_true_prd/m, B_n2.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n2.${sfx}, all_true_prd/m, B_n2.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n2.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${2*bytes} ]
	subs k_idx, k_idx, 1
#CYCLE 4
	fmla C_m0_n3.${sfx}, all_true_prd/m, B_n3.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n3.${sfx}, all_true_prd/m, B_n3.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n3.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${3*bytes} ]
#CYCLE 5
	fmla C_m0_n4.${sfx}, all_true_prd/m, B_n4.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n4.${sfx}, all_true_prd/m, B_n4.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n4.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${4*bytes} ]
#CYCLE 6
	fmla C_m0_n5.${sfx}, all_true_prd/m, B_n5.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n5.${sfx}, all_true_prd/m, B_n5.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n5.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${5*bytes} ]
#CYCLE 7
	fmla C_m0_n6.${sfx}, all_true_prd/m, B_n6.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n6.${sfx}, all_true_prd/m, B_n6.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n6.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${6*bytes} ]
#CYCLE 8
	fmla C_m0_n7.${sfx}, all_true_prd/m, B_n7.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n7.${sfx}, all_true_prd/m, B_n7.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n7.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${7*bytes} ]
#CYCLE 9
	fmla C_m0_n8.${sfx}, all_true_prd/m, B_n8.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n8.${sfx}, all_true_prd/m, B_n8.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n8.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${8*bytes} ]
	${ld1} { A_it0_m0.${sfx} }, all_true_prd/z, [ A_cur_ptr_k, 0, mul vl]
	add B_cur_ptr_k, B_cur_ptr_k, ${9*bytes}

	beq .${L}_k_loop_it1_out

#CYCLE 1
	fmla C_m0_n0.${sfx}, all_true_prd/m, B_n0.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n0.${sfx}, all_true_prd/m, B_n0.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n0.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${0*bytes} ]
	${ld1} { A_it0_m1.${sfx} }, all_true_prd/z, [ A_cur_ptr_k, 1, mul vl]
#CYCLE 2
	fmla C_m0_n1.${sfx}, all_true_prd/m, B_n1.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n1.${sfx}, all_true_prd/m, B_n1.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n1.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${1*bytes} ]
	//${inc} A_cur_ptr_k, all, mul #2
	addvl A_cur_ptr_k, A_cur_ptr_k, 2
#CYCLE 3
	fmla C_m0_n2.${sfx}, all_true_prd/m, B_n2.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n2.${sfx}, all_true_prd/m, B_n2.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n2.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${2*bytes} ]
	subs k_idx, k_idx, 1
#CYCLE 4
	fmla C_m0_n3.${sfx}, all_true_prd/m, B_n3.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n3.${sfx}, all_true_prd/m, B_n3.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n3.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${3*bytes} ]
#CYCLE 5
	fmla C_m0_n4.${sfx}, all_true_prd/m, B_n4.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n4.${sfx}, all_true_prd/m, B_n4.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n4.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${4*bytes} ]
#CYCLE 6
	fmla C_m0_n5.${sfx}, all_true_prd/m, B_n5.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n5.${sfx}, all_true_prd/m, B_n5.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n5.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${5*bytes} ]
#CYCLE 7
	fmla C_m0_n6.${sfx}, all_true_prd/m, B_n6.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n6.${sfx}, all_true_prd/m, B_n6.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n6.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${6*bytes} ]
#CYCLE 8
	fmla C_m0_n7.${sfx}, all_true_prd/m, B_n7.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n7.${sfx}, all_true_prd/m, B_n7.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n7.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${7*bytes} ]
#CYCLE 9
	fmla C_m0_n8.${sfx}, all_true_prd/m, B_n8.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n8.${sfx}, all_true_prd/m, B_n8.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n8.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${8*bytes} ]
	${ld1} { A_it1_m0.${sfx} }, all_true_prd/z, [ A_cur_ptr_k, 0, mul vl]
	add B_cur_ptr_k, B_cur_ptr_k, ${9*bytes}

	beq .${L}_k_loop_it0_out

#CYCLE 1
	fmla C_m0_n0.${sfx}, all_true_prd/m, B_n0.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n0.${sfx}, all_true_prd/m, B_n0.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n0.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${0*bytes} ]
	${ld1} { A_it1_m1.${sfx} }, all_true_prd/z, [ A_cur_ptr_k, 1, mul vl]
#CYCLE 2
	fmla C_m0_n1.${sfx}, all_true_prd/m, B_n1.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n1.${sfx}, all_true_prd/m, B_n1.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n1.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${1*bytes} ]
	//${inc} A_cur_ptr_k, all, mul #2
	addvl A_cur_ptr_k, A_cur_ptr_k, 2
#CYCLE 3
	fmla C_m0_n2.${sfx}, all_true_prd/m, B_n2.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n2.${sfx}, all_true_prd/m, B_n2.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n2.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${2*bytes} ]
	subs k_idx, k_idx, 1
#CYCLE 4
	fmla C_m0_n3.${sfx}, all_true_prd/m, B_n3.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n3.${sfx}, all_true_prd/m, B_n3.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n3.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${3*bytes} ]
#CYCLE 5
	fmla C_m0_n4.${sfx}, all_true_prd/m, B_n4.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n4.${sfx}, all_true_prd/m, B_n4.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n4.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${4*bytes} ]
#CYCLE 6
	fmla C_m0_n5.${sfx}, all_true_prd/m, B_n5.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n5.${sfx}, all_true_prd/m, B_n5.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n5.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${5*bytes} ]
#CYCLE 7
	fmla C_m0_n6.${sfx}, all_true_prd/m, B_n6.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n6.${sfx}, all_true_prd/m, B_n6.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n6.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${6*bytes} ]
#CYCLE 8
	fmla C_m0_n7.${sfx}, all_true_prd/m, B_n7.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n7.${sfx}, all_true_prd/m, B_n7.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n7.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${7*bytes} ]
#CYCLE 9
	fmla C_m0_n8.${sfx}, all_true_prd/m, B_n8.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n8.${sfx}, all_true_prd/m, B_n8.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n8.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${8*bytes} ]
	${ld1} { A_it0_m0.${sfx} }, all_true_prd/z, [ A_cur_ptr_k, 0, mul vl]
	add B_cur_ptr_k, B_cur_ptr_k, ${9*bytes}

	beq .${L}_k_loop_it1_out

#CYCLE 1
	fmla C_m0_n0.${sfx}, all_true_prd/m, B_n0.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n0.${sfx}, all_true_prd/m, B_n0.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n0.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${0*bytes} ]
	${ld1} { A_it0_m1.${sfx} }, all_true_prd/z, [ A_cur_ptr_k, 1, mul vl]
#CYCLE 2
	fmla C_m0_n1.${sfx}, all_true_prd/m, B_n1.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n1.${sfx}, all_true_prd/m, B_n1.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n1.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${1*bytes} ]
	//${inc} A_cur_ptr_k, all, mul #2
	addvl A_cur_ptr_k, A_cur_ptr_k, 2
#CYCLE 3
	fmla C_m0_n2.${sfx}, all_true_prd/m, B_n2.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n2.${sfx}, all_true_prd/m, B_n2.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n2.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${2*bytes} ]
	subs k_idx, k_idx, 1
#CYCLE 4
	fmla C_m0_n3.${sfx}, all_true_prd/m, B_n3.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n3.${sfx}, all_true_prd/m, B_n3.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n3.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${3*bytes} ]
#CYCLE 5
	fmla C_m0_n4.${sfx}, all_true_prd/m, B_n4.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n4.${sfx}, all_true_prd/m, B_n4.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n4.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${4*bytes} ]
#CYCLE 6
	fmla C_m0_n5.${sfx}, all_true_prd/m, B_n5.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n5.${sfx}, all_true_prd/m, B_n5.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n5.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${5*bytes} ]
#CYCLE 7
	fmla C_m0_n6.${sfx}, all_true_prd/m, B_n6.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n6.${sfx}, all_true_prd/m, B_n6.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n6.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${6*bytes} ]
#CYCLE 8
	fmla C_m0_n7.${sfx}, all_true_prd/m, B_n7.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n7.${sfx}, all_true_prd/m, B_n7.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n7.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${7*bytes} ]
#CYCLE 9
	fmla C_m0_n8.${sfx}, all_true_prd/m, B_n8.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n8.${sfx}, all_true_prd/m, B_n8.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n8.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${8*bytes} ]
	${ld1} { A_it1_m0.${sfx} }, all_true_prd/z, [ A_cur_ptr_k, 0, mul vl]


	add B_cur_ptr_k, B_cur_ptr_k, ${9*bytes}

	bne .${L}_k_loop_it0
	//falls through
	//b  .${L}_k_loop_it0_out

/*
 * It0 k_loop_out
 */
#CYCLE 0
.${L}_k_loop_it0_out:
#CYCLE 1
	fmla C_m0_n0.${sfx}, all_true_prd/m, B_n0.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n0.${sfx}, all_true_prd/m, B_n0.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n0.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${0*bytes} ]
	${ld1} { A_it1_m1.${sfx} }, all_true_prd/z, [ A_cur_ptr_k, 1, mul vl]
#CYCLE 2
	fmla C_m0_n1.${sfx}, all_true_prd/m, B_n1.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n1.${sfx}, all_true_prd/m, B_n1.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n1.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${1*bytes} ]
	//${inc} A_cur_ptr_k, all, mul #2
	addvl A_cur_ptr_k, A_cur_ptr_k, 2
#CYCLE 3
	fmla C_m0_n2.${sfx}, all_true_prd/m, B_n2.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n2.${sfx}, all_true_prd/m, B_n2.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n2.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${2*bytes} ]
	subs k_idx, k_idx, 1
#CYCLE 4
	fmla C_m0_n3.${sfx}, all_true_prd/m, B_n3.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n3.${sfx}, all_true_prd/m, B_n3.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n3.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${3*bytes} ]
#CYCLE 5
	fmla C_m0_n4.${sfx}, all_true_prd/m, B_n4.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n4.${sfx}, all_true_prd/m, B_n4.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n4.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${4*bytes} ]
#CYCLE 6
	fmla C_m0_n5.${sfx}, all_true_prd/m, B_n5.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n5.${sfx}, all_true_prd/m, B_n5.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n5.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${5*bytes} ]
#CYCLE 7
	fmla C_m0_n6.${sfx}, all_true_prd/m, B_n6.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n6.${sfx}, all_true_prd/m, B_n6.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n6.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${6*bytes} ]
#CYCLE 8
	fmla C_m0_n7.${sfx}, all_true_prd/m, B_n7.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n7.${sfx}, all_true_prd/m, B_n7.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n7.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${7*bytes} ]
#CYCLE 9
	fmla C_m0_n8.${sfx}, all_true_prd/m, B_n8.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n8.${sfx}, all_true_prd/m, B_n8.${sfx}, A_it0_m1.${sfx}
	${ld1r} { B_n8.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${8*bytes} ]
#CYCLE 1
	fmla C_m0_n0.${sfx}, all_true_prd/m, B_n0.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n0.${sfx}, all_true_prd/m, B_n0.${sfx}, A_it1_m1.${sfx}
#CYCLE 2
	fmla C_m0_n1.${sfx}, all_true_prd/m, B_n1.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n1.${sfx}, all_true_prd/m, B_n1.${sfx}, A_it1_m1.${sfx}
#CYCLE 3
	fmla C_m0_n2.${sfx}, all_true_prd/m, B_n2.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n2.${sfx}, all_true_prd/m, B_n2.${sfx}, A_it1_m1.${sfx}
#CYCLE 4
	fmla C_m0_n3.${sfx}, all_true_prd/m, B_n3.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n3.${sfx}, all_true_prd/m, B_n3.${sfx}, A_it1_m1.${sfx}
#CYCLE 5
	fmla C_m0_n4.${sfx}, all_true_prd/m, B_n4.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n4.${sfx}, all_true_prd/m, B_n4.${sfx}, A_it1_m1.${sfx}
#CYCLE 6
	fmla C_m0_n5.${sfx}, all_true_prd/m, B_n5.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n5.${sfx}, all_true_prd/m, B_n5.${sfx}, A_it1_m1.${sfx}
#CYCLE 7
	fmla C_m0_n6.${sfx}, all_true_prd/m, B_n6.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n6.${sfx}, all_true_prd/m, B_n6.${sfx}, A_it1_m1.${sfx}
#CYCLE 8
	fmla C_m0_n7.${sfx}, all_true_prd/m, B_n7.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n7.${sfx}, all_true_prd/m, B_n7.${sfx}, A_it1_m1.${sfx}
#CYCLE 9
	fmla C_m0_n8.${sfx}, all_true_prd/m, B_n8.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n8.${sfx}, all_true_prd/m, B_n8.${sfx}, A_it1_m1.${sfx}

	b .${L}_k_loop_end

/*
 * It1 k_loop_out
 */
#CYCLE 0
.${L}_k_loop_it1_out:
#CYCLE 1
	fmla C_m0_n0.${sfx}, all_true_prd/m, B_n0.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n0.${sfx}, all_true_prd/m, B_n0.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n0.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${0*bytes} ]
	${ld1} { A_it0_m1.${sfx} }, all_true_prd/z, [ A_cur_ptr_k, 1, mul vl]
#CYCLE 2
	fmla C_m0_n1.${sfx}, all_true_prd/m, B_n1.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n1.${sfx}, all_true_prd/m, B_n1.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n1.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${1*bytes} ]
	//${inc} A_cur_ptr_k, all, mul #2
	addvl A_cur_ptr_k, A_cur_ptr_k, 2
#CYCLE 3
	fmla C_m0_n2.${sfx}, all_true_prd/m, B_n2.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n2.${sfx}, all_true_prd/m, B_n2.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n2.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${2*bytes} ]
	subs k_idx, k_idx, 1
#CYCLE 4
	fmla C_m0_n3.${sfx}, all_true_prd/m, B_n3.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n3.${sfx}, all_true_prd/m, B_n3.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n3.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${3*bytes} ]
#CYCLE 5
	fmla C_m0_n4.${sfx}, all_true_prd/m, B_n4.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n4.${sfx}, all_true_prd/m, B_n4.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n4.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${4*bytes} ]
#CYCLE 6
	fmla C_m0_n5.${sfx}, all_true_prd/m, B_n5.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n5.${sfx}, all_true_prd/m, B_n5.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n5.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${5*bytes} ]
#CYCLE 7
	fmla C_m0_n6.${sfx}, all_true_prd/m, B_n6.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n6.${sfx}, all_true_prd/m, B_n6.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n6.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${6*bytes} ]
#CYCLE 8
	fmla C_m0_n7.${sfx}, all_true_prd/m, B_n7.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n7.${sfx}, all_true_prd/m, B_n7.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n7.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${7*bytes} ]
#CYCLE 9
	fmla C_m0_n8.${sfx}, all_true_prd/m, B_n8.${sfx}, A_it1_m0.${sfx}
	fmla C_m1_n8.${sfx}, all_true_prd/m, B_n8.${sfx}, A_it1_m1.${sfx}
	${ld1r} { B_n8.${sfx} }, all_true_prd/z, [ B_cur_ptr_k, ${8*bytes} ]
.${L}_k_loop_it0_1:
#CYCLE 1
	fmla C_m0_n0.${sfx}, all_true_prd/m, B_n0.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n0.${sfx}, all_true_prd/m, B_n0.${sfx}, A_it0_m1.${sfx}
#CYCLE 2
	fmla C_m0_n1.${sfx}, all_true_prd/m, B_n1.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n1.${sfx}, all_true_prd/m, B_n1.${sfx}, A_it0_m1.${sfx}
#CYCLE 3
	fmla C_m0_n2.${sfx}, all_true_prd/m, B_n2.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n2.${sfx}, all_true_prd/m, B_n2.${sfx}, A_it0_m1.${sfx}
#CYCLE 4
	fmla C_m0_n3.${sfx}, all_true_prd/m, B_n3.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n3.${sfx}, all_true_prd/m, B_n3.${sfx}, A_it0_m1.${sfx}
#CYCLE 5
	fmla C_m0_n4.${sfx}, all_true_prd/m, B_n4.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n4.${sfx}, all_true_prd/m, B_n4.${sfx}, A_it0_m1.${sfx}
#CYCLE 6
	fmla C_m0_n5.${sfx}, all_true_prd/m, B_n5.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n5.${sfx}, all_true_prd/m, B_n5.${sfx}, A_it0_m1.${sfx}
#CYCLE 7
	fmla C_m0_n6.${sfx}, all_true_prd/m, B_n6.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n6.${sfx}, all_true_prd/m, B_n6.${sfx}, A_it0_m1.${sfx}
#CYCLE 8
	fmla C_m0_n7.${sfx}, all_true_prd/m, B_n7.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n7.${sfx}, all_true_prd/m, B_n7.${sfx}, A_it0_m1.${sfx}
#CYCLE 9
	fmla C_m0_n8.${sfx}, all_true_prd/m, B_n8.${sfx}, A_it0_m0.${sfx}
	fmla C_m1_n8.${sfx}, all_true_prd/m, B_n8.${sfx}, A_it0_m1.${sfx}

	b .${L}_k_loop_end

.${L}_k_loop_end:

///K_LOOP_END


.${L}_c_write_out:
	madd C_cur_ptr, n_idx, ldc, m_idx
	add C_cur_ptr, C_ptr, C_cur_ptr, lsl #${shft}

	mov m_tmp, m_idx
%for crm in range(0, Creg_M):
	whilelt m_prd_${crm}.${sfx}, m_tmp, m_end
%if crm+1 != Creg_M:
	${inc} m_tmp
%endif
%endfor

//fetch alpha and beta out of the general purpose registers we were hiding them in.
%if beta != 0:
	dup beta_z.${sfx}, beta_x
%endif
	dup alpha_z.${sfx}, alpha_x

	subs n_tmp, n_end, n_idx
%for crn in range(0, Creg_N):
	beq .${L}_c_write_out_end
%for crm in range(0, Creg_M):
%if beta == 0:
	fmul C_m${crm}_n${crn}.${sfx}, all_true_prd/m, C_m${crm}_n${crn}.${sfx}, alpha_z.${sfx}
	${st1} { C_m${crm}_n${crn}.${sfx} }, m_prd_${crm}, [C_cur_ptr, #${crm}, mul vl]
%elif beta == 1:
	${ld1} { C_m${crm}.${sfx} }, m_prd_${crm}/z, [C_cur_ptr, #${crm}, mul vl]
	fmla C_m${crm}.${sfx}, all_true_prd/m, C_m${crm}_n${crn}.${sfx}, alpha_z.${sfx}
	${st1} { C_m${crm}.${sfx} }, m_prd_${crm}, [C_cur_ptr, #${crm}, mul vl]
%else:
	${ld1} { C_m${crm}.${sfx} }, m_prd_${crm}/z, [C_cur_ptr, #${crm}, mul vl]
	fmul C_m${crm}_n${crn}.${sfx}, all_true_prd/m, C_m${crm}_n${crn}.${sfx}, alpha_z.${sfx}
	fmla C_m${crm}_n${crn}.${sfx}, all_true_prd/m, C_m${crm}.${sfx}, beta_z.${sfx}
	${st1} { C_m${crm}_n${crn}.${sfx} }, m_prd_${crm}, [C_cur_ptr, #${crm}, mul vl]
%endif
	## the non-predicated SVE eor only exists with a .d suffix
	eor C_m${crm}_n${crn}.d, C_m${crm}_n${crn}.d, C_m${crm}_n${crn}.d
%endfor
	add C_cur_ptr, C_cur_ptr, ldc, lsl #${shft}
	subs n_tmp, n_tmp, #1
%endfor

.${L}_c_write_out_end:

% if ab == "A":
	add A_cur_ptr, A_cur_ptr, lda
	${loop_end('m', inc2vl_format)}
	add B_cur_ptr, B_cur_ptr, ldb
	${loop_end('n', "add {0}, {0}, " + str(Creg_N))}
% else:
	add B_cur_ptr, B_cur_ptr, ldb
	${loop_end('n', "add {0}, {0}, " + str(Creg_N))}
	add A_cur_ptr, A_cur_ptr, lda
	${loop_end('m', inc2vl_format)}

%endif ##if ab
	b .L_${name}_end
%endfor ##beta

.L_${name}_end:
	ldp x19, x20, [ sp, #224 ]
	ldp x21, x22, [ sp, #208 ]
	ldp x23, x24, [ sp, #192 ]
	ldp x25, x26, [ sp, #176 ]
	ldp x27, x28, [ sp, #160 ]
	ldp d8, d9, [ sp, #80 ]
	ldp d10, d11, [ sp, #64 ]
	ldp d12, d13, [ sp, #48 ]
	ldp d14, d15, [ sp, #32 ]
	ldp x29, x30, [sp], #320
	ret
	${epilogue(name)}
%endfor

.unreq alpha_in
.unreq beta_in

.unreq alpha_x
.unreq beta_x

.unreq all_true_prd

%for crm in range(0, Creg_M):
%for it in range(0, a_iters):
	.unreq A_it${it}_m${crm}
%endfor
	.unreq C_m${crm}
	.unreq m_prd_${crm}
%endfor
%for crn in range(0, Creg_N):
	.unreq B_n${crn}
%endfor

.unreq alpha_z
.unreq beta_z

%for crn in range(0, Creg_N):
%for crm in range(0, Creg_M):
	.unreq C_m${crm}_n${crn}
%endfor
%endfor

%endfor ## dtype = ...

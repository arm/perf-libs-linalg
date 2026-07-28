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



# Stamp out the inner loop for a given MxN dimension
<%def name="inner_loop(creg_m, creg_n, reg_lanes, b_iters, vbytes, sflane, sfx)">
#we need a version of this inner loop for each iter, they'll be identical except
#that they use different registers
<%
n_elems = creg_n*reg_lanes
m_elems = creg_m*reg_lanes
label_prefix = ".{}_k_loop_m{}_n{}".format(L, m_elems, n_elems)
%>
${label_prefix}:
	subs k_idx, k, 1
%for crn in range(0, creg_m):
	ldr A_it0_m${crn}q, [ A_cur_ptr_k, ${vbytes*crn} ]
%endfor
	add A_cur_ptr_k, A_cur_ptr_k, ${vbytes*Creg_M}

%for crn in range(0, creg_n):
	ldr B_it0_n${crn}q, [ B_cur_ptr_k, ${vbytes*crn} ]
%endfor
	add B_cur_ptr_k, B_cur_ptr_k, ${vbytes*Creg_N}

	//we only have one iterations worth of work
	beq ${label_prefix}_it0_out


%for it in range(0, b_iters):
<%
last = (it + 1) == b_iters
it_next = (it + 1) % b_iters
%>
${label_prefix}_it${it}:
	subs k_idx, k_idx, 1
// Load the NEXT iterations B values
%for n in range(0, n_elems, reg_lanes):
	ldr B_it${it_next}_n${n}q, [ B_cur_ptr_k, ${n * vbytes + it * Creg_N * vbytes} ]
%endfor

%for n in range(0, n_elems, reg_lanes):
%for m in range(0, creg_m):
	fmla C_m${m}_n${n}.${sflane}, A_it0_m${m}.${sflane}, B_it${it}_n${n}.${sfx}[0]
	fmla C_m${m}_n${n+1}.${sflane}, A_it0_m${m}.${sflane}, B_it${it}_n${n}.${sfx}[1]
	ldr A_it0_m${m}q, [ A_cur_ptr_k, ${m * vbytes + it * Creg_M * vbytes} ]
%endfor
%endfor
	beq ${label_prefix}_it${it_next}_out

%if last:
	add B_cur_ptr_k, B_cur_ptr_k, ${Creg_N * vbytes * b_iters}
	add A_cur_ptr_k, A_cur_ptr_k, ${Creg_M * vbytes * b_iters}
	b ${label_prefix}_it0
%endif
%endfor

%for it in range(0, b_iters):
${label_prefix}_it${it}_out:
%for n in range(0, n_elems, reg_lanes):
%for m in range(0, creg_m):
	fmla C_m${m}_n${n}.${sflane}, A_it0_m${m}.${sflane}, B_it${it}_n${n}.${sfx}[0]
	fmla C_m${m}_n${n+1}.${sflane}, A_it0_m${m}.${sflane}, B_it${it}_n${n}.${sfx}[1]
%endfor
%endfor
	b ${label_prefix}_end
%endfor

${label_prefix}_end:
</%def>

<%def name="write_out(btnm, creg_m, creg_n, reg_lanes, b_iters, vbytes, sflane, sfx)">
.${L}_c_write_out_${btnm}_m${m_elems}:
%if alpha == 2:
	dup alpha_v.${sflane}, alpha_x
%endif
%if beta == 2:
	dup beta_v.${sflane}, beta_x
%endif
%for n in range(0, Creg_N*lanes):
	subs n_tmpx, n_tmpx, 1
%for m_reg in range(0, full_lanes):
%if beta == 0:
	//nothing to load ${full_lanes}
%if alpha == 2:
	fmul C_m${m_reg}_n${n}.${sflane}, C_m${m_reg}_n${n}.${sflane}, alpha_v.${sfx}[0]
%endif #alpha == 2
%if alpha > 0:
	str C_m${m_reg}_n${n}q, [ C_cur_ptr, ${m_reg*vbytes} ]
%endif #alpha > 0
%elif beta == 1:
%if alpha == 0:
	//beta == 1 alpha == 0 is a no-op
%else:
	ldr C_m${m_reg}q, [ C_cur_ptr, ${m_reg*vbytes} ]
%if alpha == 1:
	fadd C_m${m_reg}.${sflane}, C_m${m_reg}.${sflane}, C_m${m_reg}_n${n}.${sflane}
%elif alpha == 2:
	fmla C_m${m_reg}.${sflane}, C_m${m_reg}_n${n}.${sflane}, alpha_v.${sfx}[0]
%endif #alpha > 0
	str C_m${m_reg}q, [ C_cur_ptr, ${m_reg*vbytes} ]
%endif
%elif beta == 2:
	ldr C_m${m_reg}q, [ C_cur_ptr, ${m_reg*vbytes} ]
%if alpha == 0:
	fmul C_m${m_reg}_n${n}.${sflane}, C_m${m_reg}.${sflane}, beta_v.${sfx}[0]
%elif alpha == 1:
	fmla C_m${m_reg}_n${n}.${sflane}, C_m${m_reg}.${sflane}, beta_v.${sfx}[0]
%elif alpha == 2:
	fmul C_m${m_reg}_n${n}.${sflane}, C_m${m_reg}_n${n}.${sflane}, alpha_v.${sfx}[0]
	fmla C_m${m_reg}_n${n}.${sflane}, C_m${m_reg}.${sflane}, beta_v.${sfx}[0]
%endif #alpha
	str C_m${m_reg}_n${n}q, [ C_cur_ptr, ${m_reg*vbytes} ]
%endif #beta
	dup C_m${m_reg}_n${n}.${sflane}, ${zeror}
%endfor
%for m_lane in range(0, modu_lanes):
%if beta == 0:
	//nothing to load
%if alpha == 2:
	fmul C_m${full_lanes}_n${n}d, C_m${full_lanes}_n${n}d, alpha_d
%endif #alpha == 2
%if alpha > 0:
	str C_m${full_lanes}_n${n}d, [ C_cur_ptr, ${full_lanes*vbytes + m_lane*bytes} ]
%endif #alpha > 0
%elif beta == 1:
%if alpha == 0:
	//beta == 1 alpha == 0 is a no-op
%else:
	ldr C_m${full_lanes}d, [ C_cur_ptr, ${full_lanes*vbytes + m_lane*bytes} ]
%if alpha == 1:
	fadd C_m${full_lanes}d, C_m${full_lanes}d, C_m${full_lanes}_n${n}d
%elif alpha == 2:
	fmadd C_m${full_lanes}d, C_m${full_lanes}_n${n}d, alpha_d, C_m${full_lanes}d
%endif #alpha > 0
	str C_m${full_lanes}d, [ C_cur_ptr, ${full_lanes*vbytes + m_lane*bytes} ]
%endif
%elif beta == 2:
	ldr C_m${full_lanes}d, [ C_cur_ptr, ${full_lanes*vbytes + m_lane*bytes} ]
%if alpha == 0:
	fmul C_m${full_lanes}_n${n}d, C_m${full_lanes}d, beta_d
%elif alpha == 1:
	fmadd C_m${full_lanes}_n${n}d, C_m${full_lanes}d, beta_d, C_m${full_lanes}_n${n}d
%elif alpha == 2:
	fmul C_m${full_lanes}_n${n}d, C_m${full_lanes}_n${n}d, alpha_d
	fmadd C_m${full_lanes}_n${n}d, C_m${full_lanes}d, beta_d, C_m${full_lanes}_n${n}d
%endif #alpha
	str C_m${full_lanes}_n${n}d, [ C_cur_ptr, ${full_lanes*vbytes + m_lane*bytes} ]
%endif #beta
	dup C_m${full_lanes}_n${n}.${sflane}, ${zeror}
%endfor
	beq .${L}_c_write_out_end
	add C_cur_ptr, C_cur_ptr, ldc, lsl #${shft}
%endfor
	b .${L}_c_write_out_end

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
n_tmpx .req x13
n_tmpw .req w13
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
## change the values req'd for alpha_x and beta_x on lines 282-3

%for dtype in [ "d", "s" ]:

<%
base_name="gemm_20x2_vanilla_no_prefetch"


if dtype == "d":
	Creg_N = 1
	Creg_M = 10
	bits=64
	sfx = "d"
	word = "d"
	x = "x"
	incvl = "incd"
	inc2vl = "incw"
	zeror = "xzr"
	one_in_hex = 0x3ff0000000000000
if dtype == "s":
	Creg_N = 1
	Creg_M = 5
	bits=32
	sfx = "s"
	word = "w"
	x = "w"
	incvl = "incw"
	inc2vl = "inch"
	zeror = "wzr"
	one_in_hex = 0x3f800000
if dtype == "h":
	bits=16
	sfx = "h"
	word = "h"
	x = "r"
	incvl = "inch"
	inc2vl = "incb"

a_iters=1
b_iters=2

ld1 = "ld1{}".format(word)
ld1r = "ld1r{}".format(word)
st1 = "st1{}".format(word)
inc = "inc{}".format(word)
lanes = int(128 / bits)
bytes = int(bits / 8)
vbytes = lanes * bytes

#suffix_lanes ie 2d, 4s..
sflane = "{}{}".format(lanes, sfx)

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

<% zreg = 0 %>

%for crm in range(0, Creg_M):
%for it in range(0, a_iters):
	A_it${it}_m${crm} .req v${zreg}
	A_it${it}_m${crm}d .req d${zreg}
	A_it${it}_m${crm}s .req s${zreg}
	A_it${it}_m${crm}q .req q${zreg}
<% zreg += 1 %>
%endfor
	C_m${crm} .req A_it0_m${crm}
	C_m${crm}d .req A_it0_m${crm}d
	C_m${crm}s .req A_it0_m${crm}s
	C_m${crm}q .req A_it0_m${crm}q
%endfor
%for crn in range(0, Creg_N):
%for it in range(0, b_iters):
	B_it${it}_n${crn} .req v${zreg}
	B_it${it}_n${crn}d .req d${zreg}
	B_it${it}_n${crn}s .req s${zreg}
	B_it${it}_n${crn}q .req q${zreg}
<% zreg += 1 %>
%endfor
%endfor

alpha_v .req B_it0_n0
alpha_d .req B_it0_n0d
alpha_s .req B_it0_n0s
beta_v .req B_it1_n0
beta_d .req B_it1_n0d
beta_s .req B_it1_n0s

//reserve the register block
%for crn in range(0, Creg_N*lanes):
%for crm in range(0, Creg_M):
	C_m${crm}_n${crn} .req v${zreg}
	C_m${crm}_n${crn}d .req d${zreg}
	C_m${crm}_n${crn}s .req s${zreg}
	C_m${crm}_n${crn}q .req q${zreg}
<% zreg += 1 %>
%endfor
%endfor

%for ab in { 'A', 'B' }:
<% name="{}{}_{}".format(dtype, base_name, ab) %>
	${prologue(name)}
${name}_prologue:
	##args 9, 10 & 11 are passed on stack
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


	fmov alpha_x, alpha_in
	fmov beta_x, beta_in

<% L = "L_{}".format(name) %>
.${L}:
//zero all of the C reg block, this is done in subsequent iterations in the save code
%for crn in range(0, Creg_N*lanes):
%for crm in range(0, Creg_M):
	dup C_m${crm}_n${crn}.${sflane}, ${zeror}
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


	/*
	 * Produce the jump table of no. of elems
	 *
	 *
	 */
	sub m_tmp, m_end, m_idx
%for creg_m in range(Creg_M, 0, -1):
<%
m_elems = creg_m * lanes
n_elems = Creg_N * lanes
%>
	cmp m_tmp, ${m_elems-2}
	bgt .${L}_k_loop_m${m_elems}_n${n_elems}
%endfor

#iterate over the  the valid reg divisions
%for m_reg in range(1, Creg_M+1):
%for n_reg in range(1, Creg_N+1):
<%
m_elems = m_reg * lanes
n_elems = n_reg * lanes
full_lanes = int(m_elems / lanes)
modu_lanes = m_elems % lanes
mod_start = full_lanes * lanes
%>
#generate an inner loop which does a (m_elems * n_elems * k) kernel
${inner_loop(m_reg, n_reg, lanes, b_iters, vbytes, sflane, sfx)}

b .${L}_c_write_out

%endfor #n_reg
%endfor #m_reg

.${L}_c_write_out:
	madd C_cur_ptr, n_idx, ldc, m_idx
	add C_cur_ptr, C_ptr, C_cur_ptr, lsl #${shft}

	cbz beta_x, .${L}_c_write_out_b0
	mov n_tmp${x}, ${one_in_hex}
	cmp beta_x, n_tmp${x}
	beq .${L}_c_write_out_b1
	b .${L}_c_write_out_b2


%for beta in range(0, 3):
.${L}_c_write_out_b${int(beta)}:
	cbz alpha_x, .${L}_c_write_out_b${beta}_a0
	mov n_tmp${x}, ${one_in_hex}
	cmp alpha_x, n_tmp${x}
	beq .${L}_c_write_out_b${beta}_a1
	b .${L}_c_write_out_b${beta}_a2

%for alpha in range(0, 3):
<%
btnm = "b{}_a{}".format(int(beta), int(alpha))
%>
.${L}_c_write_out_${btnm}:
	sub m_tmp, m_end, m_idx
	sub n_tmpx, n_end, n_idx

	cmp m_tmp, 19
	bgt .${L}_c_write_out_${btnm}_m20
	beq .${L}_c_write_out_${btnm}_m19

	cmp m_tmp, 17
	bgt .${L}_c_write_out_${btnm}_m18
	beq .${L}_c_write_out_${btnm}_m17

	cmp m_tmp, 15
	bgt .${L}_c_write_out_${btnm}_m16
	beq .${L}_c_write_out_${btnm}_m15

	cmp m_tmp, 13
	bgt .${L}_c_write_out_${btnm}_m14
	beq .${L}_c_write_out_${btnm}_m13

	cmp m_tmp, 11
	bgt .${L}_c_write_out_${btnm}_m12
	beq .${L}_c_write_out_${btnm}_m11

	cmp m_tmp, 9
	bgt .${L}_c_write_out_${btnm}_m10
	beq .${L}_c_write_out_${btnm}_m9

	cmp m_tmp, 7
	bgt .${L}_c_write_out_${btnm}_m8
	beq .${L}_c_write_out_${btnm}_m7

	cmp m_tmp, 5
	bgt .${L}_c_write_out_${btnm}_m6
	beq .${L}_c_write_out_${btnm}_m5

	cmp m_tmp, 3
	bgt .${L}_c_write_out_${btnm}_m4
	beq .${L}_c_write_out_${btnm}_m3

	cmp m_tmp, 1
	bgt .${L}_c_write_out_${btnm}_m2
	beq .${L}_c_write_out_${btnm}_m1

	b .${L}_c_write_out_end

%for m_elems in range(1, Creg_M*lanes+1):
<%
full_lanes = int(m_elems / lanes)
modu_lanes = m_elems % lanes
mod_start = full_lanes * lanes
%>
.${L}_c_write_out_${btnm}_m${m_elems}:
%if alpha == 2:
	dup alpha_v.${sflane}, alpha_x
%endif
%if beta == 2:
	dup beta_v.${sflane}, beta_x
%endif
%for n in range(0, Creg_N*lanes):
	subs n_tmpx, n_tmpx, 1
%for m_reg in range(0, full_lanes):
%if beta == 0:
	//nothing to load ${full_lanes}
%if alpha == 2:
	fmul C_m${m_reg}_n${n}.${sflane}, C_m${m_reg}_n${n}.${sflane}, alpha_v.${sfx}[0]
%endif #alpha == 2
%if alpha > 0:
	str C_m${m_reg}_n${n}q, [ C_cur_ptr, ${m_reg*vbytes} ]
%endif #alpha > 0
%elif beta == 1:
%if alpha == 0:
	//beta == 1 alpha == 0 is a no-op
%else:
	ldr C_m${m_reg}q, [ C_cur_ptr, ${m_reg*vbytes} ]
%if alpha == 1:
	fadd C_m${m_reg}.${sflane}, C_m${m_reg}.${sflane}, C_m${m_reg}_n${n}.${sflane}
%elif alpha == 2:
	fmla C_m${m_reg}.${sflane}, C_m${m_reg}_n${n}.${sflane}, alpha_v.${sfx}[0]
%endif #alpha > 0
	str C_m${m_reg}q, [ C_cur_ptr, ${m_reg*vbytes} ]
%endif
%elif beta == 2:
	ldr C_m${m_reg}q, [ C_cur_ptr, ${m_reg*vbytes} ]
%if alpha == 0:
	fmul C_m${m_reg}_n${n}.${sflane}, C_m${m_reg}.${sflane}, beta_v.${sfx}[0]
%elif alpha == 1:
	fmla C_m${m_reg}_n${n}.${sflane}, C_m${m_reg}.${sflane}, beta_v.${sfx}[0]
%elif alpha == 2:
	fmul C_m${m_reg}_n${n}.${sflane}, C_m${m_reg}_n${n}.${sflane}, alpha_v.${sfx}[0]
	fmla C_m${m_reg}_n${n}.${sflane}, C_m${m_reg}.${sflane}, beta_v.${sfx}[0]
%endif #alpha
	str C_m${m_reg}_n${n}q, [ C_cur_ptr, ${m_reg*vbytes} ]
%endif #beta
	dup C_m${m_reg}_n${n}.${sflane}, ${zeror}
%endfor
%for m_lane in range(0, modu_lanes):
%if beta == 0:
	//nothing to load
%if alpha == 2:
	fmul C_m${full_lanes}_n${n}d, C_m${full_lanes}_n${n}d, alpha_d
%endif #alpha == 2
%if alpha > 0:
	str C_m${full_lanes}_n${n}d, [ C_cur_ptr, ${full_lanes*vbytes + m_lane*bytes} ]
%endif #alpha > 0
%elif beta == 1:
%if alpha == 0:
	//beta == 1 alpha == 0 is a no-op
%else:
	ldr C_m${full_lanes}d, [ C_cur_ptr, ${full_lanes*vbytes + m_lane*bytes} ]
%if alpha == 1:
	fadd C_m${full_lanes}d, C_m${full_lanes}d, C_m${full_lanes}_n${n}d
%elif alpha == 2:
	fmadd C_m${full_lanes}d, C_m${full_lanes}_n${n}d, alpha_d, C_m${full_lanes}d
%endif #alpha > 0
	str C_m${full_lanes}d, [ C_cur_ptr, ${full_lanes*vbytes + m_lane*bytes} ]
%endif
%elif beta == 2:
	ldr C_m${full_lanes}d, [ C_cur_ptr, ${full_lanes*vbytes + m_lane*bytes} ]
%if alpha == 0:
	fmul C_m${full_lanes}_n${n}d, C_m${full_lanes}d, beta_d
%elif alpha == 1:
	fmadd C_m${full_lanes}_n${n}d, C_m${full_lanes}d, beta_d, C_m${full_lanes}_n${n}d
%elif alpha == 2:
	fmul C_m${full_lanes}_n${n}d, C_m${full_lanes}_n${n}d, alpha_d
	fmadd C_m${full_lanes}_n${n}d, C_m${full_lanes}d, beta_d, C_m${full_lanes}_n${n}d
%endif #alpha
	str C_m${full_lanes}_n${n}d, [ C_cur_ptr, ${full_lanes*vbytes + m_lane*bytes} ]
%endif #beta
	dup C_m${full_lanes}_n${n}.${sflane}, ${zeror}
%endfor
	beq .${L}_c_write_out_end
	add C_cur_ptr, C_cur_ptr, ldc, lsl #${shft}
%endfor
	b .${L}_c_write_out_end
%endfor
%endfor
%endfor
.${L}_c_write_out_end:

% if ab == "A":
	add A_cur_ptr, A_cur_ptr, lda
	${loop_end('m', "add {0}, {0}, " + str(Creg_M*lanes))}
	add B_cur_ptr, B_cur_ptr, ldb
	${loop_end('n', "add {0}, {0}, " + str(Creg_N*lanes))}
% else:
	add B_cur_ptr, B_cur_ptr, ldb
	${loop_end('n', "add {0}, {0}, " + str(Creg_N*lanes))}
	add A_cur_ptr, A_cur_ptr, lda
	${loop_end('m', "add {0}, {0}, " + str(Creg_M*lanes))}
%endif ##if ab

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

%endfor



%for crm in range(0, Creg_M):
%for it in range(0, a_iters):
	.unreq A_it${it}_m${crm}
	.unreq A_it${it}_m${crm}d
	.unreq A_it${it}_m${crm}s
	.unreq A_it${it}_m${crm}q
<% zreg += 1 %>
%endfor
	.unreq C_m${crm}
	.unreq C_m${crm}d
	.unreq C_m${crm}s
	.unreq C_m${crm}q
%endfor
%for crn in range(0, Creg_N):
%for it in range(0, b_iters):
	.unreq B_it${it}_n${crn}
	.unreq B_it${it}_n${crn}d
	.unreq B_it${it}_n${crn}s
	.unreq B_it${it}_n${crn}q
<% zreg += 1 %>
%endfor
%endfor

.unreq alpha_v
.unreq alpha_d
.unreq alpha_s
.unreq beta_v
.unreq beta_d
.unreq beta_s

//reserve the register block
%for crn in range(0, Creg_N*lanes):
%for crm in range(0, Creg_M):
	.unreq C_m${crm}_n${crn}
	.unreq C_m${crm}_n${crn}d
	.unreq C_m${crm}_n${crn}s
	.unreq C_m${crm}_n${crn}q
<% zreg += 1 %>
%endfor
%endfor


.unreq alpha_x
.unreq beta_x

.unreq alpha_in
.unreq beta_in

	${epilogue(name)}

%endfor ## dtype = ...

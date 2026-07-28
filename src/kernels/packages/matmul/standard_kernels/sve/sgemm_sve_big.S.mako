## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

#define A x0
#define B x1
#define C x2
#define K x3
#define M x4
#define N x5
#define ldc x6
#define k_idx x7
#define m_idx x8
#define n_idx x9
#define tmp1_x x10
#define tmp2_x x11
#define tmp3_x x12
#define tmp4_x x13

#define alpha_z z0
#define beta_z z1

#define Ca_m0_n0 z31
#define Ca_m0_n1 z30
#define Ca_m0_n2 z29
#define Ca_m0_n3 z28
#define Ca_m0_n4 z27
#define Ca_m0_n5 z26
#define Ca_m0_n6 z25
#define Ca_m0_n7 z24
#define Ca_m0_n8 z23
#define Ca_m0_n9 z22
#define Ca_m0_na z21
#define Ca_m0_nb z20
#define Ca_m1_n0 z19
#define Ca_m1_n1 z18
#define Ca_m1_n2 z17
#define Ca_m1_n3 z16
#define Ca_m1_n4 z15
#define Ca_m1_n5 z14
#define Ca_m1_n6 z13
#define Ca_m1_n7 z12
#define Ca_m1_n8 z11
#define Ca_m1_n9 z10
#define Ca_m1_na z9
#define Ca_m1_nb z8

#define C_m0 z2
#define C_m1 z3

#define A_it0_m0 z0
#define A_it0_m1 z1
#define A_it1_m0 z2
#define A_it1_m1 z3
#define B_n0123 z5
#define B_n4567 z6
#define B_n89ab z7

#define lda K
#define ldb K

#define XZERO(r) eor r, r, r
#define VZERO(r) eor r.16b, r.16b, r.16b
#define ZZERO(r) eor r.d, r.d, r.d

<% func_name = "sgemm_sve_big" %>
	${prologue(func_name)}
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
	stp s0, s1, [ sp, #16 ]

	ptrue p0.b

	XZERO(n_idx)

	// check if beta is 0.0 and branch to correct loop.
	fcmp s1, #0.0
	beq .Lbzero_n_loop

	// check if beta is 1.0 and branch to correct loop.
	fmov s0, #1.0
	fcmp s1, s0
	beq .Lbone_n_loop

// 2 means any value other than 0.0 or 1.0.
%for beta in ["general", "one", "zero"]:
<% L = ".Lb{}".format(beta) %>\

${L}_n_loop:
	cmp n_idx, N
	bge ${L}_n_loop_done
	XZERO(m_idx)
${L}_m_loop:
	cmp m_idx, M
	bge ${L}_m_loop_done

	mul tmp1_x, m_idx, lda
	add tmp1_x, A, tmp1_x, lsl #2
	mul tmp2_x, n_idx, ldb
	add tmp2_x, B, tmp2_x, lsl #2

	subs k_idx, K, #0
	ZZERO(Ca_m0_n0)
	ZZERO(Ca_m0_n1)
	ZZERO(Ca_m0_n2)
	ZZERO(Ca_m0_n3)
	ZZERO(Ca_m0_n4)
	ZZERO(Ca_m0_n5)
	ZZERO(Ca_m0_n6)
	ZZERO(Ca_m0_n7)
	ZZERO(Ca_m0_n8)
	ZZERO(Ca_m0_n9)
	ZZERO(Ca_m0_na)
	ZZERO(Ca_m0_nb)
	ZZERO(Ca_m1_n0)
	ZZERO(Ca_m1_n1)
	ZZERO(Ca_m1_n2)
	ZZERO(Ca_m1_n3)
	ZZERO(Ca_m1_n4)
	ZZERO(Ca_m1_n5)
	ZZERO(Ca_m1_n6)
	ZZERO(Ca_m1_n7)
	ZZERO(Ca_m1_n8)
	ZZERO(Ca_m1_n9)
	ZZERO(Ca_m1_na)
	ZZERO(Ca_m1_nb)
	beq ${L}_k_loop_done

	subs k_idx, k_idx, #1

	ld1w { A_it0_m0.s }, p0/z, [tmp1_x]
	ld1w { A_it0_m1.s }, p0/z, [tmp1_x, #1, mul vl]
	addvl tmp1_x, tmp1_x, #2

	ld1rqw { B_n0123.s }, p0/z, [tmp2_x]
	ld1rqw { B_n4567.s }, p0/z, [tmp2_x, #16]
	ld1rqw { B_n89ab.s }, p0/z, [tmp2_x, #32]
	add tmp2_x, tmp2_x, #(3 << 4)

	beq ${L}_k_loop_last_0

${L}_k_loop:
%for it in range(2):
<% next = (it+1) % 2 %>\
	fmla Ca_m0_n0.s, A_it${it}_m0.s, B_n0123.s[0]
	fmla Ca_m0_n1.s, A_it${it}_m0.s, B_n0123.s[1]
	subs k_idx, k_idx, #1
	fmla Ca_m0_n2.s, A_it${it}_m0.s, B_n0123.s[2]
	fmla Ca_m0_n3.s, A_it${it}_m0.s, B_n0123.s[3]
	addvl tmp1_x, tmp1_x, #2

	fmla Ca_m1_n0.s, A_it${it}_m1.s, B_n0123.s[0]
	fmla Ca_m1_n1.s, A_it${it}_m1.s, B_n0123.s[1]
	add tmp2_x, tmp2_x, #(3 << 4)
	fmla Ca_m1_n2.s, A_it${it}_m1.s, B_n0123.s[2]
	fmla Ca_m1_n3.s, A_it${it}_m1.s, B_n0123.s[3]
	ld1w { A_it${next}_m0.s }, p0/z, [tmp1_x, #-2, mul vl]

	fmla Ca_m0_n4.s, A_it${it}_m0.s, B_n4567.s[0]
	fmla Ca_m0_n5.s, A_it${it}_m0.s, B_n4567.s[1]
	fmla Ca_m0_n6.s, A_it${it}_m0.s, B_n4567.s[2]
	fmla Ca_m0_n7.s, A_it${it}_m0.s, B_n4567.s[3]
	ld1w { A_it${next}_m1.s }, p0/z, [tmp1_x, #-1, mul vl]

	fmla Ca_m1_n4.s, A_it${it}_m1.s, B_n4567.s[0]
	fmla Ca_m1_n5.s, A_it${it}_m1.s, B_n4567.s[1]
	fmla Ca_m1_n6.s, A_it${it}_m1.s, B_n4567.s[2]
	fmla Ca_m1_n7.s, A_it${it}_m1.s, B_n4567.s[3]
	ld1rqw { B_n0123.s }, p0/z, [tmp2_x, #-48]

	fmla Ca_m0_n8.s, A_it${it}_m0.s, B_n89ab.s[0]
	fmla Ca_m0_n9.s, A_it${it}_m0.s, B_n89ab.s[1]
	fmla Ca_m0_na.s, A_it${it}_m0.s, B_n89ab.s[2]
	fmla Ca_m0_nb.s, A_it${it}_m0.s, B_n89ab.s[3]
	ld1rqw { B_n4567.s }, p0/z, [tmp2_x, #-32]

	fmla Ca_m1_n8.s, A_it${it}_m1.s, B_n89ab.s[0]
	fmla Ca_m1_n9.s, A_it${it}_m1.s, B_n89ab.s[1]
	fmla Ca_m1_na.s, A_it${it}_m1.s, B_n89ab.s[2]
	fmla Ca_m1_nb.s, A_it${it}_m1.s, B_n89ab.s[3]
	ld1rqw { B_n89ab.s }, p0/z, [tmp2_x, #-16]
	beq ${L}_k_loop_last_${next}
%endfor
	b ${L}_k_loop

%for it in range(2):
${L}_k_loop_last_${it}:
	fmla Ca_m0_n0.s, A_it${it}_m0.s, B_n0123.s[0]
	fmla Ca_m0_n1.s, A_it${it}_m0.s, B_n0123.s[1]
	fmla Ca_m0_n2.s, A_it${it}_m0.s, B_n0123.s[2]
	fmla Ca_m0_n3.s, A_it${it}_m0.s, B_n0123.s[3]

	fmla Ca_m1_n0.s, A_it${it}_m1.s, B_n0123.s[0]
	fmla Ca_m1_n1.s, A_it${it}_m1.s, B_n0123.s[1]
	fmla Ca_m1_n2.s, A_it${it}_m1.s, B_n0123.s[2]
	fmla Ca_m1_n3.s, A_it${it}_m1.s, B_n0123.s[3]

	fmla Ca_m0_n4.s, A_it${it}_m0.s, B_n4567.s[0]
	fmla Ca_m0_n5.s, A_it${it}_m0.s, B_n4567.s[1]
	fmla Ca_m0_n6.s, A_it${it}_m0.s, B_n4567.s[2]
	fmla Ca_m0_n7.s, A_it${it}_m0.s, B_n4567.s[3]

	fmla Ca_m1_n4.s, A_it${it}_m1.s, B_n4567.s[0]
	fmla Ca_m1_n5.s, A_it${it}_m1.s, B_n4567.s[1]
	fmla Ca_m1_n6.s, A_it${it}_m1.s, B_n4567.s[2]
	fmla Ca_m1_n7.s, A_it${it}_m1.s, B_n4567.s[3]

	fmla Ca_m0_n8.s, A_it${it}_m0.s, B_n89ab.s[0]
	fmla Ca_m0_n9.s, A_it${it}_m0.s, B_n89ab.s[1]
	fmla Ca_m0_na.s, A_it${it}_m0.s, B_n89ab.s[2]
	fmla Ca_m0_nb.s, A_it${it}_m0.s, B_n89ab.s[3]

	fmla Ca_m1_n8.s, A_it${it}_m1.s, B_n89ab.s[0]
	fmla Ca_m1_n9.s, A_it${it}_m1.s, B_n89ab.s[1]
	fmla Ca_m1_na.s, A_it${it}_m1.s, B_n89ab.s[2]
	fmla Ca_m1_nb.s, A_it${it}_m1.s, B_n89ab.s[3]
	b ${L}_k_loop_done
%endfor

${L}_k_loop_done:
	mov tmp3_x, m_idx
	incw tmp3_x

	ld1rw { alpha_z.s }, p0/z, [ sp, #16 ]
	ld1rw { beta_z.s }, p0/z, [ sp, #20 ]

	whilelt p2.s, m_idx, M
	whilelt p3.s, tmp3_x, M

%for ofs in range(12):
<% c_reg0 = "Ca_m0_n{}".format(hex(ofs)[2]) %>\
<% c_reg1 = "Ca_m1_n{}".format(hex(ofs)[2]) %>\
	add tmp4_x, n_idx, #${ofs}
	cmp tmp4_x, N
	bge 1f
	madd tmp3_x, tmp4_x, ldc, m_idx
	add tmp3_x, C, tmp3_x, lsl #2
%if beta == "one":
	ld1w { C_m0.s }, p2/z, [tmp3_x]
	ld1w { C_m1.s }, p3/z, [tmp3_x, #1, mul vl]
	fmla C_m0.s, p0/m, ${c_reg0}.s, alpha_z.s
	fmla C_m1.s, p0/m, ${c_reg1}.s, alpha_z.s
	st1w { C_m0.s }, p2, [tmp3_x]
	st1w { C_m1.s }, p3, [tmp3_x, #1, mul vl]
%elif beta == "zero":
	fmul ${c_reg0}.s, p0/m, ${c_reg0}.s, alpha_z.s
	fmul ${c_reg1}.s, p0/m, ${c_reg1}.s, alpha_z.s
	st1w { ${c_reg0}.s }, p2, [tmp3_x]
	st1w { ${c_reg1}.s }, p3, [tmp3_x, #1, mul vl]
%else:
	ld1w { C_m0.s }, p2/z, [tmp3_x]
	ld1w { C_m1.s }, p3/z, [tmp3_x, #1, mul vl]
	fmul ${c_reg0}.s, p0/m, ${c_reg0}.s, alpha_z.s
	fmul ${c_reg1}.s, p0/m, ${c_reg1}.s, alpha_z.s
	fmla ${c_reg0}.s, p0/m, C_m0.s, beta_z.s
	fmla ${c_reg1}.s, p0/m, C_m1.s, beta_z.s
	st1w { ${c_reg0}.s }, p2, [tmp3_x]
	st1w { ${c_reg1}.s }, p3, [tmp3_x, #1, mul vl]
%endif
1:
%endfor

	incw m_idx, all, mul #2
	b ${L}_m_loop

${L}_m_loop_done:
	add n_idx, n_idx, #12
	b ${L}_n_loop

${L}_n_loop_done:
${L}_epilogue:
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
	${epilogue(func_name)}
%endfor

## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

/*
* ZGEMM SVE KERNEL
*
* Operation:
* C <- alpha*A*B + beta*C
*
* C Interface:
* extern void zgemm(kernel_inttype m, kernel_inttype n, kernel_inttype k, T alpha, const T *a, kernel_inttype lda, const T *b, kernel_inttype ldb, T beta, T *c, kernel_inttype ldc);
* with 64 bit pointers, 64 bit floats for the real and imaginary parts of alpha and beta.
*
* Notes:
* We are assuming A, B, and C have an r i, r i, r i,... pattern in memory, and that
* A is row major, while B and C are column major. Additionally, the matrices A and B
* are expected to have been packed into the interleaved layout required by this
* kernel.
*
* The basic GEMM computation is:
* for row, i, in A:
*	for column, j, in B:
*		C[i,j] = dot_product(row, column)
*
* We can the unroll both loops explicitly, but
* the variable vector length of the z registers results in further
* "implicit" unrolling of the loops; a 256 bit z can fit more
* values than a 128 bit one, but the length of the vector registers
* isn't known to the kernel. Otherwise the code would not be portable.
*
* This passive unrolling is a slight issue, because the number of
* C values to compute grows linearly with the unrolling, implicit or
* explicit, in A and B, and thus quadratically overall. But we still
* have accumulate the dot products in z registers which grow linearly
* with, well, the size of the z registers. As a result we are forced
* to fix the number of values in the z registers holding
* A or B values. In this case we have chosen B, and since the smallest
* z register size is 128 bits and the single complex size is 128 bits
* (64+64), with a single value.
*/

#################
# INPUT ARGUMENTS
x_A		.req	x0
x_B		.req 	x1
x_C		.req	x2

x_k		.req	x3
x_m		.req	x4
x_n		.req 	x5
x_ldc		.req	x6

d_alpha_re	.req	d0
d_alpha_im	.req	d1
d_beta_re	.req	d2
d_beta_im	.req	d3

#################
# LOCAL VARIABLES

// A value containers
z_A_i0z		.req	z0
z_A_i1z		.req	z1
z_A_i2z		.req	z2
z_A_i3z		.req	z3

// B value containers
z_B_j0		.req	z4
z_B_j1		.req	z5
z_B_j2		.req	z6
z_B_j3		.req	z7
z_B_j4		.req	z8

// Accumulators for FCMLA computations
z_AB_i0zj0	.req	z9
z_AB_i0zj1	.req	z10
z_AB_i0zj2	.req	z11
z_AB_i0zj3	.req	z12
z_AB_i0zj4	.req	z13
z_AB_i1zj0	.req	z14
z_AB_i1zj1	.req	z15
z_AB_i1zj2	.req	z16
z_AB_i1zj3	.req	z17
z_AB_i1zj4	.req	z18
z_AB_i2zj0	.req	z19
z_AB_i2zj1	.req	z20
z_AB_i2zj2	.req	z21
z_AB_i2zj3	.req	z22
z_AB_i2zj4	.req	z23
z_AB_i3zj0	.req	z24
z_AB_i3zj1	.req	z25
z_AB_i3zj2	.req	z26
z_AB_i3zj3	.req	z27
z_AB_i3zj4	.req	z28

// C value containers
z_C_i0zj0	.req	z0
z_C_i0zj1	.req	z0
z_C_i0zj2	.req	z0
z_C_i0zj3	.req	z0
z_C_i0zj4	.req	z0
z_C_i1zj0	.req	z1
z_C_i1zj1	.req	z1
z_C_i1zj2	.req	z1
z_C_i1zj3	.req	z1
z_C_i1zj4	.req	z1
z_C_i2zj0	.req	z2
z_C_i2zj1	.req	z2
z_C_i2zj2	.req	z2
z_C_i2zj3	.req	z2
z_C_i2zj4	.req	z2
z_C_i3zj0	.req	z3
z_C_i3zj1	.req	z3
z_C_i3zj2	.req	z3
z_C_i3zj3	.req	z3
z_C_i3zj4	.req	z3

z_temp_0	.req	z5
z_temp_1	.req	z6
z_temp_2	.req	z7
z_temp_3	.req	z8

z_alpha		.req	z30
z_beta		.req	z31

p_all		.req	p0
p_odd		.req	p1
p_even		.req	p2

p_C_i0		.req	p3
p_C_i1		.req	p4
p_C_i2		.req	p5
p_C_i3		.req	p6


// NB, we assume these are equal to k by the
// time the gemm call reaches this kernel
x_lda		.req	x3
x_ldb		.req	x3

x_m_idx		.req	x11
x_n_idx		.req	x12
x_k_idx		.req	x13

x_A_addr	.req	x16
x_B_addr	.req	x17
x_C_addr_0	.req	x19
x_C_addr_1	.req	x20
x_C_addr_2	.req	x21
x_C_addr_3	.req	x22

x_numquads	.req	x23
x_A_unroll	.req	x24

x_temp_0	.req	x30
x_temp_1	.req	x28
x_temp_2	.req	x27
x_temp_3	.req	x26
x_temp_4	.req	x25

// Define rotation parameters for the FCMLAs
#define _ROT1 #0
#define _ROT2 #90

###############################################################################
	<% func_name = "zgemm_sve_big" %>
	${prologue(func_name)}
	// Save State
	stp x29, x30, [sp, #-320]!
	mov x29, sp
	stp x19, x20, [ sp, #224 ]
	stp x21, x22, [ sp, #208 ]
	stp x23, x24, [ sp, #192 ]
	stp x25, x26, [ sp, #176 ]
	stp x27, x28, [ sp, #160 ]
	stp d8,  d9,  [ sp, #80 ]
	stp d10, d11, [ sp, #64 ]
	stp d12, d13, [ sp, #48 ]
	stp d14, d15, [ sp, #32 ]

	// Setup local variables
	# CONSTANTS
	ptrue	p_all.d
	ptrue	p_even.d
	pfalse	p_odd.b
	trn1	p_even.d, p_even.d, p_odd.d
	trn1	p_odd.d, p_odd.d, p_even.d

	cpy	z_alpha.d, p_even/m, d_alpha_re
	cpy	z_alpha.d, p_odd/m, d_alpha_im
	cpy	z_beta.d, p_even/m, d_beta_re
	cpy	z_beta.d, p_odd/m, d_beta_im

	# Number of complex values per SVE vector: (double lanes) / 2.
	cntd	x_numquads, all, mul #1
	lsr	x_numquads, x_numquads, #1
	mov	x_A_unroll, #4

	# VARIABLES
	mov	x_m_idx, xzr
	#

// check if beta is 0.0 and branch to correct loop.
fcmp d_beta_re, #0.0
bne .Lbgeneral_loop_m
fcmp d_beta_im, #0.0
bne .Lbgeneral_loop_m

%for beta in ["zero", "general"]:
<% L = ".Lb{}".format(beta) %>\

	${L}_loop_m:	// loop over rows in A
		mov	x_n_idx, xzr
		${L}_loop_n:	// loop over columns in B
			mul x_A_addr, x_m_idx, x_lda
			add x_A_addr, x_A, x_A_addr, lsl #4
			mul x_B_addr, x_n_idx, x_ldb
			add x_B_addr, x_B, x_B_addr, lsl #4

			mov	x_k_idx, xzr
			dup	z_AB_i0zj0.d, #0
			dup	z_AB_i0zj1.d, #0
			dup	z_AB_i0zj2.d, #0
			dup	z_AB_i0zj3.d, #0
			dup	z_AB_i0zj4.d, #0

			dup	z_AB_i1zj0.d, #0
			dup	z_AB_i1zj1.d, #0
			dup	z_AB_i1zj2.d, #0
			dup	z_AB_i1zj3.d, #0
			dup	z_AB_i1zj4.d, #0

			dup	z_AB_i2zj0.d, #0
			dup	z_AB_i2zj1.d, #0
			dup	z_AB_i2zj2.d, #0
			dup	z_AB_i2zj3.d, #0
			dup	z_AB_i2zj4.d, #0

			dup	z_AB_i3zj0.d, #0
			dup	z_AB_i3zj1.d, #0
			dup	z_AB_i3zj2.d, #0
			dup	z_AB_i3zj3.d, #0
			dup	z_AB_i3zj4.d, #0

			// While computing the loads for a given iteration of
			// the k loop, we can perform the loads for the next iteration
			// of the loop. This should reduce the time that arithmetic
			// instructions must wait for loads to complete.
			ld1d	z_A_i0z.d, p_all/z, [x_A_addr]
			ld1d	z_A_i1z.d, p_all/z, [x_A_addr, #1, mul vl]
			ld1d	z_A_i2z.d, p_all/z, [x_A_addr, #2, mul vl]
			ld1d	z_A_i3z.d, p_all/z, [x_A_addr, #3, mul vl]
			add	x_k_idx, x_k_idx, #1
			b	${L}_loop_k_cond	// compute dot products
			${L}_loop_k:
				ld1rqd	z_B_j0.d, p_all/z, [x_B_addr]
				ld1rqd	z_B_j1.d, p_all/z, [x_B_addr, #16]
				ld1rqd	z_B_j2.d, p_all/z, [x_B_addr, #32]
				ld1rqd	z_B_j3.d, p_all/z, [x_B_addr, #48]
				ld1rqd	z_B_j4.d, p_all/z, [x_B_addr, #64]
				add	x_B_addr, x_B_addr, #80
				addvl	x_A_addr, x_A_addr, #4

				fcmla	z_AB_i0zj0.d, p_all/m, z_A_i0z.d, z_B_j0.d, _ROT1
				fcmla	z_AB_i0zj0.d, p_all/m, z_A_i0z.d, z_B_j0.d, _ROT2
				fcmla	z_AB_i0zj1.d, p_all/m, z_A_i0z.d, z_B_j1.d, _ROT1
				fcmla	z_AB_i0zj1.d, p_all/m, z_A_i0z.d, z_B_j1.d, _ROT2
				fcmla	z_AB_i0zj2.d, p_all/m, z_A_i0z.d, z_B_j2.d, _ROT1
				fcmla	z_AB_i0zj2.d, p_all/m, z_A_i0z.d, z_B_j2.d, _ROT2
				fcmla	z_AB_i0zj3.d, p_all/m, z_A_i0z.d, z_B_j3.d, _ROT1
				fcmla	z_AB_i0zj3.d, p_all/m, z_A_i0z.d, z_B_j3.d, _ROT2
				fcmla	z_AB_i0zj4.d, p_all/m, z_A_i0z.d, z_B_j4.d, _ROT1
				fcmla	z_AB_i0zj4.d, p_all/m, z_A_i0z.d, z_B_j4.d, _ROT2
				ld1d	z_A_i0z.d, p_all/z, [x_A_addr]

				fcmla	z_AB_i1zj0.d, p_all/m, z_A_i1z.d, z_B_j0.d, _ROT1
				fcmla	z_AB_i1zj0.d, p_all/m, z_A_i1z.d, z_B_j0.d, _ROT2
				fcmla	z_AB_i1zj1.d, p_all/m, z_A_i1z.d, z_B_j1.d, _ROT1
				fcmla	z_AB_i1zj1.d, p_all/m, z_A_i1z.d, z_B_j1.d, _ROT2
				fcmla	z_AB_i1zj2.d, p_all/m, z_A_i1z.d, z_B_j2.d, _ROT1
				fcmla	z_AB_i1zj2.d, p_all/m, z_A_i1z.d, z_B_j2.d, _ROT2
				fcmla	z_AB_i1zj3.d, p_all/m, z_A_i1z.d, z_B_j3.d, _ROT1
				fcmla	z_AB_i1zj3.d, p_all/m, z_A_i1z.d, z_B_j3.d, _ROT2
				fcmla	z_AB_i1zj4.d, p_all/m, z_A_i1z.d, z_B_j4.d, _ROT1
				fcmla	z_AB_i1zj4.d, p_all/m, z_A_i1z.d, z_B_j4.d, _ROT2
				ld1d	z_A_i1z.d, p_all/z, [x_A_addr, #1, mul vl]

				fcmla	z_AB_i2zj0.d, p_all/m, z_A_i2z.d, z_B_j0.d, _ROT1
				fcmla	z_AB_i2zj0.d, p_all/m, z_A_i2z.d, z_B_j0.d, _ROT2
				fcmla	z_AB_i2zj1.d, p_all/m, z_A_i2z.d, z_B_j1.d, _ROT1
				fcmla	z_AB_i2zj1.d, p_all/m, z_A_i2z.d, z_B_j1.d, _ROT2
				fcmla	z_AB_i2zj2.d, p_all/m, z_A_i2z.d, z_B_j2.d, _ROT1
				fcmla	z_AB_i2zj2.d, p_all/m, z_A_i2z.d, z_B_j2.d, _ROT2
				fcmla	z_AB_i2zj3.d, p_all/m, z_A_i2z.d, z_B_j3.d, _ROT1
				fcmla	z_AB_i2zj3.d, p_all/m, z_A_i2z.d, z_B_j3.d, _ROT2
				fcmla	z_AB_i2zj4.d, p_all/m, z_A_i2z.d, z_B_j4.d, _ROT1
				fcmla	z_AB_i2zj4.d, p_all/m, z_A_i2z.d, z_B_j4.d, _ROT2
				ld1d	z_A_i2z.d, p_all/z, [x_A_addr, #2, mul vl]

				fcmla	z_AB_i3zj0.d, p_all/m, z_A_i3z.d, z_B_j0.d, _ROT1
				fcmla	z_AB_i3zj0.d, p_all/m, z_A_i3z.d, z_B_j0.d, _ROT2
				fcmla	z_AB_i3zj1.d, p_all/m, z_A_i3z.d, z_B_j1.d, _ROT1
				fcmla	z_AB_i3zj1.d, p_all/m, z_A_i3z.d, z_B_j1.d, _ROT2
				fcmla	z_AB_i3zj2.d, p_all/m, z_A_i3z.d, z_B_j2.d, _ROT1
				fcmla	z_AB_i3zj2.d, p_all/m, z_A_i3z.d, z_B_j2.d, _ROT2
				fcmla	z_AB_i3zj3.d, p_all/m, z_A_i3z.d, z_B_j3.d, _ROT1
				fcmla	z_AB_i3zj3.d, p_all/m, z_A_i3z.d, z_B_j3.d, _ROT2
				fcmla	z_AB_i3zj4.d, p_all/m, z_A_i3z.d, z_B_j4.d, _ROT1
				fcmla	z_AB_i3zj4.d, p_all/m, z_A_i3z.d, z_B_j4.d, _ROT2
				ld1d	z_A_i3z.d, p_all/z, [x_A_addr, #3, mul vl]

				add	x_k_idx, x_k_idx, #1
			${L}_loop_k_cond:
				cmp	x_k_idx, x_k
				b.lt	${L}_loop_k	// do work for current element(s)
			${L}_loop_k_end:
				ld1rqd	z_B_j0.d, p_all/z, [x_B_addr]
				ld1rqd	z_B_j1.d, p_all/z, [x_B_addr, #16]
				ld1rqd	z_B_j2.d, p_all/z, [x_B_addr, #32]
				ld1rqd	z_B_j3.d, p_all/z, [x_B_addr, #48]
				ld1rqd	z_B_j4.d, p_all/z, [x_B_addr, #64]

				fcmla	z_AB_i0zj0.d, p_all/m, z_A_i0z.d, z_B_j0.d, _ROT1
				fcmla	z_AB_i0zj0.d, p_all/m, z_A_i0z.d, z_B_j0.d, _ROT2
				fcmla	z_AB_i0zj1.d, p_all/m, z_A_i0z.d, z_B_j1.d, _ROT1
				fcmla	z_AB_i0zj1.d, p_all/m, z_A_i0z.d, z_B_j1.d, _ROT2
				fcmla	z_AB_i0zj2.d, p_all/m, z_A_i0z.d, z_B_j2.d, _ROT1
				fcmla	z_AB_i0zj2.d, p_all/m, z_A_i0z.d, z_B_j2.d, _ROT2
				fcmla	z_AB_i0zj3.d, p_all/m, z_A_i0z.d, z_B_j3.d, _ROT1
				fcmla	z_AB_i0zj3.d, p_all/m, z_A_i0z.d, z_B_j3.d, _ROT2
				fcmla	z_AB_i0zj4.d, p_all/m, z_A_i0z.d, z_B_j4.d, _ROT1
				fcmla	z_AB_i0zj4.d, p_all/m, z_A_i0z.d, z_B_j4.d, _ROT2

				fcmla	z_AB_i1zj0.d, p_all/m, z_A_i1z.d, z_B_j0.d, _ROT1
				fcmla	z_AB_i1zj0.d, p_all/m, z_A_i1z.d, z_B_j0.d, _ROT2
				fcmla	z_AB_i1zj1.d, p_all/m, z_A_i1z.d, z_B_j1.d, _ROT1
				fcmla	z_AB_i1zj1.d, p_all/m, z_A_i1z.d, z_B_j1.d, _ROT2
				fcmla	z_AB_i1zj2.d, p_all/m, z_A_i1z.d, z_B_j2.d, _ROT1
				fcmla	z_AB_i1zj2.d, p_all/m, z_A_i1z.d, z_B_j2.d, _ROT2
				fcmla	z_AB_i1zj3.d, p_all/m, z_A_i1z.d, z_B_j3.d, _ROT1
				fcmla	z_AB_i1zj3.d, p_all/m, z_A_i1z.d, z_B_j3.d, _ROT2
				fcmla	z_AB_i1zj4.d, p_all/m, z_A_i1z.d, z_B_j4.d, _ROT1
				fcmla	z_AB_i1zj4.d, p_all/m, z_A_i1z.d, z_B_j4.d, _ROT2

				fcmla	z_AB_i2zj0.d, p_all/m, z_A_i2z.d, z_B_j0.d, _ROT1
				fcmla	z_AB_i2zj0.d, p_all/m, z_A_i2z.d, z_B_j0.d, _ROT2
				fcmla	z_AB_i2zj1.d, p_all/m, z_A_i2z.d, z_B_j1.d, _ROT1
				fcmla	z_AB_i2zj1.d, p_all/m, z_A_i2z.d, z_B_j1.d, _ROT2
				fcmla	z_AB_i2zj2.d, p_all/m, z_A_i2z.d, z_B_j2.d, _ROT1
				fcmla	z_AB_i2zj2.d, p_all/m, z_A_i2z.d, z_B_j2.d, _ROT2
				fcmla	z_AB_i2zj3.d, p_all/m, z_A_i2z.d, z_B_j3.d, _ROT1
				fcmla	z_AB_i2zj3.d, p_all/m, z_A_i2z.d, z_B_j3.d, _ROT2
				fcmla	z_AB_i2zj4.d, p_all/m, z_A_i2z.d, z_B_j4.d, _ROT1
				fcmla	z_AB_i2zj4.d, p_all/m, z_A_i2z.d, z_B_j4.d, _ROT2

				fcmla	z_AB_i3zj0.d, p_all/m, z_A_i3z.d, z_B_j0.d, _ROT1
				fcmla	z_AB_i3zj0.d, p_all/m, z_A_i3z.d, z_B_j0.d, _ROT2
				fcmla	z_AB_i3zj1.d, p_all/m, z_A_i3z.d, z_B_j1.d, _ROT1
				fcmla	z_AB_i3zj1.d, p_all/m, z_A_i3z.d, z_B_j1.d, _ROT2
				fcmla	z_AB_i3zj2.d, p_all/m, z_A_i3z.d, z_B_j2.d, _ROT1
				fcmla	z_AB_i3zj2.d, p_all/m, z_A_i3z.d, z_B_j2.d, _ROT2
				fcmla	z_AB_i3zj3.d, p_all/m, z_A_i3z.d, z_B_j3.d, _ROT1
				fcmla	z_AB_i3zj3.d, p_all/m, z_A_i3z.d, z_B_j3.d, _ROT2
				fcmla	z_AB_i3zj4.d, p_all/m, z_A_i3z.d, z_B_j4.d, _ROT1
				fcmla	z_AB_i3zj4.d, p_all/m, z_A_i3z.d, z_B_j4.d, _ROT2
			${L}_loop_k_store:
				add	x_temp_0, x_m_idx, x_numquads
				add	x_temp_1, x_temp_0, x_numquads
				add	x_temp_2, x_temp_1, x_numquads

				lsl	x_temp_4, x_m, #1
				lsl	x_temp_3, x_m_idx, #1
				whilelt	p_C_i0.d, x_temp_3, x_temp_4
				lsl	x_temp_3, x_temp_0, #1
				whilelt	p_C_i1.d, x_temp_3, x_temp_4
				lsl	x_temp_3, x_temp_1, #1
				whilelt	p_C_i2.d, x_temp_3, x_temp_4
				lsl	x_temp_3, x_temp_2, #1
				whilelt	p_C_i3.d, x_temp_3, x_temp_4

				// Ideally we could give every .Lbx_iXzjY its own
				// set of registers, so the loads and stores in one section wouldn't
				// have to wait for the arithmetic instructions in another. However
				// we have a limited number of registers, so an alternating
				// series of registers is used to minimize this effect.
		${L}_j0:
			${L}_i0zj0:
				madd	x_C_addr_0, x_ldc, x_n_idx, x_m_idx
				add	x_C_addr_0, x_C, x_C_addr_0, lsl #4

				dup	z_temp_0.d, #0
				fcmla	z_temp_0.d, p_all/m, z_AB_i0zj0.d, z_alpha.d, #0
				fcmla	z_temp_0.d, p_all/m, z_AB_i0zj0.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i0zj0.d, p_C_i0/z, [x_C_addr_0]
				fcmla	z_temp_0.d, p_all/m, z_C_i0zj0.d, z_beta.d, #0
				fcmla	z_temp_0.d, p_all/m, z_C_i0zj0.d, z_beta.d, #90
%endif
				st1d	z_temp_0.d, p_C_i0, [x_C_addr_0]
			${L}_i1zj0:
				madd	x_C_addr_1, x_ldc, x_n_idx, x_temp_0
				add	x_C_addr_1, x_C, x_C_addr_1, lsl #4

				dup	z_temp_1.d, #0
				fcmla	z_temp_1.d, p_all/m, z_AB_i1zj0.d, z_alpha.d, #0
				fcmla	z_temp_1.d, p_all/m, z_AB_i1zj0.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i1zj0.d, p_C_i1/z, [x_C_addr_1]
				fcmla	z_temp_1.d, p_all/m, z_C_i1zj0.d, z_beta.d, #0
				fcmla	z_temp_1.d, p_all/m, z_C_i1zj0.d, z_beta.d, #90
%endif
				st1d	z_temp_1.d, p_C_i1, [x_C_addr_1]
			${L}_i2zj0:
				madd	x_C_addr_2, x_ldc, x_n_idx, x_temp_1
				add	x_C_addr_2, x_C, x_C_addr_2, lsl #4

				dup	z_temp_2.d, #0
				fcmla	z_temp_2.d, p_all/m, z_AB_i2zj0.d, z_alpha.d, #0
				fcmla	z_temp_2.d, p_all/m, z_AB_i2zj0.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i2zj0.d, p_C_i2/z, [x_C_addr_2]
				fcmla	z_temp_2.d, p_all/m, z_C_i2zj0.d, z_beta.d, #0
				fcmla	z_temp_2.d, p_all/m, z_C_i2zj0.d, z_beta.d, #90
%endif
				st1d	z_temp_2.d, p_C_i2, [x_C_addr_2]
			${L}_i3zj0:
				madd	x_C_addr_3, x_ldc, x_n_idx, x_temp_2
				add	x_C_addr_3, x_C, x_C_addr_3, lsl #4

				dup	z_temp_3.d, #0
				fcmla	z_temp_3.d, p_all/m, z_AB_i3zj0.d, z_alpha.d, #0
				fcmla	z_temp_3.d, p_all/m, z_AB_i3zj0.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i3zj0.d, p_C_i3/z, [x_C_addr_3]
				fcmla	z_temp_3.d, p_all/m, z_C_i3zj0.d, z_beta.d, #0
				fcmla	z_temp_3.d, p_all/m, z_C_i3zj0.d, z_beta.d, #90
%endif
				st1d	z_temp_3.d, p_C_i3, [x_C_addr_3]
		${L}_j1:
			add	x_n_idx, x_n_idx, #1
			cmp	x_n_idx, x_n
			b.ge	${L}_row_end // skip if no columns remain
			${L}_i0zj1:
				madd	x_C_addr_0, x_ldc, x_n_idx, x_m_idx
				add	x_C_addr_0, x_C, x_C_addr_0, lsl #4

				dup	z_temp_0.d, #0
				fcmla	z_temp_0.d, p_all/m, z_AB_i0zj1.d, z_alpha.d, #0
				fcmla	z_temp_0.d, p_all/m, z_AB_i0zj1.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i0zj1.d, p_C_i0/z, [x_C_addr_0]
				fcmla	z_temp_0.d, p_all/m, z_C_i0zj1.d, z_beta.d, #0
				fcmla	z_temp_0.d, p_all/m, z_C_i0zj1.d, z_beta.d, #90
%endif
				st1d	z_temp_0.d, p_C_i0, [x_C_addr_0]
			${L}_i1zj1:
				madd	x_C_addr_1, x_ldc, x_n_idx, x_temp_0
				add	x_C_addr_1, x_C, x_C_addr_1, lsl #4

				dup	z_temp_1.d, #0
				fcmla	z_temp_1.d, p_all/m, z_AB_i1zj1.d, z_alpha.d, #0
				fcmla	z_temp_1.d, p_all/m, z_AB_i1zj1.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i1zj1.d, p_C_i1/z, [x_C_addr_1]
				fcmla	z_temp_1.d, p_all/m, z_C_i1zj1.d, z_beta.d, #0
				fcmla	z_temp_1.d, p_all/m, z_C_i1zj1.d, z_beta.d, #90
%endif
				st1d	z_temp_1.d, p_C_i1, [x_C_addr_1]
			${L}_i2zj1:
				madd	x_C_addr_2, x_ldc, x_n_idx, x_temp_1
				add	x_C_addr_2, x_C, x_C_addr_2, lsl #4

				dup	z_temp_2.d, #0
				fcmla	z_temp_2.d, p_all/m, z_AB_i2zj1.d, z_alpha.d, #0
				fcmla	z_temp_2.d, p_all/m, z_AB_i2zj1.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i2zj1.d, p_C_i2/z, [x_C_addr_2]
				fcmla	z_temp_2.d, p_all/m, z_C_i2zj1.d, z_beta.d, #0
				fcmla	z_temp_2.d, p_all/m, z_C_i2zj1.d, z_beta.d, #90
%endif
				st1d	z_temp_2.d, p_C_i2, [x_C_addr_2]
			${L}_i3zj1:
				madd	x_C_addr_3, x_ldc, x_n_idx, x_temp_2
				add	x_C_addr_3, x_C, x_C_addr_3, lsl #4

				dup	z_temp_3.d, #0
				fcmla	z_temp_3.d, p_all/m, z_AB_i3zj1.d, z_alpha.d, #0
				fcmla	z_temp_3.d, p_all/m, z_AB_i3zj1.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i3zj1.d, p_C_i3/z, [x_C_addr_3]
				fcmla	z_temp_3.d, p_all/m, z_C_i3zj1.d, z_beta.d, #0
				fcmla	z_temp_3.d, p_all/m, z_C_i3zj1.d, z_beta.d, #90
%endif
				st1d	z_temp_3.d, p_C_i3, [x_C_addr_3]
		${L}_j2:
			add	x_n_idx, x_n_idx, #1
			cmp	x_n_idx, x_n
			b.ge	${L}_row_end // skip if no columns remain
			${L}_i0zj2:
				madd	x_C_addr_0, x_ldc, x_n_idx, x_m_idx
				add	x_C_addr_0, x_C, x_C_addr_0, lsl #4

				dup	z_temp_0.d, #0
				fcmla	z_temp_0.d, p_all/m, z_AB_i0zj2.d, z_alpha.d, #0
				fcmla	z_temp_0.d, p_all/m, z_AB_i0zj2.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i0zj2.d, p_C_i0/z, [x_C_addr_0]
				fcmla	z_temp_0.d, p_all/m, z_C_i0zj2.d, z_beta.d, #0
				fcmla	z_temp_0.d, p_all/m, z_C_i0zj2.d, z_beta.d, #90
%endif
				st1d	z_temp_0.d, p_C_i0, [x_C_addr_0]
			${L}_i1zj2:
				madd	x_C_addr_1, x_ldc, x_n_idx, x_temp_0
				add	x_C_addr_1, x_C, x_C_addr_1, lsl #4

				dup	z_temp_1.d, #0
				fcmla	z_temp_1.d, p_all/m, z_AB_i1zj2.d, z_alpha.d, #0
				fcmla	z_temp_1.d, p_all/m, z_AB_i1zj2.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i1zj2.d, p_C_i1/z, [x_C_addr_1]
				fcmla	z_temp_1.d, p_all/m, z_C_i1zj2.d, z_beta.d, #0
				fcmla	z_temp_1.d, p_all/m, z_C_i1zj2.d, z_beta.d, #90
%endif
				st1d	z_temp_1.d, p_C_i1, [x_C_addr_1]
			${L}_i2zj2:
				madd	x_C_addr_2, x_ldc, x_n_idx, x_temp_1
				add	x_C_addr_2, x_C, x_C_addr_2, lsl #4

				dup	z_temp_2.d, #0
				fcmla	z_temp_2.d, p_all/m, z_AB_i2zj2.d, z_alpha.d, #0
				fcmla	z_temp_2.d, p_all/m, z_AB_i2zj2.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i2zj2.d, p_C_i2/z, [x_C_addr_2]
				fcmla	z_temp_2.d, p_all/m, z_C_i2zj2.d, z_beta.d, #0
				fcmla	z_temp_2.d, p_all/m, z_C_i2zj2.d, z_beta.d, #90
%endif
				st1d	z_temp_2.d, p_C_i2, [x_C_addr_2]
			${L}_i3zj2:
				madd	x_C_addr_3, x_ldc, x_n_idx, x_temp_2
				add	x_C_addr_3, x_C, x_C_addr_3, lsl #4

				dup	z_temp_3.d, #0
				fcmla	z_temp_3.d, p_all/m, z_AB_i3zj2.d, z_alpha.d, #0
				fcmla	z_temp_3.d, p_all/m, z_AB_i3zj2.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i3zj2.d, p_C_i3/z, [x_C_addr_3]
				fcmla	z_temp_3.d, p_all/m, z_C_i3zj2.d, z_beta.d, #0
				fcmla	z_temp_3.d, p_all/m, z_C_i3zj2.d, z_beta.d, #90
%endif
				st1d	z_temp_3.d, p_C_i3, [x_C_addr_3]
		${L}_j3:
			add	x_n_idx, x_n_idx, #1
			cmp	x_n_idx, x_n
			b.ge	${L}_row_end // skip if no columns remain
			${L}_i0zj3:
				madd	x_C_addr_0, x_ldc, x_n_idx, x_m_idx
				add	x_C_addr_0, x_C, x_C_addr_0, lsl #4

				dup	z_temp_0.d, #0
				fcmla	z_temp_0.d, p_all/m, z_AB_i0zj3.d, z_alpha.d, #0
				fcmla	z_temp_0.d, p_all/m, z_AB_i0zj3.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i0zj3.d, p_C_i0/z, [x_C_addr_0]
				fcmla	z_temp_0.d, p_all/m, z_C_i0zj3.d, z_beta.d, #0
				fcmla	z_temp_0.d, p_all/m, z_C_i0zj3.d, z_beta.d, #90
%endif
				st1d	z_temp_0.d, p_C_i0, [x_C_addr_0]
			${L}_i1zj3:
				madd	x_C_addr_1, x_ldc, x_n_idx, x_temp_0
				add	x_C_addr_1, x_C, x_C_addr_1, lsl #4

				dup	z_temp_1.d, #0
				fcmla	z_temp_1.d, p_all/m, z_AB_i1zj3.d, z_alpha.d, #0
				fcmla	z_temp_1.d, p_all/m, z_AB_i1zj3.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i1zj3.d, p_C_i1/z, [x_C_addr_1]
				fcmla	z_temp_1.d, p_all/m, z_C_i1zj3.d, z_beta.d, #0
				fcmla	z_temp_1.d, p_all/m, z_C_i1zj3.d, z_beta.d, #90
%endif
				st1d	z_temp_1.d, p_C_i1, [x_C_addr_1]
			${L}_i2zj3:
				madd	x_C_addr_2, x_ldc, x_n_idx, x_temp_1
				add	x_C_addr_2, x_C, x_C_addr_2, lsl #4

				dup	z_temp_2.d, #0
				fcmla	z_temp_2.d, p_all/m, z_AB_i2zj3.d, z_alpha.d, #0
				fcmla	z_temp_2.d, p_all/m, z_AB_i2zj3.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i2zj3.d, p_C_i2/z, [x_C_addr_2]
				fcmla	z_temp_2.d, p_all/m, z_C_i2zj3.d, z_beta.d, #0
				fcmla	z_temp_2.d, p_all/m, z_C_i2zj3.d, z_beta.d, #90
%endif
				st1d	z_temp_2.d, p_C_i2, [x_C_addr_2]
			${L}_i3zj3:
				madd	x_C_addr_3, x_ldc, x_n_idx, x_temp_2
				add	x_C_addr_3, x_C, x_C_addr_3, lsl #4

				dup	z_temp_3.d, #0
				fcmla	z_temp_3.d, p_all/m, z_AB_i3zj3.d, z_alpha.d, #0
				fcmla	z_temp_3.d, p_all/m, z_AB_i3zj3.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i3zj3.d, p_C_i3/z, [x_C_addr_3]
				fcmla	z_temp_3.d, p_all/m, z_C_i3zj3.d, z_beta.d, #0
				fcmla	z_temp_3.d, p_all/m, z_C_i3zj3.d, z_beta.d, #90
%endif
				st1d	z_temp_3.d, p_C_i3, [x_C_addr_3]
		${L}_j4:
			add	x_n_idx, x_n_idx, #1
			cmp	x_n_idx, x_n
			b.ge	${L}_row_end // skip if no columns remain
			${L}_i0zj4:
				madd	x_C_addr_0, x_ldc, x_n_idx, x_m_idx
				add	x_C_addr_0, x_C, x_C_addr_0, lsl #4

				dup	z_temp_0.d, #0
				fcmla	z_temp_0.d, p_all/m, z_AB_i0zj4.d, z_alpha.d, #0
				fcmla	z_temp_0.d, p_all/m, z_AB_i0zj4.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i0zj4.d, p_C_i0/z, [x_C_addr_0]
				fcmla	z_temp_0.d, p_all/m, z_C_i0zj4.d, z_beta.d, #0
				fcmla	z_temp_0.d, p_all/m, z_C_i0zj4.d, z_beta.d, #90
%endif
				st1d	z_temp_0.d, p_C_i0, [x_C_addr_0]
			${L}_i1zj4:
				madd	x_C_addr_1, x_ldc, x_n_idx, x_temp_0
				add	x_C_addr_1, x_C, x_C_addr_1, lsl #4

				dup	z_temp_1.d, #0
				fcmla	z_temp_1.d, p_all/m, z_AB_i1zj4.d, z_alpha.d, #0
				fcmla	z_temp_1.d, p_all/m, z_AB_i1zj4.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i1zj4.d, p_C_i1/z, [x_C_addr_1]
				fcmla	z_temp_1.d, p_all/m, z_C_i1zj4.d, z_beta.d, #0
				fcmla	z_temp_1.d, p_all/m, z_C_i1zj4.d, z_beta.d, #90
%endif
				st1d	z_temp_1.d, p_C_i1, [x_C_addr_1]
			${L}_i2zj4:
				madd	x_C_addr_2, x_ldc, x_n_idx, x_temp_1
				add	x_C_addr_2, x_C, x_C_addr_2, lsl #4

				dup	z_temp_2.d, #0
				fcmla	z_temp_2.d, p_all/m, z_AB_i2zj4.d, z_alpha.d, #0
				fcmla	z_temp_2.d, p_all/m, z_AB_i2zj4.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i2zj4.d, p_C_i2/z, [x_C_addr_2]
				fcmla	z_temp_2.d, p_all/m, z_C_i2zj4.d, z_beta.d, #0
				fcmla	z_temp_2.d, p_all/m, z_C_i2zj4.d, z_beta.d, #90
%endif
				st1d	z_temp_2.d, p_C_i2, [x_C_addr_2]
			${L}_i3zj4:
				madd	x_C_addr_3, x_ldc, x_n_idx, x_temp_2
				add	x_C_addr_3, x_C, x_C_addr_3, lsl #4

				dup	z_temp_3.d, #0
				fcmla	z_temp_3.d, p_all/m, z_AB_i3zj4.d, z_alpha.d, #0
				fcmla	z_temp_3.d, p_all/m, z_AB_i3zj4.d, z_alpha.d, #90
%if beta == "general":
				ld1d	z_C_i3zj4.d, p_C_i3/z, [x_C_addr_3]
				fcmla	z_temp_3.d, p_all/m, z_C_i3zj4.d, z_beta.d, #0
				fcmla	z_temp_3.d, p_all/m, z_C_i3zj4.d, z_beta.d, #90
%endif
				st1d	z_temp_3.d, p_C_i3, [x_C_addr_3]
		${L}_loop_n_cond:
			add	x_n_idx, x_n_idx, #1
			cmp	x_n_idx, x_n
			b.lt	${L}_loop_n		// do work for rest of row(s)
		${L}_row_end:
			madd	x_m_idx, x_A_unroll, x_numquads, x_m_idx
			//b	${L}_loop_m_cond	// no work left for current row(s)
	${L}_loop_m_cond:
		cmp	x_m_idx, x_m
		b.lt	${L}_loop_m // do work for next row(s)
		b	.Lreturn // no work left
%endfor
.Lreturn:
	// Restore State
	ldp x19, x20, [ sp, #224 ]
	ldp x21, x22, [ sp, #208 ]
	ldp x23, x24, [ sp, #192 ]
	ldp x25, x26, [ sp, #176 ]
	ldp x27, x28, [ sp, #160 ]
	ldp d8,  d9,  [ sp, #80 ]
	ldp d10, d11, [ sp, #64 ]
	ldp d12, d13, [ sp, #48 ]
	ldp d14, d15, [ sp, #32 ]
	ldp x29, x30, [sp], #320
	ret
	${epilogue(func_name)}
################################################################################

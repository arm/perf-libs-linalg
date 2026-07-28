## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

// parameters
#define m x0
#define n x1
#define k x2
#define a_ptr x3
#define lda x4
#define b_ptr x5
#define ldb x6
#define c_ptr x7
#define ldc x8

// locals
#define k_block_size x9
#define k_block_idx x10 // [0,m_block_size,...,m)
#define k_block_size_max x11
#define n_block_size x11
#define n_block_idx x12 // [0,m_block_size,...,m)
#define n_block_size_max x13
#define idx_m0 x13
#define idx_m1 x14
#define idx_n_base x15 // [0,1,...,m_block_size)
#define idx_n x16 // [0,1,...,m) = m_block_idx + idx_m_base
#define x_alpha x17
#define x_beta  x19
#define lda3_minus_4 x20
// x18 may be in use for the Platform Register
#define idx_k_base x21 // [0,1,...,m_block_size)
#define idx_k x22 // [0,1,...,m) = m_block_idx + idx_m_base
#define beta_is_zero x23

#define alpha_real v31.s[0]
#define alpha_imag v31.s[1]
#define beta_real  v31.s[2]
#define beta_imag  v31.s[3]
#define alpha v31.d[0]
#define beta  v31.d[1]
#define alpha_beta v31

#define N_BLOCK_SIZE_DEFAULT 160
#define K_BLOCK_SIZE_DEFAULT 160
#define N_BLOCK_SIZE_FIRST 192
#define K_BLOCK_SIZE_FIRST 192
#define A_PTR_PREFETCH_DISTANCE 64
#define B_PTR_PREFETCH_DISTANCE 64

// we use a macro to generate the code for both transpose and transpose
// conjugate, as we simply invert whether we apply subtraction in some cases
// and addition in others
<%def name="generate_XX_kernel(func_name, fmladdsub_a, fmladdsub_b, fmlsubaddab)">
	// {{{
	${prologue(func_name)}
	stp x29, x30, [ sp, #-384 ]!
	mov x29, sp
	stp x19, x20, [ sp, #144 ]
	stp x21, x22, [ sp, #160 ]
	stp x23, x24, [ sp, #176 ]
	stp x25, x26, [ sp, #192 ]
	stp x27, x28, [ sp, #208 ]

	ldr	ldc, [ sp, #384 ]

	stp d8, d9, [ sp, #304 ]
	stp d10, d11, [ sp, #320 ]
	stp d12, d13, [ sp, #336 ]
	stp d14, d15, [ sp, #352 ]

	%if target_os in ('windows', 'windows_arm64ec'):
		// load alpha and beta from stack
		ldr s0, [sp, #392]
		ldr s1, [sp, #396]
		ldr s2, [sp, #400]
		ldr s3, [sp, #404]
	%endif

	mov v0.s[1], v1.s[0]
	mov x_alpha, v0.d[0]
	mov v2.s[1], v3.s[0]
	mov x_beta,  v2.d[0]

	fcmp s2, 0.0
	bne .L${func_name}_set_beta_not_zero
	fcmp s3, 0.0
	bne .L${func_name}_set_beta_not_zero
	mov beta_is_zero, 1
	b .L${func_name}_beta_done
.L${func_name}_set_beta_not_zero:
	mov beta_is_zero, 0
.L${func_name}_beta_done:

	add lda3_minus_4, lda, lda, lsl #1
	sub lda3_minus_4, lda3_minus_4, #4

	mov k_block_size_max, K_BLOCK_SIZE_FIRST

	eor k_block_idx, k_block_idx, k_block_idx
.L${func_name}_loop_block_k: // {{{
	cmp k_block_idx, k
	beq .L${func_name}_loop_block_k_end

	// compute next block size (in elems)
	// k_block_size = min(k_block_size, k - k_block_idx)
	sub k_block_size, k, k_block_idx
	cmp k_block_size_max, k_block_size
	mov k_block_size_max, K_BLOCK_SIZE_DEFAULT
	csel k_block_size, k_block_size_max, k_block_size, le

	mov n_block_size_max, N_BLOCK_SIZE_FIRST

	eor n_block_idx, n_block_idx, n_block_idx
.L${func_name}_loop_block_n: // {{{
	cmp n_block_idx, n
	beq .L${func_name}_loop_block_n_end

	// compute next block size (in elems)
	// m_block_size = min(m_block_size, m - m_block_idx)
	sub n_block_size, n, n_block_idx
	cmp n_block_size_max, n_block_size
	mov n_block_size_max, N_BLOCK_SIZE_DEFAULT
	csel n_block_size, n_block_size_max, n_block_size, le

	eor idx_m0, idx_m0, idx_m0
.L${func_name}_loop_m: // {{{
	cmp idx_m0, m
	bge .L${func_name}_loop_m_end

	add idx_m1, idx_m0, #4
	cmp idx_m1, m
	csel idx_m1, idx_m0, idx_m1, ge

	eor idx_n_base, idx_n_base, idx_n_base
.L${func_name}_loop_n: // {{{
	cmp idx_n_base, n_block_size
	beq .L${func_name}_loop_n_end
	add idx_n, n_block_idx, idx_n_base

.L${func_name}_loop_k_start: // {{{

#define ar01 v31
#define ar00 v30
#define ar11 v29
#define ar10 v28
#define ar21 v27
#define ar20 v26
#define ar31 v25
#define ar30 v24

#define br_im3 v23
#define br_re3 v22
#define br_im2 v21
#define br_re2 v20
#define br_im1 v19
#define br_re1 v18
#define br_im0 v17
#define br_re0 v16

#define cr0_im7 v15
#define cr0_re7 v14
#define cr0_im6 v13
#define cr0_re6 v12
#define cr0_im5 v11
#define cr0_re5 v10
#define cr0_im4 v9
#define cr0_re4 v8
#define cr0_im3 v7
#define cr0_re3 v6
#define cr0_im2 v5
#define cr0_re2 v4
#define cr0_im1 v3
#define cr0_re1 v2
#define cr0_im0 v1
#define cr0_re0 v0

#define local_b_ptr0 x28
#define local_a_ptr0 x27
#define local_a_ptr1 x26

	eor cr0_re0.16b, cr0_re0.16b, cr0_re0.16b
	eor cr0_im0.16b, cr0_im0.16b, cr0_im0.16b
	eor cr0_re1.16b, cr0_re1.16b, cr0_re1.16b
	eor cr0_im1.16b, cr0_im1.16b, cr0_im1.16b
	eor cr0_re2.16b, cr0_re2.16b, cr0_re2.16b
	eor cr0_im2.16b, cr0_im2.16b, cr0_im2.16b
	eor cr0_re3.16b, cr0_re3.16b, cr0_re3.16b
	eor cr0_im3.16b, cr0_im3.16b, cr0_im3.16b
	eor cr0_re4.16b, cr0_re4.16b, cr0_re4.16b
	eor cr0_im4.16b, cr0_im4.16b, cr0_im4.16b
	eor cr0_re5.16b, cr0_re5.16b, cr0_re5.16b
	eor cr0_im5.16b, cr0_im5.16b, cr0_im5.16b
	eor cr0_re6.16b, cr0_re6.16b, cr0_re6.16b
	eor cr0_im6.16b, cr0_im6.16b, cr0_im6.16b
	eor cr0_re7.16b, cr0_re7.16b, cr0_re7.16b
	eor cr0_im7.16b, cr0_im7.16b, cr0_im7.16b

	cmp k_block_size, #0
	beq .L${func_name}_loop_k_end

	madd local_a_ptr0, lda, idx_m0, k_block_idx
	add local_a_ptr0, a_ptr, local_a_ptr0, lsl 3
	madd local_a_ptr1, lda, idx_m1, k_block_idx
	add local_a_ptr1, a_ptr, local_a_ptr1, lsl 3

	madd local_b_ptr0, ldb, k_block_idx, idx_n
	add local_b_ptr0, b_ptr, local_b_ptr0, lsl 3

	subs idx_k, k_block_size, #4
	blt .L${func_name}_loop_k_last

	ld1 {ar00.4s, ar01.4s}, [ local_a_ptr0 ]
	add local_a_ptr0, local_a_ptr0, lda, lsl 3
	ld1 {ar10.4s, ar11.4s}, [ local_a_ptr0 ]
	add local_a_ptr0, local_a_ptr0, lda, lsl 3
	ld1 {ar20.4s, ar21.4s}, [ local_a_ptr0 ]
	add local_a_ptr0, local_a_ptr0, lda, lsl 3

	ld2 {br_re0.4s, br_im0.4s}, [ local_b_ptr0 ]
	add local_b_ptr0, local_b_ptr0, ldb, lsl 3
	ld2 {br_re1.4s, br_im1.4s}, [ local_b_ptr0 ]
	add local_b_ptr0, local_b_ptr0, ldb, lsl 3

	subs idx_k, k_block_size, #8
	blt .L${func_name}_loop_k_almost_last

.L${func_name}_loop_k: // {{{

	// cr0_re = ar_re * br_re - ar_im * br_im + cr0_re;
	// cr0_im = ar_re * br_im + ar_im * br_re + cr0_im;
	fmla         cr0_re0.4s, br_re0.4s, ar00.s[0]
	${fmladdsub_b} cr0_im0.4s, br_im0.4s, ar00.s[0]
	${fmlsubaddab} cr0_re0.4s, br_im0.4s, ar00.s[1]
	${fmladdsub_a} cr0_im0.4s, br_re0.4s, ar00.s[1]

	ld1 {ar30.4s, ar31.4s}, [ local_a_ptr0 ]

	fmla         cr0_re1.4s, br_re0.4s, ar10.s[0]
	${fmladdsub_b} cr0_im1.4s, br_im0.4s, ar10.s[0]
	${fmlsubaddab} cr0_re1.4s, br_im0.4s, ar10.s[1]
	${fmladdsub_a} cr0_im1.4s, br_re0.4s, ar10.s[1]

	prfm pldl1keep, [local_b_ptr0, B_PTR_PREFETCH_DISTANCE]

	fmla         cr0_re2.4s, br_re0.4s, ar20.s[0]
	${fmladdsub_b} cr0_im2.4s, br_im0.4s, ar20.s[0]
	${fmlsubaddab} cr0_re2.4s, br_im0.4s, ar20.s[1]
	${fmladdsub_a} cr0_im2.4s, br_re0.4s, ar20.s[1]

	ld2 {br_re2.4s, br_im2.4s}, [ local_b_ptr0 ]
	add local_b_ptr0, local_b_ptr0, ldb, lsl 3

	fmla         cr0_re3.4s, br_re0.4s, ar30.s[0]
	${fmladdsub_b} cr0_im3.4s, br_im0.4s, ar30.s[0]
	${fmlsubaddab} cr0_re3.4s, br_im0.4s, ar30.s[1]
	${fmladdsub_a} cr0_im3.4s, br_re0.4s, ar30.s[1]

	prfm pldl1keep, [local_b_ptr0, 0]
	subs idx_k, idx_k, #4

	fmla         cr0_re0.4s, br_re1.4s, ar00.s[2]
	${fmladdsub_b} cr0_im0.4s, br_im1.4s, ar00.s[2]
	${fmlsubaddab} cr0_re0.4s, br_im1.4s, ar00.s[3]
	${fmladdsub_a} cr0_im0.4s, br_re1.4s, ar00.s[3]

	prfm pldl1keep, [local_a_ptr0, A_PTR_PREFETCH_DISTANCE]
	sub local_a_ptr0, local_a_ptr0, lda3_minus_4, lsl 3

	fmla         cr0_re1.4s, br_re1.4s, ar10.s[2]
	${fmladdsub_b} cr0_im1.4s, br_im1.4s, ar10.s[2]
	${fmlsubaddab} cr0_re1.4s, br_im1.4s, ar10.s[3]
	${fmladdsub_a} cr0_im1.4s, br_re1.4s, ar10.s[3]

	fmla         cr0_re2.4s, br_re1.4s, ar20.s[2]
	${fmladdsub_b} cr0_im2.4s, br_im1.4s, ar20.s[2]
	${fmlsubaddab} cr0_re2.4s, br_im1.4s, ar20.s[3]
	${fmladdsub_a} cr0_im2.4s, br_re1.4s, ar20.s[3]

	prfm pldl1keep, [local_b_ptr0, B_PTR_PREFETCH_DISTANCE]

	fmla         cr0_re3.4s, br_re1.4s, ar30.s[2]
	${fmladdsub_b} cr0_im3.4s, br_im1.4s, ar30.s[2]
	${fmlsubaddab} cr0_re3.4s, br_im1.4s, ar30.s[3]
	${fmladdsub_a} cr0_im3.4s, br_re1.4s, ar30.s[3]

	prfm pldl1keep, [local_a_ptr1, A_PTR_PREFETCH_DISTANCE]

	fmla         cr0_re0.4s, br_re2.4s, ar01.s[0]
	${fmladdsub_b} cr0_im0.4s, br_im2.4s, ar01.s[0]
	${fmlsubaddab} cr0_re0.4s, br_im2.4s, ar01.s[1]
	${fmladdsub_a} cr0_im0.4s, br_re2.4s, ar01.s[1]

	ld2 {br_re3.4s, br_im3.4s}, [ local_b_ptr0 ]
	add local_b_ptr0, local_b_ptr0, ldb, lsl 3

	fmla         cr0_re1.4s, br_re2.4s, ar11.s[0]
	${fmladdsub_b} cr0_im1.4s, br_im2.4s, ar11.s[0]
	${fmlsubaddab} cr0_re1.4s, br_im2.4s, ar11.s[1]
	${fmladdsub_a} cr0_im1.4s, br_re2.4s, ar11.s[1]

	prfm pldl1keep, [local_b_ptr0, 0]

	fmla         cr0_re2.4s, br_re2.4s, ar21.s[0]
	${fmladdsub_b} cr0_im2.4s, br_im2.4s, ar21.s[0]
	${fmlsubaddab} cr0_re2.4s, br_im2.4s, ar21.s[1]
	${fmladdsub_a} cr0_im2.4s, br_re2.4s, ar21.s[1]

	fmla         cr0_re0.4s, br_re3.4s, ar01.s[2]
	${fmladdsub_b} cr0_im0.4s, br_im3.4s, ar01.s[2]
	${fmlsubaddab} cr0_re0.4s, br_im3.4s, ar01.s[3]
	${fmladdsub_a} cr0_im0.4s, br_re3.4s, ar01.s[3]

	ld1 {ar00.4s, ar01.4s}, [ local_a_ptr1 ]
	add local_a_ptr1, local_a_ptr1, lda, lsl 3

	fmla         cr0_re3.4s, br_re2.4s, ar31.s[0]
	${fmladdsub_b} cr0_im3.4s, br_im2.4s, ar31.s[0]
	${fmlsubaddab} cr0_re3.4s, br_im2.4s, ar31.s[1]
	${fmladdsub_a} cr0_im3.4s, br_re2.4s, ar31.s[1]

	fmla         cr0_re1.4s, br_re3.4s, ar11.s[2]
	${fmladdsub_b} cr0_im1.4s, br_im3.4s, ar11.s[2]
	${fmlsubaddab} cr0_re1.4s, br_im3.4s, ar11.s[3]
	${fmladdsub_a} cr0_im1.4s, br_re3.4s, ar11.s[3]

	prfm pldl1keep, [local_a_ptr1, A_PTR_PREFETCH_DISTANCE]

	fmla         cr0_re2.4s, br_re3.4s, ar21.s[2]
	${fmladdsub_b} cr0_im2.4s, br_im3.4s, ar21.s[2]
	${fmlsubaddab} cr0_re2.4s, br_im3.4s, ar21.s[3]
	${fmladdsub_a} cr0_im2.4s, br_re3.4s, ar21.s[3]

	ld1 {ar10.4s, ar11.4s}, [ local_a_ptr1 ]
	add local_a_ptr1, local_a_ptr1, lda, lsl 3

	fmla         cr0_re3.4s, br_re3.4s, ar31.s[2]
	${fmladdsub_b} cr0_im3.4s, br_im3.4s, ar31.s[2]
	${fmlsubaddab} cr0_re3.4s, br_im3.4s, ar31.s[3]
	${fmladdsub_a} cr0_im3.4s, br_re3.4s, ar31.s[3]

	prfm pldl1keep, [local_a_ptr1, A_PTR_PREFETCH_DISTANCE]

	fmla         cr0_re4.4s, br_re0.4s, ar00.s[0]
	${fmladdsub_b} cr0_im4.4s, br_im0.4s, ar00.s[0]
	${fmlsubaddab} cr0_re4.4s, br_im0.4s, ar00.s[1]
	${fmladdsub_a} cr0_im4.4s, br_re0.4s, ar00.s[1]

	ld1 {ar20.4s, ar21.4s}, [ local_a_ptr1 ]
	add local_a_ptr1, local_a_ptr1, lda, lsl 3

	fmla         cr0_re5.4s, br_re0.4s, ar10.s[0]
	${fmladdsub_b} cr0_im5.4s, br_im0.4s, ar10.s[0]
	${fmlsubaddab} cr0_re5.4s, br_im0.4s, ar10.s[1]
	${fmladdsub_a} cr0_im5.4s, br_re0.4s, ar10.s[1]

	prfm pldl1keep, [local_a_ptr1, A_PTR_PREFETCH_DISTANCE]

	fmla         cr0_re6.4s, br_re0.4s, ar20.s[0]
	${fmladdsub_b} cr0_im6.4s, br_im0.4s, ar20.s[0]
	${fmlsubaddab} cr0_re6.4s, br_im0.4s, ar20.s[1]
	${fmladdsub_a} cr0_im6.4s, br_re0.4s, ar20.s[1]

	ld1 {ar30.4s, ar31.4s}, [ local_a_ptr1 ]

	fmla         cr0_re7.4s, br_re0.4s, ar30.s[0]
	${fmladdsub_b} cr0_im7.4s, br_im0.4s, ar30.s[0]
	${fmlsubaddab} cr0_re7.4s, br_im0.4s, ar30.s[1]
	${fmladdsub_a} cr0_im7.4s, br_re0.4s, ar30.s[1]

	prfm pldl1keep, [local_b_ptr0, B_PTR_PREFETCH_DISTANCE]

	fmla         cr0_re4.4s, br_re1.4s, ar00.s[2]
	${fmladdsub_b} cr0_im4.4s, br_im1.4s, ar00.s[2]
	${fmlsubaddab} cr0_re4.4s, br_im1.4s, ar00.s[3]
	${fmladdsub_a} cr0_im4.4s, br_re1.4s, ar00.s[3]

	ld2 {br_re0.4s, br_im0.4s}, [ local_b_ptr0 ]
	add local_b_ptr0, local_b_ptr0, ldb, lsl 3

	fmla         cr0_re5.4s, br_re1.4s, ar10.s[2]
	${fmladdsub_b} cr0_im5.4s, br_im1.4s, ar10.s[2]
	${fmlsubaddab} cr0_re5.4s, br_im1.4s, ar10.s[3]
	${fmladdsub_a} cr0_im5.4s, br_re1.4s, ar10.s[3]

	sub local_a_ptr1, local_a_ptr1, lda3_minus_4, lsl 3
	prfm pldl1keep, [local_b_ptr0, 0]

	fmla         cr0_re6.4s, br_re1.4s, ar20.s[2]
	${fmladdsub_b} cr0_im6.4s, br_im1.4s, ar20.s[2]
	${fmlsubaddab} cr0_re6.4s, br_im1.4s, ar20.s[3]
	${fmladdsub_a} cr0_im6.4s, br_re1.4s, ar20.s[3]

	prfm pldl1keep, [local_a_ptr0, A_PTR_PREFETCH_DISTANCE]

	fmla         cr0_re7.4s, br_re1.4s, ar30.s[2]
	${fmladdsub_b} cr0_im7.4s, br_im1.4s, ar30.s[2]
	${fmlsubaddab} cr0_re7.4s, br_im1.4s, ar30.s[3]
	${fmladdsub_a} cr0_im7.4s, br_re1.4s, ar30.s[3]

	prfm pldl1keep, [local_b_ptr0, B_PTR_PREFETCH_DISTANCE]

	fmla         cr0_re4.4s, br_re2.4s, ar01.s[0]
	${fmladdsub_b} cr0_im4.4s, br_im2.4s, ar01.s[0]
	${fmlsubaddab} cr0_re4.4s, br_im2.4s, ar01.s[1]
	${fmladdsub_a} cr0_im4.4s, br_re2.4s, ar01.s[1]

	ld2 {br_re1.4s, br_im1.4s}, [ local_b_ptr0 ]
	add local_b_ptr0, local_b_ptr0, ldb, lsl 3

	fmla         cr0_re5.4s, br_re2.4s, ar11.s[0]
	${fmladdsub_b} cr0_im5.4s, br_im2.4s, ar11.s[0]
	${fmlsubaddab} cr0_re5.4s, br_im2.4s, ar11.s[1]
	${fmladdsub_a} cr0_im5.4s, br_re2.4s, ar11.s[1]

	prfm pldl1keep, [local_b_ptr0, 0]

	fmla         cr0_re6.4s, br_re2.4s, ar21.s[0]
	${fmladdsub_b} cr0_im6.4s, br_im2.4s, ar21.s[0]
	${fmlsubaddab} cr0_re6.4s, br_im2.4s, ar21.s[1]
	${fmladdsub_a} cr0_im6.4s, br_re2.4s, ar21.s[1]

	prfm pldl1keep, [local_b_ptr0, B_PTR_PREFETCH_DISTANCE]

	fmla         cr0_re4.4s, br_re3.4s, ar01.s[2]
	${fmladdsub_b} cr0_im4.4s, br_im3.4s, ar01.s[2]
	${fmlsubaddab} cr0_re4.4s, br_im3.4s, ar01.s[3]
	${fmladdsub_a} cr0_im4.4s, br_re3.4s, ar01.s[3]

	ld1 {ar00.4s, ar01.4s}, [ local_a_ptr0 ]
	add local_a_ptr0, local_a_ptr0, lda, lsl 3

	fmla         cr0_re7.4s, br_re2.4s, ar31.s[0]
	${fmladdsub_b} cr0_im7.4s, br_im2.4s, ar31.s[0]
	${fmlsubaddab} cr0_re7.4s, br_im2.4s, ar31.s[1]
	${fmladdsub_a} cr0_im7.4s, br_re2.4s, ar31.s[1]

	prfm pldl1keep, [local_a_ptr0, 0]

	fmla         cr0_re5.4s, br_re3.4s, ar11.s[2]
	${fmladdsub_b} cr0_im5.4s, br_im3.4s, ar11.s[2]
	${fmlsubaddab} cr0_re5.4s, br_im3.4s, ar11.s[3]
	${fmladdsub_a} cr0_im5.4s, br_re3.4s, ar11.s[3]

	ld1 {ar10.4s, ar11.4s}, [ local_a_ptr0 ]
	add local_a_ptr0, local_a_ptr0, lda, lsl 3

	fmla         cr0_re6.4s, br_re3.4s, ar21.s[2]
	${fmladdsub_b} cr0_im6.4s, br_im3.4s, ar21.s[2]
	${fmlsubaddab} cr0_re6.4s, br_im3.4s, ar21.s[3]
	${fmladdsub_a} cr0_im6.4s, br_re3.4s, ar21.s[3]

	ld1 {ar20.4s, ar21.4s}, [ local_a_ptr0 ]
	add local_a_ptr0, local_a_ptr0, lda, lsl 3

	fmla         cr0_re7.4s, br_re3.4s, ar31.s[2]
	${fmladdsub_b} cr0_im7.4s, br_im3.4s, ar31.s[2]
	${fmlsubaddab} cr0_re7.4s, br_im3.4s, ar31.s[3]
	${fmladdsub_a} cr0_im7.4s, br_re3.4s, ar31.s[3]

	bge .L${func_name}_loop_k

.L${func_name}_loop_k_almost_last:

	// cr0_re = ar_re * br_re - ar_im * br_im + cr0_re;
	// cr0_im = ar_re * br_im + ar_im * br_re + cr0_im;
	fmla         cr0_re0.4s, br_re0.4s, ar00.s[0]
	${fmladdsub_b} cr0_im0.4s, br_im0.4s, ar00.s[0]
	${fmlsubaddab} cr0_re0.4s, br_im0.4s, ar00.s[1]
	${fmladdsub_a} cr0_im0.4s, br_re0.4s, ar00.s[1]

	ld1 {ar30.4s, ar31.4s}, [ local_a_ptr0 ]

	fmla         cr0_re1.4s, br_re0.4s, ar10.s[0]
	${fmladdsub_b} cr0_im1.4s, br_im0.4s, ar10.s[0]
	${fmlsubaddab} cr0_re1.4s, br_im0.4s, ar10.s[1]
	${fmladdsub_a} cr0_im1.4s, br_re0.4s, ar10.s[1]

	prfm pldl1keep, [local_b_ptr0, B_PTR_PREFETCH_DISTANCE]

	fmla         cr0_re2.4s, br_re0.4s, ar20.s[0]
	${fmladdsub_b} cr0_im2.4s, br_im0.4s, ar20.s[0]
	${fmlsubaddab} cr0_re2.4s, br_im0.4s, ar20.s[1]
	${fmladdsub_a} cr0_im2.4s, br_re0.4s, ar20.s[1]

	ld2 {br_re2.4s, br_im2.4s}, [ local_b_ptr0 ]
	add local_b_ptr0, local_b_ptr0, ldb, lsl 3

	fmla         cr0_re3.4s, br_re0.4s, ar30.s[0]
	${fmladdsub_b} cr0_im3.4s, br_im0.4s, ar30.s[0]
	${fmlsubaddab} cr0_re3.4s, br_im0.4s, ar30.s[1]
	${fmladdsub_a} cr0_im3.4s, br_re0.4s, ar30.s[1]

	ld2 {br_re3.4s, br_im3.4s}, [ local_b_ptr0 ]

	fmla         cr0_re0.4s, br_re1.4s, ar00.s[2]
	${fmladdsub_b} cr0_im0.4s, br_im1.4s, ar00.s[2]
	${fmlsubaddab} cr0_re0.4s, br_im1.4s, ar00.s[3]
	${fmladdsub_a} cr0_im0.4s, br_re1.4s, ar00.s[3]

	sub local_a_ptr0, local_a_ptr0, lda3_minus_4, lsl 3

	fmla         cr0_re1.4s, br_re1.4s, ar10.s[2]
	${fmladdsub_b} cr0_im1.4s, br_im1.4s, ar10.s[2]
	${fmlsubaddab} cr0_re1.4s, br_im1.4s, ar10.s[3]
	${fmladdsub_a} cr0_im1.4s, br_re1.4s, ar10.s[3]

	fmla         cr0_re2.4s, br_re1.4s, ar20.s[2]
	${fmladdsub_b} cr0_im2.4s, br_im1.4s, ar20.s[2]
	${fmlsubaddab} cr0_re2.4s, br_im1.4s, ar20.s[3]
	${fmladdsub_a} cr0_im2.4s, br_re1.4s, ar20.s[3]

	prfm pldl1keep, [local_a_ptr0, A_PTR_PREFETCH_DISTANCE]

	fmla         cr0_re3.4s, br_re1.4s, ar30.s[2]
	${fmladdsub_b} cr0_im3.4s, br_im1.4s, ar30.s[2]
	${fmlsubaddab} cr0_re3.4s, br_im1.4s, ar30.s[3]
	${fmladdsub_a} cr0_im3.4s, br_re1.4s, ar30.s[3]

	prfm pldl1keep, [local_a_ptr1, A_PTR_PREFETCH_DISTANCE]

	fmla         cr0_re0.4s, br_re2.4s, ar01.s[0]
	${fmladdsub_b} cr0_im0.4s, br_im2.4s, ar01.s[0]
	${fmlsubaddab} cr0_re0.4s, br_im2.4s, ar01.s[1]
	${fmladdsub_a} cr0_im0.4s, br_re2.4s, ar01.s[1]

	fmla         cr0_re1.4s, br_re2.4s, ar11.s[0]
	${fmladdsub_b} cr0_im1.4s, br_im2.4s, ar11.s[0]
	${fmlsubaddab} cr0_re1.4s, br_im2.4s, ar11.s[1]
	${fmladdsub_a} cr0_im1.4s, br_re2.4s, ar11.s[1]

	fmla         cr0_re2.4s, br_re2.4s, ar21.s[0]
	${fmladdsub_b} cr0_im2.4s, br_im2.4s, ar21.s[0]
	${fmlsubaddab} cr0_re2.4s, br_im2.4s, ar21.s[1]
	${fmladdsub_a} cr0_im2.4s, br_re2.4s, ar21.s[1]

	fmla         cr0_re3.4s, br_re2.4s, ar31.s[0]
	${fmladdsub_b} cr0_im3.4s, br_im2.4s, ar31.s[0]
	${fmlsubaddab} cr0_re3.4s, br_im2.4s, ar31.s[1]
	${fmladdsub_a} cr0_im3.4s, br_re2.4s, ar31.s[1]

	fmla         cr0_re0.4s, br_re3.4s, ar01.s[2]
	${fmladdsub_b} cr0_im0.4s, br_im3.4s, ar01.s[2]
	${fmlsubaddab} cr0_re0.4s, br_im3.4s, ar01.s[3]
	${fmladdsub_a} cr0_im0.4s, br_re3.4s, ar01.s[3]

	ld1 {ar00.4s, ar01.4s}, [ local_a_ptr1 ]
	add local_a_ptr1, local_a_ptr1, lda, lsl 3

	fmla         cr0_re1.4s, br_re3.4s, ar11.s[2]
	${fmladdsub_b} cr0_im1.4s, br_im3.4s, ar11.s[2]
	${fmlsubaddab} cr0_re1.4s, br_im3.4s, ar11.s[3]
	${fmladdsub_a} cr0_im1.4s, br_re3.4s, ar11.s[3]

	prfm pldl1keep, [local_a_ptr1, A_PTR_PREFETCH_DISTANCE]

	fmla         cr0_re2.4s, br_re3.4s, ar21.s[2]
	${fmladdsub_b} cr0_im2.4s, br_im3.4s, ar21.s[2]
	${fmlsubaddab} cr0_re2.4s, br_im3.4s, ar21.s[3]
	${fmladdsub_a} cr0_im2.4s, br_re3.4s, ar21.s[3]

	ld1 {ar10.4s, ar11.4s}, [ local_a_ptr1 ]
	add local_a_ptr1, local_a_ptr1, lda, lsl 3

	fmla         cr0_re3.4s, br_re3.4s, ar31.s[2]
	${fmladdsub_b} cr0_im3.4s, br_im3.4s, ar31.s[2]
	${fmlsubaddab} cr0_re3.4s, br_im3.4s, ar31.s[3]
	${fmladdsub_a} cr0_im3.4s, br_re3.4s, ar31.s[3]

	prfm pldl1keep, [local_a_ptr1, A_PTR_PREFETCH_DISTANCE]

	fmla         cr0_re4.4s, br_re0.4s, ar00.s[0]
	${fmladdsub_b} cr0_im4.4s, br_im0.4s, ar00.s[0]
	${fmlsubaddab} cr0_re4.4s, br_im0.4s, ar00.s[1]
	${fmladdsub_a} cr0_im4.4s, br_re0.4s, ar00.s[1]

	ld1 {ar20.4s, ar21.4s}, [ local_a_ptr1 ]
	add local_a_ptr1, local_a_ptr1, lda, lsl 3

	fmla         cr0_re5.4s, br_re0.4s, ar10.s[0]
	${fmladdsub_b} cr0_im5.4s, br_im0.4s, ar10.s[0]
	${fmlsubaddab} cr0_re5.4s, br_im0.4s, ar10.s[1]
	${fmladdsub_a} cr0_im5.4s, br_re0.4s, ar10.s[1]

	ld1 {ar30.4s, ar31.4s}, [ local_a_ptr1 ]

	fmla         cr0_re6.4s, br_re0.4s, ar20.s[0]
	${fmladdsub_b} cr0_im6.4s, br_im0.4s, ar20.s[0]
	${fmlsubaddab} cr0_re6.4s, br_im0.4s, ar20.s[1]
	${fmladdsub_a} cr0_im6.4s, br_re0.4s, ar20.s[1]

	add local_b_ptr0, local_b_ptr0, ldb, lsl 3

	fmla         cr0_re7.4s, br_re0.4s, ar30.s[0]
	${fmladdsub_b} cr0_im7.4s, br_im0.4s, ar30.s[0]
	${fmlsubaddab} cr0_re7.4s, br_im0.4s, ar30.s[1]
	${fmladdsub_a} cr0_im7.4s, br_re0.4s, ar30.s[1]

	sub local_a_ptr1, local_a_ptr1, lda3_minus_4, lsl 3

	fmla         cr0_re4.4s, br_re1.4s, ar00.s[2]
	${fmladdsub_b} cr0_im4.4s, br_im1.4s, ar00.s[2]
	${fmlsubaddab} cr0_re4.4s, br_im1.4s, ar00.s[3]
	${fmladdsub_a} cr0_im4.4s, br_re1.4s, ar00.s[3]

	fmla         cr0_re5.4s, br_re1.4s, ar10.s[2]
	${fmladdsub_b} cr0_im5.4s, br_im1.4s, ar10.s[2]
	${fmlsubaddab} cr0_re5.4s, br_im1.4s, ar10.s[3]
	${fmladdsub_a} cr0_im5.4s, br_re1.4s, ar10.s[3]

	fmla         cr0_re6.4s, br_re1.4s, ar20.s[2]
	${fmladdsub_b} cr0_im6.4s, br_im1.4s, ar20.s[2]
	${fmlsubaddab} cr0_re6.4s, br_im1.4s, ar20.s[3]
	${fmladdsub_a} cr0_im6.4s, br_re1.4s, ar20.s[3]

	fmla         cr0_re7.4s, br_re1.4s, ar30.s[2]
	${fmladdsub_b} cr0_im7.4s, br_im1.4s, ar30.s[2]
	${fmlsubaddab} cr0_re7.4s, br_im1.4s, ar30.s[3]
	${fmladdsub_a} cr0_im7.4s, br_re1.4s, ar30.s[3]

	fmla         cr0_re4.4s, br_re2.4s, ar01.s[0]
	${fmladdsub_b} cr0_im4.4s, br_im2.4s, ar01.s[0]
	${fmlsubaddab} cr0_re4.4s, br_im2.4s, ar01.s[1]
	${fmladdsub_a} cr0_im4.4s, br_re2.4s, ar01.s[1]

	fmla         cr0_re5.4s, br_re2.4s, ar11.s[0]
	${fmladdsub_b} cr0_im5.4s, br_im2.4s, ar11.s[0]
	${fmlsubaddab} cr0_re5.4s, br_im2.4s, ar11.s[1]
	${fmladdsub_a} cr0_im5.4s, br_re2.4s, ar11.s[1]

	fmla         cr0_re6.4s, br_re2.4s, ar21.s[0]
	${fmladdsub_b} cr0_im6.4s, br_im2.4s, ar21.s[0]
	${fmlsubaddab} cr0_re6.4s, br_im2.4s, ar21.s[1]
	${fmladdsub_a} cr0_im6.4s, br_re2.4s, ar21.s[1]

	fmla         cr0_re7.4s, br_re2.4s, ar31.s[0]
	${fmladdsub_b} cr0_im7.4s, br_im2.4s, ar31.s[0]
	${fmlsubaddab} cr0_re7.4s, br_im2.4s, ar31.s[1]
	${fmladdsub_a} cr0_im7.4s, br_re2.4s, ar31.s[1]

	fmla         cr0_re4.4s, br_re3.4s, ar01.s[2]
	${fmladdsub_b} cr0_im4.4s, br_im3.4s, ar01.s[2]
	${fmlsubaddab} cr0_re4.4s, br_im3.4s, ar01.s[3]
	${fmladdsub_a} cr0_im4.4s, br_re3.4s, ar01.s[3]

	fmla         cr0_re5.4s, br_re3.4s, ar11.s[2]
	${fmladdsub_b} cr0_im5.4s, br_im3.4s, ar11.s[2]
	${fmlsubaddab} cr0_re5.4s, br_im3.4s, ar11.s[3]
	${fmladdsub_a} cr0_im5.4s, br_re3.4s, ar11.s[3]

	fmla         cr0_re6.4s, br_re3.4s, ar21.s[2]
	${fmladdsub_b} cr0_im6.4s, br_im3.4s, ar21.s[2]
	${fmlsubaddab} cr0_re6.4s, br_im3.4s, ar21.s[3]
	${fmladdsub_a} cr0_im6.4s, br_re3.4s, ar21.s[3]

	fmla         cr0_re7.4s, br_re3.4s, ar31.s[2]
	${fmladdsub_b} cr0_im7.4s, br_im3.4s, ar31.s[2]
	${fmlsubaddab} cr0_re7.4s, br_im3.4s, ar31.s[3]
	${fmladdsub_a} cr0_im7.4s, br_re3.4s, ar31.s[3]

.L${func_name}_loop_k_last: // }}}

	add idx_k, idx_k, #4

	cmp idx_k, #0
	beq .L${func_name}_loop_k_end
	cmp idx_k, #1
	beq .L${func_name}_loop_k_last_1

.L${func_name}_loop_k_last_2:
	ld2 {br_re0.4s, br_im0.4s}, [ local_b_ptr0 ]
	add local_b_ptr0, local_b_ptr0, ldb, lsl 3

	ld1 {ar00.4s}, [ local_a_ptr0 ]
	add local_a_ptr0, local_a_ptr0, lda, lsl 3
	ld1 {ar10.4s}, [ local_a_ptr0 ]
	add local_a_ptr0, local_a_ptr0, lda, lsl 3
	ld1 {ar20.4s}, [ local_a_ptr0 ]
	add local_a_ptr0, local_a_ptr0, lda, lsl 3
	ld1 {ar30.4s}, [ local_a_ptr0 ]

	// cr0_re = ar_re * br_re - ar_im * br_im + cr0_re;
	// cr0_im = ar_re * br_im + ar_im * br_re + cr0_im;
	fmla         cr0_re0.4s, br_re0.4s, ar00.s[0]
	${fmladdsub_b} cr0_im0.4s, br_im0.4s, ar00.s[0]
	${fmlsubaddab} cr0_re0.4s, br_im0.4s, ar00.s[1]
	${fmladdsub_a} cr0_im0.4s, br_re0.4s, ar00.s[1]

	ld2 {br_re1.4s, br_im1.4s}, [ local_b_ptr0 ]

	fmla         cr0_re1.4s, br_re0.4s, ar10.s[0]
	${fmladdsub_b} cr0_im1.4s, br_im0.4s, ar10.s[0]
	${fmlsubaddab} cr0_re1.4s, br_im0.4s, ar10.s[1]
	${fmladdsub_a} cr0_im1.4s, br_re0.4s, ar10.s[1]

	fmla         cr0_re2.4s, br_re0.4s, ar20.s[0]
	${fmladdsub_b} cr0_im2.4s, br_im0.4s, ar20.s[0]
	${fmlsubaddab} cr0_re2.4s, br_im0.4s, ar20.s[1]
	${fmladdsub_a} cr0_im2.4s, br_re0.4s, ar20.s[1]

	fmla         cr0_re3.4s, br_re0.4s, ar30.s[0]
	${fmladdsub_b} cr0_im3.4s, br_im0.4s, ar30.s[0]
	${fmlsubaddab} cr0_re3.4s, br_im0.4s, ar30.s[1]
	${fmladdsub_a} cr0_im3.4s, br_re0.4s, ar30.s[1]

	fmla         cr0_re0.4s, br_re1.4s, ar00.s[2]
	${fmladdsub_b} cr0_im0.4s, br_im1.4s, ar00.s[2]
	${fmlsubaddab} cr0_re0.4s, br_im1.4s, ar00.s[3]
	${fmladdsub_a} cr0_im0.4s, br_re1.4s, ar00.s[3]

	ld1 {ar00.4s}, [ local_a_ptr1 ]
	add local_a_ptr1, local_a_ptr1, lda, lsl 3

	fmla         cr0_re1.4s, br_re1.4s, ar10.s[2]
	${fmladdsub_b} cr0_im1.4s, br_im1.4s, ar10.s[2]
	${fmlsubaddab} cr0_re1.4s, br_im1.4s, ar10.s[3]
	${fmladdsub_a} cr0_im1.4s, br_re1.4s, ar10.s[3]

	ld1 {ar10.4s}, [ local_a_ptr1 ]
	add local_a_ptr1, local_a_ptr1, lda, lsl 3

	fmla         cr0_re2.4s, br_re1.4s, ar20.s[2]
	${fmladdsub_b} cr0_im2.4s, br_im1.4s, ar20.s[2]
	${fmlsubaddab} cr0_re2.4s, br_im1.4s, ar20.s[3]
	${fmladdsub_a} cr0_im2.4s, br_re1.4s, ar20.s[3]

	ld1 {ar20.4s}, [ local_a_ptr1 ]
	add local_a_ptr1, local_a_ptr1, lda, lsl 3

	fmla         cr0_re3.4s, br_re1.4s, ar30.s[2]
	${fmladdsub_b} cr0_im3.4s, br_im1.4s, ar30.s[2]
	${fmlsubaddab} cr0_re3.4s, br_im1.4s, ar30.s[3]
	${fmladdsub_a} cr0_im3.4s, br_re1.4s, ar30.s[3]

	ld1 {ar30.4s}, [ local_a_ptr1 ]

	fmla         cr0_re4.4s, br_re0.4s, ar00.s[0]
	${fmladdsub_b} cr0_im4.4s, br_im0.4s, ar00.s[0]
	${fmlsubaddab} cr0_re4.4s, br_im0.4s, ar00.s[1]
	${fmladdsub_a} cr0_im4.4s, br_re0.4s, ar00.s[1]

	cmp idx_k, #3

	fmla         cr0_re5.4s, br_re0.4s, ar10.s[0]
	${fmladdsub_b} cr0_im5.4s, br_im0.4s, ar10.s[0]
	${fmlsubaddab} cr0_re5.4s, br_im0.4s, ar10.s[1]
	${fmladdsub_a} cr0_im5.4s, br_re0.4s, ar10.s[1]

	fmla         cr0_re6.4s, br_re0.4s, ar20.s[0]
	${fmladdsub_b} cr0_im6.4s, br_im0.4s, ar20.s[0]
	${fmlsubaddab} cr0_re6.4s, br_im0.4s, ar20.s[1]
	${fmladdsub_a} cr0_im6.4s, br_re0.4s, ar20.s[1]

	fmla         cr0_re7.4s, br_re0.4s, ar30.s[0]
	${fmladdsub_b} cr0_im7.4s, br_im0.4s, ar30.s[0]
	${fmlsubaddab} cr0_re7.4s, br_im0.4s, ar30.s[1]
	${fmladdsub_a} cr0_im7.4s, br_re0.4s, ar30.s[1]

	fmla         cr0_re4.4s, br_re1.4s, ar00.s[2]
	${fmladdsub_b} cr0_im4.4s, br_im1.4s, ar00.s[2]
	${fmlsubaddab} cr0_re4.4s, br_im1.4s, ar00.s[3]
	${fmladdsub_a} cr0_im4.4s, br_re1.4s, ar00.s[3]

	fmla         cr0_re5.4s, br_re1.4s, ar10.s[2]
	${fmladdsub_b} cr0_im5.4s, br_im1.4s, ar10.s[2]
	${fmlsubaddab} cr0_re5.4s, br_im1.4s, ar10.s[3]
	${fmladdsub_a} cr0_im5.4s, br_re1.4s, ar10.s[3]

	fmla         cr0_re6.4s, br_re1.4s, ar20.s[2]
	${fmladdsub_b} cr0_im6.4s, br_im1.4s, ar20.s[2]
	${fmlsubaddab} cr0_re6.4s, br_im1.4s, ar20.s[3]
	${fmladdsub_a} cr0_im6.4s, br_re1.4s, ar20.s[3]

	fmla         cr0_re7.4s, br_re1.4s, ar30.s[2]
	${fmladdsub_b} cr0_im7.4s, br_im1.4s, ar30.s[2]
	${fmlsubaddab} cr0_re7.4s, br_im1.4s, ar30.s[3]
	${fmladdsub_a} cr0_im7.4s, br_re1.4s, ar30.s[3]

	bne .L${func_name}_loop_k_end

	sub local_a_ptr0, local_a_ptr0, lda3_minus_4, lsl 3
	sub local_a_ptr1, local_a_ptr1, lda3_minus_4, lsl 3
	sub local_a_ptr0, local_a_ptr0, (2 << 3)
	sub local_a_ptr1, local_a_ptr1, (2 << 3)
	add local_b_ptr0, local_b_ptr0, ldb, lsl 3

.L${func_name}_loop_k_last_1:
	ld2 {br_re0.4s, br_im0.4s}, [ local_b_ptr0 ]
	add local_b_ptr0, local_b_ptr0, ldb, lsl 3

	ld1 {ar00.2s}, [ local_a_ptr0 ]
	add local_a_ptr0, local_a_ptr0, lda, lsl 3
	ld1 {ar10.2s}, [ local_a_ptr0 ]
	add local_a_ptr0, local_a_ptr0, lda, lsl 3
	ld1 {ar20.2s}, [ local_a_ptr0 ]
	add local_a_ptr0, local_a_ptr0, lda, lsl 3
	ld1 {ar30.2s}, [ local_a_ptr0 ]

	// cr0_re = ar_re * br_re - ar_im * br_im + cr0_re;
	// cr0_im = ar_re * br_im + ar_im * br_re + cr0_im;
	fmla         cr0_re0.4s, br_re0.4s, ar00.s[0]
	${fmladdsub_b} cr0_im0.4s, br_im0.4s, ar00.s[0]
	${fmlsubaddab} cr0_re0.4s, br_im0.4s, ar00.s[1]
	${fmladdsub_a} cr0_im0.4s, br_re0.4s, ar00.s[1]

	fmla         cr0_re1.4s, br_re0.4s, ar10.s[0]
	${fmladdsub_b} cr0_im1.4s, br_im0.4s, ar10.s[0]
	${fmlsubaddab} cr0_re1.4s, br_im0.4s, ar10.s[1]
	${fmladdsub_a} cr0_im1.4s, br_re0.4s, ar10.s[1]

	ld1 {ar00.2s}, [ local_a_ptr1 ]
	add local_a_ptr1, local_a_ptr1, lda, lsl 3

	fmla         cr0_re2.4s, br_re0.4s, ar20.s[0]
	${fmladdsub_b} cr0_im2.4s, br_im0.4s, ar20.s[0]
	${fmlsubaddab} cr0_re2.4s, br_im0.4s, ar20.s[1]
	${fmladdsub_a} cr0_im2.4s, br_re0.4s, ar20.s[1]

	ld1 {ar10.2s}, [ local_a_ptr1 ]
	add local_a_ptr1, local_a_ptr1, lda, lsl 3

	fmla         cr0_re3.4s, br_re0.4s, ar30.s[0]
	${fmladdsub_b} cr0_im3.4s, br_im0.4s, ar30.s[0]
	${fmlsubaddab} cr0_re3.4s, br_im0.4s, ar30.s[1]
	${fmladdsub_a} cr0_im3.4s, br_re0.4s, ar30.s[1]

	ld1 {ar20.2s}, [ local_a_ptr1 ]
	add local_a_ptr1, local_a_ptr1, lda, lsl 3

	fmla         cr0_re4.4s, br_re0.4s, ar00.s[0]
	${fmladdsub_b} cr0_im4.4s, br_im0.4s, ar00.s[0]
	${fmlsubaddab} cr0_re4.4s, br_im0.4s, ar00.s[1]
	${fmladdsub_a} cr0_im4.4s, br_re0.4s, ar00.s[1]

	ld1 {ar30.2s}, [ local_a_ptr1 ]

	fmla         cr0_re5.4s, br_re0.4s, ar10.s[0]
	${fmladdsub_b} cr0_im5.4s, br_im0.4s, ar10.s[0]
	${fmlsubaddab} cr0_re5.4s, br_im0.4s, ar10.s[1]
	${fmladdsub_a} cr0_im5.4s, br_re0.4s, ar10.s[1]

	fmla         cr0_re6.4s, br_re0.4s, ar20.s[0]
	${fmladdsub_b} cr0_im6.4s, br_im0.4s, ar20.s[0]
	${fmlsubaddab} cr0_re6.4s, br_im0.4s, ar20.s[1]
	${fmladdsub_a} cr0_im6.4s, br_re0.4s, ar20.s[1]

	fmla         cr0_re7.4s, br_re0.4s, ar30.s[0]
	${fmladdsub_b} cr0_im7.4s, br_im0.4s, ar30.s[0]
	${fmlsubaddab} cr0_re7.4s, br_im0.4s, ar30.s[1]
	${fmladdsub_a} cr0_im7.4s, br_re0.4s, ar30.s[1]

.L${func_name}_loop_k_end: // }}}

#define local_c_ptr0 x28
#define local_c_ptr1 x27
#define local_c_ptr2 x26
#define local_c_ptr3 x25

#define cr1_im7 v30
#define cr1_re7 v29
#define cr1_im6 v28
#define cr1_re6 v27
#define cr1_im5 v26
#define cr1_re5 v25
#define cr1_im4 v24
#define cr1_re4 v23
// start again
#define cr1_im3 v30
#define cr1_re3 v29
#define cr1_im2 v28
#define cr1_re2 v27
#define cr1_im1 v26
#define cr1_re1 v25
#define cr1_im0 v24
#define cr1_re0 v23

#define cr_im7 v15
#define cr_re7 v14
#define cr_im6 v13
#define cr_re6 v12
#define cr_im5 v11
#define cr_re5 v10
#define cr_im4 v9
#define cr_re4 v8
#define cr_im3 v7
#define cr_re3 v6
#define cr_im2 v5
#define cr_re2 v4
#define cr_im1 v3
#define cr_re1 v2
#define cr_im0 v1
#define cr_re0 v0

	mov alpha, x_alpha
	mov beta, x_beta

	cmp beta_is_zero, 1
	bne .L${func_name}_beta_non_zero

	cmp k_block_idx, 0
	bne .L${func_name}_beta_non_zero

.L${func_name}_beta_zero:
	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	madd local_c_ptr0, ldc, idx_n, idx_m0
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	add local_c_ptr2, local_c_ptr1, ldc, lsl 3
	add local_c_ptr3, local_c_ptr2, ldc, lsl 3

	fmul cr1_re0.4s, cr0_re0.4s, alpha_real
	fmul cr1_im0.4s, cr0_re0.4s, alpha_imag
	fmls cr1_re0.4s, cr0_im0.4s, alpha_imag
	fmla cr1_im0.4s, cr0_im0.4s, alpha_real

	fmul cr1_re1.4s, cr0_re1.4s, alpha_real
	fmul cr1_im1.4s, cr0_re1.4s, alpha_imag
	fmls cr1_re1.4s, cr0_im1.4s, alpha_imag
	fmla cr1_im1.4s, cr0_im1.4s, alpha_real

	fmul cr1_re2.4s, cr0_re2.4s, alpha_real
	fmul cr1_im2.4s, cr0_re2.4s, alpha_imag
	fmls cr1_re2.4s, cr0_im2.4s, alpha_imag
	fmla cr1_im2.4s, cr0_im2.4s, alpha_real

	fmul cr1_re3.4s, cr0_re3.4s, alpha_real
	fmul cr1_im3.4s, cr0_re3.4s, alpha_imag
	fmls cr1_re3.4s, cr0_im3.4s, alpha_imag
	fmla cr1_im3.4s, cr0_im3.4s, alpha_real

	dup cr_re0.2d, xzr
	dup cr_im0.2d, xzr
	dup cr_re1.2d, xzr
	dup cr_im1.2d, xzr

	add local_c_ptr0, local_c_ptr0, (4*4)
	add local_c_ptr1, local_c_ptr1, (4*4)
	add local_c_ptr2, local_c_ptr2, (4*4)
	add local_c_ptr3, local_c_ptr3, (4*4)

	dup cr_re2.2d, xzr
	dup cr_im2.2d, xzr
	dup cr_re3.2d, xzr
	dup cr_im3.2d, xzr

	cmp k_block_idx, #0
	bne 10f

	// crr_re = cr1_re + cr_re * beta.real() - cr_im * beta.imag();
	// crr_im = cr1_im + cr_re * beta.imag() + cr_im * beta.real();
	fmla cr1_re0.4s, cr_re0.4s, beta_real
	fmla cr1_im0.4s, cr_re0.4s, beta_imag
	fmls cr1_re0.4s, cr_im0.4s, beta_imag
	fmla cr1_im0.4s, cr_im0.4s, beta_real

	fmla cr1_re1.4s, cr_re1.4s, beta_real
	fmla cr1_im1.4s, cr_re1.4s, beta_imag
	fmls cr1_re1.4s, cr_im1.4s, beta_imag
	fmla cr1_im1.4s, cr_im1.4s, beta_real

	fmla cr1_re2.4s, cr_re2.4s, beta_real
	fmla cr1_im2.4s, cr_re2.4s, beta_imag
	fmls cr1_re2.4s, cr_im2.4s, beta_imag
	fmla cr1_im2.4s, cr_im2.4s, beta_real

	fmla cr1_re3.4s, cr_re3.4s, beta_real
	fmla cr1_im3.4s, cr_re3.4s, beta_imag
	fmls cr1_re3.4s, cr_im3.4s, beta_imag
	fmla cr1_im3.4s, cr_im3.4s, beta_real
	b 20f

10:

	// crr_re = cr1_re + cr_re - cr_im;
	// crr_im = cr1_im + cr_re + cr_im;
	fadd cr1_re0.4s, cr1_re0.4s, cr_re0.4s
	fadd cr1_im0.4s, cr1_im0.4s, cr_im0.4s

	fadd cr1_re1.4s, cr1_re1.4s, cr_re1.4s
	fadd cr1_im1.4s, cr1_im1.4s, cr_im1.4s

	fadd cr1_re2.4s, cr1_re2.4s, cr_re2.4s
	fadd cr1_im2.4s, cr1_im2.4s, cr_im2.4s

	fadd cr1_re3.4s, cr1_re3.4s, cr_re3.4s
	fadd cr1_im3.4s, cr1_im3.4s, cr_im3.4s
20:

	st4 {cr1_re2.s, cr1_im2.s, cr1_re3.s, cr1_im3.s}[0], [ local_c_ptr0 ]
	st4 {cr1_re2.s, cr1_im2.s, cr1_re3.s, cr1_im3.s}[1], [ local_c_ptr1 ]
	st4 {cr1_re2.s, cr1_im2.s, cr1_re3.s, cr1_im3.s}[2], [ local_c_ptr2 ]
	st4 {cr1_re2.s, cr1_im2.s, cr1_re3.s, cr1_im3.s}[3], [ local_c_ptr3 ]

	sub local_c_ptr0, local_c_ptr0, (4*4)
	sub local_c_ptr1, local_c_ptr1, (4*4)
	sub local_c_ptr2, local_c_ptr2, (4*4)
	sub local_c_ptr3, local_c_ptr3, (4*4)

	st4 {cr1_re0.s, cr1_im0.s, cr1_re1.s, cr1_im1.s}[0], [ local_c_ptr0 ]
	st4 {cr1_re0.s, cr1_im0.s, cr1_re1.s, cr1_im1.s}[1], [ local_c_ptr1 ]
	st4 {cr1_re0.s, cr1_im0.s, cr1_re1.s, cr1_im1.s}[2], [ local_c_ptr2 ]
	st4 {cr1_re0.s, cr1_im0.s, cr1_re1.s, cr1_im1.s}[3], [ local_c_ptr3 ]

	cmp idx_m0, idx_m1
	beq .L${func_name}_loop_skip

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	madd local_c_ptr0, ldc, idx_n, idx_m1
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	add local_c_ptr2, local_c_ptr1, ldc, lsl 3
	add local_c_ptr3, local_c_ptr2, ldc, lsl 3

	fmul cr1_re4.4s, cr0_re4.4s, alpha_real
	fmul cr1_im4.4s, cr0_re4.4s, alpha_imag
	fmls cr1_re4.4s, cr0_im4.4s, alpha_imag
	fmla cr1_im4.4s, cr0_im4.4s, alpha_real

	fmul cr1_re5.4s, cr0_re5.4s, alpha_real
	fmul cr1_im5.4s, cr0_re5.4s, alpha_imag
	fmls cr1_re5.4s, cr0_im5.4s, alpha_imag
	fmla cr1_im5.4s, cr0_im5.4s, alpha_real

	fmul cr1_re6.4s, cr0_re6.4s, alpha_real
	fmul cr1_im6.4s, cr0_re6.4s, alpha_imag
	fmls cr1_re6.4s, cr0_im6.4s, alpha_imag
	fmla cr1_im6.4s, cr0_im6.4s, alpha_real

	fmul cr1_re7.4s, cr0_re7.4s, alpha_real
	fmul cr1_im7.4s, cr0_re7.4s, alpha_imag
	fmls cr1_re7.4s, cr0_im7.4s, alpha_imag
	fmla cr1_im7.4s, cr0_im7.4s, alpha_real

	dup cr_re4.2d, xzr
	dup cr_im4.2d, xzr
	dup cr_re5.2d, xzr
	dup cr_im5.2d, xzr

	add local_c_ptr0, local_c_ptr0, (4*4)
	add local_c_ptr1, local_c_ptr1, (4*4)
	add local_c_ptr2, local_c_ptr2, (4*4)
	add local_c_ptr3, local_c_ptr3, (4*4)

	dup cr_re6.2d, xzr
	dup cr_im6.2d, xzr
	dup cr_re7.2d, xzr
	dup cr_im7.2d, xzr

	cmp k_block_idx, #0
	bne 10f

	// crr_re = cr1_re + cr_re * beta.real() - cr_im * beta.imag();
	// crr_im = cr1_im + cr_re * beta.imag() + cr_im * beta.real();
	fmla cr1_re4.4s, cr_re4.4s, beta_real
	fmla cr1_im4.4s, cr_re4.4s, beta_imag
	fmls cr1_re4.4s, cr_im4.4s, beta_imag
	fmla cr1_im4.4s, cr_im4.4s, beta_real

	fmla cr1_re5.4s, cr_re5.4s, beta_real
	fmla cr1_im5.4s, cr_re5.4s, beta_imag
	fmls cr1_re5.4s, cr_im5.4s, beta_imag
	fmla cr1_im5.4s, cr_im5.4s, beta_real

	fmla cr1_re6.4s, cr_re6.4s, beta_real
	fmla cr1_im6.4s, cr_re6.4s, beta_imag
	fmls cr1_re6.4s, cr_im6.4s, beta_imag
	fmla cr1_im6.4s, cr_im6.4s, beta_real

	fmla cr1_re7.4s, cr_re7.4s, beta_real
	fmla cr1_im7.4s, cr_re7.4s, beta_imag
	fmls cr1_re7.4s, cr_im7.4s, beta_imag
	fmla cr1_im7.4s, cr_im7.4s, beta_real
	b 20f

10:

	// crr_re = cr1_re + cr_re - cr_im;
	// crr_im = cr1_im + cr_re + cr_im;
	fadd cr1_re4.4s, cr1_re4.4s, cr_re4.4s
	fadd cr1_im4.4s, cr1_im4.4s, cr_im4.4s

	fadd cr1_re5.4s, cr1_re5.4s, cr_re5.4s
	fadd cr1_im5.4s, cr1_im5.4s, cr_im5.4s

	fadd cr1_re6.4s, cr1_re6.4s, cr_re6.4s
	fadd cr1_im6.4s, cr1_im6.4s, cr_im6.4s

	fadd cr1_re7.4s, cr1_re7.4s, cr_re7.4s
	fadd cr1_im7.4s, cr1_im7.4s, cr_im7.4s

20:

	st4 {cr1_re6.s, cr1_im6.s, cr1_re7.s, cr1_im7.s}[0], [ local_c_ptr0 ]
	st4 {cr1_re6.s, cr1_im6.s, cr1_re7.s, cr1_im7.s}[1], [ local_c_ptr1 ]
	st4 {cr1_re6.s, cr1_im6.s, cr1_re7.s, cr1_im7.s}[2], [ local_c_ptr2 ]
	st4 {cr1_re6.s, cr1_im6.s, cr1_re7.s, cr1_im7.s}[3], [ local_c_ptr3 ]

	sub local_c_ptr0, local_c_ptr0, (4*4)
	sub local_c_ptr1, local_c_ptr1, (4*4)
	sub local_c_ptr2, local_c_ptr2, (4*4)
	sub local_c_ptr3, local_c_ptr3, (4*4)

	st4 {cr1_re4.s, cr1_im4.s, cr1_re5.s, cr1_im5.s}[0], [ local_c_ptr0 ]
	st4 {cr1_re4.s, cr1_im4.s, cr1_re5.s, cr1_im5.s}[1], [ local_c_ptr1 ]
	st4 {cr1_re4.s, cr1_im4.s, cr1_re5.s, cr1_im5.s}[2], [ local_c_ptr2 ]
	st4 {cr1_re4.s, cr1_im4.s, cr1_re5.s, cr1_im5.s}[3], [ local_c_ptr3 ]

	b .L${func_name}_loop_skip


.L${func_name}_beta_non_zero:
	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	madd local_c_ptr0, ldc, idx_n, idx_m0
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	add local_c_ptr2, local_c_ptr1, ldc, lsl 3
	add local_c_ptr3, local_c_ptr2, ldc, lsl 3

	fmul cr1_re0.4s, cr0_re0.4s, alpha_real
	fmul cr1_im0.4s, cr0_re0.4s, alpha_imag
	fmls cr1_re0.4s, cr0_im0.4s, alpha_imag
	fmla cr1_im0.4s, cr0_im0.4s, alpha_real

	fmul cr1_re1.4s, cr0_re1.4s, alpha_real
	fmul cr1_im1.4s, cr0_re1.4s, alpha_imag
	fmls cr1_re1.4s, cr0_im1.4s, alpha_imag
	fmla cr1_im1.4s, cr0_im1.4s, alpha_real

	fmul cr1_re2.4s, cr0_re2.4s, alpha_real
	fmul cr1_im2.4s, cr0_re2.4s, alpha_imag
	fmls cr1_re2.4s, cr0_im2.4s, alpha_imag
	fmla cr1_im2.4s, cr0_im2.4s, alpha_real

	fmul cr1_re3.4s, cr0_re3.4s, alpha_real
	fmul cr1_im3.4s, cr0_re3.4s, alpha_imag
	fmls cr1_re3.4s, cr0_im3.4s, alpha_imag
	fmla cr1_im3.4s, cr0_im3.4s, alpha_real

	ld4 {cr_re0.s, cr_im0.s, cr_re1.s, cr_im1.s}[0], [ local_c_ptr0 ]
	ld4 {cr_re0.s, cr_im0.s, cr_re1.s, cr_im1.s}[1], [ local_c_ptr1 ]
	ld4 {cr_re0.s, cr_im0.s, cr_re1.s, cr_im1.s}[2], [ local_c_ptr2 ]
	ld4 {cr_re0.s, cr_im0.s, cr_re1.s, cr_im1.s}[3], [ local_c_ptr3 ]

	add local_c_ptr0, local_c_ptr0, (4*4)
	add local_c_ptr1, local_c_ptr1, (4*4)
	add local_c_ptr2, local_c_ptr2, (4*4)
	add local_c_ptr3, local_c_ptr3, (4*4)

	ld4 {cr_re2.s, cr_im2.s, cr_re3.s, cr_im3.s}[0], [ local_c_ptr0 ]
	ld4 {cr_re2.s, cr_im2.s, cr_re3.s, cr_im3.s}[1], [ local_c_ptr1 ]
	ld4 {cr_re2.s, cr_im2.s, cr_re3.s, cr_im3.s}[2], [ local_c_ptr2 ]
	ld4 {cr_re2.s, cr_im2.s, cr_re3.s, cr_im3.s}[3], [ local_c_ptr3 ]

	cmp k_block_idx, #0
	bne 10f

	// crr_re = cr1_re + cr_re * beta.real() - cr_im * beta.imag();
	// crr_im = cr1_im + cr_re * beta.imag() + cr_im * beta.real();
	fmla cr1_re0.4s, cr_re0.4s, beta_real
	fmla cr1_im0.4s, cr_re0.4s, beta_imag
	fmls cr1_re0.4s, cr_im0.4s, beta_imag
	fmla cr1_im0.4s, cr_im0.4s, beta_real

	fmla cr1_re1.4s, cr_re1.4s, beta_real
	fmla cr1_im1.4s, cr_re1.4s, beta_imag
	fmls cr1_re1.4s, cr_im1.4s, beta_imag
	fmla cr1_im1.4s, cr_im1.4s, beta_real

	fmla cr1_re2.4s, cr_re2.4s, beta_real
	fmla cr1_im2.4s, cr_re2.4s, beta_imag
	fmls cr1_re2.4s, cr_im2.4s, beta_imag
	fmla cr1_im2.4s, cr_im2.4s, beta_real

	fmla cr1_re3.4s, cr_re3.4s, beta_real
	fmla cr1_im3.4s, cr_re3.4s, beta_imag
	fmls cr1_re3.4s, cr_im3.4s, beta_imag
	fmla cr1_im3.4s, cr_im3.4s, beta_real
	b 20f

10:

	// crr_re = cr1_re + cr_re - cr_im;
	// crr_im = cr1_im + cr_re + cr_im;
	fadd cr1_re0.4s, cr1_re0.4s, cr_re0.4s
	fadd cr1_im0.4s, cr1_im0.4s, cr_im0.4s

	fadd cr1_re1.4s, cr1_re1.4s, cr_re1.4s
	fadd cr1_im1.4s, cr1_im1.4s, cr_im1.4s

	fadd cr1_re2.4s, cr1_re2.4s, cr_re2.4s
	fadd cr1_im2.4s, cr1_im2.4s, cr_im2.4s

	fadd cr1_re3.4s, cr1_re3.4s, cr_re3.4s
	fadd cr1_im3.4s, cr1_im3.4s, cr_im3.4s
20:

	st4 {cr1_re2.s, cr1_im2.s, cr1_re3.s, cr1_im3.s}[0], [ local_c_ptr0 ]
	st4 {cr1_re2.s, cr1_im2.s, cr1_re3.s, cr1_im3.s}[1], [ local_c_ptr1 ]
	st4 {cr1_re2.s, cr1_im2.s, cr1_re3.s, cr1_im3.s}[2], [ local_c_ptr2 ]
	st4 {cr1_re2.s, cr1_im2.s, cr1_re3.s, cr1_im3.s}[3], [ local_c_ptr3 ]

	sub local_c_ptr0, local_c_ptr0, (4*4)
	sub local_c_ptr1, local_c_ptr1, (4*4)
	sub local_c_ptr2, local_c_ptr2, (4*4)
	sub local_c_ptr3, local_c_ptr3, (4*4)

	st4 {cr1_re0.s, cr1_im0.s, cr1_re1.s, cr1_im1.s}[0], [ local_c_ptr0 ]
	st4 {cr1_re0.s, cr1_im0.s, cr1_re1.s, cr1_im1.s}[1], [ local_c_ptr1 ]
	st4 {cr1_re0.s, cr1_im0.s, cr1_re1.s, cr1_im1.s}[2], [ local_c_ptr2 ]
	st4 {cr1_re0.s, cr1_im0.s, cr1_re1.s, cr1_im1.s}[3], [ local_c_ptr3 ]

	cmp idx_m0, idx_m1
	beq .L${func_name}_loop_skip

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	madd local_c_ptr0, ldc, idx_n, idx_m1
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	add local_c_ptr2, local_c_ptr1, ldc, lsl 3
	add local_c_ptr3, local_c_ptr2, ldc, lsl 3

	fmul cr1_re4.4s, cr0_re4.4s, alpha_real
	fmul cr1_im4.4s, cr0_re4.4s, alpha_imag
	fmls cr1_re4.4s, cr0_im4.4s, alpha_imag
	fmla cr1_im4.4s, cr0_im4.4s, alpha_real

	fmul cr1_re5.4s, cr0_re5.4s, alpha_real
	fmul cr1_im5.4s, cr0_re5.4s, alpha_imag
	fmls cr1_re5.4s, cr0_im5.4s, alpha_imag
	fmla cr1_im5.4s, cr0_im5.4s, alpha_real

	fmul cr1_re6.4s, cr0_re6.4s, alpha_real
	fmul cr1_im6.4s, cr0_re6.4s, alpha_imag
	fmls cr1_re6.4s, cr0_im6.4s, alpha_imag
	fmla cr1_im6.4s, cr0_im6.4s, alpha_real

	fmul cr1_re7.4s, cr0_re7.4s, alpha_real
	fmul cr1_im7.4s, cr0_re7.4s, alpha_imag
	fmls cr1_re7.4s, cr0_im7.4s, alpha_imag
	fmla cr1_im7.4s, cr0_im7.4s, alpha_real

	ld4 {cr_re4.s, cr_im4.s, cr_re5.s, cr_im5.s}[0], [ local_c_ptr0 ]
	ld4 {cr_re4.s, cr_im4.s, cr_re5.s, cr_im5.s}[1], [ local_c_ptr1 ]
	ld4 {cr_re4.s, cr_im4.s, cr_re5.s, cr_im5.s}[2], [ local_c_ptr2 ]
	ld4 {cr_re4.s, cr_im4.s, cr_re5.s, cr_im5.s}[3], [ local_c_ptr3 ]

	add local_c_ptr0, local_c_ptr0, (4*4)
	add local_c_ptr1, local_c_ptr1, (4*4)
	add local_c_ptr2, local_c_ptr2, (4*4)
	add local_c_ptr3, local_c_ptr3, (4*4)

	ld4 {cr_re6.s, cr_im6.s, cr_re7.s, cr_im7.s}[0], [ local_c_ptr0 ]
	ld4 {cr_re6.s, cr_im6.s, cr_re7.s, cr_im7.s}[1], [ local_c_ptr1 ]
	ld4 {cr_re6.s, cr_im6.s, cr_re7.s, cr_im7.s}[2], [ local_c_ptr2 ]
	ld4 {cr_re6.s, cr_im6.s, cr_re7.s, cr_im7.s}[3], [ local_c_ptr3 ]

	cmp k_block_idx, #0
	bne 10f

	// crr_re = cr1_re + cr_re * beta.real() - cr_im * beta.imag();
	// crr_im = cr1_im + cr_re * beta.imag() + cr_im * beta.real();
	fmla cr1_re4.4s, cr_re4.4s, beta_real
	fmla cr1_im4.4s, cr_re4.4s, beta_imag
	fmls cr1_re4.4s, cr_im4.4s, beta_imag
	fmla cr1_im4.4s, cr_im4.4s, beta_real

	fmla cr1_re5.4s, cr_re5.4s, beta_real
	fmla cr1_im5.4s, cr_re5.4s, beta_imag
	fmls cr1_re5.4s, cr_im5.4s, beta_imag
	fmla cr1_im5.4s, cr_im5.4s, beta_real

	fmla cr1_re6.4s, cr_re6.4s, beta_real
	fmla cr1_im6.4s, cr_re6.4s, beta_imag
	fmls cr1_re6.4s, cr_im6.4s, beta_imag
	fmla cr1_im6.4s, cr_im6.4s, beta_real

	fmla cr1_re7.4s, cr_re7.4s, beta_real
	fmla cr1_im7.4s, cr_re7.4s, beta_imag
	fmls cr1_re7.4s, cr_im7.4s, beta_imag
	fmla cr1_im7.4s, cr_im7.4s, beta_real
	b 20f

10:

	// crr_re = cr1_re + cr_re - cr_im;
	// crr_im = cr1_im + cr_re + cr_im;
	fadd cr1_re4.4s, cr1_re4.4s, cr_re4.4s
	fadd cr1_im4.4s, cr1_im4.4s, cr_im4.4s

	fadd cr1_re5.4s, cr1_re5.4s, cr_re5.4s
	fadd cr1_im5.4s, cr1_im5.4s, cr_im5.4s

	fadd cr1_re6.4s, cr1_re6.4s, cr_re6.4s
	fadd cr1_im6.4s, cr1_im6.4s, cr_im6.4s

	fadd cr1_re7.4s, cr1_re7.4s, cr_re7.4s
	fadd cr1_im7.4s, cr1_im7.4s, cr_im7.4s

20:

	st4 {cr1_re6.s, cr1_im6.s, cr1_re7.s, cr1_im7.s}[0], [ local_c_ptr0 ]
	st4 {cr1_re6.s, cr1_im6.s, cr1_re7.s, cr1_im7.s}[1], [ local_c_ptr1 ]
	st4 {cr1_re6.s, cr1_im6.s, cr1_re7.s, cr1_im7.s}[2], [ local_c_ptr2 ]
	st4 {cr1_re6.s, cr1_im6.s, cr1_re7.s, cr1_im7.s}[3], [ local_c_ptr3 ]

	sub local_c_ptr0, local_c_ptr0, (4*4)
	sub local_c_ptr1, local_c_ptr1, (4*4)
	sub local_c_ptr2, local_c_ptr2, (4*4)
	sub local_c_ptr3, local_c_ptr3, (4*4)

	st4 {cr1_re4.s, cr1_im4.s, cr1_re5.s, cr1_im5.s}[0], [ local_c_ptr0 ]
	st4 {cr1_re4.s, cr1_im4.s, cr1_re5.s, cr1_im5.s}[1], [ local_c_ptr1 ]
	st4 {cr1_re4.s, cr1_im4.s, cr1_re5.s, cr1_im5.s}[2], [ local_c_ptr2 ]
	st4 {cr1_re4.s, cr1_im4.s, cr1_re5.s, cr1_im5.s}[3], [ local_c_ptr3 ]

.L${func_name}_loop_skip:
	add idx_n_base, idx_n_base, #4
	b .L${func_name}_loop_n
.L${func_name}_loop_n_end: // }}}
	add idx_m0, idx_m0, #8
	b .L${func_name}_loop_m
.L${func_name}_loop_m_end: // }}}
	add n_block_idx, n_block_idx, n_block_size
	mov n_block_size_max, N_BLOCK_SIZE_DEFAULT
	b .L${func_name}_loop_block_n
.L${func_name}_loop_block_n_end: // }}}
	add k_block_idx, k_block_idx, k_block_size
	mov k_block_size_max, K_BLOCK_SIZE_DEFAULT
	b .L${func_name}_loop_block_k
.L${func_name}_loop_block_k_end: // }}}

.L${func_name}_cleanup:
	ldp x19, x20, [ sp, #144 ]
	ldp x21, x22, [ sp, #160 ]
	ldp x23, x24, [ sp, #176 ]
	ldp x25, x26, [ sp, #192 ]
	ldp x27, x28, [ sp, #208 ]
	ldp x29, x30, [ sp, #224 ]
	ldp d8, d9, [ sp, #304 ]
	ldp d10, d11, [ sp, #320 ]
	ldp d12, d13, [ sp, #336 ]
	ldp d14, d15, [ sp, #352 ]
	ldp x29, x30, [sp], #384
	ret
	${epilogue(func_name)}
</%def> // }}}

${generate_XX_kernel("cgemm_small_kernel_TT_mnk", "fmla", "fmla", "fmls")}
${generate_XX_kernel("cgemm_small_kernel_CT_mnk", "fmls", "fmla", "fmla")}
${generate_XX_kernel("cgemm_small_kernel_TC_mnk", "fmla", "fmls", "fmla")}
${generate_XX_kernel("cgemm_small_kernel_CC_mnk", "fmls", "fmls", "fmls")}

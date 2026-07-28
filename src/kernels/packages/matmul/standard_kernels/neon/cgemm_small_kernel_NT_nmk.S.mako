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

// locals
#define k_block_size x8
#define k_block_idx x9 // [0,m_block_size,...,m)
#define n_block_size x10
#define n_block_idx x11 // [0,m_block_size,...,m)
#define idx_n_base x12 // [0,1,...,m_block_size)
#define idx_n01 x13 // [0,1,...,m) = m_block_idx + idx_m_base
#define idx_n23 x14
#define idx_n45 x15
#define idx_m0 x16
#define idx_m1 x17
// x18 may be in use for the Platform Register
#define idx_k_base x19 // [0,1,...,m_block_size)
#define idx_k x20 // [0,1,...,m) = m_block_idx + idx_m_base
#define k_block_size_max x21
#define n_block_size_max x22
#define beta_is_zero x23

#define alpha_real v31.s[0]
#define alpha_imag v31.s[1]
#define beta_real  v31.s[2]
#define beta_imag  v31.s[3]

#define N_BLOCK_SIZE_DEFAULT 60
#define K_BLOCK_SIZE_DEFAULT 60
#define N_BLOCK_SIZE_FIRST N_BLOCK_SIZE_DEFAULT
#define K_BLOCK_SIZE_FIRST K_BLOCK_SIZE_DEFAULT
#define A_PTR_PREFETCH_DISTANCE 64
#define B_PTR_PREFETCH_DISTANCE 64

#if ((N_BLOCK_SIZE_DEFAULT % 6) != 0)
#error "N block size must be divisible by 6"
#endif


<%def name="loop_k_content(fmlsubadd, fmladdsub)">
	// cr0_re = ar_re * br_re - ar_im * br_im + cr0_re;
	// cr0_im = ar_re * br_im + ar_im * br_re + cr0_im;
	fmla       cr0_re0.4s, ar_re0.4s, br01.s[0]
	${fmladdsub} cr0_im0.4s, ar_re0.4s, br01.s[1]
	fmla       cr0_re1.4s, ar_re0.4s, br01.s[2]
	${fmladdsub} cr0_im1.4s, ar_re0.4s, br01.s[3]

	ld1 {br45.4s}, [ local_b_ptr45 ]
	add local_b_ptr45, local_b_ptr45, ldb, lsl 3

	${fmlsubadd} cr0_re0.4s, ar_im0.4s, br01.s[1]
	fmla       cr0_im0.4s, ar_im0.4s, br01.s[0]
	${fmlsubadd} cr0_re1.4s, ar_im0.4s, br01.s[3]
	fmla       cr0_im1.4s, ar_im0.4s, br01.s[2]

	ld2 {ar_re1.4s, ar_im1.4s}, [ local_a_ptr1 ]
	add local_a_ptr1, local_a_ptr1, lda, lsl 3

	fmla       cr0_re2.4s, ar_re0.4s, br23.s[0]
	${fmladdsub} cr0_im2.4s, ar_re0.4s, br23.s[1]
	fmla       cr0_re3.4s, ar_re0.4s, br23.s[2]
	${fmladdsub} cr0_im3.4s, ar_re0.4s, br23.s[3]

	prfm pldl1keep, [local_b_ptr45, B_PTR_PREFETCH_DISTANCE]

	${fmlsubadd} cr0_re2.4s, ar_im0.4s, br23.s[1]
	fmla       cr0_im2.4s, ar_im0.4s, br23.s[0]
	${fmlsubadd} cr0_re3.4s, ar_im0.4s, br23.s[3]
	fmla       cr0_im3.4s, ar_im0.4s, br23.s[2]

	prfm pldl1keep, [local_b_ptr01, B_PTR_PREFETCH_DISTANCE]

	fmla       cr0_re4.4s, ar_re0.4s, br45.s[0]
	${fmladdsub} cr0_im4.4s, ar_re0.4s, br45.s[1]
	fmla       cr0_re5.4s, ar_re0.4s, br45.s[2]
	${fmladdsub} cr0_im5.4s, ar_re0.4s, br45.s[3]

	subs idx_k, idx_k, #1
	prfm pldl1keep, [local_a_ptr0, A_PTR_PREFETCH_DISTANCE]

	${fmlsubadd} cr0_re4.4s, ar_im0.4s, br45.s[1]
	fmla       cr0_im4.4s, ar_im0.4s, br45.s[0]
	${fmlsubadd} cr0_re5.4s, ar_im0.4s, br45.s[3]
	fmla       cr0_im5.4s, ar_im0.4s, br45.s[2]

	ld2 {ar_re0.4s, ar_im0.4s}, [ local_a_ptr0 ]
	add local_a_ptr0, local_a_ptr0, lda, lsl 3

	fmla       cr0_re6.4s, ar_re1.4s, br01.s[0]
	${fmladdsub} cr0_im6.4s, ar_re1.4s, br01.s[1]
	fmla       cr0_re7.4s, ar_re1.4s, br01.s[2]
	${fmladdsub} cr0_im7.4s, ar_re1.4s, br01.s[3]

	prfm pldl1keep, [local_a_ptr1, 0]

	${fmlsubadd} cr0_re6.4s, ar_im1.4s, br01.s[1]
	fmla       cr0_im6.4s, ar_im1.4s, br01.s[0]
	${fmlsubadd} cr0_re7.4s, ar_im1.4s, br01.s[3]
	fmla       cr0_im7.4s, ar_im1.4s, br01.s[2]

	ld1 {br01.4s}, [ local_b_ptr01 ]
	add local_b_ptr01, local_b_ptr01, ldb, lsl 3

	fmla       cr0_re8.4s, ar_re1.4s, br23.s[0]
	${fmladdsub} cr0_im8.4s, ar_re1.4s, br23.s[1]
	fmla       cr0_re9.4s, ar_re1.4s, br23.s[2]
	${fmladdsub} cr0_im9.4s, ar_re1.4s, br23.s[3]

	prfm pldl1keep, [local_a_ptr0, 0]

	${fmlsubadd} cr0_re8.4s, ar_im1.4s, br23.s[1]
	fmla       cr0_im8.4s, ar_im1.4s, br23.s[0]
	${fmlsubadd} cr0_re9.4s, ar_im1.4s, br23.s[3]
	fmla       cr0_im9.4s, ar_im1.4s, br23.s[2]

	ld1 {br23.4s}, [ local_b_ptr23 ]
	add local_b_ptr23, local_b_ptr23, ldb, lsl 3

	fmla       cr0_re10.4s, ar_re1.4s, br45.s[0]
	${fmladdsub} cr0_im10.4s, ar_re1.4s, br45.s[1]
	fmla       cr0_re11.4s, ar_re1.4s, br45.s[2]
	${fmladdsub} cr0_im11.4s, ar_re1.4s, br45.s[3]

	prfm pldl1keep, [local_b_ptr01, B_PTR_PREFETCH_DISTANCE]

	${fmlsubadd} cr0_re10.4s, ar_im1.4s, br45.s[1]
	fmla       cr0_im10.4s, ar_im1.4s, br45.s[0]
	${fmlsubadd} cr0_re11.4s, ar_im1.4s, br45.s[3]
	fmla       cr0_im11.4s, ar_im1.4s, br45.s[2]

	prfm pldl1keep, [local_b_ptr23, 0]
</%def>

// we use a macro to generate the code for both transpose and transpose
// conjugate, as we simply invert whether we apply subtraction in some cases
// and addition in others
<%def name="generate_NX_kernel(func_name, fmlsubadd, fmladdsub)">
	${prologue(func_name)}
	stp x29, x30, [ sp, #-384 ]!
	mov x29, sp
	stp x19, x20, [ sp, #144 ]
	stp x21, x22, [ sp, #160 ]
	stp x23, x24, [ sp, #176 ]
	stp x25, x26, [ sp, #192 ]
	stp x27, x28, [ sp, #208 ]
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
	mov alpha_real, v0.s[0]
	mov alpha_imag, v1.s[0]
	mov beta_real,  v2.s[0]
	mov beta_imag,  v3.s[0]

	fcmp s2, 0.0
	bne .L${func_name}_set_beta_not_zero
	fcmp s3, 0.0
	bne .L${func_name}_set_beta_not_zero
	mov beta_is_zero, 1
	b .L${func_name}_beta_done
.L${func_name}_set_beta_not_zero:
	mov beta_is_zero, 0
.L${func_name}_beta_done:

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
	// n_block_size = min(n_block_size, n - n_block_idx)
	sub n_block_size, n, n_block_idx
	cmp n_block_size_max, n_block_size
	mov n_block_size_max, N_BLOCK_SIZE_DEFAULT
	csel n_block_size, n_block_size_max, n_block_size, le

	eor idx_m0, idx_m0, idx_m0
.L${func_name}_loop_m: // {{{
	cmp idx_m0, m
	beq .L${func_name}_loop_m_end

	add idx_m1, idx_m0, #4
	cmp idx_m1, m
	csel idx_m1, idx_m0, idx_m1, ge

	eor idx_n_base, idx_n_base, idx_n_base
.L${func_name}_loop_n: // {{{
	cmp idx_n_base, n_block_size
	bge .L${func_name}_loop_n_end
	add idx_n01, n_block_idx, idx_n_base

	add idx_n23, idx_n01, #2
	add idx_n45, idx_n01, #4

	cmp idx_n23, n
	csel idx_n23, idx_n01, idx_n23, ge
	cmp idx_n45, n
	csel idx_n45, idx_n23, idx_n45, ge

.L${func_name}_loop_first: // {{{

#define ar_im1 v30
#define ar_re1 v29
#define ar_im0 v28
#define ar_re0 v27
#define br45 v26
#define br23 v25
#define br01 v24

#define cr0_re11 v23
#define cr0_im11 v22
#define cr0_re10 v21
#define cr0_im10 v20
#define cr0_re9 v19
#define cr0_im9 v18
#define cr0_re8 v17
#define cr0_im8 v16
#define cr0_re7 v15
#define cr0_im7 v14
#define cr0_re6 v13
#define cr0_im6 v12
#define cr0_re5 v11
#define cr0_im5 v10
#define cr0_re4 v9
#define cr0_im4 v8
#define cr0_re3 v7
#define cr0_im3 v6
#define cr0_re2 v5
#define cr0_im2 v4
#define cr0_re1 v3
#define cr0_im1 v2
#define cr0_re0 v1
#define cr0_im0 v0

#define local_b_ptr01 x28
#define local_b_ptr23 x27
#define local_b_ptr45 x26
#define local_a_ptr0 x25
#define local_a_ptr1 x24

	cmp k_block_size, #0

	eor cr0_re0.16b, cr0_re0.16b, cr0_re0.16b
	eor cr0_im0.16b, cr0_im0.16b, cr0_im0.16b
	eor cr0_re1.16b, cr0_re1.16b, cr0_re1.16b
	madd local_a_ptr0, lda, k_block_idx, idx_m0
	eor cr0_im1.16b, cr0_im1.16b, cr0_im1.16b
	eor cr0_re2.16b, cr0_re2.16b, cr0_re2.16b
	eor cr0_im2.16b, cr0_im2.16b, cr0_im2.16b
	madd local_a_ptr1, lda, k_block_idx, idx_m1
	eor cr0_re3.16b, cr0_re3.16b, cr0_re3.16b
	eor cr0_im3.16b, cr0_im3.16b, cr0_im3.16b
	eor cr0_re4.16b, cr0_re4.16b, cr0_re4.16b
	madd local_b_ptr01, ldb, k_block_idx, idx_n01
	eor cr0_im4.16b, cr0_im4.16b, cr0_im4.16b
	eor cr0_re5.16b, cr0_re5.16b, cr0_re5.16b
	eor cr0_im5.16b, cr0_im5.16b, cr0_im5.16b
	madd local_b_ptr23, ldb, k_block_idx, idx_n23
	eor cr0_re6.16b, cr0_re6.16b, cr0_re6.16b
	eor cr0_im6.16b, cr0_im6.16b, cr0_im6.16b
	eor cr0_re7.16b, cr0_re7.16b, cr0_re7.16b
	madd local_b_ptr45, ldb, k_block_idx, idx_n45
	eor cr0_im7.16b, cr0_im7.16b, cr0_im7.16b
	eor cr0_re8.16b, cr0_re8.16b, cr0_re8.16b
	eor cr0_im8.16b, cr0_im8.16b, cr0_im8.16b
	eor cr0_re9.16b, cr0_re9.16b, cr0_re9.16b
	eor cr0_im9.16b, cr0_im9.16b, cr0_im9.16b
	eor cr0_re10.16b, cr0_re10.16b, cr0_re10.16b
	eor cr0_im10.16b, cr0_im10.16b, cr0_im10.16b
	eor cr0_re11.16b, cr0_re11.16b, cr0_re11.16b
	eor cr0_im11.16b, cr0_im11.16b, cr0_im11.16b

	beq .L${func_name}_loop_k_end

	subs idx_k, k_block_size, #1

	add local_a_ptr0, a_ptr, local_a_ptr0, lsl 3

	add local_a_ptr1, a_ptr, local_a_ptr1, lsl 3

	add local_b_ptr01, b_ptr, local_b_ptr01, lsl 3

	add local_b_ptr23, b_ptr, local_b_ptr23, lsl 3

	add local_b_ptr45, b_ptr, local_b_ptr45, lsl 3

	ld2 {ar_re0.4s, ar_im0.4s}, [ local_a_ptr0 ]
	add local_a_ptr0, local_a_ptr0, lda, lsl 3

	ld1 {br01.4s}, [ local_b_ptr01 ]
	add local_b_ptr01, local_b_ptr01, ldb, lsl 3

	ld1 {br23.4s}, [ local_b_ptr23 ]
	add local_b_ptr23, local_b_ptr23, ldb, lsl 3

	beq .L${func_name}_loop_k_last

.L${func_name}_loop_k: // {{{
	${loop_k_content(fmlsubadd, fmladdsub)}
	beq .L${func_name}_loop_k_last
	${loop_k_content(fmlsubadd, fmladdsub)}
	bne .L${func_name}_loop_k

.L${func_name}_loop_k_last: // }}}

	// cr0_re = ar_re * br_re - ar_im * br_im + cr0_re;
	// cr0_im = ar_re * br_im + ar_im * br_re + cr0_im;
	fmla       cr0_re0.4s, ar_re0.4s, br01.s[0]
	${fmladdsub} cr0_im0.4s, ar_re0.4s, br01.s[1]
	fmla       cr0_re1.4s, ar_re0.4s, br01.s[2]
	${fmladdsub} cr0_im1.4s, ar_re0.4s, br01.s[3]

	ld1 {br45.4s}, [ local_b_ptr45 ]
	add local_b_ptr45, local_b_ptr45, ldb, lsl 3

	${fmlsubadd} cr0_re0.4s, ar_im0.4s, br01.s[1]
	fmla       cr0_im0.4s, ar_im0.4s, br01.s[0]
	${fmlsubadd} cr0_re1.4s, ar_im0.4s, br01.s[3]
	fmla       cr0_im1.4s, ar_im0.4s, br01.s[2]

	ld2 {ar_re1.4s, ar_im1.4s}, [ local_a_ptr1 ]
	add local_a_ptr1, local_a_ptr1, lda, lsl 3

	fmla       cr0_re2.4s, ar_re0.4s, br23.s[0]
	${fmladdsub} cr0_im2.4s, ar_re0.4s, br23.s[1]
	fmla       cr0_re3.4s, ar_re0.4s, br23.s[2]
	${fmladdsub} cr0_im3.4s, ar_re0.4s, br23.s[3]

	${fmlsubadd} cr0_re2.4s, ar_im0.4s, br23.s[1]
	fmla       cr0_im2.4s, ar_im0.4s, br23.s[0]
	${fmlsubadd} cr0_re3.4s, ar_im0.4s, br23.s[3]
	fmla       cr0_im3.4s, ar_im0.4s, br23.s[2]

	fmla       cr0_re4.4s, ar_re0.4s, br45.s[0]
	${fmladdsub} cr0_im4.4s, ar_re0.4s, br45.s[1]
	fmla       cr0_re5.4s, ar_re0.4s, br45.s[2]
	${fmladdsub} cr0_im5.4s, ar_re0.4s, br45.s[3]

	${fmlsubadd} cr0_re4.4s, ar_im0.4s, br45.s[1]
	fmla       cr0_im4.4s, ar_im0.4s, br45.s[0]
	${fmlsubadd} cr0_re5.4s, ar_im0.4s, br45.s[3]
	fmla       cr0_im5.4s, ar_im0.4s, br45.s[2]

	fmla       cr0_re6.4s, ar_re1.4s, br01.s[0]
	${fmladdsub} cr0_im6.4s, ar_re1.4s, br01.s[1]
	fmla       cr0_re7.4s, ar_re1.4s, br01.s[2]
	${fmladdsub} cr0_im7.4s, ar_re1.4s, br01.s[3]

	${fmlsubadd} cr0_re6.4s, ar_im1.4s, br01.s[1]
	fmla       cr0_im6.4s, ar_im1.4s, br01.s[0]
	${fmlsubadd} cr0_re7.4s, ar_im1.4s, br01.s[3]
	fmla       cr0_im7.4s, ar_im1.4s, br01.s[2]

	fmla       cr0_re8.4s, ar_re1.4s, br23.s[0]
	${fmladdsub} cr0_im8.4s, ar_re1.4s, br23.s[1]
	fmla       cr0_re9.4s, ar_re1.4s, br23.s[2]
	${fmladdsub} cr0_im9.4s, ar_re1.4s, br23.s[3]

	${fmlsubadd} cr0_re8.4s, ar_im1.4s, br23.s[1]
	fmla       cr0_im8.4s, ar_im1.4s, br23.s[0]
	${fmlsubadd} cr0_re9.4s, ar_im1.4s, br23.s[3]
	fmla       cr0_im9.4s, ar_im1.4s, br23.s[2]

	fmla       cr0_re10.4s, ar_re1.4s, br45.s[0]
	${fmladdsub} cr0_im10.4s, ar_re1.4s, br45.s[1]
	fmla       cr0_re11.4s, ar_re1.4s, br45.s[2]
	${fmladdsub} cr0_im11.4s, ar_re1.4s, br45.s[3]

	${fmlsubadd} cr0_re10.4s, ar_im1.4s, br45.s[1]
	fmla       cr0_im10.4s, ar_im1.4s, br45.s[0]
	${fmlsubadd} cr0_re11.4s, ar_im1.4s, br45.s[3]
	fmla       cr0_im11.4s, ar_im1.4s, br45.s[2]

.L${func_name}_loop_k_end: // }}}

#define local_c_ptr0 x28
#define local_c_ptr1 x27
#define ldc x26

#define cr1_im11 v30
#define cr1_re11 v29
#define cr1_im10 v28
#define cr1_re10 v27
// start again
#define cr1_im9 v30
#define cr1_re9 v29
#define cr1_im8 v28
#define cr1_re8 v27
// start again
#define cr1_im7 v30
#define cr1_re7 v29
#define cr1_im6 v28
#define cr1_re6 v27
// start again
#define cr1_im5 v30
#define cr1_re5 v29
#define cr1_im4 v28
#define cr1_re4 v27
// start again
#define cr1_im3 v30
#define cr1_re3 v29
#define cr1_im2 v28
#define cr1_re2 v27
// start again
#define cr1_im1 v30
#define cr1_re1 v29
#define cr1_im0 v28
#define cr1_re0 v27

#define cr_im11 v23
#define cr_re11 v22
#define cr_im10 v21
#define cr_re10 v20
#define cr_im9 v19
#define cr_re9 v18
#define cr_im8 v17
#define cr_re8 v16
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

	ldr	ldc, [ sp, #384 ]

	cmp beta_is_zero, 1
	bne .L${func_name}_beta_non_zero

	cmp k_block_idx, 0
	bne .L${func_name}_beta_non_zero

.L${func_name}_beta_zero:
	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re0.4s, cr0_re0.4s, alpha_real
	fmul cr1_im0.4s, cr0_re0.4s, alpha_imag
	fmls cr1_re0.4s, cr0_im0.4s, alpha_imag
	fmla cr1_im0.4s, cr0_im0.4s, alpha_real

	fmul cr1_re1.4s, cr0_re1.4s, alpha_real
	fmul cr1_im1.4s, cr0_re1.4s, alpha_imag
	fmls cr1_re1.4s, cr0_im1.4s, alpha_imag
	fmla cr1_im1.4s, cr0_im1.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n01, idx_m0
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	dup cr_re0.2d, xzr
	dup cr_im0.2d, xzr
	dup cr_re1.2d, xzr
	dup cr_im1.2d, xzr

	// crr_re = cr1_re + cr_re;
	// crr_im = cr1_im + cr_im;
	fadd cr1_re0.4s, cr1_re0.4s, cr_re0.4s
	fadd cr1_im0.4s, cr1_im0.4s, cr_im0.4s
	st2 {cr1_re0.4s, cr1_im0.4s}, [ local_c_ptr0 ]

	fadd cr1_re1.4s, cr1_re1.4s, cr_re1.4s
	fadd cr1_im1.4s, cr1_im1.4s, cr_im1.4s
	st2 {cr1_re1.4s, cr1_im1.4s}, [ local_c_ptr1 ]

	cmp idx_n01, idx_n23
	beq 1f

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re2.4s, cr0_re2.4s, alpha_real
	fmul cr1_im2.4s, cr0_re2.4s, alpha_imag
	fmls cr1_re2.4s, cr0_im2.4s, alpha_imag
	fmla cr1_im2.4s, cr0_im2.4s, alpha_real

	fmul cr1_re3.4s, cr0_re3.4s, alpha_real
	fmul cr1_im3.4s, cr0_re3.4s, alpha_imag
	fmls cr1_re3.4s, cr0_im3.4s, alpha_imag
	fmla cr1_im3.4s, cr0_im3.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n23, idx_m0
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	dup cr_re2.2d, xzr
	dup cr_im2.2d, xzr
	dup cr_re3.2d, xzr
	dup cr_im3.2d, xzr

	// crr_re = cr1_re + cr_re;
	// crr_im = cr1_im + cr_im;
	fadd cr1_re2.4s, cr1_re2.4s, cr_re2.4s
	fadd cr1_im2.4s, cr1_im2.4s, cr_im2.4s
	st2 {cr1_re2.4s, cr1_im2.4s}, [ local_c_ptr0 ]

	fadd cr1_re3.4s, cr1_re3.4s, cr_re3.4s
	fadd cr1_im3.4s, cr1_im3.4s, cr_im3.4s
	st2 {cr1_re3.4s, cr1_im3.4s}, [ local_c_ptr1 ]

	cmp idx_n23, idx_n45
	beq 1f

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re4.4s, cr0_re4.4s, alpha_real
	fmul cr1_im4.4s, cr0_re4.4s, alpha_imag
	fmls cr1_re4.4s, cr0_im4.4s, alpha_imag
	fmla cr1_im4.4s, cr0_im4.4s, alpha_real

	fmul cr1_re5.4s, cr0_re5.4s, alpha_real
	fmul cr1_im5.4s, cr0_re5.4s, alpha_imag
	fmls cr1_re5.4s, cr0_im5.4s, alpha_imag
	fmla cr1_im5.4s, cr0_im5.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n45, idx_m0
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	dup cr_re4.2d, xzr
	dup cr_im4.2d, xzr
	dup cr_re5.2d, xzr
	dup cr_im5.2d, xzr

	// crr_re = cr1_re + cr_re;
	// crr_im = cr1_im + cr_im;
	fadd cr1_re4.4s, cr1_re4.4s, cr_re4.4s
	fadd cr1_im4.4s, cr1_im4.4s, cr_im4.4s
	st2 {cr1_re4.4s, cr1_im4.4s}, [ local_c_ptr0 ]

	fadd cr1_re5.4s, cr1_re5.4s, cr_re5.4s
	fadd cr1_im5.4s, cr1_im5.4s, cr_im5.4s
	st2 {cr1_re5.4s, cr1_im5.4s}, [ local_c_ptr1 ]

1:
	cmp idx_m0, idx_m1
	beq 99f

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re6.4s, cr0_re6.4s, alpha_real
	fmul cr1_im6.4s, cr0_re6.4s, alpha_imag
	fmls cr1_re6.4s, cr0_im6.4s, alpha_imag
	fmla cr1_im6.4s, cr0_im6.4s, alpha_real

	fmul cr1_re7.4s, cr0_re7.4s, alpha_real
	fmul cr1_im7.4s, cr0_re7.4s, alpha_imag
	fmls cr1_re7.4s, cr0_im7.4s, alpha_imag
	fmla cr1_im7.4s, cr0_im7.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n01, idx_m1
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	dup cr_re6.2d, xzr
	dup cr_im6.2d, xzr
	dup cr_re7.2d, xzr
	dup cr_im7.2d, xzr

	// crr_re = cr1_re + cr_re;
	// crr_im = cr1_im + cr_im;
	fadd cr1_re6.4s, cr1_re6.4s, cr_re6.4s
	fadd cr1_im6.4s, cr1_im6.4s, cr_im6.4s
	st2 {cr1_re6.4s, cr1_im6.4s}, [ local_c_ptr0 ]

	fadd cr1_re7.4s, cr1_re7.4s, cr_re7.4s
	fadd cr1_im7.4s, cr1_im7.4s, cr_im7.4s
	st2 {cr1_re7.4s, cr1_im7.4s}, [ local_c_ptr1 ]

	cmp idx_n01, idx_n23
	beq 99f

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re8.4s, cr0_re8.4s, alpha_real
	fmul cr1_im8.4s, cr0_re8.4s, alpha_imag
	fmls cr1_re8.4s, cr0_im8.4s, alpha_imag
	fmla cr1_im8.4s, cr0_im8.4s, alpha_real

	fmul cr1_re9.4s, cr0_re9.4s, alpha_real
	fmul cr1_im9.4s, cr0_re9.4s, alpha_imag
	fmls cr1_re9.4s, cr0_im9.4s, alpha_imag
	fmla cr1_im9.4s, cr0_im9.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n23, idx_m1
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	dup cr_re8.2d, xzr
	dup cr_im8.2d, xzr
	dup cr_re9.2d, xzr
	dup cr_im9.2d, xzr

	// crr_re = cr1_re + cr_re;
	// crr_im = cr1_im + cr_im;
	fadd cr1_re8.4s, cr1_re8.4s, cr_re8.4s
	fadd cr1_im8.4s, cr1_im8.4s, cr_im8.4s
	st2 {cr1_re8.4s, cr1_im8.4s}, [ local_c_ptr0 ]

	fadd cr1_re9.4s, cr1_re9.4s, cr_re9.4s
	fadd cr1_im9.4s, cr1_im9.4s, cr_im9.4s
	st2 {cr1_re9.4s, cr1_im9.4s}, [ local_c_ptr1 ]

	cmp idx_n23, idx_n45
	beq 99f

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re10.4s, cr0_re10.4s, alpha_real
	fmul cr1_im10.4s, cr0_re10.4s, alpha_imag
	fmls cr1_re10.4s, cr0_im10.4s, alpha_imag
	fmla cr1_im10.4s, cr0_im10.4s, alpha_real

	fmul cr1_re11.4s, cr0_re11.4s, alpha_real
	fmul cr1_im11.4s, cr0_re11.4s, alpha_imag
	fmls cr1_re11.4s, cr0_im11.4s, alpha_imag
	fmla cr1_im11.4s, cr0_im11.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n45, idx_m1
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	dup cr_re10.2d, xzr
	dup cr_im10.2d, xzr
	dup cr_re11.2d, xzr
	dup cr_im11.2d, xzr

	// crr_re = cr1_re + cr_re;
	// crr_im = cr1_im + cr_im;
	fadd cr1_re10.4s, cr1_re10.4s, cr_re10.4s
	fadd cr1_im10.4s, cr1_im10.4s, cr_im10.4s
	st2 {cr1_re10.4s, cr1_im10.4s}, [ local_c_ptr0 ]

	fadd cr1_re11.4s, cr1_re11.4s, cr_re11.4s
	fadd cr1_im11.4s, cr1_im11.4s, cr_im11.4s
	st2 {cr1_re11.4s, cr1_im11.4s}, [ local_c_ptr1 ]

	b 99f


.L${func_name}_beta_non_zero:
	cmp k_block_idx, #0
	bne .L${func_name}_loop_c_no_beta

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re0.4s, cr0_re0.4s, alpha_real
	fmul cr1_im0.4s, cr0_re0.4s, alpha_imag
	fmls cr1_re0.4s, cr0_im0.4s, alpha_imag
	fmla cr1_im0.4s, cr0_im0.4s, alpha_real

	fmul cr1_re1.4s, cr0_re1.4s, alpha_real
	fmul cr1_im1.4s, cr0_re1.4s, alpha_imag
	fmls cr1_re1.4s, cr0_im1.4s, alpha_imag
	fmla cr1_im1.4s, cr0_im1.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n01, idx_m0
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	ld2 {cr_re0.4s, cr_im0.4s}, [ local_c_ptr0 ]
	ld2 {cr_re1.4s, cr_im1.4s}, [ local_c_ptr1 ]

	// crr_re = cr1_re + cr_re * beta.real() - cr_im * beta.imag();
	// crr_im = cr1_im + cr_re * beta.imag() + cr_im * beta.real();
	fmla cr1_re0.4s, cr_re0.4s, beta_real
	fmla cr1_im0.4s, cr_re0.4s, beta_imag
	fmls cr1_re0.4s, cr_im0.4s, beta_imag
	fmla cr1_im0.4s, cr_im0.4s, beta_real
	st2 {cr1_re0.4s, cr1_im0.4s}, [ local_c_ptr0 ]

	fmla cr1_re1.4s, cr_re1.4s, beta_real
	fmla cr1_im1.4s, cr_re1.4s, beta_imag
	fmls cr1_re1.4s, cr_im1.4s, beta_imag
	fmla cr1_im1.4s, cr_im1.4s, beta_real
	st2 {cr1_re1.4s, cr1_im1.4s}, [ local_c_ptr1 ]

	cmp idx_n01, idx_n23
	beq 1f

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re2.4s, cr0_re2.4s, alpha_real
	fmul cr1_im2.4s, cr0_re2.4s, alpha_imag
	fmls cr1_re2.4s, cr0_im2.4s, alpha_imag
	fmla cr1_im2.4s, cr0_im2.4s, alpha_real

	fmul cr1_re3.4s, cr0_re3.4s, alpha_real
	fmul cr1_im3.4s, cr0_re3.4s, alpha_imag
	fmls cr1_re3.4s, cr0_im3.4s, alpha_imag
	fmla cr1_im3.4s, cr0_im3.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n23, idx_m0
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	ld2 {cr_re2.4s, cr_im2.4s}, [ local_c_ptr0 ]
	ld2 {cr_re3.4s, cr_im3.4s}, [ local_c_ptr1 ]

	// crr_re = cr1_re + cr_re * beta.real() - cr_im * beta.imag();
	// crr_im = cr1_im + cr_re * beta.imag() + cr_im * beta.real();
	fmla cr1_re2.4s, cr_re2.4s, beta_real
	fmla cr1_im2.4s, cr_re2.4s, beta_imag
	fmls cr1_re2.4s, cr_im2.4s, beta_imag
	fmla cr1_im2.4s, cr_im2.4s, beta_real
	st2 {cr1_re2.4s, cr1_im2.4s}, [ local_c_ptr0 ]

	fmla cr1_re3.4s, cr_re3.4s, beta_real
	fmla cr1_im3.4s, cr_re3.4s, beta_imag
	fmls cr1_re3.4s, cr_im3.4s, beta_imag
	fmla cr1_im3.4s, cr_im3.4s, beta_real
	st2 {cr1_re3.4s, cr1_im3.4s}, [ local_c_ptr1 ]

	cmp idx_n23, idx_n45
	beq 1f

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re4.4s, cr0_re4.4s, alpha_real
	fmul cr1_im4.4s, cr0_re4.4s, alpha_imag
	fmls cr1_re4.4s, cr0_im4.4s, alpha_imag
	fmla cr1_im4.4s, cr0_im4.4s, alpha_real

	fmul cr1_re5.4s, cr0_re5.4s, alpha_real
	fmul cr1_im5.4s, cr0_re5.4s, alpha_imag
	fmls cr1_re5.4s, cr0_im5.4s, alpha_imag
	fmla cr1_im5.4s, cr0_im5.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n45, idx_m0
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	ld2 {cr_re4.4s, cr_im4.4s}, [ local_c_ptr0 ]
	ld2 {cr_re5.4s, cr_im5.4s}, [ local_c_ptr1 ]

	// crr_re = cr1_re + cr_re * beta.real() - cr_im * beta.imag();
	// crr_im = cr1_im + cr_re * beta.imag() + cr_im * beta.real();
	fmla cr1_re4.4s, cr_re4.4s, beta_real
	fmla cr1_im4.4s, cr_re4.4s, beta_imag
	fmls cr1_re4.4s, cr_im4.4s, beta_imag
	fmla cr1_im4.4s, cr_im4.4s, beta_real
	st2 {cr1_re4.4s, cr1_im4.4s}, [ local_c_ptr0 ]

	fmla cr1_re5.4s, cr_re5.4s, beta_real
	fmla cr1_im5.4s, cr_re5.4s, beta_imag
	fmls cr1_re5.4s, cr_im5.4s, beta_imag
	fmla cr1_im5.4s, cr_im5.4s, beta_real
	st2 {cr1_re5.4s, cr1_im5.4s}, [ local_c_ptr1 ]

1:
	cmp idx_m0, idx_m1
	beq 99f

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re6.4s, cr0_re6.4s, alpha_real
	fmul cr1_im6.4s, cr0_re6.4s, alpha_imag
	fmls cr1_re6.4s, cr0_im6.4s, alpha_imag
	fmla cr1_im6.4s, cr0_im6.4s, alpha_real

	fmul cr1_re7.4s, cr0_re7.4s, alpha_real
	fmul cr1_im7.4s, cr0_re7.4s, alpha_imag
	fmls cr1_re7.4s, cr0_im7.4s, alpha_imag
	fmla cr1_im7.4s, cr0_im7.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n01, idx_m1
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	ld2 {cr_re6.4s, cr_im6.4s}, [ local_c_ptr0 ]
	ld2 {cr_re7.4s, cr_im7.4s}, [ local_c_ptr1 ]

	// crr_re = cr1_re + cr_re * beta.real() - cr_im * beta.imag();
	// crr_im = cr1_im + cr_re * beta.imag() + cr_im * beta.real();
	fmla cr1_re6.4s, cr_re6.4s, beta_real
	fmla cr1_im6.4s, cr_re6.4s, beta_imag
	fmls cr1_re6.4s, cr_im6.4s, beta_imag
	fmla cr1_im6.4s, cr_im6.4s, beta_real
	st2 {cr1_re6.4s, cr1_im6.4s}, [ local_c_ptr0 ]

	fmla cr1_re7.4s, cr_re7.4s, beta_real
	fmla cr1_im7.4s, cr_re7.4s, beta_imag
	fmls cr1_re7.4s, cr_im7.4s, beta_imag
	fmla cr1_im7.4s, cr_im7.4s, beta_real
	st2 {cr1_re7.4s, cr1_im7.4s}, [ local_c_ptr1 ]

	cmp idx_n01, idx_n23
	beq 99f

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re8.4s, cr0_re8.4s, alpha_real
	fmul cr1_im8.4s, cr0_re8.4s, alpha_imag
	fmls cr1_re8.4s, cr0_im8.4s, alpha_imag
	fmla cr1_im8.4s, cr0_im8.4s, alpha_real

	fmul cr1_re9.4s, cr0_re9.4s, alpha_real
	fmul cr1_im9.4s, cr0_re9.4s, alpha_imag
	fmls cr1_re9.4s, cr0_im9.4s, alpha_imag
	fmla cr1_im9.4s, cr0_im9.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n23, idx_m1
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	ld2 {cr_re8.4s, cr_im8.4s}, [ local_c_ptr0 ]
	ld2 {cr_re9.4s, cr_im9.4s}, [ local_c_ptr1 ]

	// crr_re = cr1_re + cr_re * beta.real() - cr_im * beta.imag();
	// crr_im = cr1_im + cr_re * beta.imag() + cr_im * beta.real();
	fmla cr1_re8.4s, cr_re8.4s, beta_real
	fmla cr1_im8.4s, cr_re8.4s, beta_imag
	fmls cr1_re8.4s, cr_im8.4s, beta_imag
	fmla cr1_im8.4s, cr_im8.4s, beta_real
	st2 {cr1_re8.4s, cr1_im8.4s}, [ local_c_ptr0 ]

	fmla cr1_re9.4s, cr_re9.4s, beta_real
	fmla cr1_im9.4s, cr_re9.4s, beta_imag
	fmls cr1_re9.4s, cr_im9.4s, beta_imag
	fmla cr1_im9.4s, cr_im9.4s, beta_real
	st2 {cr1_re9.4s, cr1_im9.4s}, [ local_c_ptr1 ]

	cmp idx_n23, idx_n45
	beq 99f

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re10.4s, cr0_re10.4s, alpha_real
	fmul cr1_im10.4s, cr0_re10.4s, alpha_imag
	fmls cr1_re10.4s, cr0_im10.4s, alpha_imag
	fmla cr1_im10.4s, cr0_im10.4s, alpha_real

	fmul cr1_re11.4s, cr0_re11.4s, alpha_real
	fmul cr1_im11.4s, cr0_re11.4s, alpha_imag
	fmls cr1_re11.4s, cr0_im11.4s, alpha_imag
	fmla cr1_im11.4s, cr0_im11.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n45, idx_m1
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	ld2 {cr_re10.4s, cr_im10.4s}, [ local_c_ptr0 ]
	ld2 {cr_re11.4s, cr_im11.4s}, [ local_c_ptr1 ]

	// crr_re = cr1_re + cr_re * beta.real() - cr_im * beta.imag();
	// crr_im = cr1_im + cr_re * beta.imag() + cr_im * beta.real();
	fmla cr1_re10.4s, cr_re10.4s, beta_real
	fmla cr1_im10.4s, cr_re10.4s, beta_imag
	fmls cr1_re10.4s, cr_im10.4s, beta_imag
	fmla cr1_im10.4s, cr_im10.4s, beta_real
	st2 {cr1_re10.4s, cr1_im10.4s}, [ local_c_ptr0 ]

	fmla cr1_re11.4s, cr_re11.4s, beta_real
	fmla cr1_im11.4s, cr_re11.4s, beta_imag
	fmls cr1_re11.4s, cr_im11.4s, beta_imag
	fmla cr1_im11.4s, cr_im11.4s, beta_real
	st2 {cr1_re11.4s, cr1_im11.4s}, [ local_c_ptr1 ]

	b 99f

.L${func_name}_loop_c_no_beta:

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re0.4s, cr0_re0.4s, alpha_real
	fmul cr1_im0.4s, cr0_re0.4s, alpha_imag
	fmls cr1_re0.4s, cr0_im0.4s, alpha_imag
	fmla cr1_im0.4s, cr0_im0.4s, alpha_real

	fmul cr1_re1.4s, cr0_re1.4s, alpha_real
	fmul cr1_im1.4s, cr0_re1.4s, alpha_imag
	fmls cr1_re1.4s, cr0_im1.4s, alpha_imag
	fmla cr1_im1.4s, cr0_im1.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n01, idx_m0
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	ld2 {cr_re0.4s, cr_im0.4s}, [ local_c_ptr0 ]
	ld2 {cr_re1.4s, cr_im1.4s}, [ local_c_ptr1 ]

	// crr_re = cr1_re + cr_re;
	// crr_im = cr1_im + cr_im;
	fadd cr1_re0.4s, cr1_re0.4s, cr_re0.4s
	fadd cr1_im0.4s, cr1_im0.4s, cr_im0.4s
	st2 {cr1_re0.4s, cr1_im0.4s}, [ local_c_ptr0 ]

	fadd cr1_re1.4s, cr1_re1.4s, cr_re1.4s
	fadd cr1_im1.4s, cr1_im1.4s, cr_im1.4s
	st2 {cr1_re1.4s, cr1_im1.4s}, [ local_c_ptr1 ]

	cmp idx_n01, idx_n23
	beq 1f

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re2.4s, cr0_re2.4s, alpha_real
	fmul cr1_im2.4s, cr0_re2.4s, alpha_imag
	fmls cr1_re2.4s, cr0_im2.4s, alpha_imag
	fmla cr1_im2.4s, cr0_im2.4s, alpha_real

	fmul cr1_re3.4s, cr0_re3.4s, alpha_real
	fmul cr1_im3.4s, cr0_re3.4s, alpha_imag
	fmls cr1_re3.4s, cr0_im3.4s, alpha_imag
	fmla cr1_im3.4s, cr0_im3.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n23, idx_m0
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	ld2 {cr_re2.4s, cr_im2.4s}, [ local_c_ptr0 ]
	ld2 {cr_re3.4s, cr_im3.4s}, [ local_c_ptr1 ]

	// crr_re = cr1_re + cr_re;
	// crr_im = cr1_im + cr_im;
	fadd cr1_re2.4s, cr1_re2.4s, cr_re2.4s
	fadd cr1_im2.4s, cr1_im2.4s, cr_im2.4s
	st2 {cr1_re2.4s, cr1_im2.4s}, [ local_c_ptr0 ]

	fadd cr1_re3.4s, cr1_re3.4s, cr_re3.4s
	fadd cr1_im3.4s, cr1_im3.4s, cr_im3.4s
	st2 {cr1_re3.4s, cr1_im3.4s}, [ local_c_ptr1 ]

	cmp idx_n23, idx_n45
	beq 1f

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re4.4s, cr0_re4.4s, alpha_real
	fmul cr1_im4.4s, cr0_re4.4s, alpha_imag
	fmls cr1_re4.4s, cr0_im4.4s, alpha_imag
	fmla cr1_im4.4s, cr0_im4.4s, alpha_real

	fmul cr1_re5.4s, cr0_re5.4s, alpha_real
	fmul cr1_im5.4s, cr0_re5.4s, alpha_imag
	fmls cr1_re5.4s, cr0_im5.4s, alpha_imag
	fmla cr1_im5.4s, cr0_im5.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n45, idx_m0
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	ld2 {cr_re4.4s, cr_im4.4s}, [ local_c_ptr0 ]
	ld2 {cr_re5.4s, cr_im5.4s}, [ local_c_ptr1 ]

	// crr_re = cr1_re + cr_re;
	// crr_im = cr1_im + cr_im;
	fadd cr1_re4.4s, cr1_re4.4s, cr_re4.4s
	fadd cr1_im4.4s, cr1_im4.4s, cr_im4.4s
	st2 {cr1_re4.4s, cr1_im4.4s}, [ local_c_ptr0 ]

	fadd cr1_re5.4s, cr1_re5.4s, cr_re5.4s
	fadd cr1_im5.4s, cr1_im5.4s, cr_im5.4s
	st2 {cr1_re5.4s, cr1_im5.4s}, [ local_c_ptr1 ]

1:
	cmp idx_m0, idx_m1
	beq 99f

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re6.4s, cr0_re6.4s, alpha_real
	fmul cr1_im6.4s, cr0_re6.4s, alpha_imag
	fmls cr1_re6.4s, cr0_im6.4s, alpha_imag
	fmla cr1_im6.4s, cr0_im6.4s, alpha_real

	fmul cr1_re7.4s, cr0_re7.4s, alpha_real
	fmul cr1_im7.4s, cr0_re7.4s, alpha_imag
	fmls cr1_re7.4s, cr0_im7.4s, alpha_imag
	fmla cr1_im7.4s, cr0_im7.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n01, idx_m1
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	ld2 {cr_re6.4s, cr_im6.4s}, [ local_c_ptr0 ]
	ld2 {cr_re7.4s, cr_im7.4s}, [ local_c_ptr1 ]

	// crr_re = cr1_re + cr_re;
	// crr_im = cr1_im + cr_im;
	fadd cr1_re6.4s, cr1_re6.4s, cr_re6.4s
	fadd cr1_im6.4s, cr1_im6.4s, cr_im6.4s
	st2 {cr1_re6.4s, cr1_im6.4s}, [ local_c_ptr0 ]

	fadd cr1_re7.4s, cr1_re7.4s, cr_re7.4s
	fadd cr1_im7.4s, cr1_im7.4s, cr_im7.4s
	st2 {cr1_re7.4s, cr1_im7.4s}, [ local_c_ptr1 ]

	cmp idx_n01, idx_n23
	beq 99f

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re8.4s, cr0_re8.4s, alpha_real
	fmul cr1_im8.4s, cr0_re8.4s, alpha_imag
	fmls cr1_re8.4s, cr0_im8.4s, alpha_imag
	fmla cr1_im8.4s, cr0_im8.4s, alpha_real

	fmul cr1_re9.4s, cr0_re9.4s, alpha_real
	fmul cr1_im9.4s, cr0_re9.4s, alpha_imag
	fmls cr1_re9.4s, cr0_im9.4s, alpha_imag
	fmla cr1_im9.4s, cr0_im9.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n23, idx_m1
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	ld2 {cr_re8.4s, cr_im8.4s}, [ local_c_ptr0 ]
	ld2 {cr_re9.4s, cr_im9.4s}, [ local_c_ptr1 ]

	// crr_re = cr1_re + cr_re;
	// crr_im = cr1_im + cr_im;
	fadd cr1_re8.4s, cr1_re8.4s, cr_re8.4s
	fadd cr1_im8.4s, cr1_im8.4s, cr_im8.4s
	st2 {cr1_re8.4s, cr1_im8.4s}, [ local_c_ptr0 ]

	fadd cr1_re9.4s, cr1_re9.4s, cr_re9.4s
	fadd cr1_im9.4s, cr1_im9.4s, cr_im9.4s
	st2 {cr1_re9.4s, cr1_im9.4s}, [ local_c_ptr1 ]

	cmp idx_n23, idx_n45
	beq 99f

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re10.4s, cr0_re10.4s, alpha_real
	fmul cr1_im10.4s, cr0_re10.4s, alpha_imag
	fmls cr1_re10.4s, cr0_im10.4s, alpha_imag
	fmla cr1_im10.4s, cr0_im10.4s, alpha_real

	fmul cr1_re11.4s, cr0_re11.4s, alpha_real
	fmul cr1_im11.4s, cr0_re11.4s, alpha_imag
	fmls cr1_re11.4s, cr0_im11.4s, alpha_imag
	fmla cr1_im11.4s, cr0_im11.4s, alpha_real

	madd local_c_ptr0, ldc, idx_n45, idx_m1
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3
	add local_c_ptr1, local_c_ptr0, ldc, lsl 3
	ld2 {cr_re10.4s, cr_im10.4s}, [ local_c_ptr0 ]
	ld2 {cr_re11.4s, cr_im11.4s}, [ local_c_ptr1 ]

	// crr_re = cr1_re + cr_re;
	// crr_im = cr1_im + cr_im;
	fadd cr1_re10.4s, cr1_re10.4s, cr_re10.4s
	fadd cr1_im10.4s, cr1_im10.4s, cr_im10.4s
	st2 {cr1_re10.4s, cr1_im10.4s}, [ local_c_ptr0 ]

	fadd cr1_re11.4s, cr1_re11.4s, cr_re11.4s
	fadd cr1_im11.4s, cr1_im11.4s, cr_im11.4s
	st2 {cr1_re11.4s, cr1_im11.4s}, [ local_c_ptr1 ]
99:
	add idx_n_base, idx_n_base, #6
	b .L${func_name}_loop_n
.L${func_name}_loop_n_end: // }}}
	add idx_m0, idx_m1, #4
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
	ldp d8, d9, [ sp, #304 ]
	ldp d10, d11, [ sp, #320 ]
	ldp d12, d13, [ sp, #336 ]
	ldp d14, d15, [ sp, #352 ]
	ldp x29, x30, [sp], #384
	ret
	${epilogue(func_name)}
</%def>

${generate_NX_kernel("cgemm_small_kernel_NT_nmk", "fmls", "fmla")}
${generate_NX_kernel("cgemm_small_kernel_NC_nmk", "fmla", "fmls")}

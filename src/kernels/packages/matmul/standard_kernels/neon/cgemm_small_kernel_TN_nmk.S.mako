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
#define k_block_idx x9
#define m_block_size x10
#define m_block_idx x11
#define idx_m_base x12
#define idx_k_base x13 // [0,1,...,m_block_size)
#define idx_m x14 // [0,1,...,m) = m_block_idx + idx_m_base
#define idx_k x15
#define ldc x15
#define idx_n01 x16
#define idx_n23 x17
// x18 may be in use for the Platform Register
#define idx_n45 x19
#define idx_n67 x20
#define ldb_minus_two x21
#define ldb_minus_one x21
#define beta_is_zero x22

#define k_block_size_max idx_n01
#define m_block_size_max idx_n01

#define alpha_real v31.s[0]
#define alpha_imag v31.s[1]
#define beta_real  v31.s[2]
#define beta_imag  v31.s[3]
#define alpha v31.d[0]
#define beta  v31.d[1]

#define M_BLOCK_SIZE_FIRST 300
#define K_BLOCK_SIZE_FIRST 300
#define M_BLOCK_SIZE_DEFAULT 256
#define K_BLOCK_SIZE_DEFAULT 256
#define A_PTR_PREFETCH_DISTANCE 48
#define B_PTR_PREFETCH_DISTANCE 48

// we use a macro to generate the code for both transpose and transpose
// conjugate, as we simply invert whether we apply subtraction in some cases
// and addition in others
<%def name="generate_XN_kernel(func_name, fmlsubadd, fmladdsub)">
	${prologue(func_name)}
	stp x29, x30, [sp, #-384]!
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

	add x23, sp, #288
	st4 { v0.s, v1.s, v2.s, v3.s}[0], [ x23 ]

	fcmp s2, 0.0
	bne .L${func_name}_set_beta_not_zero
	fcmp s3, 0.0
	bne .L${func_name}_set_beta_not_zero
	mov beta_is_zero, 1
	b .L${func_name}_beta_done
.L${func_name}_set_beta_not_zero:
	mov beta_is_zero, 0
.L${func_name}_beta_done:

	sub ldb_minus_two, ldb, #2

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

	mov m_block_size_max, M_BLOCK_SIZE_FIRST

	eor m_block_idx, m_block_idx, m_block_idx
.L${func_name}_loop_block_m: // {{{
	cmp m_block_idx, m
	beq .L${func_name}_loop_block_m_end

	// compute next block size (in elems)
	// m_block_size = min(m_block_size, m - m_block_idx)
	sub m_block_size, m, m_block_idx
	cmp m_block_size_max, m_block_size
	mov m_block_size_max, M_BLOCK_SIZE_DEFAULT
	csel m_block_size, m_block_size_max, m_block_size, le

	eor idx_n01, idx_n01, idx_n01
.L${func_name}_loop_n: // {{{
	cmp idx_n01, n
	bge .L${func_name}_loop_n_end

	add idx_n23, idx_n01, #2
	add idx_n45, idx_n01, #4
	add idx_n67, idx_n01, #6

	cmp idx_n23, n
	csel idx_n23, idx_n01, idx_n23, ge
	cmp idx_n45, n
	csel idx_n45, idx_n23, idx_n45, ge
	cmp idx_n67, n
	csel idx_n67, idx_n45, idx_n67, ge

	eor idx_m_base, idx_m_base, idx_m_base
.L${func_name}_loop_m: // {{{
	cmp idx_m_base, m_block_size
	beq .L${func_name}_loop_m_end
	add idx_m, m_block_idx, idx_m_base

#define ar_im3 v31
#define ar_re3 v30
#define ar_im2 v29
#define ar_re2 v28
#define ar_im1 v27
#define ar_re1 v26
#define ar_im0 v25
#define ar_re0 v24

#define br0 v23
#define br1 v22
#define br2 v21
#define br3 v20
#define br4 v19
#define br5 v18
#define br6 v17
#define br7 v16

#define cr0_re0 v15
#define cr0_im0 v14
#define cr0_re1 v13
#define cr0_im1 v12
#define cr0_re2 v11
#define cr0_im2 v10
#define cr0_re3 v9
#define cr0_im3 v8
#define cr0_re4 v7
#define cr0_im4 v6
#define cr0_re5 v5
#define cr0_im5 v4
#define cr0_re6 v3
#define cr0_im6 v2
#define cr0_re7 v1
#define cr0_im7 v0

#define local_b_ptr67 idx_n67
#define local_b_ptr45 idx_n45
#define local_b_ptr23 idx_n23
#define local_b_ptr01 x28
#define local_a_ptr0 x27
#define local_a_ptr1 x26
#define local_a_ptr2 x25
#define local_a_ptr3 x24

.L${func_name}_loop_k_start: // {{{

	cmp k_block_size, #0

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

	beq .L${func_name}_loop_k_end

	mov idx_k, k_block_size
	cmp k_block_size, #4

	madd local_a_ptr0, lda, idx_m, k_block_idx
	add local_a_ptr0, a_ptr, local_a_ptr0, lsl 3
	add local_a_ptr1, local_a_ptr0, lda, lsl 3
	add local_a_ptr2, local_a_ptr1, lda, lsl 3
	add local_a_ptr3, local_a_ptr2, lda, lsl 3

	madd local_b_ptr01, ldb, idx_n01, k_block_idx
	add local_b_ptr01, b_ptr, local_b_ptr01, lsl 3
	madd local_b_ptr23, ldb, idx_n23, k_block_idx
	add local_b_ptr23, b_ptr, local_b_ptr23, lsl 3
	madd local_b_ptr45, ldb, idx_n45, k_block_idx
	add local_b_ptr45, b_ptr, local_b_ptr45, lsl 3
	madd local_b_ptr67, ldb, idx_n67, k_block_idx
	add local_b_ptr67, b_ptr, local_b_ptr67, lsl 3

	blt .L${func_name}_loop_k_last

	mov idx_k, k_block_size
	cmp k_block_size, #8

	ld4 {ar_re0.s, ar_im0.s, ar_re1.s, ar_im1.s}[0], [ local_a_ptr0 ]
	add local_a_ptr0, local_a_ptr0, (4*4)
	ld4 {ar_re0.s, ar_im0.s, ar_re1.s, ar_im1.s}[1], [ local_a_ptr1 ]
	add local_a_ptr1, local_a_ptr1, (4*4)
	ld4 {ar_re0.s, ar_im0.s, ar_re1.s, ar_im1.s}[2], [ local_a_ptr2 ]
	add local_a_ptr2, local_a_ptr2, (4*4)
	ld4 {ar_re0.s, ar_im0.s, ar_re1.s, ar_im1.s}[3], [ local_a_ptr3 ]
	add local_a_ptr3, local_a_ptr3, (4*4)

	ld1 {br0.4s}, [ local_b_ptr01 ]
	add local_b_ptr01, local_b_ptr01, ldb, lsl 3
	ld1 {br1.4s}, [ local_b_ptr01 ]
	ld1 {br2.4s}, [ local_b_ptr23 ]
	add local_b_ptr23, local_b_ptr23, ldb, lsl 3
	ld1 {br3.4s}, [ local_b_ptr23 ]
	ld1 {br4.4s}, [ local_b_ptr45 ]
	add local_b_ptr45, local_b_ptr45, ldb, lsl 3
	ld1 {br5.4s}, [ local_b_ptr45 ]
	ld1 {br6.4s}, [ local_b_ptr67 ]
	add local_b_ptr67, local_b_ptr67, ldb, lsl 3
	ld1 {br7.4s}, [ local_b_ptr67 ]

	blt .L${func_name}_loop_k_almost_last
	sub idx_k, k_block_size, #8

.L${func_name}_loop_k: // {{{
	// cr0_re = ar_re * br_re - ar_im * br_im + cr0_re;
	// cr0_im = ar_re * br_im + ar_im * br_re + cr0_im;
	fmla cr0_re0.4s, ar_re0.4s, br0.s[0]
	fmla cr0_im0.4s, ar_re0.4s, br0.s[1]
	ld4 {ar_re2.s, ar_im2.s, ar_re3.s, ar_im3.s}[0], [ local_a_ptr0 ]
	${fmlsubadd} cr0_re0.4s, ar_im0.4s, br0.s[1]
	${fmladdsub} cr0_im0.4s, ar_im0.4s, br0.s[0]
	add local_a_ptr0, local_a_ptr0, (4*4)
	fmla cr0_re1.4s, ar_re0.4s, br1.s[0]
	fmla cr0_im1.4s, ar_re0.4s, br1.s[1]
	ld4 {ar_re2.s, ar_im2.s, ar_re3.s, ar_im3.s}[1], [ local_a_ptr1 ]
	${fmlsubadd} cr0_re1.4s, ar_im0.4s, br1.s[1]
	${fmladdsub} cr0_im1.4s, ar_im0.4s, br1.s[0]
	add local_a_ptr1, local_a_ptr1, (4*4)
	fmla cr0_re2.4s, ar_re0.4s, br2.s[0]
	fmla cr0_im2.4s, ar_re0.4s, br2.s[1]
	ld4 {ar_re2.s, ar_im2.s, ar_re3.s, ar_im3.s}[2], [ local_a_ptr2 ]
	${fmlsubadd} cr0_re2.4s, ar_im0.4s, br2.s[1]
	${fmladdsub} cr0_im2.4s, ar_im0.4s, br2.s[0]
	add local_a_ptr2, local_a_ptr2, (4*4)
	fmla cr0_re3.4s, ar_re0.4s, br3.s[0]
	fmla cr0_im3.4s, ar_re0.4s, br3.s[1]
	ld4 {ar_re2.s, ar_im2.s, ar_re3.s, ar_im3.s}[3], [ local_a_ptr3 ]
	${fmlsubadd} cr0_re3.4s, ar_im0.4s, br3.s[1]
	${fmladdsub} cr0_im3.4s, ar_im0.4s, br3.s[0]
	add local_a_ptr3, local_a_ptr3, (4*4)
	fmla cr0_re4.4s, ar_re0.4s, br4.s[0]
	fmla cr0_im4.4s, ar_re0.4s, br4.s[1]
	prfm pldl1keep, [local_a_ptr0, A_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re4.4s, ar_im0.4s, br4.s[1]
	${fmladdsub} cr0_im4.4s, ar_im0.4s, br4.s[0]
	sub local_b_ptr01, local_b_ptr01, ldb_minus_two, lsl 3
	fmla cr0_re5.4s, ar_re0.4s, br5.s[0]
	fmla cr0_im5.4s, ar_re0.4s, br5.s[1]
	prfm pldl1keep, [local_a_ptr1, A_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re5.4s, ar_im0.4s, br5.s[1]
	${fmladdsub} cr0_im5.4s, ar_im0.4s, br5.s[0]
	sub local_b_ptr23, local_b_ptr23, ldb_minus_two, lsl 3
	fmla cr0_re6.4s, ar_re0.4s, br6.s[0]
	fmla cr0_im6.4s, ar_re0.4s, br6.s[1]
	prfm pldl1keep, [local_a_ptr2, A_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re6.4s, ar_im0.4s, br6.s[1]
	${fmladdsub} cr0_im6.4s, ar_im0.4s, br6.s[0]
	sub local_b_ptr45, local_b_ptr45, ldb_minus_two, lsl 3
	fmla cr0_re7.4s, ar_re0.4s, br7.s[0]
	fmla cr0_im7.4s, ar_re0.4s, br7.s[1]
	prfm pldl1keep, [local_a_ptr3, A_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re7.4s, ar_im0.4s, br7.s[1]
	${fmladdsub} cr0_im7.4s, ar_im0.4s, br7.s[0]
	sub local_b_ptr67, local_b_ptr67, ldb_minus_two, lsl 3
	fmla cr0_re0.4s, ar_re1.4s, br0.s[2]
	fmla cr0_im0.4s, ar_re1.4s, br0.s[3]
	prfm pldl1keep, [local_b_ptr01, B_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re0.4s, ar_im1.4s, br0.s[3]
	${fmladdsub} cr0_im0.4s, ar_im1.4s, br0.s[2]
	ld1 {br0.4s}, [ local_b_ptr01 ]
	fmla cr0_re1.4s, ar_re1.4s, br1.s[2]
	fmla cr0_im1.4s, ar_re1.4s, br1.s[3]
	add local_b_ptr01, local_b_ptr01, ldb, lsl 3
	${fmlsubadd} cr0_re1.4s, ar_im1.4s, br1.s[3]
	${fmladdsub} cr0_im1.4s, ar_im1.4s, br1.s[2]
	ld1 {br1.4s}, [ local_b_ptr01 ]
	fmla cr0_re2.4s, ar_re1.4s, br2.s[2]
	fmla cr0_im2.4s, ar_re1.4s, br2.s[3]
	prfm pldl1keep, [local_b_ptr23, B_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re2.4s, ar_im1.4s, br2.s[3]
	${fmladdsub} cr0_im2.4s, ar_im1.4s, br2.s[2]
	ld1 {br2.4s}, [ local_b_ptr23 ]
	fmla cr0_re3.4s, ar_re1.4s, br3.s[2]
	fmla cr0_im3.4s, ar_re1.4s, br3.s[3]
	add local_b_ptr23, local_b_ptr23, ldb, lsl 3
	${fmlsubadd} cr0_re3.4s, ar_im1.4s, br3.s[3]
	${fmladdsub} cr0_im3.4s, ar_im1.4s, br3.s[2]
	ld1 {br3.4s}, [ local_b_ptr23 ]
	fmla cr0_re4.4s, ar_re1.4s, br4.s[2]
	fmla cr0_im4.4s, ar_re1.4s, br4.s[3]
	prfm pldl1keep, [local_b_ptr45, B_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re4.4s, ar_im1.4s, br4.s[3]
	${fmladdsub} cr0_im4.4s, ar_im1.4s, br4.s[2]
	ld1 {br4.4s}, [ local_b_ptr45 ]
	fmla cr0_re5.4s, ar_re1.4s, br5.s[2]
	fmla cr0_im5.4s, ar_re1.4s, br5.s[3]
	add local_b_ptr45, local_b_ptr45, ldb, lsl 3
	${fmlsubadd} cr0_re5.4s, ar_im1.4s, br5.s[3]
	${fmladdsub} cr0_im5.4s, ar_im1.4s, br5.s[2]
	ld1 {br5.4s}, [ local_b_ptr45 ]
	fmla cr0_re6.4s, ar_re1.4s, br6.s[2]
	fmla cr0_im6.4s, ar_re1.4s, br6.s[3]
	prfm pldl1keep, [local_b_ptr67, B_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re6.4s, ar_im1.4s, br6.s[3]
	${fmladdsub} cr0_im6.4s, ar_im1.4s, br6.s[2]
	ld1 {br6.4s}, [ local_b_ptr67 ]
	fmla cr0_re7.4s, ar_re1.4s, br7.s[2]
	fmla cr0_im7.4s, ar_re1.4s, br7.s[3]
	add local_b_ptr67, local_b_ptr67, ldb, lsl 3
	${fmlsubadd} cr0_re7.4s, ar_im1.4s, br7.s[3]
	${fmladdsub} cr0_im7.4s, ar_im1.4s, br7.s[2]
	ld1 {br7.4s}, [ local_b_ptr67 ]
	fmla cr0_re0.4s, ar_re2.4s, br0.s[0]
	fmla cr0_im0.4s, ar_re2.4s, br0.s[1]
	sub local_b_ptr01, local_b_ptr01, ldb_minus_two, lsl 3
	${fmlsubadd} cr0_re0.4s, ar_im2.4s, br0.s[1]
	${fmladdsub} cr0_im0.4s, ar_im2.4s, br0.s[0]
	sub local_b_ptr23, local_b_ptr23, ldb_minus_two, lsl 3
	fmla cr0_re1.4s, ar_re2.4s, br1.s[0]
	fmla cr0_im1.4s, ar_re2.4s, br1.s[1]
	sub local_b_ptr45, local_b_ptr45, ldb_minus_two, lsl 3
	${fmlsubadd} cr0_re1.4s, ar_im2.4s, br1.s[1]
	${fmladdsub} cr0_im1.4s, ar_im2.4s, br1.s[0]
	sub local_b_ptr67, local_b_ptr67, ldb_minus_two, lsl 3
	fmla cr0_re2.4s, ar_re2.4s, br2.s[0]
	fmla cr0_im2.4s, ar_re2.4s, br2.s[1]
	subs idx_k, idx_k, #4
	${fmlsubadd} cr0_re2.4s, ar_im2.4s, br2.s[1]
	${fmladdsub} cr0_im2.4s, ar_im2.4s, br2.s[0]
	ld4 {ar_re0.s, ar_im0.s, ar_re1.s, ar_im1.s}[0], [ local_a_ptr0 ]
	fmla cr0_re3.4s, ar_re2.4s, br3.s[0]
	fmla cr0_im3.4s, ar_re2.4s, br3.s[1]
	add local_a_ptr0, local_a_ptr0, (4*4)
	${fmlsubadd} cr0_re3.4s, ar_im2.4s, br3.s[1]
	${fmladdsub} cr0_im3.4s, ar_im2.4s, br3.s[0]
	ld4 {ar_re0.s, ar_im0.s, ar_re1.s, ar_im1.s}[1], [ local_a_ptr1 ]
	fmla cr0_re4.4s, ar_re2.4s, br4.s[0]
	fmla cr0_im4.4s, ar_re2.4s, br4.s[1]
	add local_a_ptr1, local_a_ptr1, (4*4)
	${fmlsubadd} cr0_re4.4s, ar_im2.4s, br4.s[1]
	${fmladdsub} cr0_im4.4s, ar_im2.4s, br4.s[0]
	ld4 {ar_re0.s, ar_im0.s, ar_re1.s, ar_im1.s}[2], [ local_a_ptr2 ]
	fmla cr0_re5.4s, ar_re2.4s, br5.s[0]
	fmla cr0_im5.4s, ar_re2.4s, br5.s[1]
	add local_a_ptr2, local_a_ptr2, (4*4)
	${fmlsubadd} cr0_re5.4s, ar_im2.4s, br5.s[1]
	${fmladdsub} cr0_im5.4s, ar_im2.4s, br5.s[0]
	ld4 {ar_re0.s, ar_im0.s, ar_re1.s, ar_im1.s}[3], [ local_a_ptr3 ]
	fmla cr0_re6.4s, ar_re2.4s, br6.s[0]
	fmla cr0_im6.4s, ar_re2.4s, br6.s[1]
	add local_a_ptr3, local_a_ptr3, (4*4)
	${fmlsubadd} cr0_re6.4s, ar_im2.4s, br6.s[1]
	${fmladdsub} cr0_im6.4s, ar_im2.4s, br6.s[0]
	prfm pldl1keep, [local_b_ptr01, B_PTR_PREFETCH_DISTANCE]
	fmla cr0_re7.4s, ar_re2.4s, br7.s[0]
	fmla cr0_im7.4s, ar_re2.4s, br7.s[1]
	prfm pldl1keep, [local_b_ptr23, B_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re7.4s, ar_im2.4s, br7.s[1]
	${fmladdsub} cr0_im7.4s, ar_im2.4s, br7.s[0]
	prfm pldl1keep, [local_b_ptr45, B_PTR_PREFETCH_DISTANCE]
	fmla cr0_re0.4s, ar_re3.4s, br0.s[2]
	fmla cr0_im0.4s, ar_re3.4s, br0.s[3]
	prfm pldl1keep, [local_b_ptr67, B_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re0.4s, ar_im3.4s, br0.s[3]
	${fmladdsub} cr0_im0.4s, ar_im3.4s, br0.s[2]
	ld1 {br0.4s}, [ local_b_ptr01 ]
	fmla cr0_re1.4s, ar_re3.4s, br1.s[2]
	fmla cr0_im1.4s, ar_re3.4s, br1.s[3]
	add local_b_ptr01, local_b_ptr01, ldb, lsl 3
	${fmlsubadd} cr0_re1.4s, ar_im3.4s, br1.s[3]
	${fmladdsub} cr0_im1.4s, ar_im3.4s, br1.s[2]
	ld1 {br1.4s}, [ local_b_ptr01 ]
	fmla cr0_re2.4s, ar_re3.4s, br2.s[2]
	fmla cr0_im2.4s, ar_re3.4s, br2.s[3]

	${fmlsubadd} cr0_re2.4s, ar_im3.4s, br2.s[3]
	${fmladdsub} cr0_im2.4s, ar_im3.4s, br2.s[2]
	ld1 {br2.4s}, [ local_b_ptr23 ]
	fmla cr0_re3.4s, ar_re3.4s, br3.s[2]
	fmla cr0_im3.4s, ar_re3.4s, br3.s[3]
	add local_b_ptr23, local_b_ptr23, ldb, lsl 3
	${fmlsubadd} cr0_re3.4s, ar_im3.4s, br3.s[3]
	${fmladdsub} cr0_im3.4s, ar_im3.4s, br3.s[2]
	ld1 {br3.4s}, [ local_b_ptr23 ]
	fmla cr0_re4.4s, ar_re3.4s, br4.s[2]
	fmla cr0_im4.4s, ar_re3.4s, br4.s[3]

	${fmlsubadd} cr0_re4.4s, ar_im3.4s, br4.s[3]
	${fmladdsub} cr0_im4.4s, ar_im3.4s, br4.s[2]
	ld1 {br4.4s}, [ local_b_ptr45 ]
	fmla cr0_re5.4s, ar_re3.4s, br5.s[2]
	fmla cr0_im5.4s, ar_re3.4s, br5.s[3]
	add local_b_ptr45, local_b_ptr45, ldb, lsl 3
	${fmlsubadd} cr0_re5.4s, ar_im3.4s, br5.s[3]
	${fmladdsub} cr0_im5.4s, ar_im3.4s, br5.s[2]
	ld1 {br5.4s}, [ local_b_ptr45 ]
	fmla cr0_re6.4s, ar_re3.4s, br6.s[2]
	fmla cr0_im6.4s, ar_re3.4s, br6.s[3]

	${fmlsubadd} cr0_re6.4s, ar_im3.4s, br6.s[3]
	${fmladdsub} cr0_im6.4s, ar_im3.4s, br6.s[2]
	ld1 {br6.4s}, [ local_b_ptr67 ]
	fmla cr0_re7.4s, ar_re3.4s, br7.s[2]
	fmla cr0_im7.4s, ar_re3.4s, br7.s[3]
	add local_b_ptr67, local_b_ptr67, ldb, lsl 3
	${fmlsubadd} cr0_re7.4s, ar_im3.4s, br7.s[3]
	${fmladdsub} cr0_im7.4s, ar_im3.4s, br7.s[2]
	ld1 {br7.4s}, [ local_b_ptr67 ]

	bge .L${func_name}_loop_k
	add idx_k, idx_k, #8


.L${func_name}_loop_k_almost_last: // }}}
	// cr0_re = ar_re * br_re - ar_im * br_im + cr0_re;
	// cr0_im = ar_re * br_im + ar_im * br_re + cr0_im;

	fmla cr0_re0.4s, ar_re0.4s, br0.s[0]
	fmla cr0_im0.4s, ar_re0.4s, br0.s[1]
	ld4 {ar_re2.s, ar_im2.s, ar_re3.s, ar_im3.s}[0], [ local_a_ptr0 ]
	${fmlsubadd} cr0_re0.4s, ar_im0.4s, br0.s[1]
	${fmladdsub} cr0_im0.4s, ar_im0.4s, br0.s[0]
	add local_a_ptr0, local_a_ptr0, (4*4)
	fmla cr0_re1.4s, ar_re0.4s, br1.s[0]
	fmla cr0_im1.4s, ar_re0.4s, br1.s[1]
	ld4 {ar_re2.s, ar_im2.s, ar_re3.s, ar_im3.s}[1], [ local_a_ptr1 ]
	${fmlsubadd} cr0_re1.4s, ar_im0.4s, br1.s[1]
	${fmladdsub} cr0_im1.4s, ar_im0.4s, br1.s[0]
	add local_a_ptr1, local_a_ptr1, (4*4)
	fmla cr0_re2.4s, ar_re0.4s, br2.s[0]
	fmla cr0_im2.4s, ar_re0.4s, br2.s[1]
	ld4 {ar_re2.s, ar_im2.s, ar_re3.s, ar_im3.s}[2], [ local_a_ptr2 ]
	${fmlsubadd} cr0_re2.4s, ar_im0.4s, br2.s[1]
	${fmladdsub} cr0_im2.4s, ar_im0.4s, br2.s[0]
	add local_a_ptr2, local_a_ptr2, (4*4)
	fmla cr0_re3.4s, ar_re0.4s, br3.s[0]
	fmla cr0_im3.4s, ar_re0.4s, br3.s[1]
	ld4 {ar_re2.s, ar_im2.s, ar_re3.s, ar_im3.s}[3], [ local_a_ptr3 ]
	${fmlsubadd} cr0_re3.4s, ar_im0.4s, br3.s[1]
	${fmladdsub} cr0_im3.4s, ar_im0.4s, br3.s[0]
	add local_a_ptr3, local_a_ptr3, (4*4)
	fmla cr0_re4.4s, ar_re0.4s, br4.s[0]
	fmla cr0_im4.4s, ar_re0.4s, br4.s[1]
	prfm pldl1keep, [local_a_ptr0, A_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re4.4s, ar_im0.4s, br4.s[1]
	${fmladdsub} cr0_im4.4s, ar_im0.4s, br4.s[0]
	sub local_b_ptr01, local_b_ptr01, ldb_minus_two, lsl 3
	fmla cr0_re5.4s, ar_re0.4s, br5.s[0]
	fmla cr0_im5.4s, ar_re0.4s, br5.s[1]
	prfm pldl1keep, [local_a_ptr1, A_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re5.4s, ar_im0.4s, br5.s[1]
	${fmladdsub} cr0_im5.4s, ar_im0.4s, br5.s[0]
	sub local_b_ptr23, local_b_ptr23, ldb_minus_two, lsl 3
	fmla cr0_re6.4s, ar_re0.4s, br6.s[0]
	fmla cr0_im6.4s, ar_re0.4s, br6.s[1]
	prfm pldl1keep, [local_a_ptr2, A_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re6.4s, ar_im0.4s, br6.s[1]
	${fmladdsub} cr0_im6.4s, ar_im0.4s, br6.s[0]
	sub local_b_ptr45, local_b_ptr45, ldb_minus_two, lsl 3
	fmla cr0_re7.4s, ar_re0.4s, br7.s[0]
	fmla cr0_im7.4s, ar_re0.4s, br7.s[1]
	prfm pldl1keep, [local_a_ptr3, A_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re7.4s, ar_im0.4s, br7.s[1]
	${fmladdsub} cr0_im7.4s, ar_im0.4s, br7.s[0]
	sub local_b_ptr67, local_b_ptr67, ldb_minus_two, lsl 3
	fmla cr0_re0.4s, ar_re1.4s, br0.s[2]
	fmla cr0_im0.4s, ar_re1.4s, br0.s[3]
	prfm pldl1keep, [local_b_ptr01, B_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re0.4s, ar_im1.4s, br0.s[3]
	${fmladdsub} cr0_im0.4s, ar_im1.4s, br0.s[2]
	ld1 {br0.4s}, [ local_b_ptr01 ]
	fmla cr0_re1.4s, ar_re1.4s, br1.s[2]
	fmla cr0_im1.4s, ar_re1.4s, br1.s[3]
	add local_b_ptr01, local_b_ptr01, ldb, lsl 3
	${fmlsubadd} cr0_re1.4s, ar_im1.4s, br1.s[3]
	${fmladdsub} cr0_im1.4s, ar_im1.4s, br1.s[2]
	ld1 {br1.4s}, [ local_b_ptr01 ]
	fmla cr0_re2.4s, ar_re1.4s, br2.s[2]
	fmla cr0_im2.4s, ar_re1.4s, br2.s[3]
	prfm pldl1keep, [local_b_ptr23, B_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re2.4s, ar_im1.4s, br2.s[3]
	${fmladdsub} cr0_im2.4s, ar_im1.4s, br2.s[2]
	ld1 {br2.4s}, [ local_b_ptr23 ]
	fmla cr0_re3.4s, ar_re1.4s, br3.s[2]
	fmla cr0_im3.4s, ar_re1.4s, br3.s[3]
	add local_b_ptr23, local_b_ptr23, ldb, lsl 3
	${fmlsubadd} cr0_re3.4s, ar_im1.4s, br3.s[3]
	${fmladdsub} cr0_im3.4s, ar_im1.4s, br3.s[2]
	ld1 {br3.4s}, [ local_b_ptr23 ]
	fmla cr0_re4.4s, ar_re1.4s, br4.s[2]
	fmla cr0_im4.4s, ar_re1.4s, br4.s[3]
	prfm pldl1keep, [local_b_ptr45, B_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re4.4s, ar_im1.4s, br4.s[3]
	${fmladdsub} cr0_im4.4s, ar_im1.4s, br4.s[2]
	ld1 {br4.4s}, [ local_b_ptr45 ]
	fmla cr0_re5.4s, ar_re1.4s, br5.s[2]
	fmla cr0_im5.4s, ar_re1.4s, br5.s[3]
	add local_b_ptr45, local_b_ptr45, ldb, lsl 3
	${fmlsubadd} cr0_re5.4s, ar_im1.4s, br5.s[3]
	${fmladdsub} cr0_im5.4s, ar_im1.4s, br5.s[2]
	ld1 {br5.4s}, [ local_b_ptr45 ]
	fmla cr0_re6.4s, ar_re1.4s, br6.s[2]
	fmla cr0_im6.4s, ar_re1.4s, br6.s[3]
	prfm pldl1keep, [local_b_ptr67, B_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re6.4s, ar_im1.4s, br6.s[3]
	${fmladdsub} cr0_im6.4s, ar_im1.4s, br6.s[2]
	ld1 {br6.4s}, [ local_b_ptr67 ]
	fmla cr0_re7.4s, ar_re1.4s, br7.s[2]
	fmla cr0_im7.4s, ar_re1.4s, br7.s[3]
	add local_b_ptr67, local_b_ptr67, ldb, lsl 3
	${fmlsubadd} cr0_re7.4s, ar_im1.4s, br7.s[3]
	${fmladdsub} cr0_im7.4s, ar_im1.4s, br7.s[2]
	ld1 {br7.4s}, [ local_b_ptr67 ]
	fmla cr0_re0.4s, ar_re2.4s, br0.s[0]
	fmla cr0_im0.4s, ar_re2.4s, br0.s[1]
	sub local_b_ptr01, local_b_ptr01, ldb_minus_two, lsl 3
	fmla cr0_re1.4s, ar_re2.4s, br1.s[0]
	fmla cr0_im1.4s, ar_re2.4s, br1.s[1]
	sub local_b_ptr23, local_b_ptr23, ldb_minus_two, lsl 3
	${fmlsubadd} cr0_re0.4s, ar_im2.4s, br0.s[1]
	${fmladdsub} cr0_im0.4s, ar_im2.4s, br0.s[0]
	sub local_b_ptr45, local_b_ptr45, ldb_minus_two, lsl 3
	${fmlsubadd} cr0_re1.4s, ar_im2.4s, br1.s[1]
	${fmladdsub} cr0_im1.4s, ar_im2.4s, br1.s[0]
	sub local_b_ptr67, local_b_ptr67, ldb_minus_two, lsl 3
	fmla cr0_re2.4s, ar_re2.4s, br2.s[0]
	fmla cr0_im2.4s, ar_re2.4s, br2.s[1]
	fmla cr0_re3.4s, ar_re2.4s, br3.s[0]
	fmla cr0_im3.4s, ar_re2.4s, br3.s[1]
	${fmlsubadd} cr0_re2.4s, ar_im2.4s, br2.s[1]
	${fmladdsub} cr0_im2.4s, ar_im2.4s, br2.s[0]
	${fmlsubadd} cr0_re3.4s, ar_im2.4s, br3.s[1]
	${fmladdsub} cr0_im3.4s, ar_im2.4s, br3.s[0]
	fmla cr0_re4.4s, ar_re2.4s, br4.s[0]
	fmla cr0_im4.4s, ar_re2.4s, br4.s[1]
	fmla cr0_re5.4s, ar_re2.4s, br5.s[0]
	fmla cr0_im5.4s, ar_re2.4s, br5.s[1]
	${fmlsubadd} cr0_re4.4s, ar_im2.4s, br4.s[1]
	${fmladdsub} cr0_im4.4s, ar_im2.4s, br4.s[0]
	${fmlsubadd} cr0_re5.4s, ar_im2.4s, br5.s[1]
	${fmladdsub} cr0_im5.4s, ar_im2.4s, br5.s[0]
	fmla cr0_re6.4s, ar_re2.4s, br6.s[0]
	fmla cr0_im6.4s, ar_re2.4s, br6.s[1]
	fmla cr0_re7.4s, ar_re2.4s, br7.s[0]
	fmla cr0_im7.4s, ar_re2.4s, br7.s[1]
	prfm pldl1keep, [local_b_ptr01, B_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re6.4s, ar_im2.4s, br6.s[1]
	${fmladdsub} cr0_im6.4s, ar_im2.4s, br6.s[0]
	prfm pldl1keep, [local_b_ptr23, B_PTR_PREFETCH_DISTANCE]
	${fmlsubadd} cr0_re7.4s, ar_im2.4s, br7.s[1]
	${fmladdsub} cr0_im7.4s, ar_im2.4s, br7.s[0]
	prfm pldl1keep, [local_b_ptr45, B_PTR_PREFETCH_DISTANCE]
	fmla cr0_re0.4s, ar_re3.4s, br0.s[2]
	fmla cr0_im0.4s, ar_re3.4s, br0.s[3]
	prfm pldl1keep, [local_b_ptr67, B_PTR_PREFETCH_DISTANCE]
	fmla cr0_re1.4s, ar_re3.4s, br1.s[2]
	fmla cr0_im1.4s, ar_re3.4s, br1.s[3]
	subs idx_k, idx_k, #4
	${fmlsubadd} cr0_re0.4s, ar_im3.4s, br0.s[3]
	${fmladdsub} cr0_im0.4s, ar_im3.4s, br0.s[2]
	${fmlsubadd} cr0_re1.4s, ar_im3.4s, br1.s[3]
	${fmladdsub} cr0_im1.4s, ar_im3.4s, br1.s[2]
	fmla cr0_re2.4s, ar_re3.4s, br2.s[2]
	fmla cr0_im2.4s, ar_re3.4s, br2.s[3]
	fmla cr0_re3.4s, ar_re3.4s, br3.s[2]
	fmla cr0_im3.4s, ar_re3.4s, br3.s[3]
	${fmlsubadd} cr0_re2.4s, ar_im3.4s, br2.s[3]
	${fmladdsub} cr0_im2.4s, ar_im3.4s, br2.s[2]
	${fmlsubadd} cr0_re3.4s, ar_im3.4s, br3.s[3]
	${fmladdsub} cr0_im3.4s, ar_im3.4s, br3.s[2]
	fmla cr0_re4.4s, ar_re3.4s, br4.s[2]
	fmla cr0_im4.4s, ar_re3.4s, br4.s[3]
	fmla cr0_re5.4s, ar_re3.4s, br5.s[2]
	fmla cr0_im5.4s, ar_re3.4s, br5.s[3]
	${fmlsubadd} cr0_re4.4s, ar_im3.4s, br4.s[3]
	${fmladdsub} cr0_im4.4s, ar_im3.4s, br4.s[2]
	${fmlsubadd} cr0_re5.4s, ar_im3.4s, br5.s[3]
	${fmladdsub} cr0_im5.4s, ar_im3.4s, br5.s[2]
	fmla cr0_re6.4s, ar_re3.4s, br6.s[2]
	fmla cr0_im6.4s, ar_re3.4s, br6.s[3]
	fmla cr0_re7.4s, ar_re3.4s, br7.s[2]
	fmla cr0_im7.4s, ar_re3.4s, br7.s[3]
	${fmlsubadd} cr0_re6.4s, ar_im3.4s, br6.s[3]
	${fmladdsub} cr0_im6.4s, ar_im3.4s, br6.s[2]
	${fmlsubadd} cr0_re7.4s, ar_im3.4s, br7.s[3]
	${fmladdsub} cr0_im7.4s, ar_im3.4s, br7.s[2]

.L${func_name}_loop_k_last:

	beq .L${func_name}_loop_k_end

	ld2 {ar_re0.s, ar_im0.s}[0], [ local_a_ptr0 ]
	ld2 {ar_re0.s, ar_im0.s}[1], [ local_a_ptr1 ]
	ld2 {ar_re0.s, ar_im0.s}[2], [ local_a_ptr2 ]
	ld2 {ar_re0.s, ar_im0.s}[3], [ local_a_ptr3 ]

	ld1 {br0.2s}, [ local_b_ptr01 ]
	add local_b_ptr01, local_b_ptr01, ldb, lsl 3
	ld1 {br2.2s}, [ local_b_ptr23 ]
	add local_b_ptr23, local_b_ptr23, ldb, lsl 3
	ld1 {br4.2s}, [ local_b_ptr45 ]
	add local_b_ptr45, local_b_ptr45, ldb, lsl 3
	ld1 {br6.2s}, [ local_b_ptr67 ]
	add local_b_ptr67, local_b_ptr67, ldb, lsl 3

	sub ldb_minus_one, ldb, #1

	subs idx_k, idx_k, #1

	fmla cr0_re0.4s, ar_re0.4s, br0.s[0]
	fmla cr0_im0.4s, ar_re0.4s, br0.s[1]
	${fmlsubadd} cr0_re0.4s, ar_im0.4s, br0.s[1]
	${fmladdsub} cr0_im0.4s, ar_im0.4s, br0.s[0]

	ld1 {br1.2s}, [ local_b_ptr01 ]
	sub local_b_ptr01, local_b_ptr01, ldb_minus_one, lsl 3

	fmla cr0_re2.4s, ar_re0.4s, br2.s[0]
	fmla cr0_im2.4s, ar_re0.4s, br2.s[1]
	${fmlsubadd} cr0_re2.4s, ar_im0.4s, br2.s[1]
	${fmladdsub} cr0_im2.4s, ar_im0.4s, br2.s[0]

	ld1 {br3.2s}, [ local_b_ptr23 ]
	sub local_b_ptr23, local_b_ptr23, ldb_minus_one, lsl 3

	fmla cr0_re4.4s, ar_re0.4s, br4.s[0]
	fmla cr0_im4.4s, ar_re0.4s, br4.s[1]
	${fmlsubadd} cr0_re4.4s, ar_im0.4s, br4.s[1]
	${fmladdsub} cr0_im4.4s, ar_im0.4s, br4.s[0]

	ld1 {br5.2s}, [ local_b_ptr45 ]
	sub local_b_ptr45, local_b_ptr45, ldb_minus_one, lsl 3

	fmla cr0_re6.4s, ar_re0.4s, br6.s[0]
	fmla cr0_im6.4s, ar_re0.4s, br6.s[1]
	${fmlsubadd} cr0_re6.4s, ar_im0.4s, br6.s[1]
	${fmladdsub} cr0_im6.4s, ar_im0.4s, br6.s[0]

	ld1 {br7.2s}, [ local_b_ptr67 ]
	sub local_b_ptr67, local_b_ptr67, ldb_minus_one, lsl 3

	fmla cr0_re1.4s, ar_re0.4s, br1.s[0]
	fmla cr0_im1.4s, ar_re0.4s, br1.s[1]
	sub ldb_minus_two, ldb, #2
	${fmlsubadd} cr0_re1.4s, ar_im0.4s, br1.s[1]
	${fmladdsub} cr0_im1.4s, ar_im0.4s, br1.s[0]

	add local_a_ptr0, local_a_ptr0, (2*4)

	fmla cr0_re3.4s, ar_re0.4s, br3.s[0]
	fmla cr0_im3.4s, ar_re0.4s, br3.s[1]
	${fmlsubadd} cr0_re3.4s, ar_im0.4s, br3.s[1]
	${fmladdsub} cr0_im3.4s, ar_im0.4s, br3.s[0]

	add local_a_ptr1, local_a_ptr1, (2*4)

	fmla cr0_re5.4s, ar_re0.4s, br5.s[0]
	fmla cr0_im5.4s, ar_re0.4s, br5.s[1]
	${fmlsubadd} cr0_re5.4s, ar_im0.4s, br5.s[1]
	${fmladdsub} cr0_im5.4s, ar_im0.4s, br5.s[0]

	add local_a_ptr2, local_a_ptr2, (2*4)

	fmla cr0_re7.4s, ar_re0.4s, br7.s[0]
	fmla cr0_im7.4s, ar_re0.4s, br7.s[1]
	${fmlsubadd} cr0_re7.4s, ar_im0.4s, br7.s[1]
	${fmladdsub} cr0_im7.4s, ar_im0.4s, br7.s[0]

	add local_a_ptr3, local_a_ptr3, (2*4)

	b .L${func_name}_loop_k_last

.L${func_name}_loop_k_end: // }}}

#define local_c_ptr01 x28
#define local_c_ptr23 x27
#define local_c_ptr45 x26
#define local_c_ptr67 x25
#define local_c_ptr_tmp x24

#define cr1_im0 v30
#define cr1_re0 v29
#define cr1_im1 v28
#define cr1_re1 v27
#define cr1_im2 v26
#define cr1_re2 v25
#define cr1_im3 v24
#define cr1_re3 v23
#define cr1_im4 v22
#define cr1_re4 v21
#define cr1_im5 v20
#define cr1_re5 v19
#define cr1_im6 v18
#define cr1_re6 v17
// we run out of registers here, so start again
#define cr1_im7 v30
#define cr1_re7 v29

#define cr_im0 v15
#define cr_re0 v14
#define cr_im1 v13
#define cr_re1 v12
#define cr_im2 v11
#define cr_re2 v10
#define cr_im3 v9
#define cr_re3 v8
#define cr_im4 v7
#define cr_re4 v6
#define cr_im5 v5
#define cr_re5 v4
#define cr_im6 v3
#define cr_re6 v2
#define cr_im7 v1
#define cr_re7 v0

	// duplicated logic here since idx_n{23,45,67} are clobbered
	// by local_{a,b}_ptr
	add idx_n23, idx_n01, #2
	add idx_n45, idx_n01, #4
	add idx_n67, idx_n01, #6

	ldr	ldc, [ sp, #384 ]

	cmp idx_n23, n
	csel idx_n23, idx_n01, idx_n23, ge
	cmp idx_n45, n
	csel idx_n45, idx_n23, idx_n45, ge
	cmp idx_n67, n
	csel idx_n67, idx_n45, idx_n67, ge

	ldr q31, [ sp, #288 ]

	cmp beta_is_zero, 1
	bne .L${func_name}_beta_non_zero

	cmp k_block_idx, 0
	bne .L${func_name}_beta_non_zero


.L${func_name}_beta_zero:
	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	madd local_c_ptr01, ldc, idx_n01, idx_m
	add local_c_ptr01, c_ptr, local_c_ptr01, lsl 3
	fmul cr1_re0.4s, cr0_re0.4s, alpha_real
	fmul cr1_im0.4s, cr0_re0.4s, alpha_imag
	fmls cr1_re0.4s, cr0_im0.4s, alpha_imag
	fmla cr1_im0.4s, cr0_im0.4s, alpha_real

	fmul cr1_re1.4s, cr0_re1.4s, alpha_real
	fmul cr1_im1.4s, cr0_re1.4s, alpha_imag
	fmls cr1_re1.4s, cr0_im1.4s, alpha_imag
	fmla cr1_im1.4s, cr0_im1.4s, alpha_real

	madd local_c_ptr23, ldc, idx_n23, idx_m
	add local_c_ptr23, c_ptr, local_c_ptr23, lsl 3
	fmul cr1_re2.4s, cr0_re2.4s, alpha_real
	fmul cr1_im2.4s, cr0_re2.4s, alpha_imag
	fmls cr1_re2.4s, cr0_im2.4s, alpha_imag
	fmla cr1_im2.4s, cr0_im2.4s, alpha_real

	fmul cr1_re3.4s, cr0_re3.4s, alpha_real
	fmul cr1_im3.4s, cr0_re3.4s, alpha_imag
	fmls cr1_re3.4s, cr0_im3.4s, alpha_imag
	fmla cr1_im3.4s, cr0_im3.4s, alpha_real

	madd local_c_ptr45, ldc, idx_n45, idx_m
	add local_c_ptr45, c_ptr, local_c_ptr45, lsl 3
	fmul cr1_re4.4s, cr0_re4.4s, alpha_real
	fmul cr1_im4.4s, cr0_re4.4s, alpha_imag
	fmls cr1_re4.4s, cr0_im4.4s, alpha_imag
	fmla cr1_im4.4s, cr0_im4.4s, alpha_real

	fmul cr1_re5.4s, cr0_re5.4s, alpha_real
	fmul cr1_im5.4s, cr0_re5.4s, alpha_imag
	fmls cr1_re5.4s, cr0_im5.4s, alpha_imag
	fmla cr1_im5.4s, cr0_im5.4s, alpha_real

	madd local_c_ptr67, ldc, idx_n67, idx_m
	add local_c_ptr67, c_ptr, local_c_ptr67, lsl 3
	fmul cr1_re6.4s, cr0_re6.4s, alpha_real
	fmul cr1_im6.4s, cr0_re6.4s, alpha_imag
	fmls cr1_re6.4s, cr0_im6.4s, alpha_imag
	fmla cr1_im6.4s, cr0_im6.4s, alpha_real
	dup cr_re0.2d, xzr
	dup cr_im0.2d, xzr
	dup cr_re1.2d, xzr
	dup cr_im1.2d, xzr
	dup cr_re2.2d, xzr
	dup cr_im2.2d, xzr
	dup cr_re3.2d, xzr
	dup cr_im3.2d, xzr
	dup cr_re4.2d, xzr
	dup cr_im4.2d, xzr
	dup cr_re5.2d, xzr
	dup cr_im5.2d, xzr
	dup cr_re6.2d, xzr
	dup cr_im6.2d, xzr


	// crr_re = cr1_re + cr_re * beta.real() - cr_im * beta.imag();
	// crr_im = cr1_im + cr_re * beta.imag() + cr_im * beta.real();
	fadd cr1_re0.4s, cr1_re0.4s, cr_re0.4s
	fadd cr1_im0.4s, cr1_im0.4s, cr_im0.4s
	st2 {cr1_re0.4s, cr1_im0.4s}, [ local_c_ptr01 ]
	add local_c_ptr01, local_c_ptr01, ldc, lsl 3

	// note: this block must occur after the store to cr1_{re,im}0
	fmul cr1_re7.4s, cr0_re7.4s, alpha_real
	fmul cr1_im7.4s, cr0_re7.4s, alpha_imag
	fmls cr1_re7.4s, cr0_im7.4s, alpha_imag
	fmla cr1_im7.4s, cr0_im7.4s, alpha_real
	dup cr_re7.2d, xzr
	dup cr_im7.2d, xzr

	fadd cr1_re1.4s, cr1_re1.4s, cr_re1.4s
	fadd cr1_im1.4s, cr1_im1.4s, cr_im1.4s
	st2 {cr1_re1.4s, cr1_im1.4s}, [ local_c_ptr01 ]

	fadd cr1_re2.4s, cr1_re2.4s, cr_re2.4s
	fadd cr1_im2.4s, cr1_im2.4s, cr_im2.4s
	st2 {cr1_re2.4s, cr1_im2.4s}, [ local_c_ptr23 ]
	add local_c_ptr23, local_c_ptr23, ldc, lsl 3

	fadd cr1_re3.4s, cr1_re3.4s, cr_re3.4s
	fadd cr1_im3.4s, cr1_im3.4s, cr_im3.4s
	st2 {cr1_re3.4s, cr1_im3.4s}, [ local_c_ptr23 ]

	fadd cr1_re4.4s, cr1_re4.4s, cr_re4.4s
	fadd cr1_im4.4s, cr1_im4.4s, cr_im4.4s
	st2 {cr1_re4.4s, cr1_im4.4s}, [ local_c_ptr45 ]
	add local_c_ptr45, local_c_ptr45, ldc, lsl 3

	fadd cr1_re5.4s, cr1_re5.4s, cr_re5.4s
	fadd cr1_im5.4s, cr1_im5.4s, cr_im5.4s
	st2 {cr1_re5.4s, cr1_im5.4s}, [ local_c_ptr45 ]

	fadd cr1_re6.4s, cr1_re6.4s, cr_re6.4s
	fadd cr1_im6.4s, cr1_im6.4s, cr_im6.4s
	st2 {cr1_re6.4s, cr1_im6.4s}, [ local_c_ptr67 ]
	add local_c_ptr67, local_c_ptr67, ldc, lsl 3

	fadd cr1_re7.4s, cr1_re7.4s, cr_re7.4s
	fadd cr1_im7.4s, cr1_im7.4s, cr_im7.4s
	st2 {cr1_re7.4s, cr1_im7.4s}, [ local_c_ptr67 ]

	add idx_m_base, idx_m_base, #4
	b .L${func_name}_loop_m


.L${func_name}_beta_non_zero:
	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	madd local_c_ptr01, ldc, idx_n01, idx_m
	add local_c_ptr01, c_ptr, local_c_ptr01, lsl 3
	fmul cr1_re0.4s, cr0_re0.4s, alpha_real
	fmul cr1_im0.4s, cr0_re0.4s, alpha_imag
	fmls cr1_re0.4s, cr0_im0.4s, alpha_imag
	fmla cr1_im0.4s, cr0_im0.4s, alpha_real
	ld2 {cr_re0.4s, cr_im0.4s}, [ local_c_ptr01 ]
	add local_c_ptr_tmp, local_c_ptr01, ldc, lsl 3

	fmul cr1_re1.4s, cr0_re1.4s, alpha_real
	fmul cr1_im1.4s, cr0_re1.4s, alpha_imag
	fmls cr1_re1.4s, cr0_im1.4s, alpha_imag
	fmla cr1_im1.4s, cr0_im1.4s, alpha_real
	ld2 {cr_re1.4s, cr_im1.4s}, [ local_c_ptr_tmp ]

	madd local_c_ptr23, ldc, idx_n23, idx_m
	add local_c_ptr23, c_ptr, local_c_ptr23, lsl 3
	fmul cr1_re2.4s, cr0_re2.4s, alpha_real
	fmul cr1_im2.4s, cr0_re2.4s, alpha_imag
	fmls cr1_re2.4s, cr0_im2.4s, alpha_imag
	fmla cr1_im2.4s, cr0_im2.4s, alpha_real
	ld2 {cr_re2.4s, cr_im2.4s}, [ local_c_ptr23 ]
	add local_c_ptr_tmp, local_c_ptr23, ldc, lsl 3

	fmul cr1_re3.4s, cr0_re3.4s, alpha_real
	fmul cr1_im3.4s, cr0_re3.4s, alpha_imag
	fmls cr1_re3.4s, cr0_im3.4s, alpha_imag
	fmla cr1_im3.4s, cr0_im3.4s, alpha_real
	ld2 {cr_re3.4s, cr_im3.4s}, [ local_c_ptr_tmp ]

	madd local_c_ptr45, ldc, idx_n45, idx_m
	add local_c_ptr45, c_ptr, local_c_ptr45, lsl 3
	fmul cr1_re4.4s, cr0_re4.4s, alpha_real
	fmul cr1_im4.4s, cr0_re4.4s, alpha_imag
	fmls cr1_re4.4s, cr0_im4.4s, alpha_imag
	fmla cr1_im4.4s, cr0_im4.4s, alpha_real
	ld2 {cr_re4.4s, cr_im4.4s}, [ local_c_ptr45 ]
	add local_c_ptr_tmp, local_c_ptr45, ldc, lsl 3

	fmul cr1_re5.4s, cr0_re5.4s, alpha_real
	fmul cr1_im5.4s, cr0_re5.4s, alpha_imag
	fmls cr1_re5.4s, cr0_im5.4s, alpha_imag
	fmla cr1_im5.4s, cr0_im5.4s, alpha_real
	ld2 {cr_re5.4s, cr_im5.4s}, [ local_c_ptr_tmp ]

	madd local_c_ptr67, ldc, idx_n67, idx_m
	add local_c_ptr67, c_ptr, local_c_ptr67, lsl 3
	fmul cr1_re6.4s, cr0_re6.4s, alpha_real
	fmul cr1_im6.4s, cr0_re6.4s, alpha_imag
	fmls cr1_re6.4s, cr0_im6.4s, alpha_imag
	fmla cr1_im6.4s, cr0_im6.4s, alpha_real
	ld2 {cr_re6.4s, cr_im6.4s}, [ local_c_ptr67 ]
	add local_c_ptr_tmp, local_c_ptr67, ldc, lsl 3

	cmp k_block_idx, #0
	beq .L${func_name}_loop_with_beta

	// crr_re = cr1_re + cr_re * beta.real() - cr_im * beta.imag();
	// crr_im = cr1_im + cr_re * beta.imag() + cr_im * beta.real();
	fadd cr1_re0.4s, cr1_re0.4s, cr_re0.4s
	fadd cr1_im0.4s, cr1_im0.4s, cr_im0.4s
	st2 {cr1_re0.4s, cr1_im0.4s}, [ local_c_ptr01 ]
	add local_c_ptr01, local_c_ptr01, ldc, lsl 3

	// note: this block must occur after the store to cr1_{re,im}0
	fmul cr1_re7.4s, cr0_re7.4s, alpha_real
	fmul cr1_im7.4s, cr0_re7.4s, alpha_imag
	fmls cr1_re7.4s, cr0_im7.4s, alpha_imag
	fmla cr1_im7.4s, cr0_im7.4s, alpha_real
	ld2 {cr_re7.4s, cr_im7.4s}, [ local_c_ptr_tmp ]

	fadd cr1_re1.4s, cr1_re1.4s, cr_re1.4s
	fadd cr1_im1.4s, cr1_im1.4s, cr_im1.4s
	st2 {cr1_re1.4s, cr1_im1.4s}, [ local_c_ptr01 ]

	fadd cr1_re2.4s, cr1_re2.4s, cr_re2.4s
	fadd cr1_im2.4s, cr1_im2.4s, cr_im2.4s
	st2 {cr1_re2.4s, cr1_im2.4s}, [ local_c_ptr23 ]
	add local_c_ptr23, local_c_ptr23, ldc, lsl 3

	fadd cr1_re3.4s, cr1_re3.4s, cr_re3.4s
	fadd cr1_im3.4s, cr1_im3.4s, cr_im3.4s
	st2 {cr1_re3.4s, cr1_im3.4s}, [ local_c_ptr23 ]

	fadd cr1_re4.4s, cr1_re4.4s, cr_re4.4s
	fadd cr1_im4.4s, cr1_im4.4s, cr_im4.4s
	st2 {cr1_re4.4s, cr1_im4.4s}, [ local_c_ptr45 ]
	add local_c_ptr45, local_c_ptr45, ldc, lsl 3

	fadd cr1_re5.4s, cr1_re5.4s, cr_re5.4s
	fadd cr1_im5.4s, cr1_im5.4s, cr_im5.4s
	st2 {cr1_re5.4s, cr1_im5.4s}, [ local_c_ptr45 ]

	fadd cr1_re6.4s, cr1_re6.4s, cr_re6.4s
	fadd cr1_im6.4s, cr1_im6.4s, cr_im6.4s
	st2 {cr1_re6.4s, cr1_im6.4s}, [ local_c_ptr67 ]
	add local_c_ptr67, local_c_ptr67, ldc, lsl 3

	fadd cr1_re7.4s, cr1_re7.4s, cr_re7.4s
	fadd cr1_im7.4s, cr1_im7.4s, cr_im7.4s
	st2 {cr1_re7.4s, cr1_im7.4s}, [ local_c_ptr67 ]

	add idx_m_base, idx_m_base, #4
	b .L${func_name}_loop_m

.L${func_name}_loop_with_beta:

	// crr_re = cr1_re + cr_re * beta.real() - cr_im * beta.imag();
	// crr_im = cr1_im + cr_re * beta.imag() + cr_im * beta.real();
	fmla cr1_re0.4s, cr_re0.4s, beta_real
	fmla cr1_im0.4s, cr_re0.4s, beta_imag
	fmls cr1_re0.4s, cr_im0.4s, beta_imag
	fmla cr1_im0.4s, cr_im0.4s, beta_real
	st2 {cr1_re0.4s, cr1_im0.4s}, [ local_c_ptr01 ]
	add local_c_ptr01, local_c_ptr01, ldc, lsl 3

	// note: this block must occur after the store to cr1_{re,im}0
	fmul cr1_re7.4s, cr0_re7.4s, alpha_real
	fmul cr1_im7.4s, cr0_re7.4s, alpha_imag
	fmls cr1_re7.4s, cr0_im7.4s, alpha_imag
	fmla cr1_im7.4s, cr0_im7.4s, alpha_real
	ld2 {cr_re7.4s, cr_im7.4s}, [ local_c_ptr_tmp ]

	fmla cr1_re1.4s, cr_re1.4s, beta_real
	fmla cr1_im1.4s, cr_re1.4s, beta_imag
	fmls cr1_re1.4s, cr_im1.4s, beta_imag
	fmla cr1_im1.4s, cr_im1.4s, beta_real
	st2 {cr1_re1.4s, cr1_im1.4s}, [ local_c_ptr01 ]

	fmla cr1_re2.4s, cr_re2.4s, beta_real
	fmla cr1_im2.4s, cr_re2.4s, beta_imag
	fmls cr1_re2.4s, cr_im2.4s, beta_imag
	fmla cr1_im2.4s, cr_im2.4s, beta_real
	st2 {cr1_re2.4s, cr1_im2.4s}, [ local_c_ptr23 ]
	add local_c_ptr23, local_c_ptr23, ldc, lsl 3

	fmla cr1_re3.4s, cr_re3.4s, beta_real
	fmla cr1_im3.4s, cr_re3.4s, beta_imag
	fmls cr1_re3.4s, cr_im3.4s, beta_imag
	fmla cr1_im3.4s, cr_im3.4s, beta_real
	st2 {cr1_re3.4s, cr1_im3.4s}, [ local_c_ptr23 ]

	fmla cr1_re4.4s, cr_re4.4s, beta_real
	fmla cr1_im4.4s, cr_re4.4s, beta_imag
	fmls cr1_re4.4s, cr_im4.4s, beta_imag
	fmla cr1_im4.4s, cr_im4.4s, beta_real
	st2 {cr1_re4.4s, cr1_im4.4s}, [ local_c_ptr45 ]
	add local_c_ptr45, local_c_ptr45, ldc, lsl 3

	fmla cr1_re5.4s, cr_re5.4s, beta_real
	fmla cr1_im5.4s, cr_re5.4s, beta_imag
	fmls cr1_re5.4s, cr_im5.4s, beta_imag
	fmla cr1_im5.4s, cr_im5.4s, beta_real
	st2 {cr1_re5.4s, cr1_im5.4s}, [ local_c_ptr45 ]

	fmla cr1_re6.4s, cr_re6.4s, beta_real
	fmla cr1_im6.4s, cr_re6.4s, beta_imag
	fmls cr1_re6.4s, cr_im6.4s, beta_imag
	fmla cr1_im6.4s, cr_im6.4s, beta_real
	st2 {cr1_re6.4s, cr1_im6.4s}, [ local_c_ptr67 ]
	add local_c_ptr67, local_c_ptr67, ldc, lsl 3

	fmla cr1_re7.4s, cr_re7.4s, beta_real
	fmla cr1_im7.4s, cr_re7.4s, beta_imag
	fmls cr1_re7.4s, cr_im7.4s, beta_imag
	fmla cr1_im7.4s, cr_im7.4s, beta_real
	st2 {cr1_re7.4s, cr1_im7.4s}, [ local_c_ptr67 ]

	add idx_m_base, idx_m_base, #4
	b .L${func_name}_loop_m
.L${func_name}_loop_m_end: // }}}
	add idx_n01, idx_n01, #8
	b .L${func_name}_loop_n
.L${func_name}_loop_n_end: // }}}
	add m_block_idx, m_block_idx, m_block_size
	mov m_block_size_max, M_BLOCK_SIZE_DEFAULT
	b .L${func_name}_loop_block_m
.L${func_name}_loop_block_m_end: // }}}
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

${generate_XN_kernel("cgemm_small_kernel_TN_nmk", "fmls", "fmla")}
${generate_XN_kernel("cgemm_small_kernel_CN_nmk", "fmla", "fmls")}

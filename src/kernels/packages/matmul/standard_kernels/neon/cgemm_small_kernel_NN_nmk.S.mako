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
#define k_block_idx x10
#define k_block_size_max x11
#define m_block_size x11
#define m_block_idx x12 // [0,m_block_size,...,m)
#define m_block_size_max x13
#define idx_m_base x13 // [0,1,...,m_block_size)
#define idx_m x14 // [0,1,...,m) = m_block_idx + idx_m_base
#define idx_k_base x15
#define idx_n01 x17
// x18 may be in use for the Platform Register
#define idx_n23 x19
#define idx_n45 x20
#define idx_n67 x21
#define ldb_minus_two x22
#define beta_is_zero x16

#define alpha_real v0.s[0]
#define alpha_imag v0.s[1]
#define beta_real v0.s[2]
#define beta_imag v0.s[3]

#define M_BLOCK_SIZE_FIRST 72
#define K_BLOCK_SIZE_FIRST 72
#define K_BLOCK_SIZE_DEFAULT 48
#define M_BLOCK_SIZE_DEFAULT 48
#define A_PTR_PREFETCH_DISTANCE 48
#define B_PTR_PREFETCH_DISTANCE 48


	<% func_name = "cgemm_small_kernel_NN_nmk" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-384]!
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
	mov alpha_real, v0.s[0]
	mov alpha_imag, v1.s[0]
	mov beta_real, v2.s[0]
	mov beta_imag, v3.s[0]

	fcmp s2, 0.0
	bne .Lset_beta_not_zero
	fcmp s3, 0.0
	bne .Lset_beta_not_zero
	mov beta_is_zero, 1
	b .Lbeta_done
.Lset_beta_not_zero:
	mov beta_is_zero, 0
.Lbeta_done:

	sub ldb_minus_two, ldb, 2

	mov k_block_size_max, K_BLOCK_SIZE_FIRST

	eor k_block_idx, k_block_idx, k_block_idx
.Loop_block_k: // {{{
	cmp k_block_idx, k
	beq .Loop_block_k_end

	// compute next block size (in elems)
	// k_block_size = min(k_block_size, k - k_block_idx)
	sub k_block_size, k, k_block_idx
	cmp k_block_size_max, k_block_size
	mov k_block_size_max, K_BLOCK_SIZE_DEFAULT
	csel k_block_size, k_block_size_max, k_block_size, le

	mov m_block_size_max, M_BLOCK_SIZE_FIRST

	eor m_block_idx, m_block_idx, m_block_idx
.Loop_block_m: // {{{
	cmp m_block_idx, m
	beq .Loop_block_m_end

	// compute next block size (in elems)
	// m_block_size = min(m_block_size, m - m_block_idx)
	sub m_block_size, m, m_block_idx
	cmp m_block_size_max, m_block_size
	mov m_block_size_max, M_BLOCK_SIZE_DEFAULT
	csel m_block_size, m_block_size_max, m_block_size, le

	eor idx_n01, idx_n01, idx_n01
.Loop_n: // {{{
	cmp idx_n01, n
	beq .Loop_n_end
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
.Loop_m: // {{{
	cmp idx_m_base, m_block_size
	beq .Loop_m_end
	add idx_m, m_block_idx, idx_m_base

.Loop_k_first_iteration: // {{{
	// do first iteration of k!

#define ar_im0 v31
#define ar_re0 v30
#define ar_im1 v29
#define ar_re1 v28
#define br0 v27
#define br1 v26
#define br2 v25
#define br3 v24
#define br4 v23
#define br5 v22
#define br6 v21
#define br7 v20
// v17-v19 inclusive unused in inner loop
#define cr0_re0 v16
#define cr0_im0 v15
#define cr0_re1 v14
#define cr0_im1 v13
#define cr0_re2 v12
#define cr0_im2 v11
#define cr0_re3 v10
#define cr0_im3 v9
#define cr0_re4 v8
#define cr0_im4 v7
#define cr0_re5 v6
#define cr0_im5 v5
#define cr0_re6 v4
#define cr0_im6 v3
#define cr0_re7 v2
#define cr0_im7 v1
// v0 reserved for holding alpha/beta

#define local_b_ptr01 x28
#define local_b_ptr23 x27
#define local_b_ptr45 x26
#define local_b_ptr67 x25
#define local_a_ptr0 x24
#define local_a_ptr1 x23

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

	// local_a_ptr = &A[lda*ik + im]
	// local_b_ptr01 = &B[ldb*in + ik]
	// local_b_ptr23 = &B[ldb*in + ik]
	// local_b_ptr45 = &B[ldb*in + ik]
	// local_b_ptr67 = &B[ldb*in + ik]
	madd local_a_ptr0, lda, k_block_idx, idx_m
	add local_a_ptr0, a_ptr, local_a_ptr0, lsl 3
	add local_a_ptr1, local_a_ptr0, lda, lsl 3

	madd local_b_ptr01, ldb, idx_n01, k_block_idx
	add local_b_ptr01, b_ptr, local_b_ptr01, lsl 3

	madd local_b_ptr23, ldb, idx_n23, k_block_idx
	add local_b_ptr23, b_ptr, local_b_ptr23, lsl 3

	madd local_b_ptr45, ldb, idx_n45, k_block_idx
	add local_b_ptr45, b_ptr, local_b_ptr45, lsl 3

	madd local_b_ptr67, ldb, idx_n67, k_block_idx
	add local_b_ptr67, b_ptr, local_b_ptr67, lsl 3

	cmp k_block_size, #1
	beq .Loop_k_last_iteration_odd
	cmp k_block_size, #2
	beq .Loop_k_last_iteration_even

	ld2 {ar_re0.4s, ar_im0.4s}, [ local_a_ptr0 ]

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

	cmp k_block_size, #4
	add idx_k_base, k_block_size, #4
	ble .Loop_k_almost_last_iteration

	mov idx_k_base, #5
.Loop_k: // {{{
.macro LOOP_K_BODY // {{{
	// cr0_re = ar_re * br_re - ar_im * br_im + cr0_re;
	// cr0_im = ar_re * br_im + ar_im * br_re + cr0_im;
	fmla cr0_re0.4s, ar_re0.4s, br0.s[0]
	fmla cr0_im0.4s, ar_re0.4s, br0.s[1]
	prfm pldl1keep, [local_a_ptr0, A_PTR_PREFETCH_DISTANCE]
	fmls cr0_re0.4s, ar_im0.4s, br0.s[1]
	fmla cr0_im0.4s, ar_im0.4s, br0.s[0]
	add idx_k_base, idx_k_base, #2
	fmla cr0_re1.4s, ar_re0.4s, br1.s[0]
	fmla cr0_im1.4s, ar_re0.4s, br1.s[1]
	prfm pldl1keep, [local_b_ptr01, B_PTR_PREFETCH_DISTANCE]
	fmls cr0_re1.4s, ar_im0.4s, br1.s[1]
	fmla cr0_im1.4s, ar_im0.4s, br1.s[0]
	cmp idx_k_base, k_block_size
	fmla cr0_re2.4s, ar_re0.4s, br2.s[0]
	fmla cr0_im2.4s, ar_re0.4s, br2.s[1]
	prfm pldl1keep, [local_a_ptr1, A_PTR_PREFETCH_DISTANCE]
	fmls cr0_re2.4s, ar_im0.4s, br2.s[1]
	fmla cr0_im2.4s, ar_im0.4s, br2.s[0]
	ld2 {ar_re1.4s, ar_im1.4s}, [ local_a_ptr1 ]
	fmla cr0_re3.4s, ar_re0.4s, br3.s[0]
	fmla cr0_im3.4s, ar_re0.4s, br3.s[1]
	add local_a_ptr0, local_a_ptr0, lda, lsl 4
	fmls cr0_re3.4s, ar_im0.4s, br3.s[1]
	fmla cr0_im3.4s, ar_im0.4s, br3.s[0]
	add local_a_ptr1, local_a_ptr1, lda, lsl 4
	fmla cr0_re4.4s, ar_re0.4s, br4.s[0]
	fmla cr0_im4.4s, ar_re0.4s, br4.s[1]
	sub local_b_ptr01, local_b_ptr01, ldb_minus_two, lsl 3
	fmls cr0_re4.4s, ar_im0.4s, br4.s[1]
	fmla cr0_im4.4s, ar_im0.4s, br4.s[0]
	prfm pldl1keep, [local_b_ptr01, B_PTR_PREFETCH_DISTANCE]
	fmla cr0_re5.4s, ar_re0.4s, br5.s[0]
	fmla cr0_im5.4s, ar_re0.4s, br5.s[1]
	prfm pldl1keep, [local_b_ptr23, B_PTR_PREFETCH_DISTANCE]
	fmls cr0_re5.4s, ar_im0.4s, br5.s[1]
	fmla cr0_im5.4s, ar_im0.4s, br5.s[0]
	sub local_b_ptr23, local_b_ptr23, ldb_minus_two, lsl 3
	fmla cr0_re6.4s, ar_re0.4s, br6.s[0]
	fmla cr0_im6.4s, ar_re0.4s, br6.s[1]
	prfm pldl1keep, [local_b_ptr23, B_PTR_PREFETCH_DISTANCE]
	fmls cr0_re6.4s, ar_im0.4s, br6.s[1]
	fmla cr0_im6.4s, ar_im0.4s, br6.s[0]
	prfm pldl1keep, [local_b_ptr45, B_PTR_PREFETCH_DISTANCE]
	fmla cr0_re7.4s, ar_re0.4s, br7.s[0]
	fmla cr0_im7.4s, ar_re0.4s, br7.s[1]
	sub local_b_ptr45, local_b_ptr45, ldb_minus_two, lsl 3
	fmls cr0_re7.4s, ar_im0.4s, br7.s[1]
	fmla cr0_im7.4s, ar_im0.4s, br7.s[0]
	sub local_b_ptr67, local_b_ptr67, ldb_minus_two, lsl 3
	fmla cr0_re0.4s, ar_re1.4s, br0.s[2]
	fmla cr0_im0.4s, ar_re1.4s, br0.s[3]
	ld2 {ar_re0.4s, ar_im0.4s}, [ local_a_ptr0 ]
	fmls cr0_re0.4s, ar_im1.4s, br0.s[3]
	fmla cr0_im0.4s, ar_im1.4s, br0.s[2]
	prfm pldl1keep, [local_b_ptr67, B_PTR_PREFETCH_DISTANCE]
	fmla cr0_re1.4s, ar_re1.4s, br1.s[2]
	fmla cr0_im1.4s, ar_re1.4s, br1.s[3]
	ld1 {br0.4s}, [ local_b_ptr01 ]
	fmls cr0_re1.4s, ar_im1.4s, br1.s[3]
	fmla cr0_im1.4s, ar_im1.4s, br1.s[2]
	add local_b_ptr01, local_b_ptr01, ldb, lsl 3
	fmla cr0_re2.4s, ar_re1.4s, br2.s[2]
	fmla cr0_im2.4s, ar_re1.4s, br2.s[3]
	ld1 {br1.4s}, [ local_b_ptr01 ]
	fmls cr0_re2.4s, ar_im1.4s, br2.s[3]
	fmla cr0_im2.4s, ar_im1.4s, br2.s[2]
	ld1 {br2.4s}, [ local_b_ptr23 ]
	fmla cr0_re3.4s, ar_re1.4s, br3.s[2]
	fmla cr0_im3.4s, ar_re1.4s, br3.s[3]
	add local_b_ptr23, local_b_ptr23, ldb, lsl 3
	fmls cr0_re3.4s, ar_im1.4s, br3.s[3]
	fmla cr0_im3.4s, ar_im1.4s, br3.s[2]
	ld1 {br3.4s}, [ local_b_ptr23 ]
	fmla cr0_re4.4s, ar_re1.4s, br4.s[2]
	fmla cr0_im4.4s, ar_re1.4s, br4.s[3]
	prfm pldl1keep, [local_b_ptr45, B_PTR_PREFETCH_DISTANCE]
	fmls cr0_re4.4s, ar_im1.4s, br4.s[3]
	fmla cr0_im4.4s, ar_im1.4s, br4.s[2]
	ld1 {br4.4s}, [ local_b_ptr45 ]
	fmla cr0_re5.4s, ar_re1.4s, br5.s[2]
	fmla cr0_im5.4s, ar_re1.4s, br5.s[3]
	add local_b_ptr45, local_b_ptr45, ldb, lsl 3
	fmls cr0_re5.4s, ar_im1.4s, br5.s[3]
	fmla cr0_im5.4s, ar_im1.4s, br5.s[2]
	ld1 {br5.4s}, [ local_b_ptr45 ]
	fmla cr0_re6.4s, ar_re1.4s, br6.s[2]
	fmla cr0_im6.4s, ar_re1.4s, br6.s[3]
	prfm pldl1keep, [local_b_ptr67, B_PTR_PREFETCH_DISTANCE]
	fmls cr0_re6.4s, ar_im1.4s, br6.s[3]
	fmla cr0_im6.4s, ar_im1.4s, br6.s[2]
	ld1 {br6.4s}, [ local_b_ptr67 ]
	fmla cr0_re7.4s, ar_re1.4s, br7.s[2]
	fmla cr0_im7.4s, ar_re1.4s, br7.s[3]
	add local_b_ptr67, local_b_ptr67, ldb, lsl 3
	fmls cr0_re7.4s, ar_im1.4s, br7.s[3]
	fmla cr0_im7.4s, ar_im1.4s, br7.s[2]
	ld1 {br7.4s}, [ local_b_ptr67 ]
.endm // }}}
    // actually use the loop body unrolled
	LOOP_K_BODY
	bgt .Loop_k_almost_last_iteration
	LOOP_K_BODY
	ble .Loop_k
.Loop_k_almost_last_iteration:

	// cr0_re = ar_re * br_re - ar_im * br_im + cr0_re;
	// cr0_im = ar_re * br_im + ar_im * br_re + cr0_im;
	fmla cr0_re0.4s, ar_re0.4s, br0.s[0]
	fmla cr0_im0.4s, ar_re0.4s, br0.s[1]
	fmls cr0_re0.4s, ar_im0.4s, br0.s[1]
	fmla cr0_im0.4s, ar_im0.4s, br0.s[0]

	fmla cr0_re1.4s, ar_re0.4s, br1.s[0]
	fmla cr0_im1.4s, ar_re0.4s, br1.s[1]
	fmls cr0_re1.4s, ar_im0.4s, br1.s[1]
	fmla cr0_im1.4s, ar_im0.4s, br1.s[0]

	fmla cr0_re2.4s, ar_re0.4s, br2.s[0]
	fmla cr0_im2.4s, ar_re0.4s, br2.s[1]
	fmls cr0_re2.4s, ar_im0.4s, br2.s[1]
	fmla cr0_im2.4s, ar_im0.4s, br2.s[0]

	ld2 {ar_re1.4s, ar_im1.4s}, [ local_a_ptr1 ]
	add local_a_ptr0, local_a_ptr0, lda, lsl 4
	add local_a_ptr1, local_a_ptr1, lda, lsl 4
	sub local_b_ptr01, local_b_ptr01, ldb_minus_two, lsl 3
	sub local_b_ptr23, local_b_ptr23, ldb_minus_two, lsl 3
	sub local_b_ptr45, local_b_ptr45, ldb_minus_two, lsl 3
	sub local_b_ptr67, local_b_ptr67, ldb_minus_two, lsl 3

	fmla cr0_re3.4s, ar_re0.4s, br3.s[0]
	fmla cr0_im3.4s, ar_re0.4s, br3.s[1]
	fmls cr0_re3.4s, ar_im0.4s, br3.s[1]
	fmla cr0_im3.4s, ar_im0.4s, br3.s[0]

	fmla cr0_re4.4s, ar_re0.4s, br4.s[0]
	fmla cr0_im4.4s, ar_re0.4s, br4.s[1]
	fmls cr0_re4.4s, ar_im0.4s, br4.s[1]
	fmla cr0_im4.4s, ar_im0.4s, br4.s[0]

	fmla cr0_re5.4s, ar_re0.4s, br5.s[0]
	fmla cr0_im5.4s, ar_re0.4s, br5.s[1]
	fmls cr0_re5.4s, ar_im0.4s, br5.s[1]
	fmla cr0_im5.4s, ar_im0.4s, br5.s[0]

	fmla cr0_re6.4s, ar_re0.4s, br6.s[0]
	fmla cr0_im6.4s, ar_re0.4s, br6.s[1]
	fmls cr0_re6.4s, ar_im0.4s, br6.s[1]
	fmla cr0_im6.4s, ar_im0.4s, br6.s[0]

	fmla cr0_re7.4s, ar_re0.4s, br7.s[0]
	fmla cr0_im7.4s, ar_re0.4s, br7.s[1]
	fmls cr0_re7.4s, ar_im0.4s, br7.s[1]
	fmla cr0_im7.4s, ar_im0.4s, br7.s[0]

	fmla cr0_re0.4s, ar_re1.4s, br0.s[2]
	fmla cr0_im0.4s, ar_re1.4s, br0.s[3]
	fmls cr0_re0.4s, ar_im1.4s, br0.s[3]
	fmla cr0_im0.4s, ar_im1.4s, br0.s[2]

	fmla cr0_re1.4s, ar_re1.4s, br1.s[2]
	fmla cr0_im1.4s, ar_re1.4s, br1.s[3]
	fmls cr0_re1.4s, ar_im1.4s, br1.s[3]
	fmla cr0_im1.4s, ar_im1.4s, br1.s[2]

	fmla cr0_re2.4s, ar_re1.4s, br2.s[2]
	fmla cr0_im2.4s, ar_re1.4s, br2.s[3]
	fmls cr0_re2.4s, ar_im1.4s, br2.s[3]
	fmla cr0_im2.4s, ar_im1.4s, br2.s[2]

	fmla cr0_re3.4s, ar_re1.4s, br3.s[2]
	fmla cr0_im3.4s, ar_re1.4s, br3.s[3]
	fmls cr0_re3.4s, ar_im1.4s, br3.s[3]
	fmla cr0_im3.4s, ar_im1.4s, br3.s[2]

	fmla cr0_re4.4s, ar_re1.4s, br4.s[2]
	fmla cr0_im4.4s, ar_re1.4s, br4.s[3]
	fmls cr0_re4.4s, ar_im1.4s, br4.s[3]
	fmla cr0_im4.4s, ar_im1.4s, br4.s[2]

	fmla cr0_re5.4s, ar_re1.4s, br5.s[2]
	fmla cr0_im5.4s, ar_re1.4s, br5.s[3]
	fmls cr0_re5.4s, ar_im1.4s, br5.s[3]
	fmla cr0_im5.4s, ar_im1.4s, br5.s[2]

	fmla cr0_re6.4s, ar_re1.4s, br6.s[2]
	fmla cr0_im6.4s, ar_re1.4s, br6.s[3]
	fmls cr0_re6.4s, ar_im1.4s, br6.s[3]
	fmla cr0_im6.4s, ar_im1.4s, br6.s[2]

	fmla cr0_re7.4s, ar_re1.4s, br7.s[2]
	fmla cr0_im7.4s, ar_re1.4s, br7.s[3]
	fmls cr0_re7.4s, ar_im1.4s, br7.s[3]
	fmla cr0_im7.4s, ar_im1.4s, br7.s[2]

	tst k_block_size, #1
	bne .Loop_k_last_iteration_odd

.Loop_k_last_iteration_even: // }}}

	// local_a_ptr = &A[lda*ik + im]
	// local_b_ptr01 = &B[ldb*in + ik]
	// local_b_ptr23 = &B[ldb*in + ik]
	// local_b_ptr45 = &B[ldb*in + ik]
	// local_b_ptr67 = &B[ldb*in + ik]

	ld2 {ar_re0.4s, ar_im0.4s}, [ local_a_ptr0 ]
	ld2 {ar_re1.4s, ar_im1.4s}, [ local_a_ptr1 ]

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

	// cr0_re = ar_re * br_re - ar_im * br_im + cr0_re;
	// cr0_im = ar_re * br_im + ar_im * br_re + cr0_im;
	fmla cr0_re0.4s, ar_re0.4s, br0.s[0]
	fmla cr0_im0.4s, ar_re0.4s, br0.s[1]
	fmls cr0_re0.4s, ar_im0.4s, br0.s[1]
	fmla cr0_im0.4s, ar_im0.4s, br0.s[0]

	fmla cr0_re1.4s, ar_re0.4s, br1.s[0]
	fmla cr0_im1.4s, ar_re0.4s, br1.s[1]
	fmls cr0_re1.4s, ar_im0.4s, br1.s[1]
	fmla cr0_im1.4s, ar_im0.4s, br1.s[0]

	fmla cr0_re2.4s, ar_re0.4s, br2.s[0]
	fmla cr0_im2.4s, ar_re0.4s, br2.s[1]
	fmls cr0_re2.4s, ar_im0.4s, br2.s[1]
	fmla cr0_im2.4s, ar_im0.4s, br2.s[0]

	fmla cr0_re3.4s, ar_re0.4s, br3.s[0]
	fmla cr0_im3.4s, ar_re0.4s, br3.s[1]
	fmls cr0_re3.4s, ar_im0.4s, br3.s[1]
	fmla cr0_im3.4s, ar_im0.4s, br3.s[0]

	fmla cr0_re4.4s, ar_re0.4s, br4.s[0]
	fmla cr0_im4.4s, ar_re0.4s, br4.s[1]
	fmls cr0_re4.4s, ar_im0.4s, br4.s[1]
	fmla cr0_im4.4s, ar_im0.4s, br4.s[0]

	fmla cr0_re5.4s, ar_re0.4s, br5.s[0]
	fmla cr0_im5.4s, ar_re0.4s, br5.s[1]
	fmls cr0_re5.4s, ar_im0.4s, br5.s[1]
	fmla cr0_im5.4s, ar_im0.4s, br5.s[0]

	fmla cr0_re6.4s, ar_re0.4s, br6.s[0]
	fmla cr0_im6.4s, ar_re0.4s, br6.s[1]
	fmls cr0_re6.4s, ar_im0.4s, br6.s[1]
	fmla cr0_im6.4s, ar_im0.4s, br6.s[0]

	fmla cr0_re7.4s, ar_re0.4s, br7.s[0]
	fmla cr0_im7.4s, ar_re0.4s, br7.s[1]
	fmls cr0_re7.4s, ar_im0.4s, br7.s[1]
	fmla cr0_im7.4s, ar_im0.4s, br7.s[0]

	fmla cr0_re0.4s, ar_re1.4s, br0.s[2]
	fmla cr0_im0.4s, ar_re1.4s, br0.s[3]
	fmls cr0_re0.4s, ar_im1.4s, br0.s[3]
	fmla cr0_im0.4s, ar_im1.4s, br0.s[2]

	fmla cr0_re1.4s, ar_re1.4s, br1.s[2]
	fmla cr0_im1.4s, ar_re1.4s, br1.s[3]
	fmls cr0_re1.4s, ar_im1.4s, br1.s[3]
	fmla cr0_im1.4s, ar_im1.4s, br1.s[2]

	fmla cr0_re2.4s, ar_re1.4s, br2.s[2]
	fmla cr0_im2.4s, ar_re1.4s, br2.s[3]
	fmls cr0_re2.4s, ar_im1.4s, br2.s[3]
	fmla cr0_im2.4s, ar_im1.4s, br2.s[2]

	fmla cr0_re3.4s, ar_re1.4s, br3.s[2]
	fmla cr0_im3.4s, ar_re1.4s, br3.s[3]
	fmls cr0_re3.4s, ar_im1.4s, br3.s[3]
	fmla cr0_im3.4s, ar_im1.4s, br3.s[2]

	fmla cr0_re4.4s, ar_re1.4s, br4.s[2]
	fmla cr0_im4.4s, ar_re1.4s, br4.s[3]
	fmls cr0_re4.4s, ar_im1.4s, br4.s[3]
	fmla cr0_im4.4s, ar_im1.4s, br4.s[2]

	fmla cr0_re5.4s, ar_re1.4s, br5.s[2]
	fmla cr0_im5.4s, ar_re1.4s, br5.s[3]
	fmls cr0_re5.4s, ar_im1.4s, br5.s[3]
	fmla cr0_im5.4s, ar_im1.4s, br5.s[2]

	fmla cr0_re6.4s, ar_re1.4s, br6.s[2]
	fmla cr0_im6.4s, ar_re1.4s, br6.s[3]
	fmls cr0_re6.4s, ar_im1.4s, br6.s[3]
	fmla cr0_im6.4s, ar_im1.4s, br6.s[2]

	fmla cr0_re7.4s, ar_re1.4s, br7.s[2]
	fmla cr0_im7.4s, ar_re1.4s, br7.s[3]
	fmls cr0_re7.4s, ar_im1.4s, br7.s[3]
	fmla cr0_im7.4s, ar_im1.4s, br7.s[2]

	b .Loop_k_end

.Loop_k_last_iteration_odd:
	// local_a_ptr = &A[lda*ik + im]
	// local_b_ptr01 = &B[ldb*in + ik]
	// local_b_ptr23 = &B[ldb*in + ik]
	// local_b_ptr45 = &B[ldb*in + ik]
	// local_b_ptr67 = &B[ldb*in + ik]

	ld2 {ar_re0.4s, ar_im0.4s}, [ local_a_ptr0 ]
	ld1 {br0.2s}, [ local_b_ptr01 ]
	add local_b_ptr01, local_b_ptr01, ldb, lsl 3
	ld1 {br1.2s}, [ local_b_ptr01 ]
	ld1 {br2.2s}, [ local_b_ptr23 ]
	add local_b_ptr23, local_b_ptr23, ldb, lsl 3
	ld1 {br3.2s}, [ local_b_ptr23 ]
	ld1 {br4.2s}, [ local_b_ptr45 ]
	add local_b_ptr45, local_b_ptr45, ldb, lsl 3
	ld1 {br5.2s}, [ local_b_ptr45 ]
	ld1 {br6.2s}, [ local_b_ptr67 ]
	add local_b_ptr67, local_b_ptr67, ldb, lsl 3
	ld1 {br7.2s}, [ local_b_ptr67 ]

	// cr0_re = ar_re * br_re - ar_im * br_im + cr0_re;
	// cr0_im = ar_re * br_im + ar_im * br_re + cr0_im;
	fmla cr0_re0.4s, ar_re0.4s, br0.s[0]
	fmla cr0_im0.4s, ar_re0.4s, br0.s[1]
	fmls cr0_re0.4s, ar_im0.4s, br0.s[1]
	fmla cr0_im0.4s, ar_im0.4s, br0.s[0]

	fmla cr0_re1.4s, ar_re0.4s, br1.s[0]
	fmla cr0_im1.4s, ar_re0.4s, br1.s[1]
	fmls cr0_re1.4s, ar_im0.4s, br1.s[1]
	fmla cr0_im1.4s, ar_im0.4s, br1.s[0]

	fmla cr0_re2.4s, ar_re0.4s, br2.s[0]
	fmla cr0_im2.4s, ar_re0.4s, br2.s[1]
	fmls cr0_re2.4s, ar_im0.4s, br2.s[1]
	fmla cr0_im2.4s, ar_im0.4s, br2.s[0]

	fmla cr0_re3.4s, ar_re0.4s, br3.s[0]
	fmla cr0_im3.4s, ar_re0.4s, br3.s[1]
	fmls cr0_re3.4s, ar_im0.4s, br3.s[1]
	fmla cr0_im3.4s, ar_im0.4s, br3.s[0]

	fmla cr0_re4.4s, ar_re0.4s, br4.s[0]
	fmla cr0_im4.4s, ar_re0.4s, br4.s[1]
	fmls cr0_re4.4s, ar_im0.4s, br4.s[1]
	fmla cr0_im4.4s, ar_im0.4s, br4.s[0]

	fmla cr0_re5.4s, ar_re0.4s, br5.s[0]
	fmla cr0_im5.4s, ar_re0.4s, br5.s[1]
	fmls cr0_re5.4s, ar_im0.4s, br5.s[1]
	fmla cr0_im5.4s, ar_im0.4s, br5.s[0]

	fmla cr0_re6.4s, ar_re0.4s, br6.s[0]
	fmla cr0_im6.4s, ar_re0.4s, br6.s[1]
	fmls cr0_re6.4s, ar_im0.4s, br6.s[1]
	fmla cr0_im6.4s, ar_im0.4s, br6.s[0]

	fmla cr0_re7.4s, ar_re0.4s, br7.s[0]
	fmla cr0_im7.4s, ar_re0.4s, br7.s[1]
	fmls cr0_re7.4s, ar_im0.4s, br7.s[1]
	fmla cr0_im7.4s, ar_im0.4s, br7.s[0]

.Loop_k_end: // }}}

#define cr1_im0 v31
#define cr1_re0 v30
#define cr1_im1 v29
#define cr1_re1 v28
#define cr1_im2 v27
#define cr1_re2 v26
#define cr1_im3 v25
#define cr1_re3 v24
#define cr1_im4 v23
#define cr1_re4 v22
#define cr1_im5 v21
#define cr1_re5 v20
#define cr1_im6 v19
#define cr1_re6 v18
// note: we run out of registers here, so start again and work back down
#define cr1_im7 v31
#define cr1_re7 v30

#define cr_im0 v16
#define cr_re0 v15
#define cr_im1 v14
#define cr_re1 v13
#define cr_im2 v12
#define cr_re2 v11
#define cr_im3 v10
#define cr_re3 v9
#define cr_im4 v8
#define cr_re4 v7
#define cr_im5 v6
#define cr_re5 v5
#define cr_im6 v4
#define cr_re6 v3
#define cr_im7 v2
#define cr_re7 v1

#define local_c_ptr01 local_b_ptr01
#define local_c_ptr23 local_b_ptr23
#define local_c_ptr45 local_b_ptr45
#define local_c_ptr67 local_b_ptr67
#define local_c_ptr_tmp local_a_ptr0


	// local_c_ptr01 = &C[ldc*in + im]
	// local_c_ptr23 = &C[ldc*in + im]
	// local_c_ptr45 = &C[ldc*in + im]
	// local_c_ptr67 = &C[ldc*in + im]
	madd local_c_ptr01, ldc, idx_n01, idx_m
	madd local_c_ptr23, ldc, idx_n23, idx_m
	madd local_c_ptr45, ldc, idx_n45, idx_m
	madd local_c_ptr67, ldc, idx_n67, idx_m
	add local_c_ptr01, c_ptr, local_c_ptr01, lsl 3
	add local_c_ptr23, c_ptr, local_c_ptr23, lsl 3
	add local_c_ptr45, c_ptr, local_c_ptr45, lsl 3
	add local_c_ptr67, c_ptr, local_c_ptr67, lsl 3

	cmp beta_is_zero, 1
	bne .Lbeta_non_zero

	cmp k_block_idx, 0
	bne .Lbeta_non_zero

.Lbeta_zero:
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

	b .Loop_m_without_beta_zero

.Loop_m_without_beta_zero:
	// already done beta multiplication, so just do the addition part

	// crr_re = cr1_re + cr_re * beta.real() - cr_im * beta.imag();
	// crr_im = cr1_im + cr_re * beta.imag() + cr_im * beta.real();
	// becomes...
	// crr_re = cr1_re + cr_re;
	// crr_im = cr1_im + cr_im;
	fadd cr1_re7.4s, cr1_re7.4s, cr_re0.4s
	fadd cr1_im7.4s, cr1_im7.4s, cr_im0.4s
	st2 {cr1_re7.4s, cr1_im7.4s}, [ local_c_ptr01 ]
	add local_c_ptr01, local_c_ptr01, ldc, lsl 3

	// Note: duplicated in both branches due to register pressure.
	fmul cr1_im7.4s, cr0_re7.4s, alpha_imag
	fmul cr1_re7.4s, cr0_re7.4s, alpha_real
	fmla cr1_im7.4s, cr0_im7.4s, alpha_real
	fmls cr1_re7.4s, cr0_im7.4s, alpha_imag
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

	add idx_m_base, idx_m_base, 4
	b .Loop_m



.Lbeta_non_zero:
	cmp k_block_idx, 0
	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re0.4s, cr0_re0.4s, alpha_real
	fmul cr1_im0.4s, cr0_re0.4s, alpha_imag
	fmls cr1_re0.4s, cr0_im0.4s, alpha_imag
	fmla cr1_im0.4s, cr0_im0.4s, alpha_real
	ld2 {cr_re0.4s, cr_im0.4s}, [ local_c_ptr01 ]
	fmul cr1_re1.4s, cr0_re1.4s, alpha_real
	fmul cr1_im1.4s, cr0_re1.4s, alpha_imag
	add local_c_ptr_tmp, local_c_ptr01, ldc, lsl 3
	fmls cr1_re1.4s, cr0_im1.4s, alpha_imag
	fmla cr1_im1.4s, cr0_im1.4s, alpha_real
	ld2 {cr_re1.4s, cr_im1.4s}, [ local_c_ptr_tmp ]
	fmul cr1_re2.4s, cr0_re2.4s, alpha_real
	fmul cr1_im2.4s, cr0_re2.4s, alpha_imag
	fmls cr1_re2.4s, cr0_im2.4s, alpha_imag
	fmla cr1_im2.4s, cr0_im2.4s, alpha_real
	ld2 {cr_re2.4s, cr_im2.4s}, [ local_c_ptr23 ]
	fmul cr1_re3.4s, cr0_re3.4s, alpha_real
	fmul cr1_im3.4s, cr0_re3.4s, alpha_imag
	add local_c_ptr_tmp, local_c_ptr23, ldc, lsl 3
	fmls cr1_re3.4s, cr0_im3.4s, alpha_imag
	fmla cr1_im3.4s, cr0_im3.4s, alpha_real
	ld2 {cr_re3.4s, cr_im3.4s}, [ local_c_ptr_tmp ]
	fmul cr1_re4.4s, cr0_re4.4s, alpha_real
	fmul cr1_im4.4s, cr0_re4.4s, alpha_imag
	fmls cr1_re4.4s, cr0_im4.4s, alpha_imag
	fmla cr1_im4.4s, cr0_im4.4s, alpha_real
	ld2 {cr_re4.4s, cr_im4.4s}, [ local_c_ptr45 ]
	fmul cr1_re5.4s, cr0_re5.4s, alpha_real
	fmul cr1_im5.4s, cr0_re5.4s, alpha_imag
	add local_c_ptr_tmp, local_c_ptr45, ldc, lsl 3
	fmls cr1_re5.4s, cr0_im5.4s, alpha_imag
	fmla cr1_im5.4s, cr0_im5.4s, alpha_real
	ld2 {cr_re5.4s, cr_im5.4s}, [ local_c_ptr_tmp ]
	fmul cr1_re6.4s, cr0_re6.4s, alpha_real
	fmul cr1_im6.4s, cr0_re6.4s, alpha_imag
	fmls cr1_re6.4s, cr0_im6.4s, alpha_imag
	fmla cr1_im6.4s, cr0_im6.4s, alpha_real
	ld2 {cr_re6.4s, cr_im6.4s}, [ local_c_ptr67 ]

	bne .Loop_m_without_beta

	// crr_re = cr1_re + cr_re * beta.real() - cr_im * beta.imag();
	// crr_im = cr1_im + cr_re * beta.imag() + cr_im * beta.real();
	fmla cr1_re7.4s, cr_re0.4s, beta_real
	fmla cr1_im7.4s, cr_re0.4s, beta_imag
	fmls cr1_re7.4s, cr_im0.4s, beta_imag
	fmla cr1_im7.4s, cr_im0.4s, beta_real
	st2 {cr1_re7.4s, cr1_im7.4s}, [ local_c_ptr01 ]
	// Note: duplicated in both branches due to register pressure.
	fmul cr1_im7.4s, cr0_re7.4s, alpha_imag
	fmul cr1_re7.4s, cr0_re7.4s, alpha_real
	add local_c_ptr_tmp, local_c_ptr67, ldc, lsl 3
	fmla cr1_im7.4s, cr0_im7.4s, alpha_real
	fmls cr1_re7.4s, cr0_im7.4s, alpha_imag
	ld2 {cr_re7.4s, cr_im7.4s}, [ local_c_ptr_tmp ]
	fmla cr1_re1.4s, cr_re1.4s, beta_real
	fmla cr1_im1.4s, cr_re1.4s, beta_imag
	add local_c_ptr01, local_c_ptr01, ldc, lsl 3
	fmls cr1_re1.4s, cr_im1.4s, beta_imag
	fmla cr1_im1.4s, cr_im1.4s, beta_real
	st2 {cr1_re1.4s, cr1_im1.4s}, [ local_c_ptr01 ]
	fmla cr1_re2.4s, cr_re2.4s, beta_real
	fmla cr1_im2.4s, cr_re2.4s, beta_imag
	fmls cr1_re2.4s, cr_im2.4s, beta_imag
	fmla cr1_im2.4s, cr_im2.4s, beta_real
	st2 {cr1_re2.4s, cr1_im2.4s}, [ local_c_ptr23 ]
	fmla cr1_re3.4s, cr_re3.4s, beta_real
	fmla cr1_im3.4s, cr_re3.4s, beta_imag
	add local_c_ptr23, local_c_ptr23, ldc, lsl 3
	fmls cr1_re3.4s, cr_im3.4s, beta_imag
	fmla cr1_im3.4s, cr_im3.4s, beta_real
	st2 {cr1_re3.4s, cr1_im3.4s}, [ local_c_ptr23 ]
	fmla cr1_re4.4s, cr_re4.4s, beta_real
	fmla cr1_im4.4s, cr_re4.4s, beta_imag
	fmls cr1_re4.4s, cr_im4.4s, beta_imag
	fmla cr1_im4.4s, cr_im4.4s, beta_real
	st2 {cr1_re4.4s, cr1_im4.4s}, [ local_c_ptr45 ]
	fmla cr1_re5.4s, cr_re5.4s, beta_real
	fmla cr1_im5.4s, cr_re5.4s, beta_imag
	add local_c_ptr45, local_c_ptr45, ldc, lsl 3
	fmls cr1_re5.4s, cr_im5.4s, beta_imag
	fmla cr1_im5.4s, cr_im5.4s, beta_real
	st2 {cr1_re5.4s, cr1_im5.4s}, [ local_c_ptr45 ]
	fmla cr1_re6.4s, cr_re6.4s, beta_real
	fmla cr1_im6.4s, cr_re6.4s, beta_imag
	fmls cr1_re6.4s, cr_im6.4s, beta_imag
	fmla cr1_im6.4s, cr_im6.4s, beta_real
	st2 {cr1_re6.4s, cr1_im6.4s}, [ local_c_ptr67 ]
	fmla cr1_re7.4s, cr_re7.4s, beta_real
	fmla cr1_im7.4s, cr_re7.4s, beta_imag
	add local_c_ptr67, local_c_ptr67, ldc, lsl 3
	fmls cr1_re7.4s, cr_im7.4s, beta_imag
	fmla cr1_im7.4s, cr_im7.4s, beta_real
	st2 {cr1_re7.4s, cr1_im7.4s}, [ local_c_ptr67 ]

	add idx_m_base, idx_m_base, 4
	b .Loop_m


.Loop_m_without_beta:
	// already done beta multiplication, so just do the addition part

	// crr_re = cr1_re + cr_re * beta.real() - cr_im * beta.imag();
	// crr_im = cr1_im + cr_re * beta.imag() + cr_im * beta.real();
	// becomes...
	// crr_re = cr1_re + cr_re;
	// crr_im = cr1_im + cr_im;
	fadd cr1_re7.4s, cr1_re7.4s, cr_re0.4s
	fadd cr1_im7.4s, cr1_im7.4s, cr_im0.4s
	st2 {cr1_re7.4s, cr1_im7.4s}, [ local_c_ptr01 ]
	add local_c_ptr01, local_c_ptr01, ldc, lsl 3

	// Note: duplicated in both branches due to register pressure.
	fmul cr1_im7.4s, cr0_re7.4s, alpha_imag
	fmul cr1_re7.4s, cr0_re7.4s, alpha_real
	add local_c_ptr_tmp, local_c_ptr67, ldc, lsl 3
	fmla cr1_im7.4s, cr0_im7.4s, alpha_real
	fmls cr1_re7.4s, cr0_im7.4s, alpha_imag
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

	add idx_m_base, idx_m_base, 4
	b .Loop_m
.Loop_m_end: // }}}
	add idx_n01, idx_n67, 2
	b .Loop_n
.Loop_n_end: // }}}
	add m_block_idx, m_block_idx, m_block_size
	mov m_block_size_max, M_BLOCK_SIZE_DEFAULT
	b .Loop_block_m
.Loop_block_m_end: // }}}
	add k_block_idx, k_block_idx, k_block_size
	mov k_block_size_max, K_BLOCK_SIZE_DEFAULT
	b .Loop_block_k
.Loop_block_k_end: // }}}

.Lcleanup:
	ldp d8, d9, [ sp, #304 ]
	ldp d10, d11, [ sp, #320 ]
	ldp d12, d13, [ sp, #336 ]
	ldp d14, d15, [ sp, #352 ]
	ldp x19, x20, [ sp, #144 ]
	ldp x21, x22, [ sp, #160 ]
	ldp x23, x24, [ sp, #176 ]
	ldp x25, x26, [ sp, #192 ]
	ldp x27, x28, [ sp, #208 ]
	ldp x29, x30, [ sp ], #384
	ret
	${epilogue(func_name)}

## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>
<%namespace file="dgemm_save_reg_block.s.inc" import="dgemm_save_reg_block"/>
<%namespace file="dgemm_save_indiv.s.inc" import="dgemm_save_indiv"/>
<%namespace file="dgemm_save_reg_block_beta1.s.inc" import="dgemm_save_reg_block_beta1"/>

A_ptr .req x1
B_ptr .req x0
C_ptr .req x2
k_size .req x3
m_size .req x5
n_size .req x4
alpha_input .req d0
beta_input .req d1

alpha_as_int .req x7
beta_as_int .req x8


A_r0 .req x9
A_ptr_persist .req x10

B_r0 .req x11
B_r1 .req x12
// x13, x14 disallowed in Arm64EC
C_r0 .req x15
C_r1 .req x16
C_r2 .req x17
// x18 may be in use for the Platform Register
C_r3 .req x19
C_r4 .req x20
C_r5 .req x21
ldc .req x6

// Safe to reuse A_r0 as t only used during stores
t .req x9

k_size_bytes .req x22
// x23, x24, x28 disallowed in Arm64EC	
a_row_idx .req x25
b_row_idx .req x26
k_idx .req x27


//#define B_PREFETCH_DIST2 #768
//#define A_PREFETCH_DIST2 #576

#define B_PREFETCH_DIST2 #1152
#define A_PREFETCH_DIST2 #768





	<% func_name = "dgemm_big_sfprfm" %>
	<% label_sfx = "" %>
	${prologue(func_name)}
	sub sp, sp, #384
	stp x29, x30, [sp]
	mov x29, sp        // This should point to the address we just stored to.

	stp x8, x9, [ sp, #64 ]
	stp x10, x11, [ sp, #80 ]
	stp x12, x13, [ sp, #96 ]
	stp x14, x15, [ sp, #112 ]
	stp x16, x17, [ sp, #128 ]
	stp x19, x20, [ sp, #144 ]
	stp x21, x22, [ sp, #160 ]
	stp x23, x24, [ sp, #176 ]
	stp x25, x26, [ sp, #192 ]
	stp x27, x28, [ sp, #208 ]

	stp d8, d9, [ sp, #304 ]
	stp d10, d11, [ sp, #320 ]
	stp d12, d13, [ sp, #336 ]
	stp d14, d15, [ sp, #352 ]

	cbz k_size, .L1${label_sfx}

	//size in bytes
	lsl k_size_bytes, k_size, #3
	lsl ldc, ldc, #3

	// k0=0
.LOOP1${label_sfx}:
	// We need to 'normalise' our alpha and beta so that
	// int(alpha) == 0 if alpha == 0.0
	fcmp alpha_input, #0.0
	fmov alpha_as_int, alpha_input
	csel alpha_as_int, alpha_as_int, xzr, ne
	// Same for beta
	fcmp beta_input, #0.0
	fmov beta_as_int, beta_input
	csel beta_as_int, beta_as_int, xzr, ne

	stp alpha_as_int, beta_as_int, [ sp, #16 ]

.LOOP2${label_sfx}:
	mov a_row_idx, xzr
.LOOP3${label_sfx}: // i (a ptr) loop
	// Set up A base pointer
	madd A_ptr_persist, a_row_idx, k_size_bytes, A_ptr     // A_ptr_base = A + (y * size)...
	// Set up C base pointer
	madd C_r0, a_row_idx, ldc, C_ptr     // C_ptr = C + (y * size)...

	add C_r1, C_r0, ldc
	add C_r2, C_r1, ldc
	add C_r3, C_r2, ldc
	add C_r4, C_r3, ldc
	add C_r5, C_r4, ldc

	mov b_row_idx, xzr
.LOOP4${label_sfx}: // j (b ptr) loop
	// Set up B pointers
	// Get the loads underway immediately.

	madd B_r0, b_row_idx, k_size_bytes, B_ptr     // B_ptr = B + (x * size)...
	add b_row_idx, b_row_idx, #8
	madd B_r1, b_row_idx, k_size_bytes, B_ptr     // B_ptr = B + (x * size)...

	cmp b_row_idx, n_size
	csel B_r1, B_ptr, B_r1, cs

	ldr q24, [B_r0]
	ldr q25, [B_r0, #16]
	ldr q26, [B_r0, #32]
	ldr q27, [B_r0, #48]


	// Set up A pointers
	mov A_r0, A_ptr_persist

	ldr q28, [A_r0]
	ldr q29, [A_r0, #16]
	ldr q30, [A_r0, #32]

	// Initialize result registers
	dup v0.2d, xzr
	prfm PLDL1KEEP, [B_r0, 64]
	//prfm	PLDL1KEEP, [B_r0, #256]
	dup v1.2d, xzr
	prfm PLDL1KEEP, [B_r0, 128]
	//prfm	PLDL1KEEP, [B_r0, #256 + 64]
	dup v2.2d, xzr
	prfm PLDL1KEEP, [B_r0, 192]
	//prfm	PLDL1KEEP, [B_r0, #256 + 128]
	dup v3.2d, xzr
	//prfm	PLDL1KEEP, [B_r0, #256 + 192]
	dup v4.2d, xzr
	dup v5.2d, xzr
	dup v6.2d, xzr
	prfm PLDL1KEEP, [A_r0, 64]
	dup v7.2d, xzr
	prfm PLDL1KEEP, [A_r0, 128]
	dup v8.2d, xzr
	prfm PLDL1KEEP, [A_r0, 192]
	dup v9.2d, xzr
	dup v10.2d, xzr
	dup v11.2d, xzr
	prfm PLDL2KEEP, [C_r0, 64]
	dup v12.2d, xzr
	prfm PLDL2KEEP, [C_r1, 64]
	dup v13.2d, xzr
	prfm PLDL2KEEP, [C_r2, 64]
	dup v14.2d, xzr
	prfm PLDL2KEEP, [C_r3, 64]
	dup v15.2d, xzr
	prfm PLDL2KEEP, [C_r4, 64]
	dup v16.2d, xzr
	prfm PLDL2KEEP, [C_r5, 64]
	dup v17.2d, xzr
	dup v18.2d, xzr
	dup v19.2d, xzr
	dup v20.2d, xzr
	dup v21.2d, xzr
	dup v22.2d, xzr
	dup v23.2d, xzr

	// Set up loop counter (n)
	mov k_idx, k_size

//	align 64
	// Inner loop
	subs  k_idx, k_idx, #1
.LOOP5${label_sfx}:
	beq .LOOP5OUTEVEN${label_sfx}

	subs k_idx, k_idx, #1

	// Unroll #0
	prfm PLDL1KEEP, [B_r1]
	fmla v0.2d, v24.2d, v28.d[0]
	fmla v1.2d, v25.2d, v28.d[0]
	fmla v2.2d, v26.2d, v28.d[0]
	prfm PLDL1KEEP, [B_r1, #64]
	fmla v3.2d, v27.2d, v28.d[0]

	fmla v4.2d, v24.2d, v28.d[1]
	fmla v5.2d, v25.2d, v28.d[1]
	prfm PLDL1KEEP, [B_r1, #128]
	fmla v6.2d, v26.2d, v28.d[1]
	fmla v7.2d, v27.2d, v28.d[1]

	fmla v8.2d, v24.2d, v29.d[0]
	prfm PLDL1KEEP, [B_r1, #192]
	fmla v9.2d, v25.2d, v29.d[0]
	fmla v10.2d, v26.2d, v29.d[0]
	fmla v11.2d, v27.2d, v29.d[0]
	ldr q28, [A_r0, #48]

	fmla v12.2d, v24.2d, v29.d[1]
	fmla v13.2d, v25.2d, v29.d[1]
	ldr q31, [A_r0, #80]
	fmla v14.2d, v26.2d, v29.d[1]
	fmla v15.2d, v27.2d, v29.d[1]
	ldr q29, [A_r0, #64]

	fmla v16.2d, v24.2d, v30.d[0]
	fmla v20.2d, v24.2d, v30.d[1]
	ldr q24, [B_r0, #64]

	fmla v17.2d, v25.2d, v30.d[0]
	fmla v21.2d, v25.2d, v30.d[1]
	ldr q25, [B_r0, #80]

	fmla v18.2d, v26.2d, v30.d[0]
	fmla v22.2d, v26.2d, v30.d[1]
	ldr q26, [B_r0, #96]

	fmla v19.2d, v27.2d, v30.d[0]
	fmla v23.2d, v27.2d, v30.d[1]
	ldr q27, [B_r0, #112]

	beq .LOOP5OUTODD${label_sfx}

	subs k_idx, k_idx, #1

	// Unroll #1
	fmla v0.2d, v24.2d, v28.d[0]
	fmla v1.2d, v25.2d, v28.d[0]

	fmla v2.2d, v26.2d, v28.d[0]
	fmla v3.2d, v27.2d, v28.d[0]

	fmla v4.2d, v24.2d, v28.d[1]
	fmla v5.2d, v25.2d, v28.d[1]
	fmla v6.2d, v26.2d, v28.d[1]
	fmla v7.2d, v27.2d, v28.d[1]

	fmla v8.2d, v24.2d, v29.d[0]
	fmla v9.2d, v25.2d, v29.d[0]
	fmla v10.2d, v26.2d, v29.d[0]
	fmla v11.2d, v27.2d, v29.d[0]
	ldr q28, [A_r0, #96]

	fmla v12.2d, v24.2d, v29.d[1]
	fmla v13.2d, v25.2d, v29.d[1]
	ldr q30, [A_r0, #128]
	fmla v14.2d, v26.2d, v29.d[1]
	fmla v15.2d, v27.2d, v29.d[1]
	ldr q29, [A_r0, #112]

	fmla v16.2d, v24.2d, v31.d[0]
	fmla v20.2d, v24.2d, v31.d[1]
	ldr q24, [B_r0, #128]

	fmla v17.2d, v25.2d, v31.d[0]
	fmla v21.2d, v25.2d, v31.d[1]
	ldr q25, [B_r0, #144]

	fmla v18.2d, v26.2d, v31.d[0]
	fmla v22.2d, v26.2d, v31.d[1]
	ldr q26, [B_r0, #160]

	fmla v19.2d, v27.2d, v31.d[0]
	fmla v23.2d, v27.2d, v31.d[1]
	ldr q27, [B_r0, #176]

	beq .LOOP5OUTEVEN${label_sfx}

	subs k_idx, k_idx, #1

	// Unroll #2
	fmla v0.2d, v24.2d, v28.d[0]
	fmla v1.2d, v25.2d, v28.d[0]
	fmla v2.2d, v26.2d, v28.d[0]
	fmla v3.2d, v27.2d, v28.d[0]

	fmla v4.2d, v24.2d, v28.d[1]
	fmla v5.2d, v25.2d, v28.d[1]
	fmla v6.2d, v26.2d, v28.d[1]
	fmla v7.2d, v27.2d, v28.d[1]

	fmla v8.2d, v24.2d, v29.d[0]
	fmla v9.2d, v25.2d, v29.d[0]
	fmla v10.2d, v26.2d, v29.d[0]
	fmla v11.2d, v27.2d, v29.d[0]
	ldr q28, [A_r0, #144]

	fmla v12.2d, v24.2d, v29.d[1]
	fmla v13.2d, v25.2d, v29.d[1]
	ldr q31, [A_r0, #176]
	fmla v14.2d, v26.2d, v29.d[1]
	fmla v15.2d, v27.2d, v29.d[1]
	ldr q29, [A_r0, #160]

	fmla v16.2d, v24.2d, v30.d[0]
	fmla v20.2d, v24.2d, v30.d[1]
	ldr q24, [B_r0, #192]

	fmla v17.2d, v25.2d, v30.d[0]
	fmla v21.2d, v25.2d, v30.d[1]
	ldr q25, [B_r0, #208]

	fmla v18.2d, v26.2d, v30.d[0]
	fmla v22.2d, v26.2d, v30.d[1]
	ldr q26, [B_r0, #224]

	fmla v19.2d, v27.2d, v30.d[0]
	fmla v23.2d, v27.2d, v30.d[1]
	ldr q27, [B_r0, #240]
	add A_r0, A_r0, #192

	beq .LOOP5OUTODD${label_sfx}

	subs k_idx, k_idx, #1

	// Unroll #3
	fmla v0.2d, v24.2d, v28.d[0]
	add B_r0, B_r0, #256
	fmla v1.2d, v25.2d, v28.d[0]
	add B_r1, B_r1, #256
	fmla v2.2d, v26.2d, v28.d[0]
	fmla v3.2d, v27.2d, v28.d[0]

	fmla v4.2d, v24.2d, v28.d[1]
	fmla v5.2d, v25.2d, v28.d[1]
	fmla v6.2d, v26.2d, v28.d[1]
	fmla v7.2d, v27.2d, v28.d[1]

	fmla v8.2d, v24.2d, v29.d[0]
	fmla v9.2d, v25.2d, v29.d[0]
	fmla v10.2d, v26.2d, v29.d[0]
	fmla v11.2d, v27.2d, v29.d[0]
	ldr q28, [A_r0]

	fmla v12.2d, v24.2d, v29.d[1]
	fmla v13.2d, v25.2d, v29.d[1]
	ldr q30, [A_r0, #32]
	fmla v14.2d, v26.2d, v29.d[1]
	fmla v15.2d, v27.2d, v29.d[1]
	ldr q29, [A_r0, #16]

	fmla v16.2d, v24.2d, v31.d[0]
	fmla v20.2d, v24.2d, v31.d[1]
	ldr q24, [B_r0]

	fmla v17.2d, v25.2d, v31.d[0]
	fmla v21.2d, v25.2d, v31.d[1]
	ldr q25, [B_r0, #16]

	fmla v18.2d, v26.2d, v31.d[0]
	fmla v22.2d, v26.2d, v31.d[1]
	ldr q26, [B_r0, #32]

	fmla v19.2d, v27.2d, v31.d[0]
	fmla v23.2d, v27.2d, v31.d[1]
	ldr q27, [B_r0, #48]

	b .LOOP5${label_sfx}

.LOOP5OUTEVEN${label_sfx}:
	fmla v0.2d, v24.2d, v28.d[0]
	prfm PLDL1KEEP, [C_r0]
	fmla v1.2d, v25.2d, v28.d[0]
	fmla v2.2d, v26.2d, v28.d[0]
	fmla v3.2d, v27.2d, v28.d[0]
	prfm PLDL1KEEP, [C_r1]

	fmla v4.2d, v24.2d, v28.d[1]
	fmla v5.2d, v25.2d, v28.d[1]
	fmla v6.2d, v26.2d, v28.d[1]
	prfm PLDL1KEEP, [C_r2]
	fmla v7.2d, v27.2d, v28.d[1]
	fmla v8.2d, v24.2d, v29.d[0]
	fmla v9.2d, v25.2d, v29.d[0]
	prfm PLDL1KEEP, [C_r3]
	fmla v10.2d, v26.2d, v29.d[0]
	fmla v11.2d, v27.2d, v29.d[0]

	fmla v12.2d, v24.2d, v29.d[1]
	prfm PLDL1KEEP, [C_r4]
	fmla v13.2d, v25.2d, v29.d[1]
	fmla v14.2d, v26.2d, v29.d[1]
	fmla v15.2d, v27.2d, v29.d[1]

	prfm PLDL1KEEP, [C_r5]
	fmla v16.2d, v24.2d, v30.d[0]
	fmla v20.2d, v24.2d, v30.d[1]

	fmla v17.2d, v25.2d, v30.d[0]
	fmla v21.2d, v25.2d, v30.d[1]

	fmla v18.2d, v26.2d, v30.d[0]
	fmla v22.2d, v26.2d, v30.d[1]

	fmla v19.2d, v27.2d, v30.d[0]
	fmla v23.2d, v27.2d, v30.d[1]

	b .LOOP5END${label_sfx}

.LOOP5OUTODD${label_sfx}:
	fmla v0.2d, v24.2d, v28.d[0]
	prfm PLDL1KEEP, [C_r0]
	fmla v1.2d, v25.2d, v28.d[0]
	fmla v2.2d, v26.2d, v28.d[0]
	fmla v3.2d, v27.2d, v28.d[0]
	prfm PLDL1KEEP, [C_r1]

	fmla v4.2d, v24.2d, v28.d[1]
	fmla v5.2d, v25.2d, v28.d[1]
	fmla v6.2d, v26.2d, v28.d[1]
	prfm PLDL1KEEP, [C_r2]
	fmla v7.2d, v27.2d, v28.d[1]
	fmla v8.2d, v24.2d, v29.d[0]
	fmla v9.2d, v25.2d, v29.d[0]
	prfm PLDL1KEEP, [C_r3]
	fmla v10.2d, v26.2d, v29.d[0]
	fmla v11.2d, v27.2d, v29.d[0]

	fmla v12.2d, v24.2d, v29.d[1]
	prfm PLDL1KEEP, [C_r4]
	fmla v13.2d, v25.2d, v29.d[1]
	fmla v14.2d, v26.2d, v29.d[1]
	fmla v15.2d, v27.2d, v29.d[1]

	prfm PLDL1KEEP, [C_r5]
	fmla v16.2d, v24.2d, v31.d[0]
	fmla v20.2d, v24.2d, v31.d[1]

	fmla v17.2d, v25.2d, v31.d[0]
	fmla v21.2d, v25.2d, v31.d[1]

	fmla v18.2d, v26.2d, v31.d[0]
	fmla v22.2d, v26.2d, v31.d[1]

	fmla v19.2d, v27.2d, v31.d[0]
	fmla v23.2d, v27.2d, v31.d[1]


.LOOP5END${label_sfx}:
	/*
	 * If we have less than 8 elements to save
	 * then we need to use the suboptimal checking store code
	 */
	//add t, b_row_idx, 8
	//cmp t, n_size
	cmp b_row_idx, n_size
	sub b_row_idx, b_row_idx, 8

	<% dgemm_save_name = "dgemm_save" %>
	bhi .L_${dgemm_save_name}_indiv

	// If beta == 1.0 the uses specific save kernel
	mov t, 0x3ff0000000000000
	cmp beta_as_int, t
	beq .L_${dgemm_save_name}_reg_block_beta1

	// Else we'll just use the normal unrolled loop
	b .L_${dgemm_save_name}_reg_block

.LOOP4END${label_sfx}:
	add	b_row_idx, b_row_idx, #8
	cmp	b_row_idx, n_size
	blt	.LOOP4${label_sfx}

	add	a_row_idx, a_row_idx, #6
	cmp	a_row_idx, m_size
	blt	.LOOP3${label_sfx}

.L1${label_sfx}:
	ldp x8, x9, [ sp, #64 ]
	ldp x10, x11, [ sp, #80 ]
	ldp x12, x13, [ sp, #96 ]
	ldp x14, x15, [ sp, #112 ]
	ldp x16, x17, [ sp, #128 ]
	ldp x19, x20, [ sp, #144 ]
	ldp x21, x22, [ sp, #160 ]
	ldp x23, x24, [ sp, #176 ]
	ldp x25, x26, [ sp, #192 ]
	ldp x27, x28, [ sp, #208 ]

	ldp d8, d9, [ sp, #304 ]
	ldp d10, d11, [ sp, #320 ]
	ldp d12, d13, [ sp, #336 ]
	ldp d14, d15, [ sp, #352 ]

	ldp   x29, x30, [sp]
	add sp, sp, #384

	ret

	${dgemm_save_reg_block(dgemm_save_name, label_sfx)}
	${dgemm_save_indiv(dgemm_save_name, label_sfx)}
	${dgemm_save_reg_block_beta1(dgemm_save_name, label_sfx)}
	${epilogue(func_name)}

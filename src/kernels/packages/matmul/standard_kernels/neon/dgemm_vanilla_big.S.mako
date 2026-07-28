## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>
<%namespace file="dgemm_save_reg_block.s.inc" import="dgemm_save_reg_block"/>
<%namespace file="dgemm_save_indiv.s.inc" import="dgemm_save_indiv"/>
<%namespace file="dgemm_save_indiv_beta1.s.inc" import="dgemm_save_indiv_beta1"/>
<%namespace file="dgemm_save_indiv_beta0.s.inc" import="dgemm_save_indiv_beta0"/>
<%namespace file="dgemm_save_indiv_beta0_alpha1.s.inc" import="dgemm_save_indiv_beta0_alpha1"/>
<%namespace file="dgemm_save_indiv_beta1_alpha1.s.inc" import="dgemm_save_indiv_beta1_alpha1"/>
<%namespace file="dgemm_save_indiv_alpha1.s.inc" import="dgemm_save_indiv_alpha1"/>
<%namespace file="dgemm_save_reg_block_beta1.s.inc" import="dgemm_save_reg_block_beta1"/>
<%namespace file="dgemm_save_reg_block_beta0.s.inc" import="dgemm_save_reg_block_beta0"/>
<%namespace file="dgemm_save_reg_block_beta0_alpha1.s.inc" import="dgemm_save_reg_block_beta0_alpha1"/>
<%namespace file="dgemm_save_reg_block_beta1_alpha1.s.inc" import="dgemm_save_reg_block_beta1_alpha1"/>
<%namespace file="dgemm_save_reg_block_alpha1.s.inc" import="dgemm_save_reg_block_alpha1"/>

A_ptr .req x0
B_ptr .req x1
C_ptr .req x2

k_size .req x3
m_size .req x4
n_size .req x5

alpha_input .req d0
beta_input .req d1

alpha_as_int .req x7
beta_as_int .req x8


B_r0 .req x9
B_ptr_persist .req x10

A_r0 .req x11

C_r0 .req x12
// x13, x14 disallowed in Arm64EC
C_r1 .req x15
C_r2 .req x16
C_r3 .req x17
// x18 may be in use for the Platform Register
C_r4 .req x19
C_r5 .req x20
ldc .req x6

t .req x21
// Safe to reuse A_r0 as tmp_one only used during stores
tmp_one .req x11

k_size_bytes .req x22
// x23, x24, x28 disallowed in Arm64EC
b_row_idx .req x25
a_row_idx .req x26
k_idx .req x27


#define A_PREFETCH_DIST #1152
#define B_PREFETCH_DIST #768

<%def name="prfm(prefetch, prfop, addr)">
%if prefetch:
	prfm	${prfop}, [${addr}]
%endif
</%def>

// cinst-> conditional instruction (depending on the unroll we are in)
// m unroll is 8, we leave all conditional instructions,
// m unroll is 6 then only leave conditional instructions involving registers of A in q{24,25,26}
// m unroll is 4 then only leave conditional instructions involving registers of A in q{24,26}
// m unroll is 2 then we omit all conditional instructions

//No need to make explicit case about q24 as it is always present by default
//(not conditionalized)
<%def name="cinst(instruction, m_unroll_id )">
	## Every instruction should be present
	%if m_unroll_id==8:
		${instruction}

	## Only instructions manipulating q24, q25, q26 should be present
	## (Leaves all instructions manipulating q27 out)
	%elif m_unroll_id==6:
		%if "q25" in instruction  :
			${instruction}
		%elif "q26" in instruction :
			${instruction}
		%elif "v25" in instruction :
			${instruction}
		%elif "v26" in instruction :
			${instruction}
		%endif

	## Only instructions manipulating q24, q25 should be present
	## (Leaves all instructions manipulating q26, q27 out)
	%elif m_unroll_id==4:
		%if "q25" in instruction  :
			${instruction}
		%elif "v25" in instruction :
			${instruction}
		%endif
	%endif
</%def>

%for prefetch, label_sfx in [ (True, ""), (False, "_no_prefetch") ]:

	<% func_name = "dgemm_vanilla_big" + label_sfx %>
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

	// Size in bytes
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
	mov b_row_idx, xzr
.LOOP3${label_sfx}: // i (a ptr) loop
	// Set up B base pointer
	madd B_ptr_persist, b_row_idx, k_size_bytes, B_ptr     // B_ptr_base = B + (y * size)...
	// Set up C base pointer
	madd C_r0, b_row_idx, ldc, C_ptr     // C_ptr = C + (y * size)...

	add C_r1, C_r0, ldc
	add C_r2, C_r1, ldc
	add C_r3, C_r2, ldc
	add C_r4, C_r3, ldc
	add C_r5, C_r4, ldc

	mov a_row_idx, xzr

.LOOP4${label_sfx}: // j (b ptr) loop
	// Set up A pointers
	// Get the loads underway immediately.

	madd A_r0, a_row_idx, k_size_bytes, A_ptr     // A_ptr = A + (x * size)...
	// Set up B pointers
	mov B_r0, B_ptr_persist
	ldr q28, [B_r0]
	ldr q29, [B_r0, #16]
	ldr q30, [B_r0, #32]

	sub	t, m_size, a_row_idx
	cmp t, #2
	ble .LOOP4${label_sfx}_2
	cmp t, #4
	ble .LOOP4${label_sfx}_4
	cmp t, #6
	ble .LOOP4${label_sfx}_6
	//b  .LOOP4${label_sfx}_8


%for opt_m in [8, 6, 4, 2]:
.LOOP4${label_sfx}_${opt_m}: // j (b ptr) loop
	ldr q24, [A_r0]
	${cinst("ldr q25, [A_r0, #16]", opt_m)}
	${cinst("ldr q26, [A_r0, #32]", opt_m)}
	${cinst("ldr q27, [A_r0, #48]", opt_m)}

	// Initialize result registers
	dup v0.2d, xzr
	${prfm(prefetch, "PLDL1KEEP", "A_r0, #256")}
	dup v1.2d, xzr
	${prfm(prefetch, "PLDL1KEEP", "A_r0, #256 + 64")}
	dup v2.2d, xzr
	${prfm(prefetch, "PLDL1KEEP", "A_r0, #256 + 128")}
	dup v3.2d, xzr
	${prfm(prefetch, "PLDL1KEEP", "A_r0, #256 + 192")}
	dup v4.2d, xzr
	dup v5.2d, xzr
	dup v6.2d, xzr
	${prfm(prefetch, "PLDL1KEEP", "B_r0, #192")}
	dup v7.2d, xzr
	${prfm(prefetch, "PLDL1KEEP", "B_r0, #192 + 64")}
	dup v8.2d, xzr
	${prfm(prefetch, "PLDL1KEEP", "B_r0, #192 + 128")}
	dup v9.2d, xzr
	dup v10.2d, xzr
	dup v11.2d, xzr
	${prfm(prefetch, "PLDL2KEEP", "C_r0, #64")}
	dup v12.2d, xzr
	${prfm(prefetch, "PLDL2KEEP", "C_r1, #64")}
	dup v13.2d, xzr
	${prfm(prefetch, "PLDL2KEEP", "C_r2, #64")}
	dup v14.2d, xzr
	${prfm(prefetch, "PLDL2KEEP", "C_r3, #64")}
	dup v15.2d, xzr
	dup v16.2d, xzr
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

.LOOP5${label_sfx}_${opt_m}:
	beq .LOOP5OUTEVEN${label_sfx}_${opt_m}

	subs k_idx, k_idx, #1

	// Unroll #0
	fmla v0.2d, v24.2d, v28.d[0]
	${prfm(prefetch, "PLDL1KEEP", "A_r0, A_PREFETCH_DIST")}
	${cinst("fmla v1.2d, v25.2d, v28.d[0]", opt_m)}
	${prfm(prefetch, "PLDL1KEEP", "A_r0, A_PREFETCH_DIST + 64")}
	${cinst("fmla v2.2d, v26.2d, v28.d[0]", opt_m)}
	${prfm(prefetch, "PLDL1KEEP", "A_r0, A_PREFETCH_DIST + 128")}
	${cinst("fmla v3.2d, v27.2d, v28.d[0]", opt_m)}

	fmla v4.2d, v24.2d, v28.d[1]
	${cinst("fmla v5.2d, v25.2d, v28.d[1]", opt_m)}
	${cinst("fmla v6.2d, v26.2d, v28.d[1]", opt_m)}
	${cinst("fmla v7.2d, v27.2d, v28.d[1]", opt_m)}

	fmla v8.2d, v24.2d, v29.d[0]
	${cinst("fmla v9.2d, v25.2d, v29.d[0]", opt_m)}
	${cinst("fmla v10.2d, v26.2d, v29.d[0]", opt_m)}
	${cinst("fmla v11.2d, v27.2d, v29.d[0]", opt_m)}
	ldr q28, [B_r0, #48]

	fmla v12.2d, v24.2d, v29.d[1]
	${cinst("fmla v13.2d, v25.2d, v29.d[1]", opt_m)}
	ldr q31, [B_r0, #80]
	${cinst("fmla v14.2d, v26.2d, v29.d[1]", opt_m)}
	${cinst("fmla v15.2d, v27.2d, v29.d[1]", opt_m)}

	ldr q29, [B_r0, #64]

	fmla v16.2d, v24.2d, v30.d[0]
	fmla v20.2d, v24.2d, v30.d[1]
	ldr q24, [A_r0, #64]
	${cinst("fmla v17.2d, v25.2d, v30.d[0]", opt_m)}
	${cinst("fmla v21.2d, v25.2d, v30.d[1]", opt_m)}
	${cinst("ldr q25, [A_r0, #80]", opt_m)}

	${cinst("fmla v18.2d, v26.2d, v30.d[0]", opt_m)}
	${cinst("fmla v22.2d, v26.2d, v30.d[1]", opt_m)}
	${cinst("ldr q26, [A_r0, #96]", opt_m)}
	${cinst("fmla v19.2d, v27.2d, v30.d[0]", opt_m)}
	${cinst("fmla v23.2d, v27.2d, v30.d[1]", opt_m)}
	${cinst("ldr q27, [A_r0, #112]", opt_m)}

	beq .LOOP5OUTODD${label_sfx}_${opt_m}

	subs k_idx, k_idx, #1

	// Unroll #1
	fmla v0.2d, v24.2d, v28.d[0]
	${prfm(prefetch, "PLDL1KEEP", "A_r0, A_PREFETCH_DIST + 192")}
	${cinst("fmla v1.2d, v25.2d, v28.d[0]", opt_m)}
	${prfm(prefetch, "PLDL1KEEP", "B_r0, B_PREFETCH_DIST")}
	${cinst("fmla v2.2d, v26.2d, v28.d[0]", opt_m)}
	${prfm(prefetch, "PLDL1KEEP", "B_r0, B_PREFETCH_DIST + 64")}
	${cinst("fmla v3.2d, v27.2d, v28.d[0]", opt_m)}

	fmla v4.2d, v24.2d, v28.d[1]
	${cinst("fmla v5.2d, v25.2d, v28.d[1]", opt_m)}
	${cinst("fmla v6.2d, v26.2d, v28.d[1]", opt_m)}
	${cinst("fmla v7.2d, v27.2d, v28.d[1]", opt_m)}


	fmla v8.2d, v24.2d, v29.d[0]
	${cinst("fmla v9.2d, v25.2d, v29.d[0]", opt_m)}
	${cinst("fmla v10.2d, v26.2d, v29.d[0]", opt_m)}
	${cinst("fmla v11.2d, v27.2d, v29.d[0]", opt_m)}
	ldr q28, [B_r0, #96]

	fmla v12.2d, v24.2d, v29.d[1]
	${cinst("fmla v13.2d, v25.2d, v29.d[1]", opt_m)}
	ldr q30, [B_r0, #128]
	${cinst("fmla v14.2d, v26.2d, v29.d[1]", opt_m)}
	${cinst("fmla v15.2d, v27.2d, v29.d[1]", opt_m)}
	ldr q29, [B_r0, #112]

	fmla v16.2d, v24.2d, v31.d[0]
	fmla v20.2d, v24.2d, v31.d[1]
	ldr q24, [A_r0, #128]
	${cinst("fmla v17.2d, v25.2d, v31.d[0]", opt_m)}
	${cinst("fmla v21.2d, v25.2d, v31.d[1]", opt_m)}
	${cinst("ldr q25, [A_r0, #144]", opt_m)}
	${cinst("fmla v18.2d, v26.2d, v31.d[0]", opt_m)}
	${cinst("fmla v22.2d, v26.2d, v31.d[1]", opt_m)}
	${cinst("ldr q26, [A_r0, #160]", opt_m)}

	${cinst("fmla v19.2d, v27.2d, v31.d[0]", opt_m)}
	${cinst("fmla v23.2d, v27.2d, v31.d[1]", opt_m)}
	${cinst("ldr q27, [A_r0, #176]", opt_m)}

	beq .LOOP5OUTEVEN${label_sfx}_${opt_m}

	subs k_idx, k_idx, #1

	// Unroll #2
	fmla v0.2d, v24.2d, v28.d[0]
	${prfm(prefetch, "PLDL1KEEP", "B_r0, B_PREFETCH_DIST + 128")}
	${cinst("fmla v1.2d, v25.2d, v28.d[0]", opt_m)}
	${cinst("fmla v2.2d, v26.2d, v28.d[0]", opt_m)}
	${cinst("fmla v3.2d, v27.2d, v28.d[0]", opt_m)}

	fmla v4.2d, v24.2d, v28.d[1]
	${cinst("fmla v5.2d, v25.2d, v28.d[1]", opt_m)}
	${cinst("fmla v6.2d, v26.2d, v28.d[1]", opt_m)}
	${cinst("fmla v7.2d, v27.2d, v28.d[1]", opt_m)}

	fmla v8.2d, v24.2d, v29.d[0]
	${cinst("fmla v9.2d, v25.2d, v29.d[0]", opt_m)}
	${cinst("fmla v10.2d, v26.2d, v29.d[0]", opt_m)}
	${cinst("fmla v11.2d, v27.2d, v29.d[0]", opt_m)}
	ldr q28, [B_r0, #144]

	fmla v12.2d, v24.2d, v29.d[1]
	${cinst("fmla v13.2d, v25.2d, v29.d[1]", opt_m)}
	ldr q31, [B_r0, #176]
	${cinst("fmla v14.2d, v26.2d, v29.d[1]", opt_m)}
	${cinst("fmla v15.2d, v27.2d, v29.d[1]", opt_m)}
	ldr q29, [B_r0, #160]

	fmla v16.2d, v24.2d, v30.d[0]
	fmla v20.2d, v24.2d, v30.d[1]
	ldr q24, [A_r0, #192]
	${cinst("fmla v17.2d, v25.2d, v30.d[0]", opt_m)}
	${cinst("fmla v21.2d, v25.2d, v30.d[1]", opt_m)}
	${cinst("ldr q25, [A_r0, #208]", opt_m)}
	${cinst("fmla v18.2d, v26.2d, v30.d[0]", opt_m)}
	${cinst("fmla v22.2d, v26.2d, v30.d[1]", opt_m)}
	${cinst("ldr q26, [A_r0, #224]", opt_m)}
	${cinst("fmla v19.2d, v27.2d, v30.d[0]", opt_m)}
	${cinst("fmla v23.2d, v27.2d, v30.d[1]", opt_m)}
	${cinst("ldr q27, [A_r0, #240]", opt_m)}
	add B_r0, B_r0, #192

	beq .LOOP5OUTODD${label_sfx}_${opt_m}

	subs k_idx, k_idx, #1

	// Unroll #3
	fmla v0.2d, v24.2d, v28.d[0]
	add A_r0, A_r0, #256
	${cinst("fmla v1.2d, v25.2d, v28.d[0]", opt_m)}
	${cinst("fmla v2.2d, v26.2d, v28.d[0]", opt_m)}
	${cinst("fmla v3.2d, v27.2d, v28.d[0]", opt_m)}

	fmla v4.2d, v24.2d, v28.d[1]
	${cinst("fmla v5.2d, v25.2d, v28.d[1]", opt_m)}
	${cinst("fmla v6.2d, v26.2d, v28.d[1]", opt_m)}
	${cinst("fmla v7.2d, v27.2d, v28.d[1]", opt_m)}

	fmla v8.2d, v24.2d, v29.d[0]
	${cinst("fmla v9.2d, v25.2d, v29.d[0]", opt_m)}
	${cinst("fmla v10.2d, v26.2d, v29.d[0]", opt_m)}
	${cinst("fmla v11.2d, v27.2d, v29.d[0]", opt_m)}
	ldr q28, [B_r0]

	fmla v12.2d, v24.2d, v29.d[1]
	${cinst("fmla v13.2d, v25.2d, v29.d[1]", opt_m)}
	ldr q30, [B_r0, #32]
	${cinst("fmla v14.2d, v26.2d, v29.d[1]", opt_m)}
	${cinst("fmla v15.2d, v27.2d, v29.d[1]", opt_m)}
	ldr q29, [B_r0, #16]

	fmla v16.2d, v24.2d, v31.d[0]
	fmla v20.2d, v24.2d, v31.d[1]
	ldr q24, [A_r0]

	${cinst("fmla v17.2d, v25.2d, v31.d[0]", opt_m)}
	${cinst("fmla v21.2d, v25.2d, v31.d[1]", opt_m)}
	${cinst("ldr q25, [A_r0, #16]", opt_m)}

	${cinst("fmla v18.2d, v26.2d, v31.d[0]", opt_m)}
	${cinst("fmla v22.2d, v26.2d, v31.d[1]", opt_m)}
	${cinst("ldr q26, [A_r0, #32]", opt_m)}

	${cinst("fmla v19.2d, v27.2d, v31.d[0]", opt_m)}
	${cinst("fmla v23.2d, v27.2d, v31.d[1]", opt_m)}
	${cinst("ldr q27, [A_r0, #48]", opt_m)}


	b .LOOP5${label_sfx}_${opt_m}

.LOOP5OUTEVEN${label_sfx}_${opt_m}:
	fmla v0.2d, v24.2d, v28.d[0]
	${cinst("fmla v1.2d, v25.2d, v28.d[0]", opt_m)}
	${cinst("fmla v2.2d, v26.2d, v28.d[0]", opt_m)}
	${cinst("fmla v3.2d, v27.2d, v28.d[0]", opt_m)}

	fmla v4.2d, v24.2d, v28.d[1]
	${cinst("fmla v5.2d, v25.2d, v28.d[1]", opt_m)}
	${cinst("fmla v6.2d, v26.2d, v28.d[1]", opt_m)}
	${cinst("fmla v7.2d, v27.2d, v28.d[1]", opt_m)}

	fmla v8.2d, v24.2d, v29.d[0]
	${cinst("fmla v9.2d, v25.2d, v29.d[0]", opt_m)}
	${cinst("fmla v10.2d, v26.2d, v29.d[0]", opt_m)}
	${cinst("fmla v11.2d, v27.2d, v29.d[0]", opt_m)}

	fmla v12.2d, v24.2d, v29.d[1]
	${cinst("fmla v13.2d, v25.2d, v29.d[1]", opt_m)}
	${cinst("fmla v14.2d, v26.2d, v29.d[1]", opt_m)}
	${cinst("fmla v15.2d, v27.2d, v29.d[1]", opt_m)}

	fmla v16.2d, v24.2d, v30.d[0]
	fmla v20.2d, v24.2d, v30.d[1]
	${cinst("fmla v17.2d, v25.2d, v30.d[0]", opt_m)}
	${cinst("fmla v21.2d, v25.2d, v30.d[1]", opt_m)}
	${cinst("fmla v18.2d, v26.2d, v30.d[0]", opt_m)}
	${cinst("fmla v22.2d, v26.2d, v30.d[1]", opt_m)}
	${cinst("fmla v19.2d, v27.2d, v30.d[0]", opt_m)}
	${cinst("fmla v23.2d, v27.2d, v30.d[1]", opt_m)}

	b .LOOP5END${label_sfx}

.LOOP5OUTODD${label_sfx}_${opt_m}:
	fmla v0.2d, v24.2d, v28.d[0]
	${cinst("fmla v1.2d, v25.2d, v28.d[0]", opt_m)}
	${cinst("fmla v2.2d, v26.2d, v28.d[0]", opt_m)}
	${cinst("fmla v3.2d, v27.2d, v28.d[0]", opt_m)}

	fmla v4.2d, v24.2d, v28.d[1]
	${cinst("fmla v5.2d, v25.2d, v28.d[1]", opt_m)}
	${cinst("fmla v6.2d, v26.2d, v28.d[1]", opt_m)}
	${cinst("fmla v7.2d, v27.2d, v28.d[1]", opt_m)}

	fmla v8.2d, v24.2d, v29.d[0]
	${cinst("fmla v9.2d, v25.2d, v29.d[0]", opt_m)}
	${cinst("fmla v10.2d, v26.2d, v29.d[0]", opt_m)}
	${cinst("fmla v11.2d, v27.2d, v29.d[0]", opt_m)}

	fmla v12.2d, v24.2d, v29.d[1]
	${cinst("fmla v13.2d, v25.2d, v29.d[1]", opt_m)}
	${cinst("fmla v14.2d, v26.2d, v29.d[1]", opt_m)}
	${cinst("fmla v15.2d, v27.2d, v29.d[1]", opt_m)}

	fmla v16.2d, v24.2d, v31.d[0]
	fmla v20.2d, v24.2d, v31.d[1]
	${cinst("fmla v17.2d, v25.2d, v31.d[0]", opt_m)}
	${cinst("fmla v21.2d, v25.2d, v31.d[1]", opt_m)}
	${cinst("fmla v18.2d, v26.2d, v31.d[0]", opt_m)}
	${cinst("fmla v22.2d, v26.2d, v31.d[1]", opt_m)}
	${cinst("fmla v19.2d, v27.2d, v31.d[0]", opt_m)}
	${cinst("fmla v23.2d, v27.2d, v31.d[1]", opt_m)}

	b .LOOP5END${label_sfx}
%endfor

.LOOP5END${label_sfx}:
	${prfm(prefetch, "PLDL2KEEP", "C_r4, #64")}
	${prfm(prefetch, "PLDL2KEEP", "C_r5, #64")}

	<% dgemm_save_name = "dgemm_save" + label_sfx %>
	// If we have less than 8 elements to save
	// then we need to use the suboptimal checking store code.
	// Note that 1.0 on float double precision is 0x3ff0000000000000
	mov tmp_one, 0x3ff0000000000000

	add t, a_row_idx, 8
	cmp t, m_size
	bhi .L_${label_sfx}_case_less_than8

	// We have at least 8 elements to save.
	// We distinguish between special cases depending
	// on the values of alpha and beta.

//Special case beta = 0
.L${label_sfx}_check_beta_0:
	cmp beta_as_int, xzr
	bne .L${label_sfx}_check_beta_1
	cmp alpha_as_int, tmp_one
	// If beta = 0 and alpha = 1, then C = A*B
	beq .L_${dgemm_save_name}_reg_block_beta0_alpha1
	// Else, if beta = 0, but alpha has a generic value, then C = alpha*A*B
	b .L_${dgemm_save_name}_reg_block_beta0

// Special case beta = 1
.L${label_sfx}_check_beta_1:
	cmp beta_as_int, tmp_one
	bne .L${label_sfx}_check_alpha_1
	cmp alpha_as_int, tmp_one
	// If beta = 1 and alpha = 1, then C = A*B + C
	beq .L_${dgemm_save_name}_reg_block_beta1_alpha1
	// Else, if beta = 1, and alpha has a generic value, then C = alpha*A*B + C
	b .L_${dgemm_save_name}_reg_block_beta1

// Special alpha = 1
.L${label_sfx}_check_alpha_1:
	cmp alpha_as_int, tmp_one
	// If alpha = 1, then C = A*B + beta*C
	beq .L_${dgemm_save_name}_reg_block_alpha1
	// Else, we have a generic case C = alpha*A*B + beta*C
	b   .L_${dgemm_save_name}_reg_block


	// If we have less than 8 elements to save
	// then we need to use the suboptimal checking store code.
	// We distinguish between special cases as well

.L_${label_sfx}_case_less_than8:
//Special case beta = 0
	cmp beta_as_int, xzr
	bne .L${label_sfx}_lt8_check_beta_1
	cmp alpha_as_int, tmp_one
	// If beta = 0 and alpha = 1, then C = A*B
	beq .L_${dgemm_save_name}_indiv_beta0_alpha1
	// Else, if beta = 0, but alpha has a generic value, then C = alpha*A*B
	b .L_${dgemm_save_name}_indiv_beta0

// Special case beta = 1
.L${label_sfx}_lt8_check_beta_1:
	cmp beta_as_int, tmp_one
	bne .L${label_sfx}_lt8_check_alpha_1
	cmp alpha_as_int, tmp_one
	// If beta = 1 and alpha = 1, then C = A*B + C
	beq .L_${dgemm_save_name}_indiv_beta1_alpha1
	// Else, if beta = 1, and alpha has a generic value, then C = alpha*A*B + C
	b .L_${dgemm_save_name}_indiv_beta1

// Special alpha = 1
.L${label_sfx}_lt8_check_alpha_1:
	cmp alpha_as_int, tmp_one
	// If alpha = 1, then C = A*B + beta*C
	beq .L_${dgemm_save_name}_indiv_alpha1
	// Else we have a generic case C = alpha*A*B + beta*C
	b .L_${dgemm_save_name}_indiv

.LOOP4END${label_sfx}:
	add a_row_idx, a_row_idx, #8
	cmp a_row_idx, m_size
	blt .LOOP4${label_sfx}

	add b_row_idx, b_row_idx, #6
	cmp b_row_idx, n_size
	blt .LOOP3${label_sfx}

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

	ldp x29, x30, [sp]
	add sp, sp, #384

	ret

	${dgemm_save_reg_block(dgemm_save_name, label_sfx)}
	${dgemm_save_indiv(dgemm_save_name, label_sfx)}
	${dgemm_save_indiv_beta1(dgemm_save_name, label_sfx)}
	${dgemm_save_indiv_beta0(dgemm_save_name, label_sfx)}
	${dgemm_save_indiv_beta0_alpha1(dgemm_save_name, label_sfx)}
	${dgemm_save_indiv_beta1_alpha1(dgemm_save_name, label_sfx)}
	${dgemm_save_indiv_alpha1(dgemm_save_name, label_sfx)}
	${dgemm_save_reg_block_beta1(dgemm_save_name, label_sfx)}
	${dgemm_save_reg_block_beta0(dgemm_save_name, label_sfx)}
	${dgemm_save_reg_block_beta0_alpha1(dgemm_save_name, label_sfx)}
	${dgemm_save_reg_block_beta1_alpha1(dgemm_save_name, label_sfx)}
	${dgemm_save_reg_block_alpha1(dgemm_save_name, label_sfx)}
	${epilogue(func_name)}
%endfor

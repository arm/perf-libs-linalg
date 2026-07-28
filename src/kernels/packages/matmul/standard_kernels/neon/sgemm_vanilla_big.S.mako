## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>
<%namespace file="sgemm_save_indiv.s.inc" import="sgemm_save_indiv"/>
<%namespace file="sgemm_save_indiv_beta1.s.inc" import="sgemm_save_indiv_beta1"/>
<%namespace file="sgemm_save_indiv_beta0.s.inc" import="sgemm_save_indiv_beta0"/>
<%namespace file="sgemm_save_indiv_beta0_alpha1.s.inc" import="sgemm_save_indiv_beta0_alpha1"/>
<%namespace file="sgemm_save_indiv_beta1_alpha1.s.inc" import="sgemm_save_indiv_beta1_alpha1"/>
<%namespace file="sgemm_save_indiv_alpha1.s.inc" import="sgemm_save_indiv_alpha1"/>

A_ptr .req x0
B_ptr .req x1
C_ptr .req x2

k_size .req x3
m_size .req x4

n_size .req x5

ldc .req x6

B_ptr_persist .req x8

A_r0 .req x9

B_r0 .req x12

b_row_idx .req x14
a_row_idx .req x15

k_size_bytes .req x16
m_size_bytes .req x17

C_r0 .req x19
C_r1 .req x20
C_r2 .req x21
C_r3 .req x22
C_r4 .req x23
C_r5 .req x24
C_r6 .req x25
C_r7 .req x26

tmp_x .req x27
tmp_w .req w27

k_idx .req x28

alpha_input .req s0
beta_input .req s1

alpha_as_int .req w10
beta_as_int .req w11

<%def name="prfm(prefetch, prfop, addr)">
%if prefetch=="P":
	prfm	${prfop}, [${addr}]
%endif
</%def>

// cinst-> conditional instruction (depending on the unroll we are in)
// m unroll is 12, we leave all conditional instructions,
// m unroll is 4 then we omit all conditional instructions
// m unroll is 8 then
// * we are only interested in the first conditional load of A
// * we are only interested in operations containing these two registers
// (typically done to q27 and q30 registers)
<%def name="cinst(instruction, m_unroll_id )">
%if m_unroll_id==12:
	${instruction}
%elif m_unroll_id==8:
	%if "q30" in instruction  :
		${instruction}
	%elif "q27" in instruction :
		${instruction}
	%elif "v30" in instruction :
		${instruction}
	%elif "v27" in instruction :
		${instruction}
	%endif
%endif
</%def>

##
## Set the prefetch distance to 4 loops ahead. Better performance could be
## squeezed by tuning these values per micro-architecture, but these values seem to
## give good results in general.
##
#define A_PREFETCH_DIST 192 * 4
#define B_PREFETCH_DIST 128 * 4

## P = Prefetching with default distances
## NP = No prefetch
%for prefetch, label_sfx in [ ("P", ""), ("NP", "_no_prefetch") ]:

	<% func_name = "sgemm_vanilla_big" + label_sfx %>
	${prologue(func_name)}
	sub sp, sp, #384
	stp x29, x30, [sp]
	mov x29, sp

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

	stp d0, d1, [ sp, #240 ]
	stp d2, d3, [ sp, #256 ]
	stp d4, d5, [ sp, #272 ]
	stp d6, d7, [ sp, #288 ]

	stp d8, d9, [ sp, #304 ]
	stp d10, d11, [ sp, #320 ]
	stp d12, d13, [ sp, #336 ]
	stp d14, d15, [ sp, #352 ]


	cbz	k_size, .L1${label_sfx}

	lsl	k_size_bytes, k_size, #2
	lsl	ldc, ldc, #2

.LOOP1${label_sfx}:
	//we need to 'normalise' our alpha and beta so that
	// int(alpha) == 0 if alpha == 0.0
	fcmp alpha_input, #0.0
	fmov alpha_as_int, alpha_input
	csel alpha_as_int, alpha_as_int, wzr, ne
	// Same for beta
	fcmp beta_input, #0.0
	fmov beta_as_int, beta_input
	csel beta_as_int, beta_as_int, wzr, ne

.LOOP2${label_sfx}:
	mov	b_row_idx, xzr

.LOOP3${label_sfx}: // i (a ptr) loop
	// Set up A base pointer
	madd	B_ptr_persist, b_row_idx, k_size_bytes, B_ptr     // B_ptr_base = A + (y * size)...

	// Set up C base pointer
	madd	C_r0, b_row_idx, ldc, C_ptr     // C_ptr = C + (y * size)...

	add	C_r1, C_r0, ldc
	add	C_r2, C_r1, ldc
	add	C_r3, C_r2, ldc
	add	C_r4, C_r3, ldc
	add	C_r5, C_r4, ldc
	add	C_r6, C_r5, ldc
	add	C_r7, C_r6, ldc

	mov	a_row_idx, xzr

.LOOP4${label_sfx}: // j (b ptr) loop
	// Set up B pointers
	// Get the loads underway immediately.

	madd	A_r0, a_row_idx, k_size_bytes, A_ptr     // A_ptr = B + (x * size)...

	// Set up B pointers
	mov	B_r0, B_ptr_persist

	ldr	q24, [B_r0]
	ldr	q25, [B_r0, #16]

	//need to make this mechanism more efficient (in harmony with what's in the end of LOOP4END)
	## //add	a_row_idx, a_row_idx, #12
	sub	tmp_x, m_size, a_row_idx
	cmp tmp_x, #4
	ble .LOOP4${label_sfx}_4
	cmp tmp_x, #8
	ble .LOOP4${label_sfx}_8
	//b  .LOOP4${label_sfx}_12

%for opt_m in [12, 8, 4]:
.LOOP4${label_sfx}_${opt_m}:

	ldr	q26, [A_r0]
	${cinst("ldr	q27, [A_r0, #16]", opt_m)}
	${cinst("ldr	q28, [A_r0, #32]", opt_m)}

	// Initialize result registers
	%for b_vrg in range(0, 8):
		dup v${b_vrg*3}.4s, wzr
		%if opt_m >= 8:
			dup v${b_vrg*3+1}.4s, wzr
			%if opt_m >= 12:
				dup v${b_vrg*3+2}.4s, wzr
			%endif
		%endif
		%if b_vrg==0:
			${prfm(prefetch, "PLDL1KEEP", "A_r0, A_PREFETCH_DIST")}
		%endif
	%endfor

	// Set up loop counter (n)
	mov k_idx, k_size

	// Inner loop
	subs k_idx, k_idx, #1

.LOOP5${label_sfx}_${opt_m}:
	beq .LOOP5OUTEVEN${label_sfx}_${opt_m}
	subs k_idx, k_idx, #1

	// Unroll 0

	fmla	v0.4s , v26.4s, v24.s[0]
	${cinst("fmla	v1.4s , v27.4s, v24.s[0]", opt_m)}
	ldr	q29, [A_r0, #48]
	${cinst("fmla	v2.4s , v28.4s, v24.s[0]", opt_m)}

	fmla	v3.4s , v26.4s, v24.s[1]
	${cinst("fmla	v4.4s , v27.4s, v24.s[1]", opt_m)}
	${prfm(prefetch, "PLDL1KEEP", "B_r0, B_PREFETCH_DIST")}
	${cinst("fmla	v5.4s , v28.4s, v24.s[1]", opt_m)}

	fmla	v6.4s , v26.4s, v24.s[2]
	${cinst("fmla	v7.4s , v27.4s, v24.s[2]", opt_m)}
	${cinst("ldr	q30, [A_r0, #64]", opt_m)}
	${cinst("fmla	v8.4s , v28.4s, v24.s[2]", opt_m)}

	fmla	v9.4s , v26.4s, v24.s[3]
	${cinst("fmla	v10.4s, v27.4s, v24.s[3]", opt_m)}
	${cinst("fmla	v11.4s, v28.4s, v24.s[3]", opt_m)}

	ldr	q24, [B_r0, #32]

	fmla	v12.4s, v26.4s, v25.s[0]
	${cinst("fmla	v13.4s, v27.4s, v25.s[0]", opt_m)}
	${cinst("fmla	v14.4s, v28.4s, v25.s[0]", opt_m)}
	${cinst("ldr	q31, [A_r0, #80]", opt_m)}

	fmla	v15.4s, v26.4s, v25.s[1]
	${cinst("fmla	v16.4s, v27.4s, v25.s[1]", opt_m)}
	${cinst("fmla	v17.4s, v28.4s, v25.s[1]", opt_m)}
	${prfm(prefetch, "PLDL1KEEP", "A_r0, A_PREFETCH_DIST + 64")}

	fmla	v18.4s, v26.4s, v25.s[2]
	${cinst("fmla	v19.4s, v27.4s, v25.s[2]", opt_m)}
	${cinst("fmla	v20.4s, v28.4s, v25.s[2]", opt_m)}

	fmla	v21.4s, v26.4s, v25.s[3]
	${cinst("fmla	v22.4s, v27.4s, v25.s[3]", opt_m)}
	${cinst("fmla	v23.4s, v28.4s, v25.s[3]", opt_m)}
	ldr	q25, [B_r0, #48]

	beq	.LOOP5OUTODD${label_sfx}_${opt_m}
	subs k_idx, k_idx, #1

	// Unroll 1

	fmla	v0.4s , v29.4s, v24.s[0]
	${cinst("fmla	v1.4s , v30.4s, v24.s[0]", opt_m)}
	${cinst("fmla	v2.4s , v31.4s, v24.s[0]", opt_m)}
	ldr	q26, [A_r0, #96]

	fmla	v3.4s , v29.4s, v24.s[1]
	${cinst("fmla	v4.4s , v30.4s, v24.s[1]", opt_m)}
	${cinst("fmla	v5.4s , v31.4s, v24.s[1]", opt_m)}
	${cinst("ldr	q27, [A_r0, #112]", opt_m)}


	fmla	v6.4s , v29.4s, v24.s[2]
	${cinst("fmla	v7.4s , v30.4s, v24.s[2]", opt_m)}
	${cinst("fmla	v8.4s , v31.4s, v24.s[2]", opt_m)}
	${cinst("ldr	q28, [A_r0, #128]", opt_m)}

	fmla	v9.4s , v29.4s, v24.s[3]
	${cinst("fmla	v10.4s, v30.4s, v24.s[3]", opt_m)}
	${cinst("fmla	v11.4s, v31.4s, v24.s[3]", opt_m)}
	ldr	q24, [B_r0, #64]

	fmla	v12.4s, v29.4s, v25.s[0]
	${cinst("fmla	v13.4s, v30.4s, v25.s[0]", opt_m)}
	${cinst("fmla	v14.4s, v31.4s, v25.s[0]", opt_m)}
	${prfm(prefetch, "PLDL1KEEP", "B_r0, B_PREFETCH_DIST + 64")}

	fmla	v15.4s, v29.4s, v25.s[1]
	${cinst("fmla	v16.4s, v30.4s, v25.s[1]", opt_m)}
	${cinst("fmla	v17.4s, v31.4s, v25.s[1]", opt_m)}
	${prfm(prefetch, "PLDL1KEEP", "A_r0, A_PREFETCH_DIST + 128")}


	fmla	v18.4s, v29.4s, v25.s[2]
	${cinst("fmla	v19.4s, v30.4s, v25.s[2]", opt_m)}
	${cinst("fmla	v20.4s, v31.4s, v25.s[2]", opt_m)}

	fmla	v21.4s, v29.4s, v25.s[3]
	${cinst("fmla	v22.4s, v30.4s, v25.s[3]", opt_m)}
	${cinst("fmla	v23.4s, v31.4s, v25.s[3]", opt_m)}
	ldr	q25, [B_r0, #80]

	beq	.LOOP5OUTEVEN${label_sfx}_${opt_m}

	subs k_idx, k_idx, #1

	// Unroll 2
	fmla	v0.4s , v26.4s, v24.s[0]
	${cinst("fmla	v1.4s , v27.4s, v24.s[0]", opt_m)}
	${cinst("fmla	v2.4s , v28.4s, v24.s[0]", opt_m)}
	ldr	q29, [A_r0, #144]

	fmla	v3.4s , v26.4s, v24.s[1]
	${cinst("fmla	v4.4s , v27.4s, v24.s[1]", opt_m)}
	${cinst("fmla	v5.4s , v28.4s, v24.s[1]", opt_m)}
	${cinst("ldr	q30, [A_r0, #160]", opt_m)}

	fmla	v6.4s , v26.4s, v24.s[2]
	${cinst("fmla	v7.4s , v27.4s, v24.s[2]", opt_m)}
	${cinst("fmla	v8.4s , v28.4s, v24.s[2]", opt_m)}

	fmla	v9.4s , v26.4s, v24.s[3]
	${cinst("fmla	v10.4s, v27.4s, v24.s[3]", opt_m)}
	${cinst("fmla	v11.4s, v28.4s, v24.s[3]", opt_m)}
	ldr	q24, [B_r0, #96]

	fmla	v12.4s, v26.4s, v25.s[0]
	${cinst("fmla	v13.4s, v27.4s, v25.s[0]", opt_m)}
	${cinst("fmla	v14.4s, v28.4s, v25.s[0]", opt_m)}
	${cinst("ldr	q31, [A_r0, #176]", opt_m)}

	fmla	v15.4s, v26.4s, v25.s[1]
	${cinst("fmla	v16.4s, v27.4s, v25.s[1]", opt_m)}
	${cinst("fmla	v17.4s, v28.4s, v25.s[1]", opt_m)}
	add	A_r0, A_r0, #192

	fmla	v18.4s, v26.4s, v25.s[2]
	${cinst("fmla	v19.4s, v27.4s, v25.s[2]", opt_m)}
	${cinst("fmla	v20.4s, v28.4s, v25.s[2]", opt_m)}

	fmla	v21.4s, v26.4s, v25.s[3]
	${cinst("fmla	v22.4s, v27.4s, v25.s[3]", opt_m)}
	${cinst("fmla	v23.4s, v28.4s, v25.s[3]", opt_m)}
	ldr	q25, [B_r0, #112]

	beq	.LOOP5OUTODD${label_sfx}_${opt_m}

	subs k_idx, k_idx, #1

	// Unroll 3

	fmla	v0.4s , v29.4s, v24.s[0]
	add	B_r0, B_r0, #128
	${cinst("fmla	v1.4s , v30.4s, v24.s[0]", opt_m)}
	${cinst("fmla	v2.4s , v31.4s, v24.s[0]", opt_m)}
	${prfm(prefetch, "PLDL1KEEP", "A_r0, A_PREFETCH_DIST")}

	fmla	v3.4s , v29.4s, v24.s[1]
	${cinst("fmla	v4.4s , v30.4s, v24.s[1]", opt_m)}
	${cinst("fmla	v5.4s , v31.4s, v24.s[1]", opt_m)}
	ldr	q26, [A_r0]

	fmla	v6.4s , v29.4s, v24.s[2]
	${cinst("fmla	v7.4s , v30.4s, v24.s[2]", opt_m)}
	${cinst("fmla	v8.4s , v31.4s, v24.s[2]", opt_m)}
	${cinst("ldr	q27, [A_r0, #16]", opt_m)}

	fmla	v9.4s , v29.4s, v24.s[3]
	${cinst("fmla	v10.4s, v30.4s, v24.s[3]", opt_m)}
	${cinst("fmla	v11.4s, v31.4s, v24.s[3]", opt_m)}
	ldr	q24, [B_r0]

	fmla	v12.4s, v29.4s, v25.s[0]
	${cinst("fmla	v13.4s, v30.4s, v25.s[0]", opt_m)}
	${cinst("fmla	v14.4s, v31.4s, v25.s[0]", opt_m)}
	${cinst("ldr	q28, [A_r0, #32]", opt_m)}

	fmla	v15.4s, v29.4s, v25.s[1]
	${cinst("fmla	v16.4s, v30.4s, v25.s[1]", opt_m)}
	${cinst("fmla	v17.4s, v31.4s, v25.s[1]", opt_m)}

	fmla	v18.4s, v29.4s, v25.s[2]
	${cinst("fmla	v19.4s, v30.4s, v25.s[2]", opt_m)}
	${cinst("fmla	v20.4s, v31.4s, v25.s[2]", opt_m)}

	fmla	v21.4s, v29.4s, v25.s[3]
	${cinst("fmla	v22.4s, v30.4s, v25.s[3]", opt_m)}
	${cinst("fmla	v23.4s, v31.4s, v25.s[3]", opt_m)}
	ldr	q25, [B_r0, 16]

	b	.LOOP5${label_sfx}_${opt_m}

.LOOP5OUTEVEN${label_sfx}_${opt_m}:
	fmla	v0.4s , v26.4s, v24.s[0]
	${cinst("fmla	v1.4s , v27.4s, v24.s[0]", opt_m)}
	${cinst("fmla	v2.4s , v28.4s, v24.s[0]", opt_m)}

	fmla	v3.4s , v26.4s, v24.s[1]
	${cinst("fmla	v4.4s , v27.4s, v24.s[1]", opt_m)}
	${cinst("fmla	v5.4s , v28.4s, v24.s[1]", opt_m)}

	fmla	v6.4s , v26.4s, v24.s[2]
	${cinst("fmla	v7.4s , v27.4s, v24.s[2]", opt_m)}
	${cinst("fmla	v8.4s , v28.4s, v24.s[2]", opt_m)}

	fmla	v9.4s , v26.4s, v24.s[3]
	${cinst("fmla	v10.4s, v27.4s, v24.s[3]", opt_m)}
	${cinst("fmla	v11.4s, v28.4s, v24.s[3]", opt_m)}

	fmla	v12.4s, v26.4s, v25.s[0]
	${cinst("fmla	v13.4s, v27.4s, v25.s[0]", opt_m)}
	${cinst("fmla	v14.4s, v28.4s, v25.s[0]", opt_m)}

	fmla	v15.4s, v26.4s, v25.s[1]
	${cinst("fmla	v16.4s, v27.4s, v25.s[1]", opt_m)}
	${cinst("fmla	v17.4s, v28.4s, v25.s[1]", opt_m)}

	fmla	v18.4s, v26.4s, v25.s[2]
	${cinst("fmla	v19.4s, v27.4s, v25.s[2]", opt_m)}
	${cinst("fmla	v20.4s, v28.4s, v25.s[2]", opt_m)}

	fmla	v21.4s, v26.4s, v25.s[3]
	${cinst("fmla	v22.4s, v27.4s, v25.s[3]", opt_m)}
	${cinst("fmla	v23.4s, v28.4s, v25.s[3]", opt_m)}

	b .LOOP5END${label_sfx}

.LOOP5OUTODD${label_sfx}_${opt_m}:

	fmla	v0.4s , v29.4s, v24.s[0]
	${cinst("fmla	v1.4s , v30.4s, v24.s[0]", opt_m)}
	${cinst("fmla	v2.4s , v31.4s, v24.s[0]", opt_m)}

	fmla	v3.4s , v29.4s, v24.s[1]
	${cinst("fmla	v4.4s , v30.4s, v24.s[1]", opt_m)}
	${cinst("fmla	v5.4s , v31.4s, v24.s[1]", opt_m)}

	fmla	v6.4s , v29.4s, v24.s[2]
	${cinst("fmla	v7.4s , v30.4s, v24.s[2]", opt_m)}
	${cinst("fmla	v8.4s , v31.4s, v24.s[2]", opt_m)}

	fmla	v9.4s , v29.4s, v24.s[3]
	${cinst("fmla	v10.4s, v30.4s, v24.s[3]", opt_m)}
	${cinst("fmla	v11.4s, v31.4s, v24.s[3]", opt_m)}

	fmla	v12.4s, v29.4s, v25.s[0]
	${cinst("fmla	v13.4s, v30.4s, v25.s[0]", opt_m)}
	${cinst("fmla	v14.4s, v31.4s, v25.s[0]", opt_m)}

	fmla	v15.4s, v29.4s, v25.s[1]
	${cinst("fmla	v16.4s, v30.4s, v25.s[1]", opt_m)}
	${cinst("fmla	v17.4s, v31.4s, v25.s[1]", opt_m)}

	fmla	v18.4s, v29.4s, v25.s[2]
	${cinst("fmla	v19.4s, v30.4s, v25.s[2]", opt_m)}
	${cinst("fmla	v20.4s, v31.4s, v25.s[2]", opt_m)}

	fmla	v21.4s, v29.4s, v25.s[3]
	${cinst("fmla	v22.4s, v30.4s, v25.s[3]", opt_m)}
	${cinst("fmla	v23.4s, v31.4s, v25.s[3]", opt_m)}

	b .LOOP5END${label_sfx}

%endfor


.LOOP5END${label_sfx}:
	<% sgemm_save_name = "sgemm_save" + label_sfx %>

	// We have 5 different kernels to write out C according to special cases
	// of Beta and Alpha.
	// there might be a more efficient way to do this!
	// Maybe look up table wita alpha and beta in the same reg?
	// Note: 1.0 in float single precision representation is 0x3f800000
	mov tmp_w, 0x3f800000

//Case 1 - Beta=0
// No mul of C content by beta
.L${label_sfx}_check_beta_0:
	cmp beta_as_int, wzr
	bne .L${label_sfx}_check_beta_1
	cmp alpha_as_int, tmp_w
	// SubCase a - Beta = 0, Alpha = 1 : C <- A.B
	// Directly overwrite contents of C with the computed dot products above
	// Without need for intermediate multiplications.
	beq .L${sgemm_save_name}_indiv_beta0_alpha1
	// SubCase b - Beta = 0, Alpha = * : C <- alpha*A.B
	// Overwrite contents of C with the scaled dot products computed
	// above. Only require use of fmla operations for this
	b .L${sgemm_save_name}_indiv_beta0

//Case 2 - Beta=1
.L${label_sfx}_check_beta_1:
	cmp beta_as_int, tmp_w
	bne .L${label_sfx}_check_alpha_1
	cmp alpha_as_int, tmp_w
	// SubCase a - Beta = 1, Alpha = 1 : C <- A.B + C
	// Accumulate dot products computed above in C contents,
	// only requiring fadd operations rather than fmul or fmla
	beq .L${sgemm_save_name}_indiv_beta1_alpha1
	// SubCase b - Beta = 1, Alpha = * : C <- alpha*A.B + C
	// Accumulate dot products computed above in C contents,
	// Without need to scale existing contents beforehand.
	// only requiring fmla operations (no need for fmla)
	b .L${sgemm_save_name}_indiv_beta1

//Case 3 - Alpha=1, Beta = * : C <- A.B + beta*C
// Use fmla involving beta rather than alpha, letting the register with
// the dot product accumulate the result and then write these out in C
// matrix
.L${label_sfx}_check_alpha_1:
	cmp alpha_as_int, tmp_w
	// Use fmla involving beta rather than alpha
	beq .L${sgemm_save_name}_indiv_alpha1
	// Default Case: Beta = *, Alpha = *
	// Only use mul of C
	b .L${sgemm_save_name}_indiv

.LOOP4END${label_sfx}:
	add	a_row_idx, a_row_idx, #12
	cmp	a_row_idx, m_size
	blt .LOOP4${label_sfx}

.LOOP4OUT${label_sfx}:
	add	b_row_idx, b_row_idx, #8
	cmp	b_row_idx, n_size
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

	ldp d0, d1, [ sp, #240 ]
	ldp d2, d3, [ sp, #256 ]
	ldp d4, d5, [ sp, #272 ]
	ldp d6, d7, [ sp, #288 ]

	ldp d8, d9, [ sp, #304 ]
	ldp d10, d11, [ sp, #320 ]
	ldp d12, d13, [ sp, #336 ]
	ldp d14, d15, [ sp, #352 ]

	ldp x29, x30, [sp]
	add sp, sp, #384

	ret

	${sgemm_save_indiv(sgemm_save_name, label_sfx)}
	${sgemm_save_indiv_beta1(sgemm_save_name, label_sfx)}
	${sgemm_save_indiv_beta0(sgemm_save_name, label_sfx)}
	${sgemm_save_indiv_beta0_alpha1(sgemm_save_name, label_sfx)}
	${sgemm_save_indiv_beta1_alpha1(sgemm_save_name, label_sfx)}
	${sgemm_save_indiv_alpha1(sgemm_save_name, label_sfx)}
	${epilogue(func_name)}
%endfor

## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%def name="hgemm_8_2_kernel_save()">
	<%
		if target_os == 'mac':
			getlo12 = lambda num : ".L_SAVE_SLOW_JMP_TBL_{}@pageoff".format(num)
			getadrp = lambda num : ".L_SAVE_SLOW_JMP_TBL_{}@page".format(num)
		elif target_os in ('linux', 'windows', 'windows_arm64ec'):
			getlo12 = lambda num : ":lo12:.L_SAVE_SLOW_JMP_TBL_{}".format(num)
			getadrp = lambda num : ".L_SAVE_SLOW_JMP_TBL_{}".format(num)
		endif
	%>

.L_SAVE_C_REG_BLOCK:
	dup b_0h, al_be.h[1]
.L_SAVE_ROW_0:
	//if this row is off the end of the C dimension then quit the save code
	cmp c_strd_rem, 0
	bls .L_SAVE_END

	cmp c_cntg_rem, 23
	//if (cntg_rem >= 24) save the whole vector out in one go
	bls .L_PARTIAL_SAVE_0


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_0_ptr, 0 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_0_0.8h, al_be.h[1]
	str a_1q, [ c_0_ptr, 0 ]


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_0_ptr, 16 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_1_0.8h, al_be.h[1]
	str a_1q, [ c_0_ptr, 16 ]


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_0_ptr, 32 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_2_0.8h, al_be.h[1]
	str a_1q, [ c_0_ptr, 32 ]


	//FAST VERSION end
	b .L_SAVE_ROW_END_0

.L_PARTIAL_SAVE_0:
	cmp c_cntg_rem, 15
	bhi .L_SAVE_1_0
	cmp c_cntg_rem, 7
	bhi .L_SAVE_0_0
	b .L_SAVE_SLOW_0

.L_SAVE_1_0:

	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_0_ptr, 16 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_1_0.8h, al_be.h[1]
	str a_1q, [ c_0_ptr, 16 ]

//If we need to save one, we need to save 0 too
.L_SAVE_0_0:

	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_0_ptr, 0 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_0_0.8h, al_be.h[1]
	str a_1q, [ c_0_ptr, 0 ]


// Save individual vector lanes.
.L_SAVE_SLOW_0:
	//load the address of the jump table
	adrp	tmpx0, ${getadrp(0)}
	//load the lower 12 bits of the address
	add	tmpx0, tmpx0, ${getlo12(0)}

	//offset into
	ldr tmpx1, [tmpx0, c_cntg_rem, lsl #3]
	br	tmpx1
.Lr_SAVE_SLOW_JMP_TBL_0:
	.data
	.p2align 3
.L_SAVE_SLOW_JMP_TBL_0:
	.quad	.L_SAVE_ROW_END_0
	.quad	.L_SAVE_SLOW_0_0
	.quad	.L_SAVE_SLOW_1_0
	.quad	.L_SAVE_SLOW_2_0
	.quad	.L_SAVE_SLOW_3_0
	.quad	.L_SAVE_SLOW_4_0
	.quad	.L_SAVE_SLOW_5_0
	.quad	.L_SAVE_SLOW_6_0
	.quad	.L_SAVE_ROW_END_0
	.quad	.L_SAVE_SLOW_8_0
	.quad	.L_SAVE_SLOW_9_0
	.quad	.L_SAVE_SLOW_10_0
	.quad	.L_SAVE_SLOW_11_0
	.quad	.L_SAVE_SLOW_12_0
	.quad	.L_SAVE_SLOW_13_0
	.quad	.L_SAVE_SLOW_14_0
	.quad	.L_SAVE_ROW_END_0
	.quad	.L_SAVE_SLOW_16_0
	.quad	.L_SAVE_SLOW_17_0
	.quad	.L_SAVE_SLOW_18_0
	.quad	.L_SAVE_SLOW_19_0
	.quad	.L_SAVE_SLOW_20_0
	.quad	.L_SAVE_SLOW_21_0
	.quad	.L_SAVE_SLOW_22_0
	.text
	.p2align 2
.L_SAVE_SLOW_22_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 44 ]
	dup a_0h, c_2_0.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 44 ]

.L_SAVE_SLOW_21_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 42 ]
	dup a_0h, c_2_0.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 42 ]

.L_SAVE_SLOW_20_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 40 ]
	dup a_0h, c_2_0.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 40 ]

.L_SAVE_SLOW_19_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 38 ]
	dup a_0h, c_2_0.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 38 ]

.L_SAVE_SLOW_18_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 36 ]
	dup a_0h, c_2_0.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 36 ]

.L_SAVE_SLOW_17_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 34 ]
	dup a_0h, c_2_0.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 34 ]

.L_SAVE_SLOW_16_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 32 ]
	dup a_0h, c_2_0.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 32 ]

	//str c_2_0h, [ c_0_ptr, 32 ]
	b .L_SAVE_ROW_END_0
	.p2align 2
.L_SAVE_SLOW_14_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 28 ]
	dup a_0h, c_1_0.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 28 ]

.L_SAVE_SLOW_13_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 26 ]
	dup a_0h, c_1_0.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 26 ]

.L_SAVE_SLOW_12_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 24 ]
	dup a_0h, c_1_0.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 24 ]

.L_SAVE_SLOW_11_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 22 ]
	dup a_0h, c_1_0.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 22 ]

.L_SAVE_SLOW_10_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 20 ]
	dup a_0h, c_1_0.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 20 ]

.L_SAVE_SLOW_9_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 18 ]
	dup a_0h, c_1_0.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 18 ]

.L_SAVE_SLOW_8_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 16 ]
	dup a_0h, c_1_0.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 16 ]

	//str c_1_0h, [ c_0_ptr, 16 ]
	b .L_SAVE_ROW_END_0
	.p2align 2
.L_SAVE_SLOW_6_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 12 ]
	dup a_0h, c_0_0.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 12 ]

.L_SAVE_SLOW_5_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 10 ]
	dup a_0h, c_0_0.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 10 ]

.L_SAVE_SLOW_4_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 8 ]
	dup a_0h, c_0_0.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 8 ]

.L_SAVE_SLOW_3_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 6 ]
	dup a_0h, c_0_0.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 6 ]

.L_SAVE_SLOW_2_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 4 ]
	dup a_0h, c_0_0.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 4 ]

.L_SAVE_SLOW_1_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 2 ]
	dup a_0h, c_0_0.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 2 ]

.L_SAVE_SLOW_0_0:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_0_ptr, 0 ]
	dup a_0h, c_0_0.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_0_ptr, 0 ]

	//str c_0_0h, [ c_0_ptr, 0 ]
	b .L_SAVE_ROW_END_0
	.p2align 2

.L_SAVE_ROW_END_0:
	//falls through to the mako'ed loop

.L_SAVE_ROW_1:
	//if this row is off the end of the C dimension then quit the save code
	cmp c_strd_rem, 1
	bls .L_SAVE_END

	cmp c_cntg_rem, 23
	//if (cntg_rem >= 24) save the whole vector out in one go
	bls .L_PARTIAL_SAVE_1


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_1_ptr, 0 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_0_1.8h, al_be.h[1]
	str a_1q, [ c_1_ptr, 0 ]


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_1_ptr, 16 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_1_1.8h, al_be.h[1]
	str a_1q, [ c_1_ptr, 16 ]


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_1_ptr, 32 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_2_1.8h, al_be.h[1]
	str a_1q, [ c_1_ptr, 32 ]


	//FAST VERSION end
	b .L_SAVE_ROW_END_1

.L_PARTIAL_SAVE_1:
	cmp c_cntg_rem, 15
	bhi .L_SAVE_1_1
	cmp c_cntg_rem, 7
	bhi .L_SAVE_0_1
	b .L_SAVE_SLOW_1

.L_SAVE_1_1:

	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_1_ptr, 16 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_1_1.8h, al_be.h[1]
	str a_1q, [ c_1_ptr, 16 ]

//If we need to save one, we need to save 0 too
.L_SAVE_0_1:

	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_1_ptr, 0 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_0_1.8h, al_be.h[1]
	str a_1q, [ c_1_ptr, 0 ]


// Save individual vector lanes.
.L_SAVE_SLOW_1:
	//load the address of the jump table
	adrp	tmpx0, ${getadrp(1)}
	//load the lower 12 bits of the address
	add	tmpx0, tmpx0, ${getlo12(1)}

	//offset into
	ldr tmpx1, [tmpx0, c_cntg_rem, lsl #3]
	br	tmpx1
.Lr_SAVE_SLOW_JMP_TBL_1:
	.data
	.p2align 3
.L_SAVE_SLOW_JMP_TBL_1:
	.quad	.L_SAVE_ROW_END_1
	.quad	.L_SAVE_SLOW_0_1
	.quad	.L_SAVE_SLOW_1_1
	.quad	.L_SAVE_SLOW_2_1
	.quad	.L_SAVE_SLOW_3_1
	.quad	.L_SAVE_SLOW_4_1
	.quad	.L_SAVE_SLOW_5_1
	.quad	.L_SAVE_SLOW_6_1
	.quad	.L_SAVE_ROW_END_1
	.quad	.L_SAVE_SLOW_8_1
	.quad	.L_SAVE_SLOW_9_1
	.quad	.L_SAVE_SLOW_10_1
	.quad	.L_SAVE_SLOW_11_1
	.quad	.L_SAVE_SLOW_12_1
	.quad	.L_SAVE_SLOW_13_1
	.quad	.L_SAVE_SLOW_14_1
	.quad	.L_SAVE_ROW_END_1
	.quad	.L_SAVE_SLOW_16_1
	.quad	.L_SAVE_SLOW_17_1
	.quad	.L_SAVE_SLOW_18_1
	.quad	.L_SAVE_SLOW_19_1
	.quad	.L_SAVE_SLOW_20_1
	.quad	.L_SAVE_SLOW_21_1
	.quad	.L_SAVE_SLOW_22_1
	.text
	.p2align 2
.L_SAVE_SLOW_22_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 44 ]
	dup a_0h, c_2_1.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 44 ]

.L_SAVE_SLOW_21_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 42 ]
	dup a_0h, c_2_1.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 42 ]

.L_SAVE_SLOW_20_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 40 ]
	dup a_0h, c_2_1.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 40 ]

.L_SAVE_SLOW_19_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 38 ]
	dup a_0h, c_2_1.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 38 ]

.L_SAVE_SLOW_18_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 36 ]
	dup a_0h, c_2_1.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 36 ]

.L_SAVE_SLOW_17_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 34 ]
	dup a_0h, c_2_1.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 34 ]

.L_SAVE_SLOW_16_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 32 ]
	dup a_0h, c_2_1.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 32 ]

	//str c_2_1h, [ c_1_ptr, 32 ]
	b .L_SAVE_ROW_END_1
	.p2align 2
.L_SAVE_SLOW_14_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 28 ]
	dup a_0h, c_1_1.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 28 ]

.L_SAVE_SLOW_13_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 26 ]
	dup a_0h, c_1_1.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 26 ]

.L_SAVE_SLOW_12_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 24 ]
	dup a_0h, c_1_1.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 24 ]

.L_SAVE_SLOW_11_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 22 ]
	dup a_0h, c_1_1.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 22 ]

.L_SAVE_SLOW_10_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 20 ]
	dup a_0h, c_1_1.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 20 ]

.L_SAVE_SLOW_9_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 18 ]
	dup a_0h, c_1_1.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 18 ]

.L_SAVE_SLOW_8_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 16 ]
	dup a_0h, c_1_1.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 16 ]

	//str c_1_1h, [ c_1_ptr, 16 ]
	b .L_SAVE_ROW_END_1
	.p2align 2
.L_SAVE_SLOW_6_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 12 ]
	dup a_0h, c_0_1.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 12 ]

.L_SAVE_SLOW_5_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 10 ]
	dup a_0h, c_0_1.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 10 ]

.L_SAVE_SLOW_4_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 8 ]
	dup a_0h, c_0_1.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 8 ]

.L_SAVE_SLOW_3_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 6 ]
	dup a_0h, c_0_1.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 6 ]

.L_SAVE_SLOW_2_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 4 ]
	dup a_0h, c_0_1.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 4 ]

.L_SAVE_SLOW_1_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 2 ]
	dup a_0h, c_0_1.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 2 ]

.L_SAVE_SLOW_0_1:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_1_ptr, 0 ]
	dup a_0h, c_0_1.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_1_ptr, 0 ]

	//str c_0_1h, [ c_1_ptr, 0 ]
	b .L_SAVE_ROW_END_1
	.p2align 2

.L_SAVE_ROW_END_1:
	//falls through to the mako'ed loop

.L_SAVE_ROW_2:
	//if this row is off the end of the C dimension then quit the save code
	cmp c_strd_rem, 2
	bls .L_SAVE_END

	cmp c_cntg_rem, 23
	//if (cntg_rem >= 24) save the whole vector out in one go
	bls .L_PARTIAL_SAVE_2


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_2_ptr, 0 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_0_2.8h, al_be.h[1]
	str a_1q, [ c_2_ptr, 0 ]


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_2_ptr, 16 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_1_2.8h, al_be.h[1]
	str a_1q, [ c_2_ptr, 16 ]


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_2_ptr, 32 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_2_2.8h, al_be.h[1]
	str a_1q, [ c_2_ptr, 32 ]


	//FAST VERSION end
	b .L_SAVE_ROW_END_2

.L_PARTIAL_SAVE_2:
	cmp c_cntg_rem, 15
	bhi .L_SAVE_1_2
	cmp c_cntg_rem, 7
	bhi .L_SAVE_0_2
	b .L_SAVE_SLOW_2

.L_SAVE_1_2:

	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_2_ptr, 16 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_1_2.8h, al_be.h[1]
	str a_1q, [ c_2_ptr, 16 ]

//If we need to save one, we need to save 0 too
.L_SAVE_0_2:

	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_2_ptr, 0 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_0_2.8h, al_be.h[1]
	str a_1q, [ c_2_ptr, 0 ]


// Save individual vector lanes.
.L_SAVE_SLOW_2:
	//load the address of the jump table
	adrp	tmpx0, ${getadrp(2)}
	//load the lower 12 bits of the address
	add	tmpx0, tmpx0, ${getlo12(2)}

	//offset into
	ldr tmpx1, [tmpx0, c_cntg_rem, lsl #3]
	br	tmpx1
.Lr_SAVE_SLOW_JMP_TBL_2:
	.data
	.p2align 3
.L_SAVE_SLOW_JMP_TBL_2:
	.quad	.L_SAVE_ROW_END_2
	.quad	.L_SAVE_SLOW_0_2
	.quad	.L_SAVE_SLOW_1_2
	.quad	.L_SAVE_SLOW_2_2
	.quad	.L_SAVE_SLOW_3_2
	.quad	.L_SAVE_SLOW_4_2
	.quad	.L_SAVE_SLOW_5_2
	.quad	.L_SAVE_SLOW_6_2
	.quad	.L_SAVE_ROW_END_2
	.quad	.L_SAVE_SLOW_8_2
	.quad	.L_SAVE_SLOW_9_2
	.quad	.L_SAVE_SLOW_10_2
	.quad	.L_SAVE_SLOW_11_2
	.quad	.L_SAVE_SLOW_12_2
	.quad	.L_SAVE_SLOW_13_2
	.quad	.L_SAVE_SLOW_14_2
	.quad	.L_SAVE_ROW_END_2
	.quad	.L_SAVE_SLOW_16_2
	.quad	.L_SAVE_SLOW_17_2
	.quad	.L_SAVE_SLOW_18_2
	.quad	.L_SAVE_SLOW_19_2
	.quad	.L_SAVE_SLOW_20_2
	.quad	.L_SAVE_SLOW_21_2
	.quad	.L_SAVE_SLOW_22_2
	.text
	.p2align 2
.L_SAVE_SLOW_22_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 44 ]
	dup a_0h, c_2_2.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 44 ]

.L_SAVE_SLOW_21_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 42 ]
	dup a_0h, c_2_2.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 42 ]

.L_SAVE_SLOW_20_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 40 ]
	dup a_0h, c_2_2.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 40 ]

.L_SAVE_SLOW_19_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 38 ]
	dup a_0h, c_2_2.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 38 ]

.L_SAVE_SLOW_18_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 36 ]
	dup a_0h, c_2_2.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 36 ]

.L_SAVE_SLOW_17_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 34 ]
	dup a_0h, c_2_2.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 34 ]

.L_SAVE_SLOW_16_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 32 ]
	dup a_0h, c_2_2.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 32 ]

	//str c_2_2h, [ c_2_ptr, 32 ]
	b .L_SAVE_ROW_END_2
	.p2align 2
.L_SAVE_SLOW_14_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 28 ]
	dup a_0h, c_1_2.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 28 ]

.L_SAVE_SLOW_13_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 26 ]
	dup a_0h, c_1_2.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 26 ]

.L_SAVE_SLOW_12_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 24 ]
	dup a_0h, c_1_2.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 24 ]

.L_SAVE_SLOW_11_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 22 ]
	dup a_0h, c_1_2.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 22 ]

.L_SAVE_SLOW_10_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 20 ]
	dup a_0h, c_1_2.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 20 ]

.L_SAVE_SLOW_9_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 18 ]
	dup a_0h, c_1_2.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 18 ]

.L_SAVE_SLOW_8_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 16 ]
	dup a_0h, c_1_2.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 16 ]

	//str c_1_2h, [ c_2_ptr, 16 ]
	b .L_SAVE_ROW_END_2
	.p2align 2
.L_SAVE_SLOW_6_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 12 ]
	dup a_0h, c_0_2.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 12 ]

.L_SAVE_SLOW_5_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 10 ]
	dup a_0h, c_0_2.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 10 ]

.L_SAVE_SLOW_4_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 8 ]
	dup a_0h, c_0_2.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 8 ]

.L_SAVE_SLOW_3_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 6 ]
	dup a_0h, c_0_2.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 6 ]

.L_SAVE_SLOW_2_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 4 ]
	dup a_0h, c_0_2.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 4 ]

.L_SAVE_SLOW_1_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 2 ]
	dup a_0h, c_0_2.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 2 ]

.L_SAVE_SLOW_0_2:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_2_ptr, 0 ]
	dup a_0h, c_0_2.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_2_ptr, 0 ]

	//str c_0_2h, [ c_2_ptr, 0 ]
	b .L_SAVE_ROW_END_2
	.p2align 2

.L_SAVE_ROW_END_2:
	//falls through to the mako'ed loop

.L_SAVE_ROW_3:
	//if this row is off the end of the C dimension then quit the save code
	cmp c_strd_rem, 3
	bls .L_SAVE_END

	cmp c_cntg_rem, 23
	//if (cntg_rem >= 24) save the whole vector out in one go
	bls .L_PARTIAL_SAVE_3


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_3_ptr, 0 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_0_3.8h, al_be.h[1]
	str a_1q, [ c_3_ptr, 0 ]


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_3_ptr, 16 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_1_3.8h, al_be.h[1]
	str a_1q, [ c_3_ptr, 16 ]


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_3_ptr, 32 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_2_3.8h, al_be.h[1]
	str a_1q, [ c_3_ptr, 32 ]


	//FAST VERSION end
	b .L_SAVE_ROW_END_3

.L_PARTIAL_SAVE_3:
	cmp c_cntg_rem, 15
	bhi .L_SAVE_1_3
	cmp c_cntg_rem, 7
	bhi .L_SAVE_0_3
	b .L_SAVE_SLOW_3

.L_SAVE_1_3:

	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_3_ptr, 16 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_1_3.8h, al_be.h[1]
	str a_1q, [ c_3_ptr, 16 ]

//If we need to save one, we need to save 0 too
.L_SAVE_0_3:

	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_3_ptr, 0 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_0_3.8h, al_be.h[1]
	str a_1q, [ c_3_ptr, 0 ]


// Save individual vector lanes.
.L_SAVE_SLOW_3:
	//load the address of the jump table
	adrp	tmpx0, ${getadrp(3)}
	//load the lower 12 bits of the address
	add	tmpx0, tmpx0, ${getlo12(3)}

	//offset into
	ldr tmpx1, [tmpx0, c_cntg_rem, lsl #3]
	br	tmpx1
.Lr_SAVE_SLOW_JMP_TBL_3:
	.data
	.p2align 3
.L_SAVE_SLOW_JMP_TBL_3:
	.quad	.L_SAVE_ROW_END_3
	.quad	.L_SAVE_SLOW_0_3
	.quad	.L_SAVE_SLOW_1_3
	.quad	.L_SAVE_SLOW_2_3
	.quad	.L_SAVE_SLOW_3_3
	.quad	.L_SAVE_SLOW_4_3
	.quad	.L_SAVE_SLOW_5_3
	.quad	.L_SAVE_SLOW_6_3
	.quad	.L_SAVE_ROW_END_3
	.quad	.L_SAVE_SLOW_8_3
	.quad	.L_SAVE_SLOW_9_3
	.quad	.L_SAVE_SLOW_10_3
	.quad	.L_SAVE_SLOW_11_3
	.quad	.L_SAVE_SLOW_12_3
	.quad	.L_SAVE_SLOW_13_3
	.quad	.L_SAVE_SLOW_14_3
	.quad	.L_SAVE_ROW_END_3
	.quad	.L_SAVE_SLOW_16_3
	.quad	.L_SAVE_SLOW_17_3
	.quad	.L_SAVE_SLOW_18_3
	.quad	.L_SAVE_SLOW_19_3
	.quad	.L_SAVE_SLOW_20_3
	.quad	.L_SAVE_SLOW_21_3
	.quad	.L_SAVE_SLOW_22_3
	.text
	.p2align 2
.L_SAVE_SLOW_22_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 44 ]
	dup a_0h, c_2_3.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 44 ]

.L_SAVE_SLOW_21_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 42 ]
	dup a_0h, c_2_3.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 42 ]

.L_SAVE_SLOW_20_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 40 ]
	dup a_0h, c_2_3.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 40 ]

.L_SAVE_SLOW_19_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 38 ]
	dup a_0h, c_2_3.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 38 ]

.L_SAVE_SLOW_18_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 36 ]
	dup a_0h, c_2_3.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 36 ]

.L_SAVE_SLOW_17_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 34 ]
	dup a_0h, c_2_3.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 34 ]

.L_SAVE_SLOW_16_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 32 ]
	dup a_0h, c_2_3.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 32 ]

	//str c_2_3h, [ c_3_ptr, 32 ]
	b .L_SAVE_ROW_END_3
	.p2align 2
.L_SAVE_SLOW_14_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 28 ]
	dup a_0h, c_1_3.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 28 ]

.L_SAVE_SLOW_13_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 26 ]
	dup a_0h, c_1_3.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 26 ]

.L_SAVE_SLOW_12_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 24 ]
	dup a_0h, c_1_3.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 24 ]

.L_SAVE_SLOW_11_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 22 ]
	dup a_0h, c_1_3.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 22 ]

.L_SAVE_SLOW_10_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 20 ]
	dup a_0h, c_1_3.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 20 ]

.L_SAVE_SLOW_9_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 18 ]
	dup a_0h, c_1_3.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 18 ]

.L_SAVE_SLOW_8_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 16 ]
	dup a_0h, c_1_3.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 16 ]

	//str c_1_3h, [ c_3_ptr, 16 ]
	b .L_SAVE_ROW_END_3
	.p2align 2
.L_SAVE_SLOW_6_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 12 ]
	dup a_0h, c_0_3.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 12 ]

.L_SAVE_SLOW_5_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 10 ]
	dup a_0h, c_0_3.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 10 ]

.L_SAVE_SLOW_4_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 8 ]
	dup a_0h, c_0_3.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 8 ]

.L_SAVE_SLOW_3_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 6 ]
	dup a_0h, c_0_3.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 6 ]

.L_SAVE_SLOW_2_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 4 ]
	dup a_0h, c_0_3.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 4 ]

.L_SAVE_SLOW_1_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 2 ]
	dup a_0h, c_0_3.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 2 ]

.L_SAVE_SLOW_0_3:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_3_ptr, 0 ]
	dup a_0h, c_0_3.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_3_ptr, 0 ]

	//str c_0_3h, [ c_3_ptr, 0 ]
	b .L_SAVE_ROW_END_3
	.p2align 2

.L_SAVE_ROW_END_3:
	//falls through to the mako'ed loop

.L_SAVE_ROW_4:
	//if this row is off the end of the C dimension then quit the save code
	cmp c_strd_rem, 4
	bls .L_SAVE_END

	cmp c_cntg_rem, 23
	//if (cntg_rem >= 24) save the whole vector out in one go
	bls .L_PARTIAL_SAVE_4


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_4_ptr, 0 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_0_4.8h, al_be.h[1]
	str a_1q, [ c_4_ptr, 0 ]


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_4_ptr, 16 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_1_4.8h, al_be.h[1]
	str a_1q, [ c_4_ptr, 16 ]


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_4_ptr, 32 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_2_4.8h, al_be.h[1]
	str a_1q, [ c_4_ptr, 32 ]


	//FAST VERSION end
	b .L_SAVE_ROW_END_4

.L_PARTIAL_SAVE_4:
	cmp c_cntg_rem, 15
	bhi .L_SAVE_1_4
	cmp c_cntg_rem, 7
	bhi .L_SAVE_0_4
	b .L_SAVE_SLOW_4

.L_SAVE_1_4:

	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_4_ptr, 16 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_1_4.8h, al_be.h[1]
	str a_1q, [ c_4_ptr, 16 ]

//If we need to save one, we need to save 0 too
.L_SAVE_0_4:

	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_4_ptr, 0 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_0_4.8h, al_be.h[1]
	str a_1q, [ c_4_ptr, 0 ]


// Save individual vector lanes.
.L_SAVE_SLOW_4:
	//load the address of the jump table
	adrp	tmpx0, ${getadrp(4)}
	//load the lower 12 bits of the address
	add	tmpx0, tmpx0, ${getlo12(4)}

	//offset into
	ldr tmpx1, [tmpx0, c_cntg_rem, lsl #3]
	br	tmpx1
.Lr_SAVE_SLOW_JMP_TBL_4:
	.data
	.p2align 3
.L_SAVE_SLOW_JMP_TBL_4:
	.quad	.L_SAVE_ROW_END_4
	.quad	.L_SAVE_SLOW_0_4
	.quad	.L_SAVE_SLOW_1_4
	.quad	.L_SAVE_SLOW_2_4
	.quad	.L_SAVE_SLOW_3_4
	.quad	.L_SAVE_SLOW_4_4
	.quad	.L_SAVE_SLOW_5_4
	.quad	.L_SAVE_SLOW_6_4
	.quad	.L_SAVE_ROW_END_4
	.quad	.L_SAVE_SLOW_8_4
	.quad	.L_SAVE_SLOW_9_4
	.quad	.L_SAVE_SLOW_10_4
	.quad	.L_SAVE_SLOW_11_4
	.quad	.L_SAVE_SLOW_12_4
	.quad	.L_SAVE_SLOW_13_4
	.quad	.L_SAVE_SLOW_14_4
	.quad	.L_SAVE_ROW_END_4
	.quad	.L_SAVE_SLOW_16_4
	.quad	.L_SAVE_SLOW_17_4
	.quad	.L_SAVE_SLOW_18_4
	.quad	.L_SAVE_SLOW_19_4
	.quad	.L_SAVE_SLOW_20_4
	.quad	.L_SAVE_SLOW_21_4
	.quad	.L_SAVE_SLOW_22_4
	.text
	.p2align 2
.L_SAVE_SLOW_22_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 44 ]
	dup a_0h, c_2_4.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 44 ]

.L_SAVE_SLOW_21_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 42 ]
	dup a_0h, c_2_4.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 42 ]

.L_SAVE_SLOW_20_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 40 ]
	dup a_0h, c_2_4.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 40 ]

.L_SAVE_SLOW_19_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 38 ]
	dup a_0h, c_2_4.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 38 ]

.L_SAVE_SLOW_18_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 36 ]
	dup a_0h, c_2_4.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 36 ]

.L_SAVE_SLOW_17_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 34 ]
	dup a_0h, c_2_4.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 34 ]

.L_SAVE_SLOW_16_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 32 ]
	dup a_0h, c_2_4.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 32 ]

	//str c_2_4h, [ c_4_ptr, 32 ]
	b .L_SAVE_ROW_END_4
	.p2align 2
.L_SAVE_SLOW_14_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 28 ]
	dup a_0h, c_1_4.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 28 ]

.L_SAVE_SLOW_13_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 26 ]
	dup a_0h, c_1_4.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 26 ]

.L_SAVE_SLOW_12_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 24 ]
	dup a_0h, c_1_4.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 24 ]

.L_SAVE_SLOW_11_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 22 ]
	dup a_0h, c_1_4.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 22 ]

.L_SAVE_SLOW_10_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 20 ]
	dup a_0h, c_1_4.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 20 ]

.L_SAVE_SLOW_9_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 18 ]
	dup a_0h, c_1_4.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 18 ]

.L_SAVE_SLOW_8_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 16 ]
	dup a_0h, c_1_4.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 16 ]

	//str c_1_4h, [ c_4_ptr, 16 ]
	b .L_SAVE_ROW_END_4
	.p2align 2
.L_SAVE_SLOW_6_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 12 ]
	dup a_0h, c_0_4.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 12 ]

.L_SAVE_SLOW_5_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 10 ]
	dup a_0h, c_0_4.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 10 ]

.L_SAVE_SLOW_4_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 8 ]
	dup a_0h, c_0_4.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 8 ]

.L_SAVE_SLOW_3_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 6 ]
	dup a_0h, c_0_4.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 6 ]

.L_SAVE_SLOW_2_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 4 ]
	dup a_0h, c_0_4.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 4 ]

.L_SAVE_SLOW_1_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 2 ]
	dup a_0h, c_0_4.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 2 ]

.L_SAVE_SLOW_0_4:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_4_ptr, 0 ]
	dup a_0h, c_0_4.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_4_ptr, 0 ]

	//str c_0_4h, [ c_4_ptr, 0 ]
	b .L_SAVE_ROW_END_4
	.p2align 2

.L_SAVE_ROW_END_4:
	//falls through to the mako'ed loop

.L_SAVE_ROW_5:
	//if this row is off the end of the C dimension then quit the save code
	cmp c_strd_rem, 5
	bls .L_SAVE_END

	cmp c_cntg_rem, 23
	//if (cntg_rem >= 24) save the whole vector out in one go
	bls .L_PARTIAL_SAVE_5


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_5_ptr, 0 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_0_5.8h, al_be.h[1]
	str a_1q, [ c_5_ptr, 0 ]


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_5_ptr, 16 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_1_5.8h, al_be.h[1]
	str a_1q, [ c_5_ptr, 16 ]


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_5_ptr, 32 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_2_5.8h, al_be.h[1]
	str a_1q, [ c_5_ptr, 32 ]


	//FAST VERSION end
	b .L_SAVE_ROW_END_5

.L_PARTIAL_SAVE_5:
	cmp c_cntg_rem, 15
	bhi .L_SAVE_1_5
	cmp c_cntg_rem, 7
	bhi .L_SAVE_0_5
	b .L_SAVE_SLOW_5

.L_SAVE_1_5:

	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_5_ptr, 16 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_1_5.8h, al_be.h[1]
	str a_1q, [ c_5_ptr, 16 ]

//If we need to save one, we need to save 0 too
.L_SAVE_0_5:

	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_5_ptr, 0 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_0_5.8h, al_be.h[1]
	str a_1q, [ c_5_ptr, 0 ]


// Save individual vector lanes.
.L_SAVE_SLOW_5:
	//load the address of the jump table
	adrp	tmpx0, ${getadrp(5)}
	//load the lower 12 bits of the address
	add	tmpx0, tmpx0, ${getlo12(5)}

	//offset into
	ldr tmpx1, [tmpx0, c_cntg_rem, lsl #3]
	br	tmpx1
.Lr_SAVE_SLOW_JMP_TBL_5:
	.data
	.p2align 3
.L_SAVE_SLOW_JMP_TBL_5:
	.quad	.L_SAVE_ROW_END_5
	.quad	.L_SAVE_SLOW_0_5
	.quad	.L_SAVE_SLOW_1_5
	.quad	.L_SAVE_SLOW_2_5
	.quad	.L_SAVE_SLOW_3_5
	.quad	.L_SAVE_SLOW_4_5
	.quad	.L_SAVE_SLOW_5_5
	.quad	.L_SAVE_SLOW_6_5
	.quad	.L_SAVE_ROW_END_5
	.quad	.L_SAVE_SLOW_8_5
	.quad	.L_SAVE_SLOW_9_5
	.quad	.L_SAVE_SLOW_10_5
	.quad	.L_SAVE_SLOW_11_5
	.quad	.L_SAVE_SLOW_12_5
	.quad	.L_SAVE_SLOW_13_5
	.quad	.L_SAVE_SLOW_14_5
	.quad	.L_SAVE_ROW_END_5
	.quad	.L_SAVE_SLOW_16_5
	.quad	.L_SAVE_SLOW_17_5
	.quad	.L_SAVE_SLOW_18_5
	.quad	.L_SAVE_SLOW_19_5
	.quad	.L_SAVE_SLOW_20_5
	.quad	.L_SAVE_SLOW_21_5
	.quad	.L_SAVE_SLOW_22_5
	.text
	.p2align 2
.L_SAVE_SLOW_22_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 44 ]
	dup a_0h, c_2_5.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 44 ]

.L_SAVE_SLOW_21_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 42 ]
	dup a_0h, c_2_5.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 42 ]

.L_SAVE_SLOW_20_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 40 ]
	dup a_0h, c_2_5.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 40 ]

.L_SAVE_SLOW_19_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 38 ]
	dup a_0h, c_2_5.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 38 ]

.L_SAVE_SLOW_18_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 36 ]
	dup a_0h, c_2_5.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 36 ]

.L_SAVE_SLOW_17_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 34 ]
	dup a_0h, c_2_5.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 34 ]

.L_SAVE_SLOW_16_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 32 ]
	dup a_0h, c_2_5.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 32 ]

	//str c_2_5h, [ c_5_ptr, 32 ]
	b .L_SAVE_ROW_END_5
	.p2align 2
.L_SAVE_SLOW_14_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 28 ]
	dup a_0h, c_1_5.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 28 ]

.L_SAVE_SLOW_13_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 26 ]
	dup a_0h, c_1_5.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 26 ]

.L_SAVE_SLOW_12_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 24 ]
	dup a_0h, c_1_5.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 24 ]

.L_SAVE_SLOW_11_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 22 ]
	dup a_0h, c_1_5.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 22 ]

.L_SAVE_SLOW_10_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 20 ]
	dup a_0h, c_1_5.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 20 ]

.L_SAVE_SLOW_9_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 18 ]
	dup a_0h, c_1_5.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 18 ]

.L_SAVE_SLOW_8_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 16 ]
	dup a_0h, c_1_5.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 16 ]

	//str c_1_5h, [ c_5_ptr, 16 ]
	b .L_SAVE_ROW_END_5
	.p2align 2
.L_SAVE_SLOW_6_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 12 ]
	dup a_0h, c_0_5.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 12 ]

.L_SAVE_SLOW_5_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 10 ]
	dup a_0h, c_0_5.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 10 ]

.L_SAVE_SLOW_4_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 8 ]
	dup a_0h, c_0_5.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 8 ]

.L_SAVE_SLOW_3_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 6 ]
	dup a_0h, c_0_5.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 6 ]

.L_SAVE_SLOW_2_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 4 ]
	dup a_0h, c_0_5.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 4 ]

.L_SAVE_SLOW_1_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 2 ]
	dup a_0h, c_0_5.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 2 ]

.L_SAVE_SLOW_0_5:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_5_ptr, 0 ]
	dup a_0h, c_0_5.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_5_ptr, 0 ]

	//str c_0_5h, [ c_5_ptr, 0 ]
	b .L_SAVE_ROW_END_5
	.p2align 2

.L_SAVE_ROW_END_5:
	//falls through to the mako'ed loop

.L_SAVE_ROW_6:
	//if this row is off the end of the C dimension then quit the save code
	cmp c_strd_rem, 6
	bls .L_SAVE_END

	cmp c_cntg_rem, 23
	//if (cntg_rem >= 24) save the whole vector out in one go
	bls .L_PARTIAL_SAVE_6


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_6_ptr, 0 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_0_6.8h, al_be.h[1]
	str a_1q, [ c_6_ptr, 0 ]


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_6_ptr, 16 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_1_6.8h, al_be.h[1]
	str a_1q, [ c_6_ptr, 16 ]


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_6_ptr, 32 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_2_6.8h, al_be.h[1]
	str a_1q, [ c_6_ptr, 32 ]


	//FAST VERSION end
	b .L_SAVE_ROW_END_6

.L_PARTIAL_SAVE_6:
	cmp c_cntg_rem, 15
	bhi .L_SAVE_1_6
	cmp c_cntg_rem, 7
	bhi .L_SAVE_0_6
	b .L_SAVE_SLOW_6

.L_SAVE_1_6:

	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_6_ptr, 16 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_1_6.8h, al_be.h[1]
	str a_1q, [ c_6_ptr, 16 ]

//If we need to save one, we need to save 0 too
.L_SAVE_0_6:

	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_6_ptr, 0 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_0_6.8h, al_be.h[1]
	str a_1q, [ c_6_ptr, 0 ]


// Save individual vector lanes.
.L_SAVE_SLOW_6:
	//load the address of the jump table
	adrp	tmpx0, ${getadrp(6)}
	//load the lower 12 bits of the address
	add	tmpx0, tmpx0, ${getlo12(6)}

	//offset into
	ldr tmpx1, [tmpx0, c_cntg_rem, lsl #3]
	br	tmpx1
.Lr_SAVE_SLOW_JMP_TBL_6:
	.data
	.p2align 3
.L_SAVE_SLOW_JMP_TBL_6:
	.quad	.L_SAVE_ROW_END_6
	.quad	.L_SAVE_SLOW_0_6
	.quad	.L_SAVE_SLOW_1_6
	.quad	.L_SAVE_SLOW_2_6
	.quad	.L_SAVE_SLOW_3_6
	.quad	.L_SAVE_SLOW_4_6
	.quad	.L_SAVE_SLOW_5_6
	.quad	.L_SAVE_SLOW_6_6
	.quad	.L_SAVE_ROW_END_6
	.quad	.L_SAVE_SLOW_8_6
	.quad	.L_SAVE_SLOW_9_6
	.quad	.L_SAVE_SLOW_10_6
	.quad	.L_SAVE_SLOW_11_6
	.quad	.L_SAVE_SLOW_12_6
	.quad	.L_SAVE_SLOW_13_6
	.quad	.L_SAVE_SLOW_14_6
	.quad	.L_SAVE_ROW_END_6
	.quad	.L_SAVE_SLOW_16_6
	.quad	.L_SAVE_SLOW_17_6
	.quad	.L_SAVE_SLOW_18_6
	.quad	.L_SAVE_SLOW_19_6
	.quad	.L_SAVE_SLOW_20_6
	.quad	.L_SAVE_SLOW_21_6
	.quad	.L_SAVE_SLOW_22_6
	.text
	.p2align 2
.L_SAVE_SLOW_22_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 44 ]
	dup a_0h, c_2_6.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 44 ]

.L_SAVE_SLOW_21_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 42 ]
	dup a_0h, c_2_6.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 42 ]

.L_SAVE_SLOW_20_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 40 ]
	dup a_0h, c_2_6.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 40 ]

.L_SAVE_SLOW_19_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 38 ]
	dup a_0h, c_2_6.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 38 ]

.L_SAVE_SLOW_18_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 36 ]
	dup a_0h, c_2_6.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 36 ]

.L_SAVE_SLOW_17_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 34 ]
	dup a_0h, c_2_6.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 34 ]

.L_SAVE_SLOW_16_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 32 ]
	dup a_0h, c_2_6.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 32 ]

	//str c_2_6h, [ c_6_ptr, 32 ]
	b .L_SAVE_ROW_END_6
	.p2align 2
.L_SAVE_SLOW_14_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 28 ]
	dup a_0h, c_1_6.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 28 ]

.L_SAVE_SLOW_13_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 26 ]
	dup a_0h, c_1_6.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 26 ]

.L_SAVE_SLOW_12_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 24 ]
	dup a_0h, c_1_6.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 24 ]

.L_SAVE_SLOW_11_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 22 ]
	dup a_0h, c_1_6.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 22 ]

.L_SAVE_SLOW_10_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 20 ]
	dup a_0h, c_1_6.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 20 ]

.L_SAVE_SLOW_9_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 18 ]
	dup a_0h, c_1_6.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 18 ]

.L_SAVE_SLOW_8_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 16 ]
	dup a_0h, c_1_6.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 16 ]

	//str c_1_6h, [ c_6_ptr, 16 ]
	b .L_SAVE_ROW_END_6
	.p2align 2
.L_SAVE_SLOW_6_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 12 ]
	dup a_0h, c_0_6.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 12 ]

.L_SAVE_SLOW_5_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 10 ]
	dup a_0h, c_0_6.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 10 ]

.L_SAVE_SLOW_4_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 8 ]
	dup a_0h, c_0_6.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 8 ]

.L_SAVE_SLOW_3_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 6 ]
	dup a_0h, c_0_6.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 6 ]

.L_SAVE_SLOW_2_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 4 ]
	dup a_0h, c_0_6.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 4 ]

.L_SAVE_SLOW_1_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 2 ]
	dup a_0h, c_0_6.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 2 ]

.L_SAVE_SLOW_0_6:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_6_ptr, 0 ]
	dup a_0h, c_0_6.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_6_ptr, 0 ]

	//str c_0_6h, [ c_6_ptr, 0 ]
	b .L_SAVE_ROW_END_6
	.p2align 2

.L_SAVE_ROW_END_6:
	//falls through to the mako'ed loop

.L_SAVE_ROW_7:
	//if this row is off the end of the C dimension then quit the save code
	cmp c_strd_rem, 7
	bls .L_SAVE_END

	cmp c_cntg_rem, 23
	//if (cntg_rem >= 24) save the whole vector out in one go
	bls .L_PARTIAL_SAVE_7


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_7_ptr, 0 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_0_7.8h, al_be.h[1]
	str a_1q, [ c_7_ptr, 0 ]


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_7_ptr, 16 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_1_7.8h, al_be.h[1]
	str a_1q, [ c_7_ptr, 16 ]


	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_7_ptr, 32 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_2_7.8h, al_be.h[1]
	str a_1q, [ c_7_ptr, 32 ]


	//FAST VERSION end
	b .L_SAVE_ROW_END_7

.L_PARTIAL_SAVE_7:
	cmp c_cntg_rem, 15
	bhi .L_SAVE_1_7
	cmp c_cntg_rem, 7
	bhi .L_SAVE_0_7
	b .L_SAVE_SLOW_7

.L_SAVE_1_7:

	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_7_ptr, 16 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_1_7.8h, al_be.h[1]
	str a_1q, [ c_7_ptr, 16 ]

//If we need to save one, we need to save 0 too
.L_SAVE_0_7:

	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_7_ptr, 0 ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_0_7.8h, al_be.h[1]
	str a_1q, [ c_7_ptr, 0 ]


// Save individual vector lanes.
.L_SAVE_SLOW_7:
	//load the address of the jump table
	adrp	tmpx0, ${getadrp(7)}
	//load the lower 12 bits of the address
	add	tmpx0, tmpx0, ${getlo12(7)}

	//offset into
	ldr tmpx1, [tmpx0, c_cntg_rem, lsl #3]
	br	tmpx1
.Lr_SAVE_SLOW_JMP_TBL_7:
	.data
	.p2align 3
.L_SAVE_SLOW_JMP_TBL_7:
	.quad	.L_SAVE_ROW_END_7
	.quad	.L_SAVE_SLOW_0_7
	.quad	.L_SAVE_SLOW_1_7
	.quad	.L_SAVE_SLOW_2_7
	.quad	.L_SAVE_SLOW_3_7
	.quad	.L_SAVE_SLOW_4_7
	.quad	.L_SAVE_SLOW_5_7
	.quad	.L_SAVE_SLOW_6_7
	.quad	.L_SAVE_ROW_END_7
	.quad	.L_SAVE_SLOW_8_7
	.quad	.L_SAVE_SLOW_9_7
	.quad	.L_SAVE_SLOW_10_7
	.quad	.L_SAVE_SLOW_11_7
	.quad	.L_SAVE_SLOW_12_7
	.quad	.L_SAVE_SLOW_13_7
	.quad	.L_SAVE_SLOW_14_7
	.quad	.L_SAVE_ROW_END_7
	.quad	.L_SAVE_SLOW_16_7
	.quad	.L_SAVE_SLOW_17_7
	.quad	.L_SAVE_SLOW_18_7
	.quad	.L_SAVE_SLOW_19_7
	.quad	.L_SAVE_SLOW_20_7
	.quad	.L_SAVE_SLOW_21_7
	.quad	.L_SAVE_SLOW_22_7
	.text
	.p2align 2
.L_SAVE_SLOW_22_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 44 ]
	dup a_0h, c_2_7.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 44 ]

.L_SAVE_SLOW_21_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 42 ]
	dup a_0h, c_2_7.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 42 ]

.L_SAVE_SLOW_20_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 40 ]
	dup a_0h, c_2_7.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 40 ]

.L_SAVE_SLOW_19_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 38 ]
	dup a_0h, c_2_7.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 38 ]

.L_SAVE_SLOW_18_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 36 ]
	dup a_0h, c_2_7.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 36 ]

.L_SAVE_SLOW_17_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 34 ]
	dup a_0h, c_2_7.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 34 ]

.L_SAVE_SLOW_16_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 32 ]
	dup a_0h, c_2_7.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 32 ]

	//str c_2_7h, [ c_7_ptr, 32 ]
	b .L_SAVE_ROW_END_7
	.p2align 2
.L_SAVE_SLOW_14_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 28 ]
	dup a_0h, c_1_7.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 28 ]

.L_SAVE_SLOW_13_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 26 ]
	dup a_0h, c_1_7.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 26 ]

.L_SAVE_SLOW_12_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 24 ]
	dup a_0h, c_1_7.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 24 ]

.L_SAVE_SLOW_11_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 22 ]
	dup a_0h, c_1_7.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 22 ]

.L_SAVE_SLOW_10_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 20 ]
	dup a_0h, c_1_7.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 20 ]

.L_SAVE_SLOW_9_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 18 ]
	dup a_0h, c_1_7.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 18 ]

.L_SAVE_SLOW_8_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 16 ]
	dup a_0h, c_1_7.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 16 ]

	//str c_1_7h, [ c_7_ptr, 16 ]
	b .L_SAVE_ROW_END_7
	.p2align 2
.L_SAVE_SLOW_6_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 12 ]
	dup a_0h, c_0_7.h[6]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 12 ]

.L_SAVE_SLOW_5_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 10 ]
	dup a_0h, c_0_7.h[5]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 10 ]

.L_SAVE_SLOW_4_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 8 ]
	dup a_0h, c_0_7.h[4]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 8 ]

.L_SAVE_SLOW_3_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 6 ]
	dup a_0h, c_0_7.h[3]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 6 ]

.L_SAVE_SLOW_2_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 4 ]
	dup a_0h, c_0_7.h[2]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 4 ]

.L_SAVE_SLOW_1_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 2 ]
	dup a_0h, c_0_7.h[1]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 2 ]

.L_SAVE_SLOW_0_7:

// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_7_ptr, 0 ]
	dup a_0h, c_0_7.h[0]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_7_ptr, 0 ]

	//str c_0_7h, [ c_7_ptr, 0 ]
	b .L_SAVE_ROW_END_7
	.p2align 2

.L_SAVE_ROW_END_7:
	//falls through to the mako'ed loop

.L_SAVE_END:
	b .L_SAVE_RETURN

</%def>

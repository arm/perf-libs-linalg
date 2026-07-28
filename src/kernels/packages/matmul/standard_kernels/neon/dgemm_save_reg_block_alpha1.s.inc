## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%def name="dgemm_save_reg_block_alpha1(dgemm_save_name, label_sfx)">
.L_${dgemm_save_name}_reg_block_alpha1:
   	//load  beta  into float reg
	ins v28.d[1], beta_as_int

	ldr	q24, [C_r0]
	ldr	q25, [C_r0, #16]
	ldr	q26, [C_r0, #32]
	ldr	q27, [C_r0, #48]

	fmla v0.2d, v24.2d, v28.d[1]
	fmla v1.2d, v25.2d, v28.d[1]
	fmla v2.2d, v26.2d, v28.d[1]
	fmla v3.2d, v27.2d, v28.d[1]

	str	q0, [C_r0]
	str	q1, [C_r0, #16]
	str	q2, [C_r0, #32]
	str	q3, [C_r0, #48]

	add	C_r0, C_r0, #64

	add	t, b_row_idx, #1
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

	ldr	q24, [C_r1]
	ldr	q25, [C_r1, #16]
	ldr	q26, [C_r1, #32]
	ldr	q27, [C_r1, #48]

	fmla v4.2d, v24.2d, v28.d[1]
	fmla v5.2d, v25.2d, v28.d[1]
	fmla v6.2d, v26.2d, v28.d[1]
	fmla v7.2d, v27.2d, v28.d[1]

	str	q4, [C_r1]
	str	q5, [C_r1, #16]
	str	q6, [C_r1, #32]
	str	q7, [C_r1, #48]

	add	C_r1, C_r1, #64

	add	t, b_row_idx, #2
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

	ldr	q24, [C_r2]
	ldr	q25, [C_r2, #16]
	ldr	q26, [C_r2, #32]
	ldr	q27, [C_r2, #48]

	fmla v8.2d, v24.2d, v28.d[1]
	fmla v9.2d, v25.2d, v28.d[1]
	fmla v10.2d, v26.2d, v28.d[1]
	fmla v11.2d, v27.2d, v28.d[1]

	str	q8, [C_r2]
	str	q9, [C_r2, #16]
	str	q10, [C_r2, #32]
	str	q11, [C_r2, #48]

	add	C_r2, C_r2, #64

	add	t, b_row_idx, #3
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

	ldr	q24, [C_r3]
	ldr	q25, [C_r3, #16]
	ldr	q26, [C_r3, #32]
	ldr	q27, [C_r3, #48]

	fmla v12.2d, v24.2d, v28.d[1]
	fmla v13.2d, v25.2d, v28.d[1]
	fmla v14.2d, v26.2d, v28.d[1]
	fmla v15.2d, v27.2d, v28.d[1]

	str	q12, [C_r3]
	str	q13, [C_r3, #16]
	str	q14, [C_r3, #32]
	str	q15, [C_r3, #48]

	add	C_r3, C_r3, #64


	add	t, b_row_idx, #4
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

	ldr	q24, [C_r4]
	ldr	q25, [C_r4, #16]
	ldr	q26, [C_r4, #32]
	ldr	q27, [C_r4, #48]

	fmla v16.2d, v24.2d, v28.d[1]
	fmla v17.2d, v25.2d, v28.d[1]
	fmla v18.2d, v26.2d, v28.d[1]
	fmla v19.2d, v27.2d, v28.d[1]

	str	q16, [C_r4]
	str	q17, [C_r4, #16]
	str	q18, [C_r4, #32]
	str	q19, [C_r4, #48]

	add	C_r4, C_r4, #64

	add	t, b_row_idx, #5
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

	ldr	q24, [C_r5]
	ldr	q25, [C_r5, #16]
	ldr	q26, [C_r5, #32]
	ldr	q27, [C_r5, #48]

	fmla v20.2d, v24.2d, v28.d[1]
	fmla v21.2d, v25.2d, v28.d[1]
	fmla v22.2d, v26.2d, v28.d[1]
	fmla v23.2d, v27.2d, v28.d[1]

	str	q20, [C_r5]
	str	q21, [C_r5, #16]
	str	q22, [C_r5, #32]
	str	q23, [C_r5, #48]

	add	C_r5, C_r5, #64

	b .LOOP4END${label_sfx}
</%def>

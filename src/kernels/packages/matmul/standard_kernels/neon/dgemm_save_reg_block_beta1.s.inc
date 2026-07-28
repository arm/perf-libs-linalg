## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%def name="dgemm_save_reg_block_beta1(dgemm_save_name, label_sfx)">
.L_${dgemm_save_name}_reg_block_beta1:
	//load alpha and beta into float regs
	ins v28.d[0], alpha_as_int
	//we do not need beta
	//ins v29.d[0], beta_as_int

	ldr	q24, [C_r0]
	ldr	q25, [C_r0, #16]
	ldr	q26, [C_r0, #32]
	ldr	q27, [C_r0, #48]

	fmla	v24.2d, v0.2d, v28.d[0]
	fmla	v25.2d, v1.2d, v28.d[0]
	fmla	v26.2d, v2.2d, v28.d[0]
	fmla	v27.2d, v3.2d, v28.d[0]

	str	q24, [C_r0]
	str	q25, [C_r0, #16]
	str	q26, [C_r0, #32]
	str	q27, [C_r0, #48]

	add	C_r0, C_r0, #64

	add	t, b_row_idx, #1
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

	ins v24.d[0], alpha_as_int

	ldr	q28, [C_r1]
	ldr	q29, [C_r1, #16]
	ldr	q30, [C_r1, #32]
	ldr	q31, [C_r1, #48]

	fmla	v28.2d, v4.2d, v24.d[0]
	fmla	v29.2d, v5.2d, v24.d[0]
	fmla	v30.2d, v6.2d, v24.d[0]
	fmla	v31.2d, v7.2d, v24.d[0]

	str	q28, [C_r1]
	str	q29, [C_r1, #16]
	str	q30, [C_r1, #32]
	str	q31, [C_r1, #48]

	add	C_r1, C_r1, #64


	add	t, b_row_idx, #2
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

	ins v28.d[0], alpha_as_int

	ldr	q24, [C_r2]
	ldr	q25, [C_r2, #16]
	ldr	q26, [C_r2, #32]
	ldr	q27, [C_r2, #48]

	fmla	v24.2d, v8.2d, v28.d[0]
	fmla	v25.2d, v9.2d, v28.d[0]
	fmla	v26.2d, v10.2d, v28.d[0]
	fmla	v27.2d, v11.2d, v28.d[0]

	str	q24, [C_r2]
	str	q25, [C_r2, #16]
	str	q26, [C_r2, #32]
	str	q27, [C_r2, #48]

	add	C_r2, C_r2, #64


	add	t, b_row_idx, #3
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

	ins v24.d[0], alpha_as_int

	ldr	q28, [C_r3]
	ldr	q29, [C_r3, #16]
	ldr	q30, [C_r3, #32]
	ldr	q31, [C_r3, #48]

	fmla	v28.2d, v12.2d, v24.d[0]
	fmla	v29.2d, v13.2d, v24.d[0]
	fmla	v30.2d, v14.2d, v24.d[0]
	fmla	v31.2d, v15.2d, v24.d[0]

	str	q28, [C_r3]
	str	q29, [C_r3, #16]
	str	q30, [C_r3, #32]
	str	q31, [C_r3, #48]

	add	C_r3, C_r3, #64

	add	t, b_row_idx, #4
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

	ins v28.d[0], alpha_as_int

	ldr	q24, [C_r4]
	ldr	q25, [C_r4, #16]
	ldr	q26, [C_r4, #32]
	ldr	q27, [C_r4, #48]

	fmla	v24.2d, v16.2d, v28.d[0]
	fmla	v25.2d, v17.2d, v28.d[0]
	fmla	v26.2d, v18.2d, v28.d[0]
	fmla	v27.2d, v19.2d, v28.d[0]

	str	q24, [C_r4]
	str	q25, [C_r4, #16]
	str	q26, [C_r4, #32]
	str	q27, [C_r4, #48]

	add	C_r4, C_r4, #64

	add	t, b_row_idx, #5
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

	ins v24.d[0], alpha_as_int

	ldr	q28, [C_r5]
	ldr	q29, [C_r5, #16]
	ldr	q30, [C_r5, #32]
	ldr	q31, [C_r5, #48]

	fmla	v28.2d, v20.2d, v24.d[0]
	fmla	v29.2d, v21.2d, v24.d[0]
	fmla	v30.2d, v22.2d, v24.d[0]
	fmla	v31.2d, v23.2d, v24.d[0]

	str	q28, [C_r5]
	str	q29, [C_r5, #16]
	str	q30, [C_r5, #32]
	str	q31, [C_r5, #48]

	add	C_r5, C_r5, #64

	b .LOOP4END${label_sfx}
</%def>

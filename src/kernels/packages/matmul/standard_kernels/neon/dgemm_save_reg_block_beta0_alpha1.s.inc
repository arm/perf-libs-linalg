## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%def name="dgemm_save_reg_block_beta0_alpha1(dgemm_save_name, label_sfx)">
.L_${dgemm_save_name}_reg_block_beta0_alpha1:
	//No need to load alpha and beta

	str	q0, [C_r0]
	str	q1, [C_r0, #16]
	str	q2, [C_r0, #32]
	str	q3, [C_r0, #48]

	add	C_r0, C_r0, #64

	add	t, b_row_idx, #1
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

	str	q4, [C_r1]
	str	q5, [C_r1, #16]
	str	q6, [C_r1, #32]
	str	q7, [C_r1, #48]

	add	C_r1, C_r1, #64

	add	t, b_row_idx, #2
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

	str	q8, [C_r2]
	str	q9, [C_r2, #16]
	str	q10, [C_r2, #32]
	str	q11, [C_r2, #48]

	add	C_r2, C_r2, #64

	add	t, b_row_idx, #3
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

	str	q12, [C_r3]
	str	q13, [C_r3, #16]
	str	q14, [C_r3, #32]
	str	q15, [C_r3, #48]

	add	C_r3, C_r3, #64

	add	t, b_row_idx, #4
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

	str	q16, [C_r4]
	str	q17, [C_r4, #16]
	str	q18, [C_r4, #32]
	str	q19, [C_r4, #48]

	add	C_r4, C_r4, #64

	add	t, b_row_idx, #5
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

	str	q20, [C_r5]
	str	q21, [C_r5, #16]
	str	q22, [C_r5, #32]
	str	q23, [C_r5, #48]

	add	C_r5, C_r5, #64

	b .LOOP4END${label_sfx}
</%def>

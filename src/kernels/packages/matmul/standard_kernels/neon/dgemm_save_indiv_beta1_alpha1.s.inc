## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%def name="dgemm_save_indiv_beta1_alpha1(dgemm_save_name, label_sfx)">
.L_${dgemm_save_name}_indiv_beta1_alpha1:

	add t, b_row_idx, #0
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r0_even${label_sfx}_beta1_alpha1:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	bls .Lc_r0_str8${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r0_str6${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r0_str4${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r0_str2${label_sfx}_beta1_alpha1

	b .Lc_r0_odd${label_sfx}_beta1_alpha1

.Lc_r0_str8${label_sfx}_beta1_alpha1:
	ldr	q27, [C_r0, #48]
	fadd v27.2d, v27.2d, v3.2d
	str	q27, [C_r0, #48]
.Lc_r0_str6${label_sfx}_beta1_alpha1:
	ldr	q26, [C_r0, #32]
	fadd  v26.2d, v26.2d, v2.2d
	str q26, [C_r0, #32]
.Lc_r0_str4${label_sfx}_beta1_alpha1:
	ldr	q25, [C_r0, #16]
	fadd v25.2d, v25.2d, v1.2d
	str q25, [C_r0, #16]
.Lc_r0_str2${label_sfx}_beta1_alpha1:
	ldr	q24, [C_r0]
	fadd    v24.2d,    v24.2d, v0.2d
	str q24, [C_r0]

.Lc_r0_odd${label_sfx}_beta1_alpha1:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r0_str1${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r0_str3${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r0_str5${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r0_str7${label_sfx}_beta1_alpha1

	b .Lc_r0_odd_end${label_sfx}_beta1_alpha1

.Lc_r0_str1${label_sfx}_beta1_alpha1:
	ldr d24, [C_r0]
	fadd d0, d0, d24
	str d0, [C_r0]
	b .Lc_r0_odd_end${label_sfx}_beta1_alpha1
.Lc_r0_str3${label_sfx}_beta1_alpha1:
	ldr d25, [C_r0, #16]
	fadd d1, d1, d25
	str d1, [C_r0, #16]
	b .Lc_r0_odd_end${label_sfx}_beta1_alpha1
.Lc_r0_str5${label_sfx}_beta1_alpha1:
	ldr d26, [C_r0, #32]
	fadd d2, d2, d26
	str d2, [C_r0, #32]
	b .Lc_r0_odd_end${label_sfx}_beta1_alpha1
.Lc_r0_str7${label_sfx}_beta1_alpha1:
	ldr d27, [C_r0, #48]
	fadd d3, d3, d27
	str d3, [C_r0, #48]
	b .Lc_r0_odd_end${label_sfx}_beta1_alpha1
.Lc_r0_odd_end${label_sfx}_beta1_alpha1:
	add	C_r0, C_r0, #64

	add t, b_row_idx, #1
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r1_even${label_sfx}_beta1_alpha1:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	bls .Lc_r1_str8${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r1_str6${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r1_str4${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r1_str2${label_sfx}_beta1_alpha1

	b .Lc_r1_odd${label_sfx}_beta1_alpha1

.Lc_r1_str8${label_sfx}_beta1_alpha1:
	ldr	q31, [C_r1, #48]
	fadd  v31.2d, v31.2d, v7.2d
	str	q31, [C_r1, #48]
.Lc_r1_str6${label_sfx}_beta1_alpha1:
	ldr	q30, [C_r1, #32]
	fadd v30.2d, v30.2d, v6.2d
	str q30, [C_r1, #32]
.Lc_r1_str4${label_sfx}_beta1_alpha1:
	ldr	q29, [C_r1, #16]
	fadd v29.2d, v29.2d, v5.2d
	str q29, [C_r1, #16]
.Lc_r1_str2${label_sfx}_beta1_alpha1:
	ldr	q28, [C_r1]
	fadd v28.2d, v28.2d, v4.2d
	str q28, [C_r1]

.Lc_r1_odd${label_sfx}_beta1_alpha1:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r1_str1${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r1_str3${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r1_str5${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r1_str7${label_sfx}_beta1_alpha1

	b .Lc_r1_odd_end${label_sfx}_beta1_alpha1

.Lc_r1_str1${label_sfx}_beta1_alpha1:
	ldr d28, [C_r1]
	fadd d4, d4, d28
	str d4, [C_r1]
	b .Lc_r1_odd_end${label_sfx}_beta1_alpha1
.Lc_r1_str3${label_sfx}_beta1_alpha1:
	ldr d29, [C_r1, #16]
	fadd d5, d5, d29
	str d5, [C_r1, #16]
	b .Lc_r1_odd_end${label_sfx}_beta1_alpha1
.Lc_r1_str5${label_sfx}_beta1_alpha1:
	ldr d30, [C_r1, #32]
	fadd d6, d6, d30
	str d6, [C_r1, #32]
	b .Lc_r1_odd_end${label_sfx}_beta1_alpha1
.Lc_r1_str7${label_sfx}_beta1_alpha1:
	ldr d31, [C_r1, #48]
	fadd d7, d7, d31
	str d7, [C_r1, #48]
	b .Lc_r1_odd_end${label_sfx}_beta1_alpha1
.Lc_r1_odd_end${label_sfx}_beta1_alpha1:
	add	C_r1, C_r1, #64

	add t, b_row_idx, #2
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r2_even${label_sfx}_beta1_alpha1:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	bls .Lc_r2_str8${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r2_str6${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r2_str4${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r2_str2${label_sfx}_beta1_alpha1

	b .Lc_r2_odd${label_sfx}_beta1_alpha1

.Lc_r2_str8${label_sfx}_beta1_alpha1:
	ldr	q27, [C_r2, #48]
	fadd v27.2d, v27.2d, v11.2d
	str	q27, [C_r2, #48]
.Lc_r2_str6${label_sfx}_beta1_alpha1:
	ldr	q26, [C_r2, #32]
	fadd    v26.2d,    v26.2d, v10.2d
	str q26, [C_r2, #32]
.Lc_r2_str4${label_sfx}_beta1_alpha1:
	ldr	q25, [C_r2, #16]
	fadd v25.2d, v25.2d, v9.2d
	str q25, [C_r2, #16]
.Lc_r2_str2${label_sfx}_beta1_alpha1:
	ldr	q24, [C_r2]
	fadd    v24.2d,    v24.2d, v8.2d
	str q24, [C_r2]

.Lc_r2_odd${label_sfx}_beta1_alpha1:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r2_str1${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r2_str3${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r2_str5${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r2_str7${label_sfx}_beta1_alpha1

	b .Lc_r2_odd_end${label_sfx}_beta1_alpha1

.Lc_r2_str1${label_sfx}_beta1_alpha1:
	ldr d24, [C_r2]
	fadd d8, d8, d24
	str d8, [C_r2]
	b .Lc_r2_odd_end${label_sfx}_beta1_alpha1
.Lc_r2_str3${label_sfx}_beta1_alpha1:
	ldr d25, [C_r2, #16]
	fadd d9, d9, d25
	str d9, [C_r2, #16]
	b .Lc_r2_odd_end${label_sfx}_beta1_alpha1
.Lc_r2_str5${label_sfx}_beta1_alpha1:
	ldr d26, [C_r2, #32]
	fadd d10, d10, d26
	str d10, [C_r2, #32]
	b .Lc_r2_odd_end${label_sfx}_beta1_alpha1
.Lc_r2_str7${label_sfx}_beta1_alpha1:
	ldr d27, [C_r2, #48]
	fadd d11, d11, d27
	str d11, [C_r2, #48]
	b .Lc_r2_odd_end${label_sfx}_beta1_alpha1
.Lc_r2_odd_end${label_sfx}_beta1_alpha1:
	add	C_r2, C_r2, #64

	add t, b_row_idx, #3
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r3_even${label_sfx}_beta1_alpha1:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	bls .Lc_r3_str8${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r3_str6${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r3_str4${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r3_str2${label_sfx}_beta1_alpha1

	b .Lc_r3_odd${label_sfx}_beta1_alpha1

.Lc_r3_str8${label_sfx}_beta1_alpha1:
	ldr	q31, [C_r3, #48]
	fadd    v31.2d,    v31.2d, v15.2d
	str	q31, [C_r3, #48]
.Lc_r3_str6${label_sfx}_beta1_alpha1:
	ldr	q30, [C_r3, #32]
	fadd    v30.2d,    v30.2d, v14.2d
	str q30, [C_r3, #32]
.Lc_r3_str4${label_sfx}_beta1_alpha1:
	ldr	q29, [C_r3, #16]
	fadd v29.2d, v29.2d, v13.2d
	str q29, [C_r3, #16]
.Lc_r3_str2${label_sfx}_beta1_alpha1:
	ldr	q28, [C_r3]
	fadd    v28.2d,    v28.2d, v12.2d
	str q28, [C_r3]

.Lc_r3_odd${label_sfx}_beta1_alpha1:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r3_str1${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r3_str3${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r3_str5${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r3_str7${label_sfx}_beta1_alpha1

	b .Lc_r3_odd_end${label_sfx}_beta1_alpha1

.Lc_r3_str1${label_sfx}_beta1_alpha1:
	ldr d28, [C_r3]
	fadd d12, d12, d28
	str d12, [C_r3]
	b .Lc_r3_odd_end${label_sfx}_beta1_alpha1
.Lc_r3_str3${label_sfx}_beta1_alpha1:
	ldr d29, [C_r3, #16]
	fadd d13, d13, d29
	str d13, [C_r3, #16]
	b .Lc_r3_odd_end${label_sfx}_beta1_alpha1
.Lc_r3_str5${label_sfx}_beta1_alpha1:
	ldr d30, [C_r3, #32]
	fadd d14, d14, d30
	str d14, [C_r3, #32]
	b .Lc_r3_odd_end${label_sfx}_beta1_alpha1
.Lc_r3_str7${label_sfx}_beta1_alpha1:
	ldr d31, [C_r3, #48]
	fadd d15, d15, d31
	str d15, [C_r3, #48]
	b .Lc_r3_odd_end${label_sfx}_beta1_alpha1
.Lc_r3_odd_end${label_sfx}_beta1_alpha1:
	add	C_r3, C_r3, #64

	add t, b_row_idx, #4
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r4_even${label_sfx}_beta1_alpha1:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	bls .Lc_r4_str8${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r4_str6${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r4_str4${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r4_str2${label_sfx}_beta1_alpha1

	b .Lc_r4_odd${label_sfx}_beta1_alpha1

.Lc_r4_str8${label_sfx}_beta1_alpha1:
	ldr	q27, [C_r4, #48]
	fadd    v27.2d,    v27.2d, v19.2d
	str	q27, [C_r4, #48]
.Lc_r4_str6${label_sfx}_beta1_alpha1:
	ldr	q26, [C_r4, #32]
	fadd    v26.2d,    v26.2d, v18.2d
	str q26, [C_r4, #32]
.Lc_r4_str4${label_sfx}_beta1_alpha1:
	ldr	q25, [C_r4, #16]
	fadd v25.2d, v25.2d, v17.2d
	str q25, [C_r4, #16]
.Lc_r4_str2${label_sfx}_beta1_alpha1:
	ldr	q24, [C_r4]
	fadd    v24.2d,    v24.2d, v16.2d
	str q24, [C_r4]

.Lc_r4_odd${label_sfx}_beta1_alpha1:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r4_str1${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r4_str3${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r4_str5${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r4_str7${label_sfx}_beta1_alpha1

	b .Lc_r4_odd_end${label_sfx}_beta1_alpha1

.Lc_r4_str1${label_sfx}_beta1_alpha1:
	ldr d24, [C_r4]
	fadd d16, d16, d24
	str d16, [C_r4]
	b .Lc_r4_odd_end${label_sfx}_beta1_alpha1
.Lc_r4_str3${label_sfx}_beta1_alpha1:
	ldr d25, [C_r4, #16]
	fadd d17, d17, d25
	str d17, [C_r4, #16]
	b .Lc_r4_odd_end${label_sfx}_beta1_alpha1
.Lc_r4_str5${label_sfx}_beta1_alpha1:
	ldr d26, [C_r4, #32]
	fadd d18, d18, d26
	str d18, [C_r4, #32]
	b .Lc_r4_odd_end${label_sfx}_beta1_alpha1
.Lc_r4_str7${label_sfx}_beta1_alpha1:
	ldr d27, [C_r4, #48]
	fadd d19, d19, d27
	str d19, [C_r4, #48]
	b .Lc_r4_odd_end${label_sfx}_beta1_alpha1
.Lc_r4_odd_end${label_sfx}_beta1_alpha1:
	add	C_r4, C_r4, #64

	add t, b_row_idx, #5
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r5_even${label_sfx}_beta1_alpha1:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	bls .Lc_r5_str8${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r5_str6${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r5_str4${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r5_str2${label_sfx}_beta1_alpha1

	b .Lc_r5_odd${label_sfx}_beta1_alpha1

.Lc_r5_str8${label_sfx}_beta1_alpha1:
	ldr	q31, [C_r5, #48]
	fadd    v31.2d,    v31.2d, v23.2d
	str	q31, [C_r5, #48]
.Lc_r5_str6${label_sfx}_beta1_alpha1:
	ldr	q30, [C_r5, #32]
	fadd    v30.2d,    v30.2d, v22.2d
	str q30, [C_r5, #32]
.Lc_r5_str4${label_sfx}_beta1_alpha1:
	ldr	q29, [C_r5, #16]
	fadd v29.2d, v29.2d, v21.2d
	str q29, [C_r5, #16]
.Lc_r5_str2${label_sfx}_beta1_alpha1:
	ldr	q28, [C_r5]
	fadd    v28.2d,    v28.2d, v20.2d
	str q28, [C_r5]

.Lc_r5_odd${label_sfx}_beta1_alpha1:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r5_str1${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r5_str3${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r5_str5${label_sfx}_beta1_alpha1
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r5_str7${label_sfx}_beta1_alpha1

	b .Lc_r5_odd_end${label_sfx}_beta1_alpha1

.Lc_r5_str1${label_sfx}_beta1_alpha1:
	ldr d28, [C_r5]
	fadd d20, d20, d28
	str d20, [C_r5]
	b .Lc_r5_odd_end${label_sfx}_beta1_alpha1
.Lc_r5_str3${label_sfx}_beta1_alpha1:
	ldr d29, [C_r5, #16]
	fadd d21, d21, d29
	str d21, [C_r5, #16]
	b .Lc_r5_odd_end${label_sfx}_beta1_alpha1
.Lc_r5_str5${label_sfx}_beta1_alpha1:
	ldr d30, [C_r5, #32]
	fadd d22, d22, d30
	str d22, [C_r5, #32]
	b .Lc_r5_odd_end${label_sfx}_beta1_alpha1
.Lc_r5_str7${label_sfx}_beta1_alpha1:
	ldr d31, [C_r5, #48]
	fadd d23, d23, d31
	str d23, [C_r5, #48]
	b .Lc_r5_odd_end${label_sfx}_beta1_alpha1
.Lc_r5_odd_end${label_sfx}_beta1_alpha1:
	add	C_r5, C_r5, #64
	b .LOOP4END${label_sfx}
</%def>

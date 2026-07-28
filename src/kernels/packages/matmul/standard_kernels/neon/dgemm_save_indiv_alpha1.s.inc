## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%def name="dgemm_save_indiv_alpha1(dgemm_save_name, label_sfx)">
.L_${dgemm_save_name}_indiv_alpha1:

	add t, b_row_idx, #0
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r0_even${label_sfx}_alpha1:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	ins v29.d[0], beta_as_int

	bls .Lc_r0_str8${label_sfx}_alpha1
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r0_str6${label_sfx}_alpha1
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r0_str4${label_sfx}_alpha1
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r0_str2${label_sfx}_alpha1

	b .Lc_r0_odd${label_sfx}_alpha1

.Lc_r0_str8${label_sfx}_alpha1:
	ldr	q27, [C_r0, #48]
	fmla v3.2d, v27.2d, v29.d[0]
	str	q3, [C_r0, #48]
.Lc_r0_str6${label_sfx}_alpha1:
	ldr	q26, [C_r0, #32]
	fmla v2.2d, v26.2d, v29.d[0]
	str q2, [C_r0, #32]
.Lc_r0_str4${label_sfx}_alpha1:
	ldr	q25, [C_r0, #16]
	fmla v1.2d, v25.2d, v29.d[0]
	str q1, [C_r0, #16]
.Lc_r0_str2${label_sfx}_alpha1:
	ldr	q24, [C_r0]
	fmla v0.2d, v24.2d, v29.d[0]
	str q0, [C_r0]

.Lc_r0_odd${label_sfx}_alpha1:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r0_str1${label_sfx}_alpha1
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r0_str3${label_sfx}_alpha1
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r0_str5${label_sfx}_alpha1
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r0_str7${label_sfx}_alpha1

	b .Lc_r0_odd_end${label_sfx}_alpha1

.Lc_r0_str1${label_sfx}_alpha1:
	ldr d24, [C_r0]
	fmadd d0, d24, d29, d0
	str d0, [C_r0]
	b .Lc_r0_odd_end${label_sfx}_alpha1
.Lc_r0_str3${label_sfx}_alpha1:
	ldr d25, [C_r0, #16]
	fmadd d1, d25, d29, d1
	str d1, [C_r0, #16]
	b .Lc_r0_odd_end${label_sfx}_alpha1
.Lc_r0_str5${label_sfx}_alpha1:
	ldr d26, [C_r0, #32]
	fmadd d2, d26, d29, d2
	str d2, [C_r0, #32]
	b .Lc_r0_odd_end${label_sfx}_alpha1
.Lc_r0_str7${label_sfx}_alpha1:
	ldr d27, [C_r0, #48]
	fmadd d3, d27, d29, d3
	str d3, [C_r0, #48]
	b .Lc_r0_odd_end${label_sfx}_alpha1
.Lc_r0_odd_end${label_sfx}_alpha1:
	add	C_r0, C_r0, #64

	add t, b_row_idx, #1
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r1_even${label_sfx}_alpha1:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	ins v25.d[0], beta_as_int

	bls .Lc_r1_str8${label_sfx}_alpha1
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r1_str6${label_sfx}_alpha1
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r1_str4${label_sfx}_alpha1
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r1_str2${label_sfx}_alpha1

	b .Lc_r1_odd${label_sfx}_alpha1

.Lc_r1_str8${label_sfx}_alpha1:
	ldr	q31, [C_r1, #48]
	fmla v7.2d, v31.2d, v25.d[0]
	str	q7, [C_r1, #48]
.Lc_r1_str6${label_sfx}_alpha1:
	ldr	q30, [C_r1, #32]
	fmla v6.2d, v30.2d, v25.d[0]
	str q6, [C_r1, #32]
.Lc_r1_str4${label_sfx}_alpha1:
	ldr	q29, [C_r1, #16]
	fmla v5.2d, v29.2d, v25.d[0]
	str q5, [C_r1, #16]
.Lc_r1_str2${label_sfx}_alpha1:
	ldr	q28, [C_r1]
	fmla v4.2d, v28.2d, v25.d[0]
	str q4, [C_r1]

.Lc_r1_odd${label_sfx}_alpha1:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r1_str1${label_sfx}_alpha1
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r1_str3${label_sfx}_alpha1
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r1_str5${label_sfx}_alpha1
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r1_str7${label_sfx}_alpha1

	b .Lc_r1_odd_end${label_sfx}_alpha1

.Lc_r1_str1${label_sfx}_alpha1:
	ldr d28, [C_r1]
	fmadd d4, d28, d25, d4
	str d4, [C_r1]
	b .Lc_r1_odd_end${label_sfx}_alpha1
.Lc_r1_str3${label_sfx}_alpha1:
	ldr d29, [C_r1, #16]
	fmadd d5, d29, d25, d5
	str d5, [C_r1, #16]
	b .Lc_r1_odd_end${label_sfx}_alpha1
.Lc_r1_str5${label_sfx}_alpha1:
	ldr d30, [C_r1, #32]
	fmadd d6, d30, d25, d6
	str d6, [C_r1, #32]
	b .Lc_r1_odd_end${label_sfx}_alpha1
.Lc_r1_str7${label_sfx}_alpha1:
	ldr d31, [C_r1, #48]
	fmadd d7, d31, d25, d7
	str d7, [C_r1, #48]
	b .Lc_r1_odd_end${label_sfx}_alpha1
.Lc_r1_odd_end${label_sfx}_alpha1:
	add	C_r1, C_r1, #64

	add t, b_row_idx, #2
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r2_even${label_sfx}_alpha1:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	ins v29.d[0], beta_as_int

	bls .Lc_r2_str8${label_sfx}_alpha1
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r2_str6${label_sfx}_alpha1
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r2_str4${label_sfx}_alpha1
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r2_str2${label_sfx}_alpha1

	b .Lc_r2_odd${label_sfx}_alpha1

.Lc_r2_str8${label_sfx}_alpha1:
	ldr	q27, [C_r2, #48]
	fmla v11.2d, v27.2d, v29.d[0]
	str	q11, [C_r2, #48]
.Lc_r2_str6${label_sfx}_alpha1:
	ldr	q26, [C_r2, #32]
	fmla v10.2d, v26.2d, v29.d[0]
	str q10, [C_r2, #32]
.Lc_r2_str4${label_sfx}_alpha1:
	ldr	q25, [C_r2, #16]
	fmla v9.2d, v25.2d, v29.d[0]
	str q9, [C_r2, #16]
.Lc_r2_str2${label_sfx}_alpha1:
	ldr	q24, [C_r2]
	fmla v8.2d, v24.2d, v29.d[0]
	str q8, [C_r2]

.Lc_r2_odd${label_sfx}_alpha1:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r2_str1${label_sfx}_alpha1
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r2_str3${label_sfx}_alpha1
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r2_str5${label_sfx}_alpha1
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r2_str7${label_sfx}_alpha1

	b .Lc_r2_odd_end${label_sfx}_alpha1

.Lc_r2_str1${label_sfx}_alpha1:
	ldr d24, [C_r2]
	fmadd d8, d24, d29, d8
	str d8, [C_r2]
	b .Lc_r2_odd_end${label_sfx}_alpha1
.Lc_r2_str3${label_sfx}_alpha1:
	ldr d25, [C_r2, #16]
	fmadd d9, d25, d29, d9
	str d9, [C_r2, #16]
	b .Lc_r2_odd_end${label_sfx}_alpha1
.Lc_r2_str5${label_sfx}_alpha1:
	ldr d26, [C_r2, #32]
	fmadd d10, d26, d29, d10
	str d10, [C_r2, #32]
	b .Lc_r2_odd_end${label_sfx}_alpha1
.Lc_r2_str7${label_sfx}_alpha1:
	ldr d27, [C_r2, #48]
	fmadd d11, d27, d29, d11
	str d11, [C_r2, #48]
	b .Lc_r2_odd_end${label_sfx}_alpha1
.Lc_r2_odd_end${label_sfx}_alpha1:
	add	C_r2, C_r2, #64

	add t, b_row_idx, #3
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r3_even${label_sfx}_alpha1:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	ins v25.d[0], beta_as_int

	bls .Lc_r3_str8${label_sfx}_alpha1
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r3_str6${label_sfx}_alpha1
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r3_str4${label_sfx}_alpha1
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r3_str2${label_sfx}_alpha1

	b .Lc_r3_odd${label_sfx}_alpha1

.Lc_r3_str8${label_sfx}_alpha1:
	ldr	q31, [C_r3, #48]
	fmla v15.2d, v31.2d, v25.d[0]
	str	q15, [C_r3, #48]
.Lc_r3_str6${label_sfx}_alpha1:
	ldr	q30, [C_r3, #32]
	fmla v14.2d, v30.2d, v25.d[0]
	str q14, [C_r3, #32]
.Lc_r3_str4${label_sfx}_alpha1:
	ldr	q29, [C_r3, #16]
	fmla v13.2d, v29.2d, v25.d[0]
	str q13, [C_r3, #16]
.Lc_r3_str2${label_sfx}_alpha1:
	ldr	q28, [C_r3]
	fmla v12.2d, v28.2d, v25.d[0]
	str q12, [C_r3]

.Lc_r3_odd${label_sfx}_alpha1:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r3_str1${label_sfx}_alpha1
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r3_str3${label_sfx}_alpha1
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r3_str5${label_sfx}_alpha1
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r3_str7${label_sfx}_alpha1

	b .Lc_r3_odd_end${label_sfx}_alpha1

.Lc_r3_str1${label_sfx}_alpha1:
	ldr d28, [C_r3]
	fmadd d12, d28, d25, d12
	str d12, [C_r3]
	b .Lc_r3_odd_end${label_sfx}_alpha1
.Lc_r3_str3${label_sfx}_alpha1:
	ldr d29, [C_r3, #16]
	fmadd d13, d29, d25, d13
	str d13, [C_r3, #16]
	b .Lc_r3_odd_end${label_sfx}_alpha1
.Lc_r3_str5${label_sfx}_alpha1:
	ldr d30, [C_r3, #32]
	fmadd d14, d30, d25, d14
	str d14, [C_r3, #32]
	b .Lc_r3_odd_end${label_sfx}_alpha1
.Lc_r3_str7${label_sfx}_alpha1:
	ldr d31, [C_r3, #48]
	fmadd d15, d31, d25, d15
	str d15, [C_r3, #48]
	b .Lc_r3_odd_end${label_sfx}_alpha1
.Lc_r3_odd_end${label_sfx}_alpha1:
	add	C_r3, C_r3, #64

	add t, b_row_idx, #4
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r4_even${label_sfx}_alpha1:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	ins v29.d[0], beta_as_int

	bls .Lc_r4_str8${label_sfx}_alpha1
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r4_str6${label_sfx}_alpha1
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r4_str4${label_sfx}_alpha1
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r4_str2${label_sfx}_alpha1

	b .Lc_r4_odd${label_sfx}_alpha1

.Lc_r4_str8${label_sfx}_alpha1:
	ldr	q27, [C_r4, #48]
	fmla v19.2d, v27.2d, v29.d[0]
	str	q19, [C_r4, #48]
.Lc_r4_str6${label_sfx}_alpha1:
	ldr	q26, [C_r4, #32]
	fmla v18.2d, v26.2d, v29.d[0]
	str q18, [C_r4, #32]
.Lc_r4_str4${label_sfx}_alpha1:
	ldr	q25, [C_r4, #16]
	fmla v17.2d, v25.2d, v29.d[0]
	str q17, [C_r4, #16]
.Lc_r4_str2${label_sfx}_alpha1:
	ldr	q24, [C_r4]
	fmla v16.2d, v24.2d, v29.d[0]
	str q16, [C_r4]

.Lc_r4_odd${label_sfx}_alpha1:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r4_str1${label_sfx}_alpha1
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r4_str3${label_sfx}_alpha1
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r4_str5${label_sfx}_alpha1
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r4_str7${label_sfx}_alpha1

	b .Lc_r4_odd_end${label_sfx}_alpha1

.Lc_r4_str1${label_sfx}_alpha1:
	ldr d24, [C_r4]
	fmadd d16, d24, d29, d16
	str d16, [C_r4]
	b .Lc_r4_odd_end${label_sfx}_alpha1
.Lc_r4_str3${label_sfx}_alpha1:
	ldr d25, [C_r4, #16]
	fmadd d17, d25, d29, d17
	str d17, [C_r4, #16]
	b .Lc_r4_odd_end${label_sfx}_alpha1
.Lc_r4_str5${label_sfx}_alpha1:
	ldr d26, [C_r4, #32]
	fmadd d18, d26, d29, d18
	str d18, [C_r4, #32]
	b .Lc_r4_odd_end${label_sfx}_alpha1
.Lc_r4_str7${label_sfx}_alpha1:
	ldr d27, [C_r4, #48]
	fmadd d19, d27, d29, d19
	str d19, [C_r4, #48]
	b .Lc_r4_odd_end${label_sfx}_alpha1
.Lc_r4_odd_end${label_sfx}_alpha1:
	add	C_r4, C_r4, #64

	add t, b_row_idx, #5
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r5_even${label_sfx}_alpha1:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	ins v25.d[0], beta_as_int

	bls .Lc_r5_str8${label_sfx}_alpha1
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r5_str6${label_sfx}_alpha1
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r5_str4${label_sfx}_alpha1
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r5_str2${label_sfx}_alpha1

	b .Lc_r5_odd${label_sfx}_alpha1

.Lc_r5_str8${label_sfx}_alpha1:
	ldr	q31, [C_r5, #48]
	fmla v23.2d, v31.2d, v25.d[0]
	str	q23, [C_r5, #48]
.Lc_r5_str6${label_sfx}_alpha1:
	ldr	q30, [C_r5, #32]
	fmla v22.2d, v30.2d, v25.d[0]
	str q22, [C_r5, #32]
.Lc_r5_str4${label_sfx}_alpha1:
	ldr	q29, [C_r5, #16]
	fmla v21.2d, v29.2d, v25.d[0]
	str q21, [C_r5, #16]
.Lc_r5_str2${label_sfx}_alpha1:
	ldr	q28, [C_r5]
	fmla v20.2d, v28.2d, v25.d[0]
	str q20, [C_r5]

.Lc_r5_odd${label_sfx}_alpha1:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r5_str1${label_sfx}_alpha1
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r5_str3${label_sfx}_alpha1
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r5_str5${label_sfx}_alpha1
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r5_str7${label_sfx}_alpha1

	b .Lc_r5_odd_end${label_sfx}_alpha1

.Lc_r5_str1${label_sfx}_alpha1:
	ldr d28, [C_r5]
	fmadd d20, d28, d25, d20
	str d20, [C_r5]
	b .Lc_r5_odd_end${label_sfx}_alpha1
.Lc_r5_str3${label_sfx}_alpha1:
	ldr d29, [C_r5, #16]
	fmadd d21, d29, d25, d21
	str d21, [C_r5, #16]
	b .Lc_r5_odd_end${label_sfx}_alpha1
.Lc_r5_str5${label_sfx}_alpha1:
	ldr d30, [C_r5, #32]
	fmadd d22, d30, d25, d22
	str d22, [C_r5, #32]
	b .Lc_r5_odd_end${label_sfx}_alpha1
.Lc_r5_str7${label_sfx}_alpha1:
	ldr d31, [C_r5, #48]
	fmadd d23, d31, d25, d23
	str d23, [C_r5, #48]
	b .Lc_r5_odd_end${label_sfx}_alpha1
.Lc_r5_odd_end${label_sfx}_alpha1:
	add	C_r5, C_r5, #64
	b .LOOP4END${label_sfx}
</%def>

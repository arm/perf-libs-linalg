## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%def name="dgemm_save_indiv(dgemm_save_name, label_sfx)">
.L_${dgemm_save_name}_indiv:

	add t, b_row_idx, #0
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r0_even${label_sfx}:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	ins v28.d[0], alpha_as_int
	ins v29.d[0], beta_as_int

	bls .Lc_r0_str8${label_sfx}
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r0_str6${label_sfx}
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r0_str4${label_sfx}
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r0_str2${label_sfx}

	b .Lc_r0_odd${label_sfx}

.Lc_r0_str8${label_sfx}:
	ldr	q27, [C_r0, #48]
	fmul v27.2d, v27.2d, v29.d[0]
	fmla    v27.2d, v3.2d, v28.d[0]
	str	q27, [C_r0, #48]
.Lc_r0_str6${label_sfx}:
	ldr	q26, [C_r0, #32]
	fmul v26.2d, v26.2d, v29.d[0]
	fmla    v26.2d, v2.2d, v28.d[0]
	str q26, [C_r0, #32]
.Lc_r0_str4${label_sfx}:
	ldr	q25, [C_r0, #16]
	fmul v25.2d, v25.2d, v29.d[0]
	fmla v25.2d, v1.2d, v28.d[0]
	str q25, [C_r0, #16]
.Lc_r0_str2${label_sfx}:
	ldr	q24, [C_r0]
	fmul v24.2d, v24.2d, v29.d[0]
	fmla    v24.2d, v0.2d, v28.d[0]
	str q24, [C_r0]

.Lc_r0_odd${label_sfx}:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r0_str1${label_sfx}
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r0_str3${label_sfx}
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r0_str5${label_sfx}
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r0_str7${label_sfx}

	b .Lc_r0_odd_end${label_sfx}

.Lc_r0_str1${label_sfx}:
	ldr d24, [C_r0]
	fmul d24, d24, d29
	fmadd d0, d0, d28, d24
	str d0, [C_r0]
	b .Lc_r0_odd_end${label_sfx}
.Lc_r0_str3${label_sfx}:
	ldr d25, [C_r0, #16]
	fmul d25, d25, d29
	fmadd d1, d1, d28, d25
	str d1, [C_r0, #16]
	b .Lc_r0_odd_end${label_sfx}
.Lc_r0_str5${label_sfx}:
	ldr d26, [C_r0, #32]
	fmul d26, d26, d29
	fmadd d2, d2, d28, d26
	str d2, [C_r0, #32]
	b .Lc_r0_odd_end${label_sfx}
.Lc_r0_str7${label_sfx}:
	ldr d27, [C_r0, #48]
	fmul d27, d27, d29
	fmadd d3, d3, d28, d27
	str d3, [C_r0, #48]
	b .Lc_r0_odd_end${label_sfx}
.Lc_r0_odd_end${label_sfx}:
	add	C_r0, C_r0, #64

	add t, b_row_idx, #1
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r1_even${label_sfx}:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	ins v24.d[0], alpha_as_int
	ins v25.d[0], beta_as_int

	bls .Lc_r1_str8${label_sfx}
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r1_str6${label_sfx}
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r1_str4${label_sfx}
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r1_str2${label_sfx}

	b .Lc_r1_odd${label_sfx}

.Lc_r1_str8${label_sfx}:
	ldr	q31, [C_r1, #48]
	fmul v31.2d, v31.2d, v25.d[0]
	fmla    v31.2d, v7.2d, v24.d[0]
	str	q31, [C_r1, #48]
.Lc_r1_str6${label_sfx}:
	ldr	q30, [C_r1, #32]
	fmul v30.2d, v30.2d, v25.d[0]
	fmla    v30.2d, v6.2d, v24.d[0]
	str q30, [C_r1, #32]
.Lc_r1_str4${label_sfx}:
	ldr	q29, [C_r1, #16]
	fmul v29.2d, v29.2d, v25.d[0]
	fmla v29.2d, v5.2d, v24.d[0]
	str q29, [C_r1, #16]
.Lc_r1_str2${label_sfx}:
	ldr	q28, [C_r1]
	fmul v28.2d, v28.2d, v25.d[0]
	fmla    v28.2d, v4.2d, v24.d[0]
	str q28, [C_r1]

.Lc_r1_odd${label_sfx}:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r1_str1${label_sfx}
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r1_str3${label_sfx}
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r1_str5${label_sfx}
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r1_str7${label_sfx}

	b .Lc_r1_odd_end${label_sfx}

.Lc_r1_str1${label_sfx}:
	ldr d28, [C_r1]
	fmul d28, d28, d25
	fmadd d4, d4, d24, d28
	str d4, [C_r1]
	b .Lc_r1_odd_end${label_sfx}
.Lc_r1_str3${label_sfx}:
	ldr d29, [C_r1, #16]
	fmul d29, d29, d25
	fmadd d5, d5, d24, d29
	str d5, [C_r1, #16]
	b .Lc_r1_odd_end${label_sfx}
.Lc_r1_str5${label_sfx}:
	ldr d30, [C_r1, #32]
	fmul d30, d30, d25
	fmadd d6, d6, d24, d30
	str d6, [C_r1, #32]
	b .Lc_r1_odd_end${label_sfx}
.Lc_r1_str7${label_sfx}:
	ldr d31, [C_r1, #48]
	fmul d31, d31, d25
	fmadd d7, d7, d24, d31
	str d7, [C_r1, #48]
	b .Lc_r1_odd_end${label_sfx}
.Lc_r1_odd_end${label_sfx}:
	add	C_r1, C_r1, #64

	add t, b_row_idx, #2
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r2_even${label_sfx}:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	ins v28.d[0], alpha_as_int
	ins v29.d[0], beta_as_int

	bls .Lc_r2_str8${label_sfx}
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r2_str6${label_sfx}
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r2_str4${label_sfx}
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r2_str2${label_sfx}

	b .Lc_r2_odd${label_sfx}

.Lc_r2_str8${label_sfx}:
	ldr	q27, [C_r2, #48]
	fmul v27.2d, v27.2d, v29.d[0]
	fmla    v27.2d, v11.2d, v28.d[0]
	str	q27, [C_r2, #48]
.Lc_r2_str6${label_sfx}:
	ldr	q26, [C_r2, #32]
	fmul v26.2d, v26.2d, v29.d[0]
	fmla    v26.2d, v10.2d, v28.d[0]
	str q26, [C_r2, #32]
.Lc_r2_str4${label_sfx}:
	ldr	q25, [C_r2, #16]
	fmul v25.2d, v25.2d, v29.d[0]
	fmla v25.2d, v9.2d, v28.d[0]
	str q25, [C_r2, #16]
.Lc_r2_str2${label_sfx}:
	ldr	q24, [C_r2]
	fmul v24.2d, v24.2d, v29.d[0]
	fmla    v24.2d, v8.2d, v28.d[0]
	str q24, [C_r2]

.Lc_r2_odd${label_sfx}:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r2_str1${label_sfx}
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r2_str3${label_sfx}
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r2_str5${label_sfx}
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r2_str7${label_sfx}

	b .Lc_r2_odd_end${label_sfx}

.Lc_r2_str1${label_sfx}:
	ldr d24, [C_r2]
	fmul d24, d24, d29
	fmadd d8, d8, d28, d24
	str d8, [C_r2]
	b .Lc_r2_odd_end${label_sfx}
.Lc_r2_str3${label_sfx}:
	ldr d25, [C_r2, #16]
	fmul d25, d25, d29
	fmadd d9, d9, d28, d25
	str d9, [C_r2, #16]
	b .Lc_r2_odd_end${label_sfx}
.Lc_r2_str5${label_sfx}:
	ldr d26, [C_r2, #32]
	fmul d26, d26, d29
	fmadd d10, d10, d28, d26
	str d10, [C_r2, #32]
	b .Lc_r2_odd_end${label_sfx}
.Lc_r2_str7${label_sfx}:
	ldr d27, [C_r2, #48]
	fmul d27, d27, d29
	fmadd d11, d11, d28, d27
	str d11, [C_r2, #48]
	b .Lc_r2_odd_end${label_sfx}
.Lc_r2_odd_end${label_sfx}:
	add	C_r2, C_r2, #64

	add t, b_row_idx, #3
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r3_even${label_sfx}:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	ins v24.d[0], alpha_as_int
	ins v25.d[0], beta_as_int

	bls .Lc_r3_str8${label_sfx}
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r3_str6${label_sfx}
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r3_str4${label_sfx}
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r3_str2${label_sfx}

	b .Lc_r3_odd${label_sfx}

.Lc_r3_str8${label_sfx}:
	ldr	q31, [C_r3, #48]
	fmul v31.2d, v31.2d, v25.d[0]
	fmla    v31.2d, v15.2d, v24.d[0]
	str	q31, [C_r3, #48]
.Lc_r3_str6${label_sfx}:
	ldr	q30, [C_r3, #32]
	fmul v30.2d, v30.2d, v25.d[0]
	fmla    v30.2d, v14.2d, v24.d[0]
	str q30, [C_r3, #32]
.Lc_r3_str4${label_sfx}:
	ldr	q29, [C_r3, #16]
	fmul v29.2d, v29.2d, v25.d[0]
	fmla v29.2d, v13.2d, v24.d[0]
	str q29, [C_r3, #16]
.Lc_r3_str2${label_sfx}:
	ldr	q28, [C_r3]
	fmul v28.2d, v28.2d, v25.d[0]
	fmla    v28.2d, v12.2d, v24.d[0]
	str q28, [C_r3]

.Lc_r3_odd${label_sfx}:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r3_str1${label_sfx}
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r3_str3${label_sfx}
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r3_str5${label_sfx}
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r3_str7${label_sfx}

	b .Lc_r3_odd_end${label_sfx}

.Lc_r3_str1${label_sfx}:
	ldr d28, [C_r3]
	fmul d28, d28, d25
	fmadd d12, d12, d24, d28
	str d12, [C_r3]
	b .Lc_r3_odd_end${label_sfx}
.Lc_r3_str3${label_sfx}:
	ldr d29, [C_r3, #16]
	fmul d29, d29, d25
	fmadd d13, d13, d24, d29
	str d13, [C_r3, #16]
	b .Lc_r3_odd_end${label_sfx}
.Lc_r3_str5${label_sfx}:
	ldr d30, [C_r3, #32]
	fmul d30, d30, d25
	fmadd d14, d14, d24, d30
	str d14, [C_r3, #32]
	b .Lc_r3_odd_end${label_sfx}
.Lc_r3_str7${label_sfx}:
	ldr d31, [C_r3, #48]
	fmul d31, d31, d25
	fmadd d15, d15, d24, d31
	str d15, [C_r3, #48]
	b .Lc_r3_odd_end${label_sfx}
.Lc_r3_odd_end${label_sfx}:
	add	C_r3, C_r3, #64

	add t, b_row_idx, #4
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r4_even${label_sfx}:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	ins v28.d[0], alpha_as_int
	ins v29.d[0], beta_as_int

	bls .Lc_r4_str8${label_sfx}
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r4_str6${label_sfx}
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r4_str4${label_sfx}
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r4_str2${label_sfx}

	b .Lc_r4_odd${label_sfx}

.Lc_r4_str8${label_sfx}:
	ldr	q27, [C_r4, #48]
	fmul v27.2d, v27.2d, v29.d[0]
	fmla    v27.2d, v19.2d, v28.d[0]
	str	q27, [C_r4, #48]
.Lc_r4_str6${label_sfx}:
	ldr	q26, [C_r4, #32]
	fmul v26.2d, v26.2d, v29.d[0]
	fmla    v26.2d, v18.2d, v28.d[0]
	str q26, [C_r4, #32]
.Lc_r4_str4${label_sfx}:
	ldr	q25, [C_r4, #16]
	fmul v25.2d, v25.2d, v29.d[0]
	fmla v25.2d, v17.2d, v28.d[0]
	str q25, [C_r4, #16]
.Lc_r4_str2${label_sfx}:
	ldr	q24, [C_r4]
	fmul v24.2d, v24.2d, v29.d[0]
	fmla    v24.2d, v16.2d, v28.d[0]
	str q24, [C_r4]

.Lc_r4_odd${label_sfx}:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r4_str1${label_sfx}
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r4_str3${label_sfx}
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r4_str5${label_sfx}
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r4_str7${label_sfx}

	b .Lc_r4_odd_end${label_sfx}

.Lc_r4_str1${label_sfx}:
	ldr d24, [C_r4]
	fmul d24, d24, d29
	fmadd d16, d16, d28, d24
	str d16, [C_r4]
	b .Lc_r4_odd_end${label_sfx}
.Lc_r4_str3${label_sfx}:
	ldr d25, [C_r4, #16]
	fmul d25, d25, d29
	fmadd d17, d17, d28, d25
	str d17, [C_r4, #16]
	b .Lc_r4_odd_end${label_sfx}
.Lc_r4_str5${label_sfx}:
	ldr d26, [C_r4, #32]
	fmul d26, d26, d29
	fmadd d18, d18, d28, d26
	str d18, [C_r4, #32]
	b .Lc_r4_odd_end${label_sfx}
.Lc_r4_str7${label_sfx}:
	ldr d27, [C_r4, #48]
	fmul d27, d27, d29
	fmadd d19, d19, d28, d27
	str d19, [C_r4, #48]
	b .Lc_r4_odd_end${label_sfx}
.Lc_r4_odd_end${label_sfx}:
	add	C_r4, C_r4, #64

	add t, b_row_idx, #5
	cmp	t, n_size
	beq	.LOOP4END${label_sfx}

.Lc_r5_even${label_sfx}:
	mov	t, a_row_idx
	add	t, t, 8
	cmp	t, m_size

	ins v24.d[0], alpha_as_int
	ins v25.d[0], beta_as_int

	bls .Lc_r5_str8${label_sfx}
	add	t, a_row_idx, 6
	cmp	m_size, t
	bcs	.Lc_r5_str6${label_sfx}
	add	t, a_row_idx, 4
	cmp	m_size, t
	bcs	.Lc_r5_str4${label_sfx}
	add	t, a_row_idx, 2
	cmp	m_size, t
	bcs	.Lc_r5_str2${label_sfx}

	b .Lc_r5_odd${label_sfx}

.Lc_r5_str8${label_sfx}:
	ldr	q31, [C_r5, #48]
	fmul v31.2d, v31.2d, v25.d[0]
	fmla    v31.2d, v23.2d, v24.d[0]
	str	q31, [C_r5, #48]
.Lc_r5_str6${label_sfx}:
	ldr	q30, [C_r5, #32]
	fmul v30.2d, v30.2d, v25.d[0]
	fmla    v30.2d, v22.2d, v24.d[0]
	str q30, [C_r5, #32]
.Lc_r5_str4${label_sfx}:
	ldr	q29, [C_r5, #16]
	fmul v29.2d, v29.2d, v25.d[0]
	fmla v29.2d, v21.2d, v24.d[0]
	str q29, [C_r5, #16]
.Lc_r5_str2${label_sfx}:
	ldr	q28, [C_r5]
	fmul v28.2d, v28.2d, v25.d[0]
	fmla    v28.2d, v20.2d, v24.d[0]
	str q28, [C_r5]

.Lc_r5_odd${label_sfx}:
	add	t, a_row_idx, 1
	cmp	m_size, t
	beq	.Lc_r5_str1${label_sfx}
	add	t, a_row_idx, 3
	cmp	m_size, t
	beq	.Lc_r5_str3${label_sfx}
	add	t, a_row_idx, 5
	cmp	m_size, t
	beq	.Lc_r5_str5${label_sfx}
	add	t, a_row_idx, 7
	cmp	m_size, t
	beq	.Lc_r5_str7${label_sfx}

	b .Lc_r5_odd_end${label_sfx}

.Lc_r5_str1${label_sfx}:
	ldr d28, [C_r5]
	fmul d28, d28, d25
	fmadd d20, d20, d24, d28
	str d20, [C_r5]
	b .Lc_r5_odd_end${label_sfx}
.Lc_r5_str3${label_sfx}:
	ldr d29, [C_r5, #16]
	fmul d29, d29, d25
	fmadd d21, d21, d24, d29
	str d21, [C_r5, #16]
	b .Lc_r5_odd_end${label_sfx}
.Lc_r5_str5${label_sfx}:
	ldr d30, [C_r5, #32]
	fmul d30, d30, d25
	fmadd d22, d22, d24, d30
	str d22, [C_r5, #32]
	b .Lc_r5_odd_end${label_sfx}
.Lc_r5_str7${label_sfx}:
	ldr d31, [C_r5, #48]
	fmul d31, d31, d25
	fmadd d23, d23, d24, d31
	str d23, [C_r5, #48]
	b .Lc_r5_odd_end${label_sfx}
.Lc_r5_odd_end${label_sfx}:
	add	C_r5, C_r5, #64
	b .LOOP4END${label_sfx}
</%def>

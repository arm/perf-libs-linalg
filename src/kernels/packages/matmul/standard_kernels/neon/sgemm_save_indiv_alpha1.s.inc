## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%def name="sgemm_save_indiv_alpha1(sgemm_save_name, label_sfx)">
.L${sgemm_save_name}_indiv_alpha1:
	add tmp_x, b_row_idx, #0
	cmp	tmp_x, n_size
	beq	.LOOP4END${label_sfx}

.Lsgemm_c_r0_even${label_sfx}_alpha1:
	ins v31.s[1], beta_as_int

	add	tmp_x, a_row_idx, 12
	cmp	tmp_x, m_size
	bls .Lsgemm_c_r0_str12${label_sfx}_alpha1

	add	tmp_x, a_row_idx, 8
	cmp	m_size, tmp_x
	bcs .Lsgemm_c_r0_str8${label_sfx}_alpha1

	add	tmp_x, a_row_idx, 4
	cmp	m_size, tmp_x
	bcs .Lsgemm_c_r0_str4${label_sfx}_alpha1

	b .Lsgemm_c_r0_odd${label_sfx}_alpha1

.Lsgemm_c_r0_str12${label_sfx}_alpha1:
	ldr	q26, [C_r0, #32]
	fmla    v2.4s, v26.4s, v31.s[1]
	str q2, [C_r0, #32]
.Lsgemm_c_r0_str8${label_sfx}_alpha1:
	ldr	q25, [C_r0, #16]
	fmla v1.4s, v25.4s, v31.s[1]
	str q1, [C_r0, #16]
.Lsgemm_c_r0_str4${label_sfx}_alpha1:
	ldr	q24, [C_r0]
	fmla    v0.4s, v24.4s,  v31.s[1]
	str q0, [C_r0]

.Lsgemm_c_r0_odd${label_sfx}_alpha1:
	add	tmp_x, a_row_idx, 1
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r0_str1${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 2
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r0_str2${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 3
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r0_str3${label_sfx}_alpha1


	add	tmp_x, a_row_idx, 5
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r0_str5${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 6
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r0_str6${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 7
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r0_str7${label_sfx}_alpha1


	add	tmp_x, a_row_idx, 9
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r0_str9${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 10
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r0_str10${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 11
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r0_str11${label_sfx}_alpha1


	b .Lsgemm_c_r0_odd_end${label_sfx}_alpha1

.Lsgemm_c_r0_str11${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v2.s[2]
	ldr	s24, [C_r0, #40]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r0, #40]
.Lsgemm_c_r0_str10${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v2.s[1]
	ldr	s24, [C_r0, #36]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r0, #36]
.Lsgemm_c_r0_str9${label_sfx}_alpha1:
	//Load out two values
	ldr	s24, [C_r0, #32]
	//Apply Alpha to our element of C
	fmla	s2, s24, v31.s[1]
	str	s2, [C_r0, #32]

	b .Lsgemm_c_r0_odd_end${label_sfx}_alpha1


.Lsgemm_c_r0_str7${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v1.s[2]
	ldr	s24, [C_r0, #24]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r0, #24]
.Lsgemm_c_r0_str6${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v1.s[1]
	ldr	s24, [C_r0, #20]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r0, #20]
.Lsgemm_c_r0_str5${label_sfx}_alpha1:
	//Load out two values
	ldr	s24, [C_r0, #16]
	//Apply Alpha to our element of C
	fmla	s1, s24, v31.s[1]
	str	s1, [C_r0, #16]

	b .Lsgemm_c_r0_odd_end${label_sfx}_alpha1

.Lsgemm_c_r0_str3${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v0.s[2]
	ldr	s24, [C_r0, 8]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r0, 8]
.Lsgemm_c_r0_str2${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v0.s[1]
	ldr	s24, [C_r0, 4]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r0, 4]
.Lsgemm_c_r0_str1${label_sfx}_alpha1:
	//Load out two values
	ldr	s24, [C_r0]
	//Apply Alpha to our element of C
	fmla	s0, s24, v31.s[1]
	str	s0, [C_r0]

	b .Lsgemm_c_r0_odd_end${label_sfx}_alpha1


.Lsgemm_c_r0_odd_end${label_sfx}_alpha1:
	add	C_r0, C_r0, #48

	add tmp_x, b_row_idx, #1
	cmp	tmp_x, n_size
	beq	.LOOP4END${label_sfx}

.Lsgemm_c_r1_even${label_sfx}_alpha1:

	ins v31.s[1], beta_as_int

	add	tmp_x, a_row_idx, 12
	cmp	tmp_x, m_size
	bls .Lsgemm_c_r1_str12${label_sfx}_alpha1

	add	tmp_x, a_row_idx, 8
	cmp	m_size, tmp_x
	bcs .Lsgemm_c_r1_str8${label_sfx}_alpha1

	add	tmp_x, a_row_idx, 4
	cmp	m_size, tmp_x
	bcs .Lsgemm_c_r1_str4${label_sfx}_alpha1

	b .Lsgemm_c_r1_odd${label_sfx}_alpha1


.Lsgemm_c_r1_str12${label_sfx}_alpha1:
	ldr	q29, [C_r1, #32]
	fmla    v5.4s, v29.4s, v31.s[1]
	str q5, [C_r1, #32]
.Lsgemm_c_r1_str8${label_sfx}_alpha1:
	ldr	q28, [C_r1, #16]
	fmla v4.4s, v28.4s,  v31.s[1]
	str q4, [C_r1, #16]
.Lsgemm_c_r1_str4${label_sfx}_alpha1:
	ldr	q27, [C_r1]
	fmla    v3.4s, v27.4s,  v31.s[1]
	str q3, [C_r1]

.Lsgemm_c_r1_odd${label_sfx}_alpha1:
	add	tmp_x, a_row_idx, 1
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r1_str1${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 2
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r1_str2${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 3
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r1_str3${label_sfx}_alpha1


	add	tmp_x, a_row_idx, 5
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r1_str5${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 6
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r1_str6${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 7
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r1_str7${label_sfx}_alpha1


	add	tmp_x, a_row_idx, 9
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r1_str9${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 10
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r1_str10${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 11
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r1_str11${label_sfx}_alpha1


	b .Lsgemm_c_r1_odd_end${label_sfx}_alpha1

.Lsgemm_c_r1_str11${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v5.s[2]
	ldr	s27, [C_r1, #40]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r1, #40]
.Lsgemm_c_r1_str10${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v5.s[1]
	ldr	s27, [C_r1, #36]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r1, #36]
.Lsgemm_c_r1_str9${label_sfx}_alpha1:
	//Load out two values
	ldr	s27, [C_r1, #32]
	//Apply Alpha to our element of C
	fmla	s5, s27, v31.s[1]
	str	s5, [C_r1, #32]

	b .Lsgemm_c_r1_odd_end${label_sfx}_alpha1


.Lsgemm_c_r1_str7${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v4.s[2]
	ldr	s27, [C_r1, #24]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r1, #24]
.Lsgemm_c_r1_str6${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v4.s[1]
	ldr	s27, [C_r1, #20]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r1, #20]
.Lsgemm_c_r1_str5${label_sfx}_alpha1:
	//Load out two values
	ldr	s27, [C_r1, #16]
	//Apply Alpha to our element of C
	fmla	s4, s27, v31.s[1]
	str	s4, [C_r1, #16]

	b .Lsgemm_c_r1_odd_end${label_sfx}_alpha1

.Lsgemm_c_r1_str3${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v3.s[2]
	ldr	s27, [C_r1, 8]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r1, 8]
.Lsgemm_c_r1_str2${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v3.s[1]
	ldr	s27, [C_r1, 4]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r1, 4]
.Lsgemm_c_r1_str1${label_sfx}_alpha1:
	//Load out two values
	ldr	s27, [C_r1]
	//Apply Alpha to our element of C
	fmla	s3, s27, v31.s[1]
	str	s3, [C_r1]

	b .Lsgemm_c_r1_odd_end${label_sfx}_alpha1


.Lsgemm_c_r1_odd_end${label_sfx}_alpha1:
	add	C_r1, C_r1, #48

	add tmp_x, b_row_idx, #2
	cmp	tmp_x, n_size
	beq	.LOOP4END${label_sfx}

.Lsgemm_c_r2_even${label_sfx}_alpha1:
	ins v31.s[1], beta_as_int

	add	tmp_x, a_row_idx, 12
	cmp	tmp_x, m_size
	bls .Lsgemm_c_r2_str12${label_sfx}_alpha1

	add	tmp_x, a_row_idx, 8
	cmp	m_size, tmp_x
	bcs .Lsgemm_c_r2_str8${label_sfx}_alpha1

	add	tmp_x, a_row_idx, 4
	cmp	m_size, tmp_x
	bcs .Lsgemm_c_r2_str4${label_sfx}_alpha1

	b .Lsgemm_c_r2_odd${label_sfx}_alpha1


.Lsgemm_c_r2_str12${label_sfx}_alpha1:
	ldr	q26, [C_r2, #32]
	fmla    v8.4s, v26.4s,  v31.s[1]
	str q8, [C_r2, #32]
.Lsgemm_c_r2_str8${label_sfx}_alpha1:
	ldr	q25, [C_r2, #16]
	fmla v7.4s, v25.4s, v31.s[1]
	str q7, [C_r2, #16]
.Lsgemm_c_r2_str4${label_sfx}_alpha1:
	ldr	q24, [C_r2]
	fmla    v6.4s, v24.4s, v31.s[1]
	str q6, [C_r2]

.Lsgemm_c_r2_odd${label_sfx}_alpha1:
	add	tmp_x, a_row_idx, 1
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r2_str1${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 2
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r2_str2${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 3
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r2_str3${label_sfx}_alpha1


	add	tmp_x, a_row_idx, 5
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r2_str5${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 6
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r2_str6${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 7
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r2_str7${label_sfx}_alpha1


	add	tmp_x, a_row_idx, 9
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r2_str9${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 10
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r2_str10${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 11
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r2_str11${label_sfx}_alpha1


	b .Lsgemm_c_r2_odd_end${label_sfx}_alpha1

.Lsgemm_c_r2_str11${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v8.s[2]
	ldr	s24, [C_r2, #40]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r2, #40]
.Lsgemm_c_r2_str10${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v8.s[1]
	ldr	s24, [C_r2, #36]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r2, #36]
.Lsgemm_c_r2_str9${label_sfx}_alpha1:
	//Load out two values
	ldr	s24, [C_r2, #32]
	//Apply Alpha to our element of C
	fmla	s8, s24, v31.s[1]
	str	s8, [C_r2, #32]

	b .Lsgemm_c_r2_odd_end${label_sfx}_alpha1


.Lsgemm_c_r2_str7${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v7.s[2]
	ldr	s24, [C_r2, #24]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r2, #24]
.Lsgemm_c_r2_str6${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v7.s[1]
	ldr	s24, [C_r2, #20]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r2, #20]
.Lsgemm_c_r2_str5${label_sfx}_alpha1:
	//Load out two values
	ldr	s24, [C_r2, #16]
	//Apply Alpha to our element of C
	fmla	s7, s24, v31.s[1]
	str	s7, [C_r2, #16]

	b .Lsgemm_c_r2_odd_end${label_sfx}_alpha1

.Lsgemm_c_r2_str3${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v6.s[2]
	ldr	s24, [C_r2, 8]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r2, 8]
.Lsgemm_c_r2_str2${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v6.s[1]
	ldr	s24, [C_r2, 4]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r2, 4]
.Lsgemm_c_r2_str1${label_sfx}_alpha1:
	//Load out two values
	ldr	s24, [C_r2]
	//Apply Alpha to our element of C
	fmla	s6, s24, v31.s[1]
	str	s6, [C_r2]

	b .Lsgemm_c_r2_odd_end${label_sfx}_alpha1


.Lsgemm_c_r2_odd_end${label_sfx}_alpha1:
	add	C_r2, C_r2, #48

	add tmp_x, b_row_idx, #3
	cmp	tmp_x, n_size
	beq	.LOOP4END${label_sfx}

.Lsgemm_c_r3_even${label_sfx}_alpha1:

	ins v31.s[1], beta_as_int

	add	tmp_x, a_row_idx, 12
	cmp	tmp_x, m_size
	bls .Lsgemm_c_r3_str12${label_sfx}_alpha1

	add	tmp_x, a_row_idx, 8
	cmp	m_size, tmp_x
	bcs .Lsgemm_c_r3_str8${label_sfx}_alpha1

	add	tmp_x, a_row_idx, 4
	cmp	m_size, tmp_x
	bcs .Lsgemm_c_r3_str4${label_sfx}_alpha1

	b .Lsgemm_c_r3_odd${label_sfx}_alpha1


.Lsgemm_c_r3_str12${label_sfx}_alpha1:
	ldr	q29, [C_r3, #32]
	fmla    v11.4s, v29.4s, v31.s[1]
	str q11, [C_r3, #32]
.Lsgemm_c_r3_str8${label_sfx}_alpha1:
	ldr	q28, [C_r3, #16]
	fmla    v10.4s, v28.4s, v31.s[1]
	str q10, [C_r3, #16]
.Lsgemm_c_r3_str4${label_sfx}_alpha1:
	ldr	q27, [C_r3]
	fmla    v9.4s, v27.4s, v31.s[1]
	str q9, [C_r3]

.Lsgemm_c_r3_odd${label_sfx}_alpha1:
	add	tmp_x, a_row_idx, 1
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r3_str1${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 2
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r3_str2${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 3
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r3_str3${label_sfx}_alpha1


	add	tmp_x, a_row_idx, 5
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r3_str5${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 6
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r3_str6${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 7
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r3_str7${label_sfx}_alpha1


	add	tmp_x, a_row_idx, 9
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r3_str9${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 10
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r3_str10${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 11
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r3_str11${label_sfx}_alpha1


	b .Lsgemm_c_r3_odd_end${label_sfx}_alpha1

.Lsgemm_c_r3_str11${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v11.s[2]
	ldr	s27, [C_r3, #40]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r3, #40]
.Lsgemm_c_r3_str10${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v11.s[1]
	ldr	s27, [C_r3, #36]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r3, #36]
.Lsgemm_c_r3_str9${label_sfx}_alpha1:
	//Load out two values
	ldr	s27, [C_r3, #32]
	//Apply Alpha to our element of C
	fmla	s11, s27, v31.s[1]
	str	s11, [C_r3, #32]

	b .Lsgemm_c_r3_odd_end${label_sfx}_alpha1


.Lsgemm_c_r3_str7${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v10.s[2]
	ldr	s27, [C_r3, #24]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r3, #24]
.Lsgemm_c_r3_str6${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v10.s[1]
	ldr	s27, [C_r3, #20]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r3, #20]
.Lsgemm_c_r3_str5${label_sfx}_alpha1:
	//Load out two values
	ldr	s27, [C_r3, #16]
	//Apply Alpha to our element of C
	fmla	s10, s27, v31.s[1]
	str	s10, [C_r3, #16]

	b .Lsgemm_c_r3_odd_end${label_sfx}_alpha1

.Lsgemm_c_r3_str3${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v9.s[2]
	ldr	s27, [C_r3, 8]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r3, 8]
.Lsgemm_c_r3_str2${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v9.s[1]
	ldr	s27, [C_r3, 4]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r3, 4]
.Lsgemm_c_r3_str1${label_sfx}_alpha1:
	//Load out two values
	ldr	s27, [C_r3]
	//Apply Alpha to our element of C
	fmla	s9, s27, v31.s[1]
	str	s9, [C_r3]

	b .Lsgemm_c_r3_odd_end${label_sfx}_alpha1


.Lsgemm_c_r3_odd_end${label_sfx}_alpha1:
	add	C_r3, C_r3, #48

	add tmp_x, b_row_idx, #4
	cmp	tmp_x, n_size
	beq	.LOOP4END${label_sfx}

.Lsgemm_c_r4_even${label_sfx}_alpha1:

	ins v31.s[1], beta_as_int

	add	tmp_x, a_row_idx, 12
	cmp	tmp_x, m_size
	bls .Lsgemm_c_r4_str12${label_sfx}_alpha1

	add	tmp_x, a_row_idx, 8
	cmp	m_size, tmp_x
	bcs .Lsgemm_c_r4_str8${label_sfx}_alpha1

	add	tmp_x, a_row_idx, 4
	cmp	m_size, tmp_x
	bcs .Lsgemm_c_r4_str4${label_sfx}_alpha1

	b .Lsgemm_c_r4_odd${label_sfx}_alpha1


.Lsgemm_c_r4_str12${label_sfx}_alpha1:
	ldr	q26, [C_r4, #32]
	fmla    v14.4s, v26.4s, v31.s[1]
	str q14, [C_r4, #32]
.Lsgemm_c_r4_str8${label_sfx}_alpha1:
	ldr	q25, [C_r4, #16]
	fmla v13.4s, v25.4s, v31.s[1]
	str q13, [C_r4, #16]
.Lsgemm_c_r4_str4${label_sfx}_alpha1:
	ldr	q24, [C_r4]
	fmla    v12.4s, v24.4s, v31.s[1]
	str q12, [C_r4]

.Lsgemm_c_r4_odd${label_sfx}_alpha1:
	add	tmp_x, a_row_idx, 1
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r4_str1${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 2
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r4_str2${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 3
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r4_str3${label_sfx}_alpha1


	add	tmp_x, a_row_idx, 5
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r4_str5${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 6
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r4_str6${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 7
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r4_str7${label_sfx}_alpha1


	add	tmp_x, a_row_idx, 9
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r4_str9${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 10
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r4_str10${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 11
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r4_str11${label_sfx}_alpha1


	b .Lsgemm_c_r4_odd_end${label_sfx}_alpha1

.Lsgemm_c_r4_str11${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v14.s[2]
	ldr	s24, [C_r4, #40]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r4, #40]
.Lsgemm_c_r4_str10${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v14.s[1]
	ldr	s24, [C_r4, #36]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r4, #36]
.Lsgemm_c_r4_str9${label_sfx}_alpha1:
	//Load out two values
	ldr	s24, [C_r4, #32]
	//Apply Alpha to our element of C
	fmla	s14, s24, v31.s[1]
	str	s14, [C_r4, #32]

	b .Lsgemm_c_r4_odd_end${label_sfx}_alpha1


.Lsgemm_c_r4_str7${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v13.s[2]
	ldr	s24, [C_r4, #24]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r4, #24]
.Lsgemm_c_r4_str6${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v13.s[1]
	ldr	s24, [C_r4, #20]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r4, #20]
.Lsgemm_c_r4_str5${label_sfx}_alpha1:
	//Load out two values
	ldr	s24, [C_r4, #16]
	//Apply Alpha to our element of C
	fmla	s13, s24, v31.s[1]
	str	s13, [C_r4, #16]

	b .Lsgemm_c_r4_odd_end${label_sfx}_alpha1

.Lsgemm_c_r4_str3${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v12.s[2]
	ldr	s24, [C_r4, 8]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r4, 8]
.Lsgemm_c_r4_str2${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v12.s[1]
	ldr	s24, [C_r4, 4]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r4, 4]
.Lsgemm_c_r4_str1${label_sfx}_alpha1:
	//Load out two values
	ldr	s24, [C_r4]
	//Apply Alpha to our element of C
	fmla	s12, s24, v31.s[1]
	str	s12, [C_r4]

	b .Lsgemm_c_r4_odd_end${label_sfx}_alpha1


.Lsgemm_c_r4_odd_end${label_sfx}_alpha1:
	add	C_r4, C_r4, #48

	add tmp_x, b_row_idx, #5
	cmp	tmp_x, n_size
	beq	.LOOP4END${label_sfx}

.Lsgemm_c_r5_even${label_sfx}_alpha1:

	ins v31.s[1], beta_as_int

	add	tmp_x, a_row_idx, 12
	cmp	tmp_x, m_size
	bls .Lsgemm_c_r5_str12${label_sfx}_alpha1

	add	tmp_x, a_row_idx, 8
	cmp	m_size, tmp_x
	bcs .Lsgemm_c_r5_str8${label_sfx}_alpha1

	add	tmp_x, a_row_idx, 4
	cmp	m_size, tmp_x
	bcs .Lsgemm_c_r5_str4${label_sfx}_alpha1

	b .Lsgemm_c_r5_odd${label_sfx}_alpha1


.Lsgemm_c_r5_str12${label_sfx}_alpha1:
	ldr	q29, [C_r5, #32]
	fmla    v17.4s, v29.4s, v31.s[1]
	str q17, [C_r5, #32]
.Lsgemm_c_r5_str8${label_sfx}_alpha1:
	ldr	q28, [C_r5, #16]
	fmla v16.4s, v28.4s, v31.s[1]
	str q16, [C_r5, #16]
.Lsgemm_c_r5_str4${label_sfx}_alpha1:
	ldr	q27, [C_r5]
	fmla    v15.4s, v27.4s, v31.s[1]
	str q15, [C_r5]

.Lsgemm_c_r5_odd${label_sfx}_alpha1:
	add	tmp_x, a_row_idx, 1
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r5_str1${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 2
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r5_str2${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 3
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r5_str3${label_sfx}_alpha1


	add	tmp_x, a_row_idx, 5
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r5_str5${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 6
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r5_str6${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 7
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r5_str7${label_sfx}_alpha1


	add	tmp_x, a_row_idx, 9
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r5_str9${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 10
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r5_str10${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 11
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r5_str11${label_sfx}_alpha1


	b .Lsgemm_c_r5_odd_end${label_sfx}_alpha1

.Lsgemm_c_r5_str11${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v17.s[2]
	ldr	s27, [C_r5, #40]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r5, #40]
.Lsgemm_c_r5_str10${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v17.s[1]
	ldr	s27, [C_r5, #36]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r5, #36]
.Lsgemm_c_r5_str9${label_sfx}_alpha1:
	//Load out two values
	ldr	s27, [C_r5, #32]
	//Apply Alpha to our element of C
	fmla	s17, s27, v31.s[1]
	str	s17, [C_r5, #32]

	b .Lsgemm_c_r5_odd_end${label_sfx}_alpha1


.Lsgemm_c_r5_str7${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v16.s[2]
	ldr	s27, [C_r5, #24]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r5, #24]
.Lsgemm_c_r5_str6${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v16.s[1]
	ldr	s27, [C_r5, #20]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r5, #20]
.Lsgemm_c_r5_str5${label_sfx}_alpha1:
	//Load out two values
	ldr	s27, [C_r5, #16]
	//Apply Alpha to our element of C
	fmla	s16, s27, v31.s[1]
	str	s16, [C_r5, #16]

	b .Lsgemm_c_r5_odd_end${label_sfx}_alpha1

.Lsgemm_c_r5_str3${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v15.s[2]
	ldr	s27, [C_r5, 8]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r5, 8]
.Lsgemm_c_r5_str2${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v15.s[1]
	ldr	s27, [C_r5, 4]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r5, 4]
.Lsgemm_c_r5_str1${label_sfx}_alpha1:
	//Load out two values
	ldr	s27, [C_r5]
	//Apply Alpha to our element of C
	fmla	s15, s27, v31.s[1]
	str	s15, [C_r5]

	b .Lsgemm_c_r5_odd_end${label_sfx}_alpha1


.Lsgemm_c_r5_odd_end${label_sfx}_alpha1:
	add	C_r5, C_r5, #48

	add tmp_x, b_row_idx, #6
	cmp	tmp_x, n_size
	beq	.LOOP4END${label_sfx}

.Lsgemm_c_r6_even${label_sfx}_alpha1:

	ins v31.s[1], beta_as_int

	add	tmp_x, a_row_idx, 12
	cmp	tmp_x, m_size
	bls .Lsgemm_c_r6_str12${label_sfx}_alpha1

	add	tmp_x, a_row_idx, 8
	cmp	m_size, tmp_x
	bcs .Lsgemm_c_r6_str8${label_sfx}_alpha1

	add	tmp_x, a_row_idx, 4
	cmp	m_size, tmp_x
	bcs .Lsgemm_c_r6_str4${label_sfx}_alpha1

	b .Lsgemm_c_r6_odd${label_sfx}_alpha1


.Lsgemm_c_r6_str12${label_sfx}_alpha1:
	ldr	q26, [C_r6, #32]
	fmla    v20.4s, v26.4s, v31.s[1]
	str q20, [C_r6, #32]
.Lsgemm_c_r6_str8${label_sfx}_alpha1:
	ldr	q25, [C_r6, #16]
	fmla v19.4s, v25.4s, v31.s[1]
	str q19, [C_r6, #16]
.Lsgemm_c_r6_str4${label_sfx}_alpha1:
	ldr	q24, [C_r6]
	fmla    v18.4s, v24.4s, v31.s[1]
	str q18, [C_r6]

.Lsgemm_c_r6_odd${label_sfx}_alpha1:
	add	tmp_x, a_row_idx, 1
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r6_str1${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 2
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r6_str2${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 3
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r6_str3${label_sfx}_alpha1


	add	tmp_x, a_row_idx, 5
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r6_str5${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 6
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r6_str6${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 7
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r6_str7${label_sfx}_alpha1


	add	tmp_x, a_row_idx, 9
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r6_str9${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 10
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r6_str10${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 11
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r6_str11${label_sfx}_alpha1


	b .Lsgemm_c_r6_odd_end${label_sfx}_alpha1

.Lsgemm_c_r6_str11${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v20.s[2]
	ldr	s24, [C_r6, #40]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r6, #40]
.Lsgemm_c_r6_str10${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v20.s[1]
	ldr	s24, [C_r6, #36]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r6, #36]
.Lsgemm_c_r6_str9${label_sfx}_alpha1:
	//Load out two values
	ldr	s24, [C_r6, #32]
	//Apply Alpha to our element of C
	fmla	s20, s24, v31.s[1]
	str	s20, [C_r6, #32]

	b .Lsgemm_c_r6_odd_end${label_sfx}_alpha1


.Lsgemm_c_r6_str7${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v19.s[2]
	ldr	s24, [C_r6, #24]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r6, #24]
.Lsgemm_c_r6_str6${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v19.s[1]
	ldr	s24, [C_r6, #20]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r6, #20]
.Lsgemm_c_r6_str5${label_sfx}_alpha1:
	//Load out two values
	ldr	s24, [C_r6, #16]
	//Apply Alpha to our element of C
	fmla	s19, s24, v31.s[1]
	str	s19, [C_r6, #16]

	b .Lsgemm_c_r6_odd_end${label_sfx}_alpha1

.Lsgemm_c_r6_str3${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v18.s[2]
	ldr	s24, [C_r6, 8]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r6, 8]
.Lsgemm_c_r6_str2${label_sfx}_alpha1:
	//Load out two values
	dup	s25, v18.s[1]
	ldr	s24, [C_r6, 4]
	//Apply Alpha to our element of C
	fmla	s25, s24, v31.s[1]
	str	s25, [C_r6, 4]
.Lsgemm_c_r6_str1${label_sfx}_alpha1:
	//Load out two values
	ldr	s24, [C_r6]
	//Apply Alpha to our element of C
	fmla	s18, s24, v31.s[1]
	str	s18, [C_r6]

	b .Lsgemm_c_r6_odd_end${label_sfx}_alpha1


.Lsgemm_c_r6_odd_end${label_sfx}_alpha1:
	add	C_r6, C_r6, #48

	add tmp_x, b_row_idx, #7
	cmp	tmp_x, n_size
	beq	.LOOP4END${label_sfx}

.Lsgemm_c_r7_even${label_sfx}_alpha1:

	ins v31.s[1], beta_as_int

	add	tmp_x, a_row_idx, 12
	cmp	tmp_x, m_size
	bls .Lsgemm_c_r7_str12${label_sfx}_alpha1

	add	tmp_x, a_row_idx, 8
	cmp	m_size, tmp_x
	bcs .Lsgemm_c_r7_str8${label_sfx}_alpha1

	add	tmp_x, a_row_idx, 4
	cmp	m_size, tmp_x
	bcs .Lsgemm_c_r7_str4${label_sfx}_alpha1

	b .Lsgemm_c_r7_odd${label_sfx}_alpha1


.Lsgemm_c_r7_str12${label_sfx}_alpha1:
	ldr	q29, [C_r7, #32]
	fmla    v23.4s, v29.4s, v31.s[1]
	str q23, [C_r7, #32]
.Lsgemm_c_r7_str8${label_sfx}_alpha1:
	ldr	q28, [C_r7, #16]
	fmla v22.4s, v28.4s, v31.s[1]
	str q22, [C_r7, #16]
.Lsgemm_c_r7_str4${label_sfx}_alpha1:
	ldr	q27, [C_r7]
	fmla    v21.4s, v27.4s, v31.s[1]
	str q21, [C_r7]

.Lsgemm_c_r7_odd${label_sfx}_alpha1:
	add	tmp_x, a_row_idx, 1
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r7_str1${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 2
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r7_str2${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 3
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r7_str3${label_sfx}_alpha1


	add	tmp_x, a_row_idx, 5
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r7_str5${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 6
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r7_str6${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 7
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r7_str7${label_sfx}_alpha1


	add	tmp_x, a_row_idx, 9
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r7_str9${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 10
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r7_str10${label_sfx}_alpha1
	add	tmp_x, a_row_idx, 11
	cmp	m_size, tmp_x
	beq .Lsgemm_c_r7_str11${label_sfx}_alpha1


	b .Lsgemm_c_r7_odd_end${label_sfx}_alpha1

.Lsgemm_c_r7_str11${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v23.s[2]
	ldr	s27, [C_r7, #40]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r7, #40]
.Lsgemm_c_r7_str10${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v23.s[1]
	ldr	s27, [C_r7, #36]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r7, #36]
.Lsgemm_c_r7_str9${label_sfx}_alpha1:
	//Load out two values
	ldr	s27, [C_r7, #32]
	//Apply Alpha to our element of C
	fmla	s23, s27, v31.s[1]
	str	s23, [C_r7, #32]

	b .Lsgemm_c_r7_odd_end${label_sfx}_alpha1


.Lsgemm_c_r7_str7${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v22.s[2]
	ldr	s27, [C_r7, #24]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r7, #24]
.Lsgemm_c_r7_str6${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v22.s[1]
	ldr	s27, [C_r7, #20]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r7, #20]
.Lsgemm_c_r7_str5${label_sfx}_alpha1:
	//Load out two values
	ldr	s27, [C_r7, #16]
	//Apply Alpha to our element of C
	fmla	s22, s27, v31.s[1]
	str	s22, [C_r7, #16]

	b .Lsgemm_c_r7_odd_end${label_sfx}_alpha1

.Lsgemm_c_r7_str3${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v21.s[2]
	ldr	s27, [C_r7, 8]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r7, 8]
.Lsgemm_c_r7_str2${label_sfx}_alpha1:
	//Load out two values
	dup	s28, v21.s[1]
	ldr	s27, [C_r7, 4]
	//Apply Alpha to our element of C
	fmla	s28, s27, v31.s[1]
	str	s28, [C_r7, 4]
.Lsgemm_c_r7_str1${label_sfx}_alpha1:
	//Load out two values
	ldr	s27, [C_r7]
	//Apply Alpha to our element of C
	fmla	s21, s27, v31.s[1]
	str	s21, [C_r7]

	b .Lsgemm_c_r7_odd_end${label_sfx}_alpha1


.Lsgemm_c_r7_odd_end${label_sfx}_alpha1:
	add	C_r7, C_r7, #48

	b .LOOP4END${label_sfx}
</%def>

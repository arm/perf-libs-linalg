## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

// arg0 : k  (int64)
// arg1 : a_ptr (float *)
// arg2 : lda (int64)
// arg3 : b_ptr (float *)
// arg4 : ldb (int64)
// arg5 : c_ptr (float *)
// arg6 : ldc (int64)
// arg7 : alpha (float)
// arg8 : beta (float) or fixed 0.0 or 1.0

#define k x0
#define a_ptr0 x1
#define lda x2
#define b_ptr0 x3
#define ldb x4
#define c_ptr0 x5
#define ldc x6
#define a_ptr1 x7
#define a_ptr2 x8
#define a_ptr3 x9
#define b_ptr1 x13
#define b_ptr2 x14
#define b_ptr3 x15
#define c_ptr1 x16
#define c_ptr2 x17
#define c_ptr3 x19
#define counter x20
#define lda4 x21
#define ldapf x22
#define ldamul x23

#define PRFCL2 64
#define PRFBL2 512
#define PRFBL1 128

%for beta, label_sfx in [ ("0", "_bet0"), ("1", "_bet1"), ("any", "") ]:
	<% func_name = "sgemm_alt_kernel_NN_n4m12k8" + label_sfx %>
	${prologue(func_name)}
	stp x29, x30, [sp, -256]!
	mov x29, sp
	stp x19, x20, [sp,  16]
	stp x21, x22, [sp,  32]
	stp x23, x24, [sp,  48]
	stp x25, x26, [sp,  64]
	stp x27, x28, [sp,  80]
	stp d8,  d9,  [sp,  96]
	stp d10, d11, [sp, 112]
	stp d12, d13, [sp, 128]
	stp d14, d15, [sp, 144]

%if beta == "any":
	// save alpha, beta
	stp s0, s1, [sp, 160]
%else:
	// save alpha
	str s0, [sp, 160]
%endif

	mov counter, k

	lsl lda, lda, 2
	lsl ldb, ldb, 2
	lsl ldc, ldc, 2

	mov ldamul, 4

	add b_ptr1, b_ptr0, ldb
	add b_ptr2, b_ptr1, ldb
	add b_ptr3, b_ptr2, ldb

	add a_ptr1, a_ptr0, lda
	add a_ptr2, a_ptr1, lda
	add a_ptr3, a_ptr2, lda

	add c_ptr1, c_ptr0, ldc
	add c_ptr2, c_ptr1, ldc
	add c_ptr3, c_ptr2, ldc

	lsl lda4, lda, 2
	mul ldapf, lda, ldamul

	ldr q12, [a_ptr0]
	ldr q13, [a_ptr0, 16]
	ldr q14, [a_ptr0, 32]
	ldr q15, [a_ptr1]
	ldr q16, [a_ptr1, 16]
	ldr q17, [a_ptr1, 32]
	ldr q18, [a_ptr2]
	ldr q19, [a_ptr2, 16]
	ldr q20, [a_ptr2, 32]
//	ldr q21, [a_ptr3]
//	ldr q22, [a_ptr3, 16]
//	ldr q23, [a_ptr3, 32]
	add a_ptr0, a_ptr0, lda4
	add a_ptr1, a_ptr1, lda4
	add a_ptr2, a_ptr2, lda4
//	add a_ptr3, a_ptr3, lda4

	ldr q24, [b_ptr0]
	ldr q26, [b_ptr1]
	ldr q28, [b_ptr2]
	ldr q30, [b_ptr3]
//	ldr q25, [b_ptr0, 16]
//	ldr q27, [b_ptr1, 16]
//	ldr q29, [b_ptr2, 16]
//	ldr q31, [b_ptr3, 16]

	dup v0.2d, xzr
	prfm pldl2keep, [b_ptr0, PRFBL2]
	dup v1.2d, xzr
	prfm pldl2keep, [b_ptr1, PRFBL2]
	dup v2.2d, xzr
	prfm pldl2keep, [b_ptr2, PRFBL2]
	dup v3.2d, xzr
	prfm pldl2keep, [b_ptr3, PRFBL2]
	dup v4.2d, xzr
	prfm pldl2keep, [c_ptr1, PRFCL2]
	dup v5.2d, xzr
	prfm pldl2keep, [c_ptr0, PRFCL2]
	dup v6.2d, xzr
	prfm pldl2keep, [c_ptr2, PRFCL2]
	dup v7.2d, xzr
	prfm pldl2keep, [c_ptr3, PRFCL2]
	dup v8.2d, xzr
//	prfm pldl2keep, [a_ptr0, PRFBL2]
	dup v9.2d, xzr
//	prfm pldl2keep, [a_ptr1, PRFBL2]
	dup v10.2d, xzr
//	prfm pldl2keep, [a_ptr2, PRFBL2]
	dup v11.2d, xzr
//	prfm pldl2keep, [a_ptr3, PRFBL2]

.Loop${label_sfx}:
	subs counter, counter, 8

	// k = 0
	ldr q25, [b_ptr0, 16]
	fmla v0.4s, v12.4s, v24.s[0]
	prfm pldl1keep, [b_ptr1, PRFBL1]
	fmla v1.4s, v13.4s, v24.s[0]
	ldr q27, [b_ptr1, 16]
	fmla v2.4s, v14.4s, v24.s[0]
	fmla v3.4s, v12.4s, v26.s[0]
	ldr q29, [b_ptr2, 16]
	fmla v4.4s, v13.4s, v26.s[0]
	prfm pldl1keep, [b_ptr2, PRFBL1]
	fmla v5.4s, v14.4s, v26.s[0]
	ldr q31, [b_ptr3, 16]
	fmla v6.4s, v12.4s, v28.s[0]
	fmla v7.4s, v13.4s, v28.s[0]
	ldr q21, [a_ptr3]
	fmla v8.4s, v14.4s, v28.s[0]
	prfm pldl1keep, [b_ptr3, PRFBL1]
	fmla v9.4s, v12.4s, v30.s[0]
	ldr q22, [a_ptr3, 16]
	fmla v10.4s, v13.4s, v30.s[0]
	fmla v11.4s, v14.4s, v30.s[0]
	ldr q23, [a_ptr3, 32]

	// k = 1
	fmla v0.4s, v15.4s, v24.s[1]
	fmla v1.4s, v16.4s, v24.s[1]
	prfm pldl1keep, [a_ptr0, ldapf]
	add a_ptr3, a_ptr3, lda4
	fmla v2.4s, v17.4s, v24.s[1]
	fmla v3.4s, v15.4s, v26.s[1]
	prfm pldl1keep, [a_ptr1, ldapf]
	fmla v4.4s, v16.4s, v26.s[1]
	fmla v5.4s, v17.4s, v26.s[1]
	ldr q12, [a_ptr0]
	fmla v6.4s, v15.4s, v28.s[1]
	fmla v7.4s, v16.4s, v28.s[1]
	ldr q13, [a_ptr0, 16]
	fmla v8.4s, v17.4s, v28.s[1]
	fmla v9.4s, v15.4s, v30.s[1]
	ldr q14, [a_ptr0, 32]
	fmla v10.4s, v16.4s, v30.s[1]
	fmla v11.4s, v17.4s, v30.s[1]
	add a_ptr0, a_ptr0, lda4

	// k = 2
	ldr q15, [a_ptr1]
	fmla v0.4s, v18.4s, v24.s[2]
	fmla v1.4s, v19.4s, v24.s[2]
	ldr q16, [a_ptr1, 16]
	fmla v2.4s, v20.4s, v24.s[2]
	fmla v3.4s, v18.4s, v26.s[2]
	ldr q17, [a_ptr1, 32]
	fmla v4.4s, v19.4s, v26.s[2]
	fmla v5.4s, v20.4s, v26.s[2]
	add a_ptr1, a_ptr1, lda4
	fmla v6.4s, v18.4s, v28.s[2]
	fmla v7.4s, v19.4s, v28.s[2]
	fmla v8.4s, v20.4s, v28.s[2]
	fmla v9.4s, v18.4s, v30.s[2]
	fmla v10.4s, v19.4s, v30.s[2]
	fmla v11.4s, v20.4s, v30.s[2]

	// k = 3
	fmla v0.4s, v21.4s, v24.s[3]
	fmla v1.4s, v22.4s, v24.s[3]
	prfm pldl1keep, [a_ptr2, ldapf]
	fmla v2.4s, v23.4s, v24.s[3]
	fmla v3.4s, v21.4s, v26.s[3]
	prfm pldl1keep, [a_ptr3, ldapf]
	fmla v4.4s, v22.4s, v26.s[3]
	fmla v5.4s, v23.4s, v26.s[3]
	ldr q18, [a_ptr2]
	fmla v6.4s, v21.4s, v28.s[3]
	fmla v7.4s, v22.4s, v28.s[3]
	ldr q19, [a_ptr2, 16]
	fmla v8.4s, v23.4s, v28.s[3]
	fmla v9.4s, v21.4s, v30.s[3]
	ldr q20, [a_ptr2, 32]
	fmla v10.4s, v22.4s, v30.s[3]
	fmla v11.4s, v23.4s, v30.s[3]
	prfm pldl1keep, [b_ptr0, PRFBL1]
	add a_ptr2, a_ptr2, lda4

	// k = 4
	ldr q24, [b_ptr0, 32]!
	fmla v0.4s, v12.4s, v25.s[0]
	prfm pldl1keep, [b_ptr1, PRFBL1]
	fmla v1.4s, v13.4s, v25.s[0]
	ldr q26, [b_ptr1, 32]!
	fmla v2.4s, v14.4s, v25.s[0]
	fmla v3.4s, v12.4s, v27.s[0]
	ldr q28, [b_ptr2, 32]!
	fmla v4.4s, v13.4s, v27.s[0]
	prfm pldl1keep, [b_ptr2, PRFBL1]
	fmla v5.4s, v14.4s, v27.s[0]
	fmla v6.4s, v12.4s, v29.s[0]
	fmla v7.4s, v13.4s, v29.s[0]
	ldr q21, [a_ptr3]
	fmla v8.4s, v14.4s, v29.s[0]
	prfm pldl1keep, [b_ptr3, PRFBL1]
	fmla v9.4s, v12.4s, v31.s[0]
	ldr q22, [a_ptr3, 16]
	fmla v10.4s, v13.4s, v31.s[0]
	fmla v11.4s, v14.4s, v31.s[0]
	ldr q23, [a_ptr3, 32]

	beq .Lwrite${label_sfx}

	// k = 5
	fmla v0.4s, v15.4s, v25.s[1]
	ldr q30, [b_ptr3, 32]!
	fmla v1.4s, v16.4s, v25.s[1]
	prfm pldl1keep, [a_ptr0, ldapf]
	add a_ptr3, a_ptr3, lda4
	fmla v2.4s, v17.4s, v25.s[1]
	fmla v3.4s, v15.4s, v27.s[1]
	prfm pldl1keep, [a_ptr1, ldapf]
	fmla v4.4s, v16.4s, v27.s[1]
	fmla v5.4s, v17.4s, v27.s[1]
	ldr q12, [a_ptr0]
	fmla v6.4s, v15.4s, v29.s[1]
	fmla v7.4s, v16.4s, v29.s[1]
	ldr q13, [a_ptr0, 16]
	fmla v8.4s, v17.4s, v29.s[1]
	fmla v9.4s, v15.4s, v31.s[1]
	ldr q14, [a_ptr0, 32]
	fmla v10.4s, v16.4s, v31.s[1]
	fmla v11.4s, v17.4s, v31.s[1]
	add a_ptr0, a_ptr0, lda4

	// k = 6
	ldr q15, [a_ptr1]
	fmla v0.4s, v18.4s, v25.s[2]
	fmla v1.4s, v19.4s, v25.s[2]
	ldr q16, [a_ptr1, 16]
	fmla v2.4s, v20.4s, v25.s[2]
	fmla v3.4s, v18.4s, v27.s[2]
	ldr q17, [a_ptr1, 32]
	fmla v4.4s, v19.4s, v27.s[2]
	fmla v5.4s, v20.4s, v27.s[2]
	add a_ptr1, a_ptr1, lda4
	fmla v6.4s, v18.4s, v29.s[2]
	fmla v7.4s, v19.4s, v29.s[2]
	fmla v8.4s, v20.4s, v29.s[2]
	fmla v9.4s, v18.4s, v31.s[2]
	fmla v10.4s, v19.4s, v31.s[2]
	fmla v11.4s, v20.4s, v31.s[2]

	// k = 7
	fmla v0.4s, v21.4s, v25.s[3]
	fmla v1.4s, v22.4s, v25.s[3]
	prfm pldl1keep, [a_ptr2, ldapf]
	fmla v2.4s, v23.4s, v25.s[3]
	fmla v3.4s, v21.4s, v27.s[3]
	prfm pldl1keep, [a_ptr3, ldapf]
	fmla v4.4s, v22.4s, v27.s[3]
	fmla v5.4s, v23.4s, v27.s[3]
	ldr q18, [a_ptr2]
	fmla v6.4s, v21.4s, v29.s[3]
	fmla v7.4s, v22.4s, v29.s[3]
	ldr q19, [a_ptr2, 16]
	fmla v8.4s, v23.4s, v29.s[3]
	fmla v9.4s, v21.4s, v31.s[3]
	ldr q20, [a_ptr2, 32]
	fmla v10.4s, v22.4s, v31.s[3]
	fmla v11.4s, v23.4s, v31.s[3]
	prfm pldl1keep, [b_ptr0, PRFBL1]
	add a_ptr2, a_ptr2, lda4

	b .Loop${label_sfx}

.Lwrite${label_sfx}:

%if beta == "any":
	// load alpha, beta
	ldp s24, s26, [sp, 160]
%else:
	// load alpha
	ldr s24, [sp, 160]
%endif

	// k = 5
	fmla v0.4s, v15.4s, v25.s[1]
	fmla v1.4s, v16.4s, v25.s[1]
	fmla v2.4s, v17.4s, v25.s[1]
	fmla v3.4s, v15.4s, v27.s[1]
	fmla v4.4s, v16.4s, v27.s[1]
	fmla v5.4s, v17.4s, v27.s[1]
	fmla v6.4s, v15.4s, v29.s[1]
	fmla v7.4s, v16.4s, v29.s[1]
	fmla v8.4s, v17.4s, v29.s[1]
	fmla v9.4s, v15.4s, v31.s[1]
	fmla v10.4s, v16.4s, v31.s[1]
	fmla v11.4s, v17.4s, v31.s[1]

	// k = 6
	fmla v0.4s, v18.4s, v25.s[2]
	fmla v1.4s, v19.4s, v25.s[2]
	fmla v2.4s, v20.4s, v25.s[2]
	fmla v3.4s, v18.4s, v27.s[2]
	fmla v4.4s, v19.4s, v27.s[2]
	fmla v5.4s, v20.4s, v27.s[2]
	fmla v6.4s, v18.4s, v29.s[2]
	fmla v7.4s, v19.4s, v29.s[2]
	fmla v8.4s, v20.4s, v29.s[2]
	fmla v9.4s, v18.4s, v31.s[2]
	fmla v10.4s, v19.4s, v31.s[2]
	fmla v11.4s, v20.4s, v31.s[2]

	// k = 7
	fmla v0.4s, v21.4s, v25.s[3]
	fmla v1.4s, v22.4s, v25.s[3]
	fmla v2.4s, v23.4s, v25.s[3]
	fmla v3.4s, v21.4s, v27.s[3]
	fmla v4.4s, v22.4s, v27.s[3]
	fmla v5.4s, v23.4s, v27.s[3]
	fmla v6.4s, v21.4s, v29.s[3]
	fmla v7.4s, v22.4s, v29.s[3]
	fmla v8.4s, v23.4s, v29.s[3]
	fmla v9.4s, v21.4s, v31.s[3]
	fmla v10.4s, v22.4s, v31.s[3]
	fmla v11.4s, v23.4s, v31.s[3]

%if beta == "0":
	// multiply A*B result by alpha and store in C (beta==0)
	fmul v12.4s, v0.4s, v24.s[0]
	fmul v13.4s, v1.4s, v24.s[0]
	fmul v14.4s, v2.4s, v24.s[0]
	fmul v15.4s, v3.4s, v24.s[0]
	fmul v16.4s, v4.4s, v24.s[0]
	fmul v17.4s, v5.4s, v24.s[0]
	fmul v18.4s, v6.4s, v24.s[0]
	fmul v19.4s, v7.4s, v24.s[0]
	fmul v20.4s, v8.4s, v24.s[0]
	fmul v21.4s, v9.4s, v24.s[0]
	fmul v22.4s, v10.4s, v24.s[0]
	fmul v23.4s, v11.4s, v24.s[0]
%else:
	ldp q12, q13, [c_ptr0]
	ldr q14, [c_ptr0, 32]
	ldp q15, q16, [c_ptr1]
	ldr q17, [c_ptr1, 32]
	ldp q18, q19, [c_ptr2]
	ldr q20, [c_ptr2, 32]
	ldp q21, q22, [c_ptr3]
	ldr q23, [c_ptr3, 32]

%if beta == "any":
	// multiply C by beta
	fmul v12.4s, v12.4s, v26.s[0]
	fmul v13.4s, v13.4s, v26.s[0]
	fmul v14.4s, v14.4s, v26.s[0]
	fmul v15.4s, v15.4s, v26.s[0]
	fmul v16.4s, v16.4s, v26.s[0]
	fmul v17.4s, v17.4s, v26.s[0]
	fmul v18.4s, v18.4s, v26.s[0]
	fmul v19.4s, v19.4s, v26.s[0]
	fmul v20.4s, v20.4s, v26.s[0]
	fmul v21.4s, v21.4s, v26.s[0]
	fmul v22.4s, v22.4s, v26.s[0]
	fmul v23.4s, v23.4s, v26.s[0]
%endif
	// multiply A*B result by alpha and add in to C
	fmla v12.4s, v0.4s, v24.s[0]
	fmla v13.4s, v1.4s, v24.s[0]
	fmla v14.4s, v2.4s, v24.s[0]
	fmla v15.4s, v3.4s, v24.s[0]
	fmla v16.4s, v4.4s, v24.s[0]
	fmla v17.4s, v5.4s, v24.s[0]
	fmla v18.4s, v6.4s, v24.s[0]
	fmla v19.4s, v7.4s, v24.s[0]
	fmla v20.4s, v8.4s, v24.s[0]
	fmla v21.4s, v9.4s, v24.s[0]
	fmla v22.4s, v10.4s, v24.s[0]
	fmla v23.4s, v11.4s, v24.s[0]
%endif
	stp q12, q13, [c_ptr0]
	str q14, [c_ptr0, 32]
	stp q15, q16, [c_ptr1]
	str q17, [c_ptr1, 32]
	stp q18, q19, [c_ptr2]
	str q20, [c_ptr2, 32]
	stp q21, q22, [c_ptr3]
	str q23, [c_ptr3, 32]

.Lend${label_sfx}:
	ldp x19, x20, [sp, #16]
	ldp x21, x22, [sp, #32]
	ldp x23, x24, [sp, #48]
	ldp x25, x26, [sp, #64]
	ldp x27, x28, [sp, #80]
	ldp d8,  d9,  [sp,  96]
	ldp d10, d11, [sp, 112]
	ldp d12, d13, [sp, 128]
	ldp d14, d15, [sp, 144]
	ldp x29, x30, [sp], 256
	ret
	${epilogue(func_name)}
%endfor

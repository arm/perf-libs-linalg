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
#define a_ptr4 x10
#define b_ptr1 x13
#define b_ptr2 x14
#define c_ptr1 x15
#define c_ptr2 x16
#define counter x20

#define PRFAL1 128
#define PRFAL2 512
#define PRFBL2 512
#define PRFBL1 128
#define PRFCL2 64

%for beta, label_sfx in [ ("0", "_bet0"), ("1", "_bet1"), ("any", "") ]:
	<% func_name = "sgemm_alt_kernel_TN_n3m5k8" + label_sfx %>
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

	add b_ptr1, b_ptr0, ldb
	add b_ptr2, b_ptr1, ldb

	add a_ptr1, a_ptr0, lda
	add a_ptr2, a_ptr1, lda
	add a_ptr3, a_ptr2, lda
	add a_ptr4, a_ptr3, lda

	add c_ptr1, c_ptr0, ldc
	add c_ptr2, c_ptr1, ldc

	ldr q15, [a_ptr0]
	ldr q17, [a_ptr1]
	ldr q19, [a_ptr2]
	ldr q21, [a_ptr3]
	ldr q23, [a_ptr4]

	ldr q25, [b_ptr0]
	ldr q27, [b_ptr1]
	ldr q29, [b_ptr2]

	dup v0.2d, xzr
	prfm pldl2keep, [c_ptr0, PRFCL2]
	dup v1.2d, xzr
//	prfm pldl2keep, [b_ptr0, PRFBL2]
	dup v2.2d, xzr
	prfm pldl2keep, [c_ptr1, PRFCL2]
	dup v3.2d, xzr
//	prfm pldl2keep, [b_ptr1, PRFBL2]
	dup v4.2d, xzr
	prfm pldl2keep, [c_ptr2, PRFCL2]
	dup v5.2d, xzr
//	prfm pldl2keep, [b_ptr2, PRFBL2]
	dup v6.2d, xzr
	prfm pldl2keep, [a_ptr0, PRFCL2]
	dup v7.2d, xzr
	dup v8.2d, xzr
	prfm pldl2keep, [a_ptr1, PRFAL2]
	dup v9.2d, xzr
	dup v10.2d, xzr
	prfm pldl2keep, [a_ptr2, PRFAL2]
	dup v11.2d, xzr
	dup v12.2d, xzr
	prfm pldl2keep, [a_ptr3, PRFAL2]
	dup v13.2d, xzr
	dup v14.2d, xzr
	prfm pldl2keep, [a_ptr4, PRFAL2]

.Loop${label_sfx}:
	subs counter, counter, 8

	ldr q26, [b_ptr0, 16]
	fmla v0.4s, v15.4s, v25.4s
	prfm pldl1keep, [b_ptr0, PRFBL1]
	fmla v1.4s, v17.4s, v25.4s
	ldr q28, [b_ptr1, 16]
	fmla v2.4s, v19.4s, v25.4s
	prfm pldl1keep, [b_ptr1, PRFBL1]
	fmla v3.4s, v21.4s, v25.4s
	ldr q30, [b_ptr2, 16]
	fmla v4.4s, v23.4s, v25.4s
	prfm pldl1keep, [b_ptr2, PRFBL1]
	fmla v5.4s, v15.4s, v27.4s
	ldr q16, [a_ptr0, 16]
	fmla v6.4s, v17.4s, v27.4s
	prfm pldl1keep, [a_ptr0, PRFAL1]
	fmla v7.4s, v19.4s, v27.4s
	ldr q18, [a_ptr1, 16]
	fmla v8.4s, v21.4s, v27.4s
	prfm pldl1keep, [a_ptr1, PRFAL1]
	fmla v9.4s, v23.4s, v27.4s
	ldr q20, [a_ptr2, 16]
	fmla v10.4s, v15.4s, v29.4s
	prfm pldl1keep, [a_ptr2, PRFAL1]
	fmla v11.4s, v17.4s, v29.4s
	ldr q22, [a_ptr3, 16]
	fmla v12.4s, v19.4s, v29.4s
	prfm pldl1keep, [a_ptr3, PRFAL1]
	fmla v13.4s, v21.4s, v29.4s
	ldr q24, [a_ptr4, 16]
	fmla v14.4s, v23.4s, v29.4s
	prfm pldl1keep, [a_ptr4, PRFAL1]

	beq .Lwrite${label_sfx}

	ldr q25, [b_ptr0, 32]!
	fmla v0.4s, v16.4s, v26.4s
	fmla v1.4s, v18.4s, v26.4s
	ldr q27, [b_ptr1, 32]!
	fmla v2.4s, v20.4s, v26.4s
	fmla v3.4s, v22.4s, v26.4s
	ldr q29, [b_ptr2, 32]!
	fmla v4.4s, v24.4s, v26.4s
	fmla v5.4s, v16.4s, v28.4s
	ldr q15, [a_ptr0, 32]!
	fmla v6.4s, v18.4s, v28.4s
	fmla v7.4s, v20.4s, v28.4s
	ldr q17, [a_ptr1, 32]!
	fmla v8.4s, v22.4s, v28.4s
	fmla v9.4s, v24.4s, v28.4s
	ldr q19, [a_ptr2, 32]!
	fmla v10.4s, v16.4s, v30.4s
	fmla v11.4s, v18.4s, v30.4s
	ldr q21, [a_ptr3, 32]!
	fmla v12.4s, v20.4s, v30.4s
	fmla v13.4s, v22.4s, v30.4s
	ldr q23, [a_ptr4, 32]!
	fmla v14.4s, v24.4s, v30.4s

	b .Loop${label_sfx}

.Lwrite${label_sfx}:

	fmla v0.4s, v16.4s, v26.4s
	fmla v1.4s, v18.4s, v26.4s
	fmla v2.4s, v20.4s, v26.4s
	fmla v3.4s, v22.4s, v26.4s
	fmla v4.4s, v24.4s, v26.4s
	fmla v5.4s, v16.4s, v28.4s
	fmla v6.4s, v18.4s, v28.4s
	fmla v7.4s, v20.4s, v28.4s
	fmla v8.4s, v22.4s, v28.4s
	fmla v9.4s, v24.4s, v28.4s
	fmla v10.4s, v16.4s, v30.4s
	fmla v11.4s, v18.4s, v30.4s
	fmla v12.4s, v20.4s, v30.4s
	fmla v13.4s, v22.4s, v30.4s
	fmla v14.4s, v24.4s, v30.4s

%if beta != "any":
	// load alpha
	ldr s30, [sp, 160]
%endif

	dup v31.2d, xzr
	faddp v15.4s, v0.4s, v31.4s
	faddp v16.4s, v1.4s, v31.4s
	faddp v17.4s, v2.4s, v31.4s
	faddp v18.4s, v3.4s, v31.4s
	faddp v19.4s, v4.4s, v31.4s
	faddp v20.4s, v5.4s, v31.4s
	faddp v21.4s, v6.4s, v31.4s
	faddp v22.4s, v7.4s, v31.4s
	faddp v23.4s, v8.4s, v31.4s
	faddp v24.4s, v9.4s, v31.4s
	faddp v25.4s, v10.4s, v31.4s
	faddp v26.4s, v11.4s, v31.4s
	faddp v27.4s, v12.4s, v31.4s
	faddp v28.4s, v13.4s, v31.4s
	faddp v29.4s, v14.4s, v31.4s
	faddp s0, v15.2s
	faddp s1, v16.2s
	faddp s2, v17.2s
	faddp s3, v18.2s
	faddp s4, v19.2s
	faddp s5, v20.2s
	faddp s6, v21.2s
	faddp s7, v22.2s
	faddp s8, v23.2s
	faddp s9, v24.2s
	faddp s10, v25.2s
	faddp s11, v26.2s
	faddp s12, v27.2s
	faddp s13, v28.2s
	faddp s14, v29.2s

%if beta == "0":
	// multiple A*B by alpha
	fmul s15, s0, s30
	fmul s16, s1, s30
	fmul s17, s2, s30
	fmul s18, s3, s30
	fmul s19, s4, s30
	fmul s20, s5, s30
	fmul s21, s6, s30
	fmul s22, s7, s30
	fmul s23, s8, s30
	fmul s24, s9, s30
	fmul s25, s10, s30
	fmul s26, s11, s30
	fmul s27, s12, s30
	fmul s28, s13, s30
	fmul s29, s14, s30
%else:
%if beta == "any":
	// load beta
	ldr s30, [sp, 164]
%endif
	// load C
	ldr s15, [c_ptr0]
	ldr s16, [c_ptr0, 4]
	ldr s17, [c_ptr0, 8]
	ldr s18, [c_ptr0, 12]
	ldr s19, [c_ptr0, 16]
	ldr s20, [c_ptr1]
	ldr s21, [c_ptr1, 4]
	ldr s22, [c_ptr1, 8]
	ldr s23, [c_ptr1, 12]
	ldr s24, [c_ptr1, 16]
	ldr s25, [c_ptr2]
	ldr s26, [c_ptr2, 4]
	ldr s27, [c_ptr2, 8]
	ldr s28, [c_ptr2, 12]
	ldr s29, [c_ptr2, 16]
%if beta == "any":
	// multiply C by beta
	fmul s15, s15, s30
	fmul s16, s16, s30
	fmul s17, s17, s30
	fmul s18, s18, s30
	fmul s19, s19, s30
	fmul s20, s20, s30
	fmul s21, s21, s30
	fmul s22, s22, s30
	fmul s23, s23, s30
	fmul s24, s24, s30
	fmul s25, s25, s30
	fmul s26, s26, s30
	fmul s27, s27, s30
	fmul s28, s28, s30
	fmul s29, s29, s30

	// load alpha
	ldr s30, [sp, 160]
%endif
	// multiply AB with alpha and accumulate into C
	fmadd s15, s0, s30, s15
	fmadd s16, s1, s30, s16
	fmadd s17, s2, s30, s17
	fmadd s18, s3, s30, s18
	fmadd s19, s4, s30, s19
	fmadd s20, s5, s30, s20
	fmadd s21, s6, s30, s21
	fmadd s22, s7, s30, s22
	fmadd s23, s8, s30, s23
	fmadd s24, s9, s30, s24
	fmadd s25, s10, s30, s25
	fmadd s26, s11, s30, s26
	fmadd s27, s12, s30, s27
	fmadd s28, s13, s30, s28
	fmadd s29, s14, s30, s29
%endif
	str s15, [c_ptr0]
	str s16, [c_ptr0, 4]
	str s17, [c_ptr0, 8]
	str s18, [c_ptr0, 12]
	str s19, [c_ptr0, 16]
	str s20, [c_ptr1]
	str s21, [c_ptr1, 4]
	str s22, [c_ptr1, 8]
	str s23, [c_ptr1, 12]
	str s24, [c_ptr1, 16]
	str s25, [c_ptr2]
	str s26, [c_ptr2, 4]
	str s27, [c_ptr2, 8]
	str s28, [c_ptr2, 12]
	str s29, [c_ptr2, 16]

.Lend${label_sfx}:
	ldp x19, x20, [sp, 16]
	ldp x21, x22, [sp, 32]
	ldp x23, x24, [sp, 48]
	ldp x25, x26, [sp, 64]
	ldp x27, x28, [sp, 80]
	ldp d8,  d9,  [sp,  96]
	ldp d10, d11, [sp, 112]
	ldp d12, d13, [sp, 128]
	ldp d14, d15, [sp, 144]
	ldp x29, x30, [sp], 256
	ret
	${epilogue(func_name)}
%endfor

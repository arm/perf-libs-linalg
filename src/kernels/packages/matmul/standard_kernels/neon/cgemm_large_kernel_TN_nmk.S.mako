## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>
<%namespace file="cgemm_save_indiv.s.inc" import="cgemm_save_indiv"/>

// parameters
#define a_ptr x0
#define b_ptr x1
#define c_ptr x2

#define k x3
#define m x4
#define n x5

#define ldc x6
#define lda k
#define ldb k

// locals
// x13, x14 disallowed in Arm64EC
#define idx_m x15 // [0,4,8,...,m)
#define idx_k x16
#define idx_n x17
// x18 may be in use for the Platform Register

#define alpha v0
#define t x21

<%def name="loop_k_body(fmlx1, fmlx2, fmlx3, fmlx4, ar_re, ar_im, ar_re_other, ar_im_other, n_remainder)">

	${fmlx1} cr0_re0.4s, ${ar_re}, br01.s[0]
	subs idx_k, idx_k, #1
	${fmlx2} cr0_im0.4s, ${ar_re}, br01.s[1]
	${fmlx3} cr0_re0.4s, ${ar_im}, br01.s[1]
	ldr br89_q, [ local_b_ptr01, 64 ]
	${fmlx4} cr0_im0.4s, ${ar_im}, br01.s[0]
	add local_b_ptr01, local_b_ptr01, 80

%if n_remainder>1:
	${fmlx1} cr0_re1.4s, ${ar_re}, br01.s[2]
	${fmlx2} cr0_im1.4s, ${ar_re}, br01.s[3]
	${fmlx3} cr0_re1.4s, ${ar_im}, br01.s[3]
	${fmlx4} cr0_im1.4s, ${ar_im}, br01.s[2]
%endif
	add local_a_ptr0, local_a_ptr0, (1 << 5)

%if n_remainder>2:
	${fmlx1} cr0_re2.4s, ${ar_re}, br23.s[0]
	${fmlx2} cr0_im2.4s, ${ar_re}, br23.s[1]
	${fmlx3} cr0_re2.4s, ${ar_im}, br23.s[1]
	${fmlx4} cr0_im2.4s, ${ar_im}, br23.s[0]
%endif
	ldr br01_q, [ local_b_ptr01 ]

%if n_remainder>3:
	${fmlx1} cr0_re3.4s, ${ar_re}, br23.s[2]
	${fmlx2} cr0_im3.4s, ${ar_re}, br23.s[3]
	${fmlx3} cr0_re3.4s, ${ar_im}, br23.s[3]
	${fmlx4} cr0_im3.4s, ${ar_im}, br23.s[2]
%endif

%if n_remainder>4:
	${fmlx1} cr0_re4.4s, ${ar_re}, br45.s[0]
	${fmlx2} cr0_im4.4s, ${ar_re}, br45.s[1]
	${fmlx3} cr0_re4.4s, ${ar_im}, br45.s[1]
	${fmlx4} cr0_im4.4s, ${ar_im}, br45.s[0]
%endif
	ldr br23_q, [ local_b_ptr01, 16 ]

%if n_remainder>5:
	${fmlx1} cr0_re5.4s, ${ar_re}, br45.s[2]
	${fmlx2} cr0_im5.4s, ${ar_re}, br45.s[3]
	${fmlx3} cr0_re5.4s, ${ar_im}, br45.s[3]
	${fmlx4} cr0_im5.4s, ${ar_im}, br45.s[2]
%endif
	ldr ${ar_re_other}, [ local_a_ptr0 ]

%if n_remainder>6:
	${fmlx1} cr0_re6.4s, ${ar_re}, br67.s[0]
	${fmlx2} cr0_im6.4s, ${ar_re}, br67.s[1]
	${fmlx3} cr0_re6.4s, ${ar_im}, br67.s[1]
	${fmlx4} cr0_im6.4s, ${ar_im}, br67.s[0]
%endif
	ldr br45_q, [ local_b_ptr01, 32 ]

%if n_remainder>6:
	${fmlx1} cr0_re7.4s, ${ar_re}, br67.s[2]
	${fmlx2} cr0_im7.4s, ${ar_re}, br67.s[3]
	${fmlx3} cr0_re7.4s, ${ar_im}, br67.s[3]
	${fmlx4} cr0_im7.4s, ${ar_im}, br67.s[2]
%endif

%if n_remainder>7:
	${fmlx1} cr0_re8.4s, ${ar_re}, br89.s[0]
	${fmlx2} cr0_im8.4s, ${ar_re}, br89.s[1]
	${fmlx1} cr0_re9.4s, ${ar_re}, br89.s[2]
	${fmlx2} cr0_im9.4s, ${ar_re}, br89.s[3]
%endif
	ldr br67_q, [ local_b_ptr01, 48 ]

%if n_remainder>8:
	${fmlx3} cr0_re8.4s, ${ar_im}, br89.s[1]
	${fmlx4} cr0_im8.4s, ${ar_im}, br89.s[0]
	${fmlx3} cr0_re9.4s, ${ar_im}, br89.s[3]
	${fmlx4} cr0_im9.4s, ${ar_im}, br89.s[2]
%endif
	ldr ${ar_im_other}, [ local_a_ptr0, (1 << 4) ]
</%def>

<%def name="loop_k_last(fmlx1, fmlx2, fmlx3, fmlx4, ar_re, ar_im, n_remainder)">
## %if "0" in ar_re:
## .Lloop_k_last_n${n_remainder}_a0:
## %else :
## .Lloop_k_last_n${n_remainder}_a1:
## %endif
	${fmlx1} cr0_re0.4s, ${ar_re}, br01.s[0]
	${fmlx2} cr0_im0.4s, ${ar_re}, br01.s[1]
	${fmlx3} cr0_re0.4s, ${ar_im}, br01.s[1]
	${fmlx4} cr0_im0.4s, ${ar_im}, br01.s[0]

	ldr br89_q, [ local_b_ptr01, 64 ]

%if n_remainder>1:
	${fmlx1} cr0_re1.4s, ${ar_re}, br01.s[2]
	${fmlx2} cr0_im1.4s, ${ar_re}, br01.s[3]
	${fmlx3} cr0_re1.4s, ${ar_im}, br01.s[3]
	${fmlx4} cr0_im1.4s, ${ar_im}, br01.s[2]
%endif

%if n_remainder>2:
	${fmlx1} cr0_re2.4s, ${ar_re}, br23.s[0]
	${fmlx2} cr0_im2.4s, ${ar_re}, br23.s[1]
	${fmlx3} cr0_re2.4s, ${ar_im}, br23.s[1]
	${fmlx4} cr0_im2.4s, ${ar_im}, br23.s[0]
%endif

%if n_remainder>3:
	${fmlx1} cr0_re3.4s, ${ar_re}, br23.s[2]
	${fmlx2} cr0_im3.4s, ${ar_re}, br23.s[3]
	${fmlx3} cr0_re3.4s, ${ar_im}, br23.s[3]
	${fmlx4} cr0_im3.4s, ${ar_im}, br23.s[2]
%endif

%if n_remainder>4:
	${fmlx1} cr0_re4.4s, ${ar_re}, br45.s[0]
	${fmlx2} cr0_im4.4s, ${ar_re}, br45.s[1]
	${fmlx3} cr0_re4.4s, ${ar_im}, br45.s[1]
	${fmlx4} cr0_im4.4s, ${ar_im}, br45.s[0]
%endif

%if n_remainder>5:
	${fmlx1} cr0_re5.4s, ${ar_re}, br45.s[2]
	${fmlx2} cr0_im5.4s, ${ar_re}, br45.s[3]
	${fmlx3} cr0_re5.4s, ${ar_im}, br45.s[3]
	${fmlx4} cr0_im5.4s, ${ar_im}, br45.s[2]
%endif

%if n_remainder>6:
	${fmlx1} cr0_re6.4s, ${ar_re}, br67.s[0]
	${fmlx2} cr0_im6.4s, ${ar_re}, br67.s[1]
	${fmlx3} cr0_re6.4s, ${ar_im}, br67.s[1]
	${fmlx4} cr0_im6.4s, ${ar_im}, br67.s[0]
%endif

%if n_remainder>7:
	${fmlx1} cr0_re7.4s, ${ar_re}, br67.s[2]
	${fmlx2} cr0_im7.4s, ${ar_re}, br67.s[3]
	${fmlx3} cr0_re7.4s, ${ar_im}, br67.s[3]
	${fmlx4} cr0_im7.4s, ${ar_im}, br67.s[2]
%endif

%if n_remainder>8:
	${fmlx1} cr0_re8.4s, ${ar_re}, br89.s[0]
	${fmlx2} cr0_im8.4s, ${ar_re}, br89.s[1]
	${fmlx3} cr0_re8.4s, ${ar_im}, br89.s[1]
	${fmlx4} cr0_im8.4s, ${ar_im}, br89.s[0]
%endif

%if n_remainder>9:
	${fmlx1} cr0_re9.4s, ${ar_re}, br89.s[2]
	${fmlx2} cr0_im9.4s, ${ar_re}, br89.s[3]
	${fmlx3} cr0_re9.4s, ${ar_im}, br89.s[3]
	${fmlx4} cr0_im9.4s, ${ar_im}, br89.s[2]
%endif
</%def>

<%def name="generate_XN_kernel(func_name, fmlx1, fmlx2, fmlx3, fmlx4)">
	${prologue(func_name)}
	stp x29, x30, [ sp, #-384 ]!
	mov x29, sp
	stp x19, x20, [ sp, #144 ]
	stp x21, x22, [ sp, #160 ]
	stp x23, x24, [ sp, #176 ]
	stp x25, x26, [ sp, #192 ]
	stp x27, x28, [ sp, #208 ]
	stp d8, d9, [ sp, #304 ]
	stp d10, d11, [ sp, #320 ]
	stp d12, d13, [ sp, #336 ]
	stp d14, d15, [ sp, #352 ]

	%if target_os in ('windows', 'windows_arm64ec'):
		// move alpha and beta from GPR or load from stack
		lsr x8, x7, #0x20
		fmov s0, w7
		fmov s1, w8
		ldr s2, [sp, #384]
		ldr s3, [sp, #388]
	%endif

	mov alpha.s[0], v0.s[0]
	mov alpha.s[1], v1.s[0]

	eor idx_n, idx_n, idx_n
.L${func_name}_loop_n: // {{{
	add t, n, #1
	lsr t, t, #1

	//Need to decide which circuit to pursue
	subs t, t, idx_n
	ble .L${func_name}_loop_n_end
	eor idx_m, idx_m, idx_m
	cmp t, #1
	beq .L${func_name}_n2_loop_m
	cmp t, #2
	beq .L${func_name}_n4_loop_m
	cmp t, #3
	beq .L${func_name}_n6_loop_m
	cmp t, #4
	beq .L${func_name}_n8_loop_m

%for remaining_n in [10, 8, 6, 4, 2]:
.L${func_name}_n${remaining_n}_loop_m: // {{{
	cmp idx_m, m
	bge .L${func_name}_loop_m_end

#define ar_im0 v31
#define ar_re0 v30
#define ar_im1 v29
#define ar_re1 v28

#define ar_im0_q q31
#define ar_re0_q q30
#define ar_im1_q q29
#define ar_re1_q q28

#define br01 v27
#define br23 v26
#define br45 v25
#define br67 v24
#define br89 v23

#define br01_q q27
#define br23_q q26
#define br45_q q25
#define br67_q q24
#define br89_q q23

// x23, x24, x28 disallowed in Arm64EC
#define local_b_ptr01 x25
#define local_a_ptr0 x26

#define cr0_re0 v20
#define cr0_im0 v19
#define cr0_re1 v18
#define cr0_im1 v17
#define cr0_re2 v16
#define cr0_im2 v15
#define cr0_re3 v14
#define cr0_im3 v13
#define cr0_re4 v12
#define cr0_im4 v11
#define cr0_re5 v10
#define cr0_im5 v9
#define cr0_re6 v8
#define cr0_im6 v7
#define cr0_re7 v6
#define cr0_im7 v5
#define cr0_re8 v4
#define cr0_im8 v3
#define cr0_re9 v2
#define cr0_im9 v1

	adds idx_k, k, 0

	eor cr0_re0.16b, cr0_re0.16b, cr0_re0.16b
	eor cr0_im0.16b, cr0_im0.16b, cr0_im0.16b
	eor cr0_re1.16b, cr0_re1.16b, cr0_re1.16b
	eor cr0_im1.16b, cr0_im1.16b, cr0_im1.16b
	eor cr0_re2.16b, cr0_re2.16b, cr0_re2.16b
	eor cr0_im2.16b, cr0_im2.16b, cr0_im2.16b
	eor cr0_re3.16b, cr0_re3.16b, cr0_re3.16b
	eor cr0_im3.16b, cr0_im3.16b, cr0_im3.16b
	eor cr0_re4.16b, cr0_re4.16b, cr0_re4.16b
	eor cr0_im4.16b, cr0_im4.16b, cr0_im4.16b
	eor cr0_re5.16b, cr0_re5.16b, cr0_re5.16b
	eor cr0_im5.16b, cr0_im5.16b, cr0_im5.16b
	eor cr0_re6.16b, cr0_re6.16b, cr0_re6.16b
	eor cr0_im6.16b, cr0_im6.16b, cr0_im6.16b
	eor cr0_re7.16b, cr0_re7.16b, cr0_re7.16b
	eor cr0_im7.16b, cr0_im7.16b, cr0_im7.16b
	eor cr0_re8.16b, cr0_re8.16b, cr0_re8.16b
	eor cr0_im8.16b, cr0_im8.16b, cr0_im8.16b
	eor cr0_re9.16b, cr0_re9.16b, cr0_re9.16b
	eor cr0_im9.16b, cr0_im9.16b, cr0_im9.16b

	beq .L${func_name}_n${remaining_n}_loop_k_end
	subs idx_k, idx_k, #1

	mul local_a_ptr0, idx_m, lda
	add local_a_ptr0, a_ptr, local_a_ptr0, lsl 3

	eor local_b_ptr01, local_b_ptr01, local_b_ptr01
	madd local_b_ptr01, ldb, idx_n, local_b_ptr01
	add local_b_ptr01, b_ptr, local_b_ptr01, lsl 4

	ldp ar_re0_q, ar_im0_q, [ local_a_ptr0 ]
	ldr br01_q, [ local_b_ptr01 ]
	ldr br23_q, [ local_b_ptr01, 16 ]
	ldr br45_q, [ local_b_ptr01, 32 ]
	ldr br67_q, [ local_b_ptr01, 48 ]
	//8 & 9 are loaded in the k_loop

	beq .L${func_name}_n${remaining_n}_loop_k_last_a0

.L${func_name}_n${remaining_n}_loop_k: // {{{
	${loop_k_body(fmlx1, fmlx2, fmlx3, fmlx4, "ar_re0.4s", "ar_im0.4s", "ar_re1_q", "ar_im1_q", remaining_n)}
	ble .L${func_name}_n${remaining_n}_loop_k_last_a1
	${loop_k_body(fmlx1, fmlx2, fmlx3, fmlx4, "ar_re1.4s", "ar_im1.4s", "ar_re0_q", "ar_im0_q", remaining_n)}
	bgt .L${func_name}_n${remaining_n}_loop_k
.L${func_name}_n${remaining_n}_loop_k_last_a0:
// }}}
	${loop_k_last(fmlx1, fmlx2, fmlx3, fmlx4, "ar_re0.4s", "ar_im0.4s", remaining_n)}
	b .L${func_name}_n${remaining_n}_loop_k_end
.L${func_name}_n${remaining_n}_loop_k_last_a1:
	${loop_k_last(fmlx1, fmlx2, fmlx3, fmlx4, "ar_re1.4s", "ar_im1.4s", remaining_n)}
.L${func_name}_n${remaining_n}_loop_k_end: // }}}

#define local_c_ptr0 local_b_ptr01

#define cr1_im0 v30
#define cr1_re0 v29
#define cr1_im1 v28
#define cr1_re1 v27
#define cr1_im2 v26
#define cr1_re2 v25
#define cr1_im3 v24
#define cr1_re3 v23
#define cr1_im4 v22
#define cr1_re4 v21
#define cr1_im5 v20
#define cr1_re5 v19
#define cr1_im6 v18
#define cr1_re6 v17
#define cr1_im7 v16
#define cr1_re7 v15
#define cr1_im8 v14
#define cr1_re8 v13
#define cr1_im9 v12
#define cr1_re9 v11

#define cr_im v2
#define cr_re v1


#define t0 s1
#define t1 s2
#define t2 s3
#define t3 s4


	lsl idx_n, idx_n, #1

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
	fmul cr1_re0.4s, cr0_re0.4s, alpha.s[0]
	fmul cr1_im0.4s, cr0_re0.4s, alpha.s[1]
	fmls cr1_re0.4s, cr0_im0.4s, alpha.s[1]
	fmla cr1_im0.4s, cr0_im0.4s, alpha.s[0]

	fmul cr1_re1.4s, cr0_re1.4s, alpha.s[0]
	fmul cr1_im1.4s, cr0_re1.4s, alpha.s[1]
	fmls cr1_re1.4s, cr0_im1.4s, alpha.s[1]
	fmla cr1_im1.4s, cr0_im1.4s, alpha.s[0]

	fmul cr1_re2.4s, cr0_re2.4s, alpha.s[0]
	fmul cr1_im2.4s, cr0_re2.4s, alpha.s[1]
	fmls cr1_re2.4s, cr0_im2.4s, alpha.s[1]
	fmla cr1_im2.4s, cr0_im2.4s, alpha.s[0]

	fmul cr1_re3.4s, cr0_re3.4s, alpha.s[0]
	fmul cr1_im3.4s, cr0_re3.4s, alpha.s[1]
	fmls cr1_re3.4s, cr0_im3.4s, alpha.s[1]
	fmla cr1_im3.4s, cr0_im3.4s, alpha.s[0]

	fmul cr1_re4.4s, cr0_re4.4s, alpha.s[0]
	fmul cr1_im4.4s, cr0_re4.4s, alpha.s[1]
	fmls cr1_re4.4s, cr0_im4.4s, alpha.s[1]
	fmla cr1_im4.4s, cr0_im4.4s, alpha.s[0]

	fmul cr1_re5.4s, cr0_re5.4s, alpha.s[0]
	fmul cr1_im5.4s, cr0_re5.4s, alpha.s[1]
	fmls cr1_re5.4s, cr0_im5.4s, alpha.s[1]
	fmla cr1_im5.4s, cr0_im5.4s, alpha.s[0]

	fmul cr1_re6.4s, cr0_re6.4s, alpha.s[0]
	fmul cr1_im6.4s, cr0_re6.4s, alpha.s[1]
	fmls cr1_re6.4s, cr0_im6.4s, alpha.s[1]
	fmla cr1_im6.4s, cr0_im6.4s, alpha.s[0]

	fmul cr1_re7.4s, cr0_re7.4s, alpha.s[0]
	fmul cr1_im7.4s, cr0_re7.4s, alpha.s[1]
	fmls cr1_re7.4s, cr0_im7.4s, alpha.s[1]
	fmla cr1_im7.4s, cr0_im7.4s, alpha.s[0]

	fmul cr1_re8.4s, cr0_re8.4s, alpha.s[0]
	fmul cr1_im8.4s, cr0_re8.4s, alpha.s[1]
	fmls cr1_re8.4s, cr0_im8.4s, alpha.s[1]
	fmla cr1_im8.4s, cr0_im8.4s, alpha.s[0]

	fmul cr1_re9.4s, cr0_re9.4s, alpha.s[0]
	fmul cr1_im9.4s, cr0_re9.4s, alpha.s[1]
	fmls cr1_re9.4s, cr0_im9.4s, alpha.s[1]
	fmla cr1_im9.4s, cr0_im9.4s, alpha.s[0]

	madd local_c_ptr0, ldc, idx_n, idx_m
	add local_c_ptr0, c_ptr, local_c_ptr0, lsl 3

	${cgemm_save_indiv(func_name + "_n" + str(remaining_n) + "_save_indiv")}

	lsr idx_n, idx_n, #1
	add idx_m, idx_m, #4

	b .L${func_name}_n${remaining_n}_loop_m
%endfor

.L${func_name}_loop_m_end: // }}}
	add idx_n, idx_n, #5
	b .L${func_name}_loop_n
.L${func_name}_loop_n_end: // }}}


.L${func_name}_cleanup:
	ldp d8, d9, [ sp, #304 ]
	ldp d10, d11, [ sp, #320 ]
	ldp d12, d13, [ sp, #336 ]
	ldp d14, d15, [ sp, #352 ]
	ldp x19, x20, [ sp, #144 ]
	ldp x21, x22, [ sp, #160 ]
	ldp x23, x24, [ sp, #176 ]
	ldp x25, x26, [ sp, #192 ]
	ldp x27, x28, [ sp, #208 ]
	ldp x29, x30, [ sp ], #384
	ret
	${epilogue(func_name)}
</%def>

${generate_XN_kernel("cgemm_large_kernel_TN_nmk", "fmla", "fmla", "fmls", "fmla")}

## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

//the number of rows of A & B that we step through respectively

#define A_STEP 4
#define B_STEP 4
#define K_UNROLL_STEP 1
//std::complex<double>
#define DATA_TYPE_LSL 4


a_ptr .req x0
b_ptr .req x1
c_ptr .req x2


//the k dimension of a & b
k_cntg .req x3
//number of strides in a
a_strd .req x4
//number of strides in b
b_strd  .req x5


ldc .req x6


a_idx .req x7
b_idx .req x8
k_idx .req x9
t .req x10


A_cur_ptr .req x11
B_cur_ptr .req x12
// x13, x14 disallowed in Arm64EC
C_ptr0 .req x15
C_ptr1 .req x16
C_ptr2 .req x17
// x18 may be in use for the Platform Register	
C_ptr3 .req x19

lda .req x20
ldb .req x21

alpha_re_int .req x22
// x23, x24, x28 disallowed in Arm64EC
alpha_im_int .req x25

alpha_re_input .req d0
alpha_im_input .req d1

//warming this stamps over a B load register
alpha .req v20
vt .req v21

//the real register block accumulated in the K loop a stored into C
C_re_0_0 .req v0
C_re_0_1 .req v2
C_re_1_0 .req v4
C_re_1_1 .req v6
C_re_2_0 .req v8
C_re_2_1 .req v10
C_re_3_0 .req v12
C_re_3_1 .req v14
//the imag register block accumulated in the K loop a stored into C
C_im_0_0 .req v1
C_im_0_1 .req v3
C_im_1_0 .req v5
C_im_1_1 .req v7
C_im_2_0 .req v9
C_im_2_1 .req v11
C_im_3_0 .req v13
C_im_3_1 .req v15


//the elements which we load for the use C matrix
//which we will accumulate with the reg block at the end
//of the k loop
C_ld_re_0_0 .req v24
C_ld_re_0_1 .req v26
C_ld_re_1_0 .req v28
C_ld_re_1_1 .req v30

C_ld_im_0_0 .req v25
C_ld_im_0_1 .req v27
C_ld_im_1_0 .req v29
C_ld_im_1_1 .req v31


//These are the d versions of the above registers
//and are used to save out individual elements in
//save_indiv section of the kernel
C_re_0_0d .req d0
C_re_0_1d .req d2
C_re_1_0d .req d4
C_re_1_1d .req d6
C_re_2_0d .req d8
C_re_2_1d .req d10
C_re_3_0d .req d12
C_re_3_1d .req d14

C_im_0_0d .req d1
C_im_0_1d .req d3
C_im_1_0d .req d5
C_im_1_1d .req d7
C_im_2_0d .req d9
C_im_2_1d .req d11
C_im_3_0d .req d13
C_im_3_1d .req d15

C_ld_re_0_0d .req d24
C_ld_re_0_1d .req d26
C_ld_re_1_0d .req d28
C_ld_re_1_1d .req d30

C_ld_im_0_0d .req d25
C_ld_im_0_1d .req d27
C_ld_im_1_0d .req d29
C_ld_im_1_1d .req d31







//real elements of interleaved rows of A
A_re_0 .req v16
A_re_1 .req v17
//imag elements of interleave rows of A
A_im_0 .req v18
A_im_1 .req v19

//REAL, IMAG pairs of B
B_r0 .req v20
B_r1 .req v21
B_r2 .req v22
B_r3 .req v23


//The Q register form of the above
A_re_0q .req q16
A_re_1q .req q17
A_im_0q .req q18
A_im_1q .req q19

B_r0q .req q20
B_r1q .req q21
B_r2q .req q22
B_r3q .req q23


// labels
#define b_loop(x) 1##x
#define a_loop(x) 2##x
#define k_loop(x) 3##x
#define k_loop_end(x) 4##x
#define store_c_fast(x) 5##x
#define store_c_pred(x) 6##x
#define c_0_str2(x) 7##x
#define c_0_str4(x) 8##x
#define c_0_str_odd_start(x) 9##x
#define c_0_str_odd_end(x) 10##x
#define c_0_str1(x) 11##x
#define c_0_str3(x) 12##x
#define c_1_str2(x) 13##x
#define c_1_str4(x) 14##x
#define c_1_str_odd_start(x) 15##x
#define c_1_str_odd_end(x) 16##x
#define c_1_str1(x) 17##x
#define c_1_str3(x) 18##x
#define c_2_str2(x) 19##x
#define c_2_str4(x) 20##x
#define c_2_str_odd_start(x) 21##x
#define c_2_str_odd_end(x) 22##x
#define c_2_str1(x) 23##x
#define c_2_str3(x) 24##x
#define c_3_str2(x) 25##x
#define c_3_str4(x) 26##x
#define c_3_str_odd_start(x) 27##x
#define c_3_str_odd_end(x) 28##x
#define c_3_str1(x) 29##x
#define c_3_str3(x) 30##x
#define save_indiv_end(x) 31##x
#define store_c_end(x) 32##x
#define b_loop_end(x) 33##x

<%def name="loop_k_body(m_remainder)">
	//load from A_ptr REAL0, REAL1, REAL2, REAL3, IMAG0, IMAG1, IMAG2, IMAG3,

	${FMLX1} C_re_0_0.2d, A_re_0.2d, B_r0.d[0]
	add A_cur_ptr, A_cur_ptr, #64
	${FMLX2} C_im_0_0.2d, A_re_0.2d, B_r0.d[1]
	add B_cur_ptr, B_cur_ptr, #64
	${FMLX3} C_re_0_0.2d, A_im_0.2d, B_r0.d[1]
	${FMLX4} C_im_0_0.2d, A_im_0.2d, B_r0.d[0]

%if m_remainder>0:
	${FMLX1} C_re_0_1.2d, A_re_1.2d, B_r0.d[0]
	${FMLX2} C_im_0_1.2d, A_re_1.2d, B_r0.d[1]
	${FMLX3} C_re_0_1.2d, A_im_1.2d, B_r0.d[1]
	${FMLX4} C_im_0_1.2d, A_im_1.2d, B_r0.d[0]
%endif

	ldr B_r0q, [ B_cur_ptr ]

//c_cntg_1
	${FMLX1} C_re_1_0.2d, A_re_0.2d, B_r1.d[0]
	${FMLX2} C_im_1_0.2d, A_re_0.2d, B_r1.d[1]
	prfm PLDL1KEEP, [ B_cur_ptr, #256 ]
	${FMLX3} C_re_1_0.2d, A_im_0.2d, B_r1.d[1]
	prfm PLDL1KEEP, [ B_cur_ptr, #256 + 64]
	${FMLX4} C_im_1_0.2d, A_im_0.2d, B_r1.d[0]

%if m_remainder>0:
	prfm PLDL1KEEP, [ B_cur_ptr, #256 + 128]
	${FMLX1} C_re_1_1.2d, A_re_1.2d, B_r1.d[0]
	prfm PLDL1KEEP, [ B_cur_ptr, #256 + 192]
	${FMLX2} C_im_1_1.2d, A_re_1.2d, B_r1.d[1]
	prfm PLDL1KEEP, [ A_cur_ptr, #256 ]
	${FMLX3} C_re_1_1.2d, A_im_1.2d, B_r1.d[1]
	prfm PLDL1KEEP, [ A_cur_ptr, #256 + 64 ]
	${FMLX4} C_im_1_1.2d, A_im_1.2d, B_r1.d[0]
%endif


	ldr B_r1q, [ B_cur_ptr, #16 ]

//c_cntg_2
	${FMLX1} C_re_2_0.2d, A_re_0.2d, B_r2.d[0]
	${FMLX2} C_im_2_0.2d, A_re_0.2d, B_r2.d[1]
	prfm PLDL1KEEP, [ A_cur_ptr, #256 + 128 ]
	${FMLX3} C_re_2_0.2d, A_im_0.2d, B_r2.d[1]
	prfm PLDL1KEEP, [ A_cur_ptr, #256  + 192 ]
	${FMLX4} C_im_2_0.2d, A_im_0.2d, B_r2.d[0]

	add k_idx, k_idx, 1

%if m_remainder>0:
	${FMLX1} C_re_2_1.2d, A_re_1.2d, B_r2.d[0]
	${FMLX2} C_im_2_1.2d, A_re_1.2d, B_r2.d[1]

	${FMLX3} C_re_2_1.2d, A_im_1.2d, B_r2.d[1]
	${FMLX4} C_im_2_1.2d, A_im_1.2d, B_r2.d[0]
%endif
	cmp k_idx, k_cntg

	ldr B_r2q, [ B_cur_ptr, #32 ]

//c_cntg_3
	${FMLX1} C_re_3_0.2d, A_re_0.2d, B_r3.d[0]
	${FMLX2} C_im_3_0.2d, A_re_0.2d, B_r3.d[1]

	ldr A_re_0q, [ A_cur_ptr ]

	${FMLX3} C_re_3_0.2d, A_im_0.2d, B_r3.d[1]
	${FMLX4} C_im_3_0.2d, A_im_0.2d, B_r3.d[0]

	ldr A_im_0q, [ A_cur_ptr, #32 ]

%if m_remainder>0:
	${FMLX1} C_re_3_1.2d, A_re_1.2d, B_r3.d[0]
	${FMLX2} C_im_3_1.2d, A_re_1.2d, B_r3.d[1]
	%endif
	ldr A_re_1q, [ A_cur_ptr, #16 ]
	%if m_remainder>0:
	${FMLX3} C_re_3_1.2d, A_im_1.2d, B_r3.d[1]
	${FMLX4} C_im_3_1.2d, A_im_1.2d, B_r3.d[0]
%endif

	ldr A_im_1q, [ A_cur_ptr, #48 ]
	ldr B_r3q, [ B_cur_ptr, #48 ]

	bcc .L${func_name}_k_loop_m${m_remainder}
	b .L${func_name}_k_loop_end_m${m_remainder}
</%def>

<%def name="loop_k_end(m_remainder)">
//c_cntg_0
	${FMLX1} C_re_0_0.2d, A_re_0.2d, B_r0.d[0]
	prfm PLDL1KEEP, [C_ptr0]
	${FMLX2} C_im_0_0.2d, A_re_0.2d, B_r0.d[1]
	prfm PLDL1KEEP, [C_ptr1]
	${FMLX3} C_re_0_0.2d, A_im_0.2d, B_r0.d[1]
	prfm PLDL1KEEP, [C_ptr2]
	${FMLX4} C_im_0_0.2d, A_im_0.2d, B_r0.d[0]
	prfm PLDL1KEEP, [C_ptr3]

%if m_remainder>0:
	${FMLX1} C_re_0_1.2d, A_re_1.2d, B_r0.d[0]
	${FMLX2} C_im_0_1.2d, A_re_1.2d, B_r0.d[1]
	${FMLX3} C_re_0_1.2d, A_im_1.2d, B_r0.d[1]
	${FMLX4} C_im_0_1.2d, A_im_1.2d, B_r0.d[0]
%endif

//c_cntg_1
	${FMLX1} C_re_1_0.2d, A_re_0.2d, B_r1.d[0]
	${FMLX2} C_im_1_0.2d, A_re_0.2d, B_r1.d[1]
	${FMLX3} C_re_1_0.2d, A_im_0.2d, B_r1.d[1]
	${FMLX4} C_im_1_0.2d, A_im_0.2d, B_r1.d[0]


%if m_remainder>0:
	${FMLX1} C_re_1_1.2d, A_re_1.2d, B_r1.d[0]
	${FMLX2} C_im_1_1.2d, A_re_1.2d, B_r1.d[1]
	${FMLX3} C_re_1_1.2d, A_im_1.2d, B_r1.d[1]
	${FMLX4} C_im_1_1.2d, A_im_1.2d, B_r1.d[0]
%endif

//c_cntg_2
	${FMLX1} C_re_2_0.2d, A_re_0.2d, B_r2.d[0]
	${FMLX2} C_im_2_0.2d, A_re_0.2d, B_r2.d[1]
	${FMLX3} C_re_2_0.2d, A_im_0.2d, B_r2.d[1]
	${FMLX4} C_im_2_0.2d, A_im_0.2d, B_r2.d[0]


%if m_remainder>0:
	${FMLX1} C_re_2_1.2d, A_re_1.2d, B_r2.d[0]
	${FMLX2} C_im_2_1.2d, A_re_1.2d, B_r2.d[1]
	${FMLX3} C_re_2_1.2d, A_im_1.2d, B_r2.d[1]
	${FMLX4} C_im_2_1.2d, A_im_1.2d, B_r2.d[0]
%endif

//c_cntg_3
	${FMLX1} C_re_3_0.2d, A_re_0.2d, B_r3.d[0]
	${FMLX2} C_im_3_0.2d, A_re_0.2d, B_r3.d[1]
	${FMLX3} C_re_3_0.2d, A_im_0.2d, B_r3.d[1]
	${FMLX4} C_im_3_0.2d, A_im_0.2d, B_r3.d[0]


%if m_remainder>0:
	${FMLX1} C_re_3_1.2d, A_re_1.2d, B_r3.d[0]
	${FMLX2} C_im_3_1.2d, A_re_1.2d, B_r3.d[1]
	${FMLX3} C_re_3_1.2d, A_im_1.2d, B_r3.d[1]
	${FMLX4} C_im_3_1.2d, A_im_1.2d, B_r3.d[0]
%endif

	b .L${func_name}_b_loop_end_m${m_remainder}
</%def>


%for transopt in ["TN"]:
<% FMLX1 = "fmla" %>\
<% FMLX2 = "fmls" if transopt[1] == 'C' else "fmla" %>\
<% FMLX3 = "fmls" if (transopt[0] == 'C') == (transopt[1] == 'C') else "fmla" %>\
<% FMLX4 = "fmls" if transopt[0] == 'C' else "fmla" %>\

	<% func_name = "zgemm_vanilla_big_" + transopt %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-384]!
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
		// load alpha and beta from stack
		ldr d0, [sp, #384]
		ldr d1, [sp, #392]
		ldr d2, [sp, #400]
		ldr d3, [sp, #408]
	%endif

	//pre shift LDA and LDB so that the dimension is in bytes
	//rather than in elements
	lsl lda, k_cntg, DATA_TYPE_LSL
	lsl ldb, k_cntg, DATA_TYPE_LSL

	fmov alpha_re_int, alpha_re_input
	fmov alpha_im_int, alpha_im_input

//loop over all of the strds of A
	mov b_idx, #0
b_loop():

//loop over all of the strds of A
	mov a_idx, #0
a_loop():
	sub t, a_strd, a_idx
	cmp t, #2
	ble .L${func_name}_a_loop_m0
	//b .L${func_name}_a_loop_m1
%for remaining_m in [1, 0]:
.L${func_name}_a_loop_m${remaining_m}:
	//complex_t *A_cur_ptr = A_ptr + a_idx * lda
	//LDA has already been LSLed by DATA_TYPE_LSL
	madd A_cur_ptr, a_idx, lda, a_ptr

	ldp A_re_0q, A_re_1q, [ A_cur_ptr ]

	//complex_t *B_cur_ptr = B_ptr + b_idx * ldb
	//LDB has already been LSLed by DATA_TYPE_LSL
	madd B_cur_ptr, b_idx, ldb, b_ptr


	ldp A_im_0q, A_im_1q, [ A_cur_ptr, #32 ]

	madd C_ptr0, b_idx, ldc, a_idx
	//Load from B_ptr REAL0, IMAG0, REAL1, IMAG1, REAL2, IMAG2, REAL3, IMAG3
	ldp B_r0q, B_r1q, [ B_cur_ptr ]
	add C_ptr0, c_ptr,  C_ptr0, lsl DATA_TYPE_LSL
	ldp B_r2q, B_r3q, [ B_cur_ptr, #32 ]

	add C_ptr1, C_ptr0, ldc,    lsl DATA_TYPE_LSL

	//zero out the C reg block on every iteration of of the A & B loops
	//the values in the reg block are accumulated over the k_loop and
	//stored at k_loop end
	//zero real elements
	dup C_re_0_0.2d, xzr
	add C_ptr2, C_ptr1, ldc,    lsl DATA_TYPE_LSL
	prfm PLDL1KEEP, [ A_cur_ptr, #256 ]
	add C_ptr3, C_ptr2, ldc,    lsl DATA_TYPE_LSL
	dup C_re_0_1.2d, xzr
	prfm PLDL1KEEP, [ A_cur_ptr, #256 + 64 ]
	dup C_re_1_0.2d, xzr
	prfm PLDL1KEEP, [ A_cur_ptr, #256 + 128 ]
	dup C_re_1_1.2d, xzr
	prfm PLDL1KEEP, [ A_cur_ptr, #256 + 192 ]
	dup C_re_2_0.2d, xzr
	prfm PLDL1KEEP, [ B_cur_ptr, #256 ]
	dup C_re_2_1.2d, xzr
	prfm PLDL1KEEP, [ B_cur_ptr, #256 + 64 ]
	dup C_re_3_0.2d, xzr
	prfm PLDL1KEEP, [ B_cur_ptr, #256 + 128 ]
	dup C_re_3_1.2d, xzr
	prfm PLDL1KEEP, [ B_cur_ptr, #256 + 192 ]
	//zero imag element
	dup C_im_0_0.2d, xzr
	prfm    PLDL2KEEP, [C_ptr0, #64]
	dup C_im_0_1.2d, xzr
	prfm    PLDL2KEEP, [C_ptr1, #64]
	dup C_im_1_0.2d, xzr
	prfm    PLDL2KEEP, [C_ptr2, #64]
	dup C_im_1_1.2d, xzr
	prfm    PLDL2KEEP, [C_ptr3, #64]
	dup C_im_2_0.2d, xzr
	mov k_idx, #1
	dup C_im_2_1.2d, xzr
	cmp k_idx, k_cntg
	dup C_im_3_0.2d, xzr
	dup C_im_3_1.2d, xzr

	beq .L${func_name}_k_loop_end_m${remaining_m}

.L${func_name}_k_loop_m${remaining_m}:
##k_loop(): //for(uint64_t k_idx=0u; k_idx<k_cntg; k_idx+=K_UNROLL_STEP)
	${loop_k_body(remaining_m)}


.L${func_name}_k_loop_end_m${remaining_m}:
	${loop_k_end(remaining_m)}


.L${func_name}_b_loop_end_m${remaining_m}:
	b b_loop_end(f)

%endfor

b_loop_end():
	ins alpha.d[0], alpha_re_int
	ins alpha.d[1], alpha_im_int

	// cr1_re = cr0_re * alpha.real() - cr0_im * alpha.imag();
	// cr1_im = cr0_re * alpha.imag() + cr0_im * alpha.real();
//cntg0
	fmul vt.2d,       c_re_0_0.2d, alpha.d[0] //real
	fmls vt.2d,       c_im_0_0.2d, alpha.d[1] //imag
	fmul c_im_0_0.2d, c_im_0_0.2d, alpha.d[0] //real
	fmla c_im_0_0.2d, c_re_0_0.2d, alpha.d[1] //imag
	mov  c_re_0_0.16b, vt.16b

	fmul vt.2d,       c_re_0_1.2d, alpha.d[0] //real
	fmls vt.2d,       c_im_0_1.2d, alpha.d[1] //imag
	fmul c_im_0_1.2d, c_im_0_1.2d, alpha.d[0] //real
	fmla c_im_0_1.2d, c_re_0_1.2d, alpha.d[1] //imag
	mov  c_re_0_1.16b, vt.16b

//cntg1
	fmul vt.2d,       c_re_1_0.2d, alpha.d[0] //real
	fmls vt.2d,       c_im_1_0.2d, alpha.d[1] //imag
	fmul c_im_1_0.2d, c_im_1_0.2d, alpha.d[0] //real
	fmla c_im_1_0.2d, c_re_1_0.2d, alpha.d[1] //imag
	mov c_re_1_0.16b, vt.16b

	fmul vt.2d,       c_re_1_1.2d, alpha.d[0] //real
	fmls vt.2d,       c_im_1_1.2d, alpha.d[1] //imag
	fmul c_im_1_1.2d, c_im_1_1.2d, alpha.d[0] //real
	fmla c_im_1_1.2d, c_re_1_1.2d, alpha.d[1] //imag
	mov c_re_1_1.16b, vt.16b

//cntg2
	fmul vt.2d,       c_re_2_0.2d, alpha.d[0] //real
	fmls vt.2d,       c_im_2_0.2d, alpha.d[1] //imag
	fmul c_im_2_0.2d, c_im_2_0.2d, alpha.d[0] //real
	fmla c_im_2_0.2d, c_re_2_0.2d, alpha.d[1] //imag
	mov c_re_2_0.16b, vt.16b

	fmul vt.2d,       c_re_2_1.2d, alpha.d[0] //real
	fmls vt.2d,       c_im_2_1.2d, alpha.d[1] //imag
	fmul c_im_2_1.2d, c_im_2_1.2d, alpha.d[0] //real
	fmla c_im_2_1.2d, c_re_2_1.2d, alpha.d[1] //imag
	mov c_re_2_1.16b, vt.16b

//cntg3
	fmul vt.2d,       c_re_3_0.2d, alpha.d[0] //real
	fmls vt.2d,       c_im_3_0.2d, alpha.d[1] //imag
	fmul c_im_3_0.2d, c_im_3_0.2d, alpha.d[0] //real
	fmla c_im_3_0.2d, c_re_3_0.2d, alpha.d[1] //imag
	mov c_re_3_0.16b, vt.16b

	fmul vt.2d,       c_re_3_1.2d, alpha.d[0] //real
	fmls vt.2d,       c_im_3_1.2d, alpha.d[1] //imag
	fmul c_im_3_1.2d, c_im_3_1.2d, alpha.d[0] //real
	fmla c_im_3_1.2d, c_re_3_1.2d, alpha.d[1] //imag
	mov c_re_3_1.16b, vt.16b


	//save reg block into B
	//if(b_idx + B_STEP > b_strd || a_idx + A_STEP >  a_strd)
	//	store_c_pred(f)
	//else
	//	store_c
	add t, b_idx, B_STEP
	cmp b_strd, t
	bcc store_c_pred(f)

	add t, a_idx, A_STEP
	cmp a_strd, t
	bcc store_c_pred(f)
store_c_fast():

//c_cntg_0
	ld2 { C_ld_re_0_0.2d, C_ld_im_0_0.2d }, [ C_ptr0 ]
	fadd C_re_0_0.2d, C_re_0_0.2d, C_ld_re_0_0.2d
	fadd C_im_0_0.2d, C_im_0_0.2d, C_ld_im_0_0.2d
	st2 { C_re_0_0.2d, C_im_0_0.2d }, [ C_ptr0 ]

	add C_ptr0, C_ptr0, #32

	ld2 { C_ld_re_0_1.2d, C_ld_im_0_1.2d }, [ C_ptr0 ]
	fadd C_re_0_1.2d, C_re_0_1.2d, C_ld_re_0_1.2d
	fadd C_im_0_1.2d, C_im_0_1.2d, C_ld_im_0_1.2d
	st2 { C_re_0_1.2d, C_im_0_1.2d }, [ C_ptr0 ]


//c_cntg_1
	ld2 { C_ld_re_1_0.2d, C_ld_im_1_0.2d }, [ C_ptr1 ]
	fadd C_re_1_0.2d, C_re_1_0.2d, C_ld_re_1_0.2d
	fadd C_im_1_0.2d, C_im_1_0.2d, C_ld_im_1_0.2d
	st2 { C_re_1_0.2d, C_im_1_0.2d }, [ C_ptr1 ]

	add C_ptr1, C_ptr1, #32

	ld2 { C_ld_re_1_1.2d, C_ld_im_1_1.2d }, [ C_ptr1 ]
	fadd C_re_1_1.2d, C_re_1_1.2d, C_ld_re_1_1.2d
	fadd C_im_1_1.2d, C_im_1_1.2d, C_ld_im_1_1.2d
	st2 { C_re_1_1.2d, C_im_1_1.2d }, [ C_ptr1 ]


//c_cntg_2
	ld2 { C_ld_re_0_0.2d, C_ld_im_0_0.2d }, [ C_ptr2 ]
	fadd C_re_2_0.2d, C_re_2_0.2d, C_ld_re_0_0.2d
	fadd C_im_2_0.2d, C_im_2_0.2d, C_ld_im_0_0.2d
	st2 { C_re_2_0.2d, C_im_2_0.2d }, [ C_ptr2 ]

	add C_ptr2, C_ptr2, #32

	ld2 { C_ld_re_0_1.2d, C_ld_im_0_1.2d }, [ C_ptr2 ]
	fadd C_re_2_1.2d, C_re_2_1.2d, C_ld_re_0_1.2d
	fadd C_im_2_1.2d, C_im_2_1.2d, C_ld_im_0_1.2d
	st2 { C_re_2_1.2d, C_im_2_1.2d }, [ C_ptr2 ]


//c_cntg_3
	ld2 { C_ld_re_1_0.2d, C_ld_im_1_0.2d }, [ C_ptr3 ]
	fadd C_re_3_0.2d, C_re_3_0.2d, C_ld_re_1_0.2d
	fadd C_im_3_0.2d, C_im_3_0.2d, C_ld_im_1_0.2d
	st2 { C_re_3_0.2d, C_im_3_0.2d }, [ C_ptr3 ]

	add C_ptr3, C_ptr3, #32

	ld2 { C_ld_re_1_1.2d, C_ld_im_1_1.2d }, [ C_ptr3 ]
	fadd C_re_3_1.2d, C_re_3_1.2d, C_ld_re_1_1.2d
	fadd C_im_3_1.2d, C_im_3_1.2d, C_ld_im_1_1.2d
	st2 { C_re_3_1.2d, C_im_3_1.2d }, [ C_ptr3 ]

	b store_c_end(f)

store_c_pred():


// c_0_str_start:
	cmp b_idx, b_strd
	beq save_indiv_end(f)

	add t, a_idx, 4
	cmp t, a_strd
	bls c_0_str4(f)

	add t, a_idx, 2
	cmp t, a_strd
	bls c_0_str2(f)

	//no full vectors to store, need to save the first halves.
	b c_0_str_odd_start(f)

c_0_str4():
	add t, C_ptr0, #32
	ld2 { C_ld_re_0_1.2d, C_ld_im_0_1.2d }, [ t ]
	fadd C_re_0_1.2d, C_re_0_1.2d, C_ld_re_0_1.2d
	fadd C_im_0_1.2d, C_im_0_1.2d, C_ld_im_0_1.2d
	st2 { C_re_0_1.2d, C_im_0_1.2d }, [ t ]

c_0_str2():
	ld2 { C_ld_re_0_0.2d, C_ld_im_0_0.2d }, [ C_ptr0 ]
	fadd C_re_0_0.2d, C_re_0_0.2d, C_ld_re_0_0.2d
	fadd C_im_0_0.2d, C_im_0_0.2d, C_ld_im_0_0.2d
	st2 { C_re_0_0.2d, C_im_0_0.2d }, [ C_ptr0 ]


c_0_str_odd_start():
	add t, a_idx, 1
	cmp a_strd, t
	beq c_0_str1(f)

	add t, a_idx, 3
	cmp a_strd, t
	beq c_0_str3(f)

	b c_0_str_odd_end(f)

c_0_str1():
	ldp C_ld_re_0_0d, C_ld_im_0_0d, [ C_ptr0 ]
	fadd C_re_0_0d, C_re_0_0d, C_ld_re_0_0d
	fadd C_im_0_0d, C_im_0_0d, C_ld_im_0_0d
	stp C_re_0_0d, C_im_0_0d, [ C_ptr0 ]

	b c_0_str_odd_end(f)

c_0_str3():
	add C_ptr0, C_ptr0, #32
	ldp C_ld_re_0_1d, C_ld_im_0_1d, [ C_ptr0 ]
	fadd C_re_0_1d, C_re_0_1d, C_ld_re_0_1d
	fadd C_im_0_1d, C_im_0_1d, C_ld_im_0_1d
	stp C_re_0_1d, C_im_0_1d, [ C_ptr0 ]

	b c_0_str_odd_end(f)

c_0_str_odd_end():




// c_1_str_start:
	add t, b_idx, #1
	cmp t, b_strd
	beq save_indiv_end(f)

	add t, a_idx, 4
	cmp t, a_strd
	bls c_1_str4(f)

	add t, a_idx, 2
	cmp t, a_strd
	bls c_1_str2(f)

	//no full vectors to store, need to save the first halves.
	b c_1_str_odd_start(f)

c_1_str4():
	add t, C_ptr1, #32
	ld2 { C_ld_re_1_1.2d, C_ld_im_1_1.2d }, [ t ]
	fadd C_re_1_1.2d, C_re_1_1.2d, C_ld_re_1_1.2d
	fadd C_im_1_1.2d, C_im_1_1.2d, C_ld_im_1_1.2d
	st2 { C_re_1_1.2d, C_im_1_1.2d }, [ t ]

c_1_str2():
	ld2 { C_ld_re_1_0.2d, C_ld_im_1_0.2d }, [ C_ptr1 ]
	fadd C_re_1_0.2d, C_re_1_0.2d, C_ld_re_1_0.2d
	fadd C_im_1_0.2d, C_im_1_0.2d, C_ld_im_1_0.2d
	st2 { C_re_1_0.2d, C_im_1_0.2d }, [ C_ptr1 ]


c_1_str_odd_start():
	add t, a_idx, 1
	cmp a_strd, t
	beq c_1_str1(f)

	add t, a_idx, 3
	cmp a_strd, t
	beq c_1_str3(f)

	b c_1_str_odd_end(f)

c_1_str1():
	ldp C_ld_re_1_0d, C_ld_im_1_0d, [ C_ptr1 ]
	fadd C_re_1_0d, C_re_1_0d, C_ld_re_1_0d
	fadd C_im_1_0d, C_im_1_0d, C_ld_im_1_0d
	stp C_re_1_0d, C_im_1_0d, [ C_ptr1 ]

	b c_1_str_odd_end(f)

c_1_str3():
	add C_ptr1, C_ptr1, #32
	ldp C_ld_re_1_1d, C_ld_im_1_1d, [ C_ptr1 ]
	fadd C_re_1_1d, C_re_1_1d, C_ld_re_1_1d
	fadd C_im_1_1d, C_im_1_1d, C_ld_im_1_1d
	stp C_re_1_1d, C_im_1_1d, [ C_ptr1 ]

	b c_1_str_odd_end(f)

c_1_str_odd_end():




// c_2_str_start:
	add t, b_idx, #2
	cmp t, b_strd
	beq save_indiv_end(f)

	add t, a_idx, 4
	cmp t, a_strd
	bls c_2_str4(f)

	add t, a_idx, 2
	cmp t, a_strd
	bls c_2_str2(f)

	//no full vectors to store, need to save the first halves.
	b c_2_str_odd_start(f)

c_2_str4():
	add t, C_ptr2, #32
	ld2 { C_ld_re_0_1.2d, C_ld_im_0_1.2d }, [ t ]
	fadd C_re_2_1.2d, C_re_2_1.2d, C_ld_re_0_1.2d
	fadd C_im_2_1.2d, C_im_2_1.2d, C_ld_im_0_1.2d
	st2 { C_re_2_1.2d, C_im_2_1.2d }, [ t ]

c_2_str2():
	ld2 { C_ld_re_0_0.2d, C_ld_im_0_0.2d }, [ C_ptr2 ]
	fadd C_re_2_0.2d, C_re_2_0.2d, C_ld_re_0_0.2d
	fadd C_im_2_0.2d, C_im_2_0.2d, C_ld_im_0_0.2d
	st2 { C_re_2_0.2d, C_im_2_0.2d }, [ C_ptr2 ]


c_2_str_odd_start():
	add t, a_idx, 1
	cmp a_strd, t
	beq c_2_str1(f)

	add t, a_idx, 3
	cmp a_strd, t
	beq c_2_str3(f)

	b c_2_str_odd_end(f)

c_2_str1():
	ldp C_ld_re_0_0d, C_ld_im_0_0d, [ C_ptr2 ]
	fadd C_re_2_0d, C_re_2_0d, C_ld_re_0_0d
	fadd C_im_2_0d, C_im_2_0d, C_ld_im_0_0d
	stp C_re_2_0d, C_im_2_0d, [ C_ptr2 ]

	b c_2_str_odd_end(f)

c_2_str3():
	add C_ptr2, C_ptr2, #32
	ldp C_ld_re_0_1d, C_ld_im_0_1d, [ C_ptr2 ]
	fadd C_re_2_1d, C_re_2_1d, C_ld_re_0_1d
	fadd C_im_2_1d, C_im_2_1d, C_ld_im_0_1d
	stp C_re_2_1d, C_im_2_1d, [ C_ptr2 ]

	b c_2_str_odd_end(f)

c_2_str_odd_end():




// c_3_str_start:
	add t, b_idx, #3
	cmp t, b_strd
	beq save_indiv_end(f)

	add t, a_idx, 4
	cmp t, a_strd
	bls c_3_str4(f)

	add t, a_idx, 2
	cmp t, a_strd
	bls c_3_str2(f)

	//no full vectors to store, need to save the first halves.
	b c_3_str_odd_start(f)

c_3_str4():
	add t, C_ptr3, #32
	ld2 { C_ld_re_1_1.2d, C_ld_im_1_1.2d }, [ t ]
	fadd C_re_3_1.2d, C_re_3_1.2d, C_ld_re_1_1.2d
	fadd C_im_3_1.2d, C_im_3_1.2d, C_ld_im_1_1.2d
	st2 { C_re_3_1.2d, C_im_3_1.2d }, [ t ]

c_3_str2():
	ld2 { C_ld_re_1_0.2d, C_ld_im_1_0.2d }, [ C_ptr3 ]
	fadd C_re_3_0.2d, C_re_3_0.2d, C_ld_re_1_0.2d
	fadd C_im_3_0.2d, C_im_3_0.2d, C_ld_im_1_0.2d
	st2 { C_re_3_0.2d, C_im_3_0.2d }, [ C_ptr3 ]

c_3_str_odd_start():
	add t, a_idx, 1
	cmp a_strd, t
	beq c_3_str1(f)

	add t, a_idx, 3
	cmp a_strd, t
	beq c_3_str3(f)

	b c_3_str_odd_end(f)

c_3_str1():
	ldp C_ld_re_1_0d, C_ld_im_1_0d, [ C_ptr3 ]
	fadd C_re_3_0d, C_re_3_0d, C_ld_re_1_0d
	fadd C_im_3_0d, C_im_3_0d, C_ld_im_1_0d
	stp C_re_3_0d, C_im_3_0d, [ C_ptr3 ]

	b c_3_str_odd_end(f)

c_3_str3():
	add C_ptr3, C_ptr3, #32
	ldp C_ld_re_1_1d, C_ld_im_1_1d, [ C_ptr3 ]
	fadd C_re_3_1d, C_re_3_1d, C_ld_re_1_1d
	fadd C_im_3_1d, C_im_3_1d, C_ld_im_1_1d
	stp C_re_3_1d, C_im_3_1d, [ C_ptr3 ]

	b c_3_str_odd_end(f)

c_3_str_odd_end():





save_indiv_end():

store_c_end():


// a_loop_end():
	add a_idx, a_idx, A_STEP
	cmp a_strd, a_idx
	bhi a_loop(b)



// b_loop_end:
	add b_idx, b_idx, B_STEP
	cmp b_strd, b_idx
	bhi b_loop(b)





// cleanup:
	ldp x19, x20, [ sp, #144 ]
	ldp x21, x22, [ sp, #160 ]
	ldp x23, x24, [ sp, #176 ]
	ldp x25, x26, [ sp, #192 ]
	ldp x27, x28, [ sp, #208 ]
	ldp d8, d9, [ sp, #304 ]
	ldp d10, d11, [ sp, #320 ]
	ldp d12, d13, [ sp, #336 ]
	ldp d14, d15, [ sp, #352 ]
	ldp x29, x30, [sp], #384
	ret

	${epilogue(func_name)}

%endfor

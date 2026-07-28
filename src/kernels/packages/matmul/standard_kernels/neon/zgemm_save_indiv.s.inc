## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

	<%def name="zgemm_save_indiv(func_name)">
    //save reg block into B
	//if(b_idx + B_STEP > b_strd || a_idx + A_STEP >  a_strd)
	//	.L${func_name}_store_c_pred
	//else
	//	.L${func_name}_store_c
	add t, b_idx, B_STEP
	cmp b_strd, t
	bcc .L${func_name}_store_c_pred

	add t, a_idx, A_STEP
	cmp a_strd, t
	bcc .L${func_name}_store_c_pred
.L${func_name}_store_c_fast:

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

	b .L${func_name}_store_c_end

.L${func_name}_store_c_pred:
	cmp b_idx, b_strd
	beq .L${func_name}_save_indiv_end

	add t, a_idx, 4
	cmp t, a_strd
	bls .L${func_name}_c_0_str4

	add t, a_idx, 2
	cmp t, a_strd
	bls .L${func_name}_c_0_str2

	//no full vectors to store, need to save the first halves.
	b .L${func_name}_c_0_str_odd_start

.L${func_name}_c_0_str4:
	add t, C_ptr0, #32
	ld2 { C_ld_re_0_1.2d, C_ld_im_0_1.2d }, [ t ]
	fadd C_re_0_1.2d, C_re_0_1.2d, C_ld_re_0_1.2d
	fadd C_im_0_1.2d, C_im_0_1.2d, C_ld_im_0_1.2d
	st2 { C_re_0_1.2d, C_im_0_1.2d }, [ t ]

.L${func_name}_c_0_str2:
	ld2 { C_ld_re_0_0.2d, C_ld_im_0_0.2d }, [ C_ptr0 ]
	fadd C_re_0_0.2d, C_re_0_0.2d, C_ld_re_0_0.2d
	fadd C_im_0_0.2d, C_im_0_0.2d, C_ld_im_0_0.2d
	st2 { C_re_0_0.2d, C_im_0_0.2d }, [ C_ptr0 ]


.L${func_name}_c_0_str_odd_start:
	add t, a_idx, 1
	cmp a_strd, t
	beq .L${func_name}_c_0_str1

	add t, a_idx, 3
	cmp a_strd, t
	beq .L${func_name}_c_0_str3

	b .L${func_name}_c_0_str_odd_end

.L${func_name}_c_0_str1:
	ldp C_ld_re_0_0d, C_ld_im_0_0d, [ C_ptr0 ]
	fadd C_re_0_0d, C_re_0_0d, C_ld_re_0_0d
	fadd C_im_0_0d, C_im_0_0d, C_ld_im_0_0d
	stp C_re_0_0d, C_im_0_0d, [ C_ptr0 ]

	b .L${func_name}_c_0_str_odd_end

.L${func_name}_c_0_str3:
	add C_ptr0, C_ptr0, #32
	ldp C_ld_re_0_1d, C_ld_im_0_1d, [ C_ptr0 ]
	fadd C_re_0_1d, C_re_0_1d, C_ld_re_0_1d
	fadd C_im_0_1d, C_im_0_1d, C_ld_im_0_1d
	stp C_re_0_1d, C_im_0_1d, [ C_ptr0 ]

	b .L${func_name}_c_0_str_odd_end

.L${func_name}_c_0_str_odd_end:
	add t, b_idx, #1
	cmp t, b_strd
	beq .L${func_name}_save_indiv_end

	add t, a_idx, 4
	cmp t, a_strd
	bls .L${func_name}_c_1_str4

	add t, a_idx, 2
	cmp t, a_strd
	bls .L${func_name}_c_1_str2

	//no full vectors to store, need to save the first halves.
	b .L${func_name}_c_1_str_odd_start

.L${func_name}_c_1_str4:
	add t, C_ptr1, #32
	ld2 { C_ld_re_1_1.2d, C_ld_im_1_1.2d }, [ t ]
	fadd C_re_1_1.2d, C_re_1_1.2d, C_ld_re_1_1.2d
	fadd C_im_1_1.2d, C_im_1_1.2d, C_ld_im_1_1.2d
	st2 { C_re_1_1.2d, C_im_1_1.2d }, [ t ]

.L${func_name}_c_1_str2:
	ld2 { C_ld_re_1_0.2d, C_ld_im_1_0.2d }, [ C_ptr1 ]
	fadd C_re_1_0.2d, C_re_1_0.2d, C_ld_re_1_0.2d
	fadd C_im_1_0.2d, C_im_1_0.2d, C_ld_im_1_0.2d
	st2 { C_re_1_0.2d, C_im_1_0.2d }, [ C_ptr1 ]


.L${func_name}_c_1_str_odd_start:
	add t, a_idx, 1
	cmp a_strd, t
	beq .L${func_name}_c_1_str1

	add t, a_idx, 3
	cmp a_strd, t
	beq .L${func_name}_c_1_str3

	b .L${func_name}_c_1_str_odd_end

.L${func_name}_c_1_str1:
	ldp C_ld_re_1_0d, C_ld_im_1_0d, [ C_ptr1 ]
	fadd C_re_1_0d, C_re_1_0d, C_ld_re_1_0d
	fadd C_im_1_0d, C_im_1_0d, C_ld_im_1_0d
	stp C_re_1_0d, C_im_1_0d, [ C_ptr1 ]

	b .L${func_name}_c_1_str_odd_end

.L${func_name}_c_1_str3:
	add C_ptr1, C_ptr1, #32
	ldp C_ld_re_1_1d, C_ld_im_1_1d, [ C_ptr1 ]
	fadd C_re_1_1d, C_re_1_1d, C_ld_re_1_1d
	fadd C_im_1_1d, C_im_1_1d, C_ld_im_1_1d
	stp C_re_1_1d, C_im_1_1d, [ C_ptr1 ]

	b .L${func_name}_c_1_str_odd_end

.L${func_name}_c_1_str_odd_end:




// .L${func_name}_c_2_str_start:
	add t, b_idx, #2
	cmp t, b_strd
	beq .L${func_name}_save_indiv_end

	add t, a_idx, 4
	cmp t, a_strd
	bls .L${func_name}_c_2_str4

	add t, a_idx, 2
	cmp t, a_strd
	bls .L${func_name}_c_2_str2

	//no full vectors to store, need to save the first halves.
	b .L${func_name}_c_2_str_odd_start

.L${func_name}_c_2_str4:
	add t, C_ptr2, #32
	ld2 { C_ld_re_0_1.2d, C_ld_im_0_1.2d }, [ t ]
	fadd C_re_2_1.2d, C_re_2_1.2d, C_ld_re_0_1.2d
	fadd C_im_2_1.2d, C_im_2_1.2d, C_ld_im_0_1.2d
	st2 { C_re_2_1.2d, C_im_2_1.2d }, [ t ]

.L${func_name}_c_2_str2:
	ld2 { C_ld_re_0_0.2d, C_ld_im_0_0.2d }, [ C_ptr2 ]
	fadd C_re_2_0.2d, C_re_2_0.2d, C_ld_re_0_0.2d
	fadd C_im_2_0.2d, C_im_2_0.2d, C_ld_im_0_0.2d
	st2 { C_re_2_0.2d, C_im_2_0.2d }, [ C_ptr2 ]


.L${func_name}_c_2_str_odd_start:
	add t, a_idx, 1
	cmp a_strd, t
	beq .L${func_name}_c_2_str1

	add t, a_idx, 3
	cmp a_strd, t
	beq .L${func_name}_c_2_str3

	b .L${func_name}_c_2_str_odd_end

.L${func_name}_c_2_str1:
	ldp C_ld_re_0_0d, C_ld_im_0_0d, [ C_ptr2 ]
	fadd C_re_2_0d, C_re_2_0d, C_ld_re_0_0d
	fadd C_im_2_0d, C_im_2_0d, C_ld_im_0_0d
	stp C_re_2_0d, C_im_2_0d, [ C_ptr2 ]

	b .L${func_name}_c_2_str_odd_end

.L${func_name}_c_2_str3:
	add C_ptr2, C_ptr2, #32
	ldp C_ld_re_0_1d, C_ld_im_0_1d, [ C_ptr2 ]
	fadd C_re_2_1d, C_re_2_1d, C_ld_re_0_1d
	fadd C_im_2_1d, C_im_2_1d, C_ld_im_0_1d
	stp C_re_2_1d, C_im_2_1d, [ C_ptr2 ]

	b .L${func_name}_c_2_str_odd_end

.L${func_name}_c_2_str_odd_end:




// .L${func_name}_c_3_str_start:
	add t, b_idx, #3
	cmp t, b_strd
	beq .L${func_name}_save_indiv_end

	add t, a_idx, 4
	cmp t, a_strd
	bls .L${func_name}_c_3_str4

	add t, a_idx, 2
	cmp t, a_strd
	bls .L${func_name}_c_3_str2

	//no full vectors to store, need to save the first halves.
	b .L${func_name}_c_3_str_odd_start

.L${func_name}_c_3_str4:
	add t, C_ptr3, #32
	ld2 { C_ld_re_1_1.2d, C_ld_im_1_1.2d }, [ t ]
	fadd C_re_3_1.2d, C_re_3_1.2d, C_ld_re_1_1.2d
	fadd C_im_3_1.2d, C_im_3_1.2d, C_ld_im_1_1.2d
	st2 { C_re_3_1.2d, C_im_3_1.2d }, [ t ]

.L${func_name}_c_3_str2:
	ld2 { C_ld_re_1_0.2d, C_ld_im_1_0.2d }, [ C_ptr3 ]
	fadd C_re_3_0.2d, C_re_3_0.2d, C_ld_re_1_0.2d
	fadd C_im_3_0.2d, C_im_3_0.2d, C_ld_im_1_0.2d
	st2 { C_re_3_0.2d, C_im_3_0.2d }, [ C_ptr3 ]

.L${func_name}_c_3_str_odd_start:
	add t, a_idx, 1
	cmp a_strd, t
	beq .L${func_name}_c_3_str1

	add t, a_idx, 3
	cmp a_strd, t
	beq .L${func_name}_c_3_str3

	b .L${func_name}_c_3_str_odd_end

.L${func_name}_c_3_str1:
	ldp C_ld_re_1_0d, C_ld_im_1_0d, [ C_ptr3 ]
	fadd C_re_3_0d, C_re_3_0d, C_ld_re_1_0d
	fadd C_im_3_0d, C_im_3_0d, C_ld_im_1_0d
	stp C_re_3_0d, C_im_3_0d, [ C_ptr3 ]

	b .L${func_name}_c_3_str_odd_end

.L${func_name}_c_3_str3:
	add C_ptr3, C_ptr3, #32
	ldp C_ld_re_1_1d, C_ld_im_1_1d, [ C_ptr3 ]
	fadd C_re_3_1d, C_re_3_1d, C_ld_re_1_1d
	fadd C_im_3_1d, C_im_3_1d, C_ld_im_1_1d
	stp C_re_3_1d, C_im_3_1d, [ C_ptr3 ]

	b .L${func_name}_c_3_str_odd_end

.L${func_name}_c_3_str_odd_end:


.L${func_name}_save_indiv_end:

.L${func_name}_store_c_end:
</%def>

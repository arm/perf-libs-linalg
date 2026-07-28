## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%def name="save_lane(vector, row_num, lane, type)">
%if type == "A1B0":
//alpha == 1.0 and beta == 0.0 -- no scalars are applied
	dup a_0h, c_${vector}_${row_num}.h[${lane}]
	str a_0h, [ c_${row_num}_ptr, ${vector * 16 + lane * 2} ]
%elif type == "AXB0":
//beta == 0.0 & alpha != 1.0 -- no load from C
	dup a_0h, c_${vector}_${row_num}.h[${lane}]
	fmul a_0h, a_0h, al_beh;
	str a_0h, [ c_${row_num}_ptr, ${vector * 16 + lane * 2} ]
%else:
// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1h, [ c_${row_num}_ptr, ${vector * 16 +  lane * 2} ]
	dup a_0h, c_${vector}_${row_num}.h[${lane}]
	//load c value
	//multiply c value by beta
	fmul a_1h, a_1h, al_beh
	fmadd a_1h, a_0h, b_0h, a_1h
	str a_1h, [ c_${row_num}_ptr, ${vector * 16 + lane * 2} ]
%endif
</%def>

<%def name="save_vector(vector, row_num, type)">
%if type == "A1B0":
//alpha == 1.0 and beta == 0.0 -- no scalars are applied
	//str c_${vector}_${row_num}, [ c_${row_num}_ptr, ${vector * 16} ]
	str a_1q, [ c_${row_num}_ptr, ${vector * 16} ]
%elif type == "AXB0":
//beta == 0.0 & alpha != 1.0 -- no load from C
	fmul a_1.8h, c_${vector}_${row_num}.8h, al_be.h[1]
	str a_1q, [ c_${row_num}_ptr, ${vector * 16} ]
%else:
	// c is loaded from memory and scalars are applied
	//copy value out of lane
	ldr a_1q, [ c_${row_num}_ptr, ${vector * 16} ]
	fmul a_1.8h, a_1.8h, al_be.h[0]
	fmla a_1.8h, c_${vector}_${row_num}.8h, al_be.h[1]
	str a_1q, [ c_${row_num}_ptr, ${vector * 16} ]
%endif
</%def>

.L_SAVE_C_REG_BLOCK:
dup b_0h, al_be.h[1]
%for ROW_NUM in range(8):
.L_SAVE_ROW_${ROW_NUM}:
	//if this row is off the end of the C dimension then quit the save code
	cmp c_strd_rem, ${ROW_NUM}
	bls .L_SAVE_END

	cmp c_cntg_rem, 23
	//if (cntg_rem >= 24) save the whole vector out in one go
	bls .L_PARTIAL_SAVE_${ROW_NUM}

	${save_vector(0, ROW_NUM, "AxBx")}
	${save_vector(1, ROW_NUM, "AxBx")}
	${save_vector(2, ROW_NUM, "AxBx")}

	//FAST VERSION end
	b .L_SAVE_ROW_END_${ROW_NUM}

.L_PARTIAL_SAVE_${ROW_NUM}:
	cmp c_cntg_rem, 15
	bhi .L_SAVE_1_${ROW_NUM}
	cmp c_cntg_rem, 7
	bhi .L_SAVE_0_${ROW_NUM}
	b .L_SAVE_SLOW_${ROW_NUM}

.L_SAVE_1_${ROW_NUM}:
	${save_vector(1, ROW_NUM, "AxBx")}
//If we need to save one, we need to save 0 too
.L_SAVE_0_${ROW_NUM}:
	${save_vector(0, ROW_NUM, "AxBx")}

// Save individual vector lanes.
.L_SAVE_SLOW_${ROW_NUM}:
	//load the address of the jump table
	adrp	tmpx0, .L_SAVE_SLOW_JMP_TBL_${ROW_NUM}
	//load the lower 12 bits of the address
	add	tmpx0, tmpx0, :lo12:.L_SAVE_SLOW_JMP_TBL_${ROW_NUM}

	//offset into
	//ldrb	tmpx1, [tmpx0,c_cntg_rem,uxtw]
	ldrh	tmpx1w, [tmpx0,c_cntg_remw,uxtw #1]
	adr	tmpx0, .Lr_SAVE_SLOW_JMP_TBL_${ROW_NUM}
	add	tmpx1, tmpx0, tmpx1w, sxth #2
	br	tmpx1
// Jump table generated from compiler output.
.Lr_SAVE_SLOW_JMP_TBL_${ROW_NUM}:
	.section	.rodata
	.align	0
	.align	2
.L_SAVE_SLOW_JMP_TBL_${ROW_NUM}:
	.2byte	(.L_SAVE_ROW_END_${ROW_NUM} - .Lr_SAVE_SLOW_JMP_TBL_${ROW_NUM}) / 4
%for LANE in [ x for x in range(0, 23, 1) ]:
	%if (LANE+1)%8 == 0:
	.2byte	(.L_SAVE_ROW_END_${ROW_NUM} - .Lr_SAVE_SLOW_JMP_TBL_${ROW_NUM}) / 4
	%else:
	.2byte	(.L_SAVE_SLOW_${LANE}_${ROW_NUM} - .Lr_SAVE_SLOW_JMP_TBL_${ROW_NUM}) / 4
	%endif
%endfor
	.text
	.p2align 2
%for VECTOR in range(2, -1, -1):
.L_SAVE_SLOW_${VECTOR * 8 + 6}_${ROW_NUM}:
	${save_lane(VECTOR, ROW_NUM, 6,  "AxBx") }
.L_SAVE_SLOW_${VECTOR * 8 + 5}_${ROW_NUM}:
	${save_lane(VECTOR, ROW_NUM, 5,  "AxBx") }
.L_SAVE_SLOW_${VECTOR * 8 + 4}_${ROW_NUM}:
	${save_lane(VECTOR, ROW_NUM, 4,  "AxBx") }
.L_SAVE_SLOW_${VECTOR * 8 + 3}_${ROW_NUM}:
	${save_lane(VECTOR, ROW_NUM, 3,  "AxBx") }
.L_SAVE_SLOW_${VECTOR * 8 + 2}_${ROW_NUM}:
	${save_lane(VECTOR, ROW_NUM, 2,  "AxBx") }
.L_SAVE_SLOW_${VECTOR * 8 + 1}_${ROW_NUM}:
	${save_lane(VECTOR, ROW_NUM, 1,  "AxBx") }
.L_SAVE_SLOW_${VECTOR * 8 + 0}_${ROW_NUM}:
	${save_lane(VECTOR, ROW_NUM, 0,  "AxBx") }
	//str c_${VECTOR}_${ROW_NUM}h, [ c_${ROW_NUM}_ptr, ${VECTOR * 16} ]
	b .L_SAVE_ROW_END_${ROW_NUM}
	.p2align 2
%endfor

.L_SAVE_ROW_END_${ROW_NUM}:
	//falls through to the mako'ed loop

%endfor
.L_SAVE_END:
	b .L_SAVE_RETURN

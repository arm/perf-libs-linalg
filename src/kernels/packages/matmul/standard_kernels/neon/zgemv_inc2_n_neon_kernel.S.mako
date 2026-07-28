## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

/* PREFACE: This should be largely identical to zgemv_n, since we don't need ld2
* for loading from x in the non-transposed case. Changes are highlighted by
* comments starting with 'INC2:'. */

// How many columns do we process per loop?
<% COLUMNS_PER_LOOP = 6 %>
// Keep this list length at 4 or fewer, otherwise code generation fails.
// Include 1 in this list.
<% UNROLL_PROGRESSION = [4, 2, 1] %>
<% assert len(UNROLL_PROGRESSION) <= 4 %>
<% assert 1 in UNROLL_PROGRESSION %>

M                      .req x0
N                      .req x1
A                      .req x2
LDA                    .req x3
X                      .req x4
Y                      .req x6
ALPHA                  .req v0
BETA                   .req v1
Y_ADDR                 .req x7
COLUMN                 .req x8
%for ix, unroll in enumerate(UNROLL_PROGRESSION):
	M_ROUND_DOWN_${unroll} .req x${ix+9}
%endfor
N_ROUND_DOWN           .req x12
## x13, x14 unavailable on Arm64EC
INTER_COLUMN_STEP      .req x15
INTER_COLUMN_STEP_TAIL .req x16
## Avoid using x18, x23, x24, x28 and x29 when allocating general-purpose registers
## so that our code remains portable, including on Arm64EC. We also disallow x30,
## because it can break backtraces and perf profiles.
<% gpreg = 16 %>
%for c in range(COLUMNS_PER_LOOP):
	<% gpreg += 1 %>
	%if gpreg == 18:
		<% gpreg += 1 %>
	%endif
	%if gpreg == 23:
		<% gpreg += 2 %>
	%endif
	<% assert gpreg <= 27, f"gpreg exceeded 27 (got {gpreg})" %>
	A_ADDR_${c} .req x${gpreg}
%endfor
// .req X registers
%for c in range(COLUMNS_PER_LOOP):
	A_REG_X_${c}_V .req v${2*c+2}
	A_REG_X_${c}_Q .req q${2*c+2}
	A_REG_X_${c}_D .req d${2*c+2}
	A_REG_Y_${c}_V .req v${2*c+3}
	A_REG_Y_${c}_Q .req q${2*c+3}
	A_REG_Y_${c}_D .req d${2*c+3}
%endfor
%for c in range(COLUMNS_PER_LOOP):
	X_REG_${c}_V .req v${2*COLUMNS_PER_LOOP+2+c}
	X_REG_${c}_Q .req q${2*COLUMNS_PER_LOOP+2+c}
	X_REG_${c}_D .req d${2*COLUMNS_PER_LOOP+2+c}
%endfor
%for l in range(max(UNROLL_PROGRESSION)):
	Y_REG_X_${l}_V .req v${COLUMNS_PER_LOOP*3+2+l*2}
	Y_REG_X_${l}_Q .req q${COLUMNS_PER_LOOP*3+2+l*2}
	Y_REG_X_${l}_D .req d${COLUMNS_PER_LOOP*3+2+l*2}
	Y_REG_Y_${l}_V .req v${COLUMNS_PER_LOOP*3+3+l*2}
	Y_REG_Y_${l}_Q .req q${COLUMNS_PER_LOOP*3+3+l*2}
	Y_REG_Y_${l}_D .req d${COLUMNS_PER_LOOP*3+3+l*2}
%endfor
// .req accumulator for the tail (just Y_REG_0)
TAIL_ACC_X_V .req v${COLUMNS_PER_LOOP*3+2}
TAIL_ACC_X_Q .req q${COLUMNS_PER_LOOP*3+2}
TAIL_ACC_X_D .req d${COLUMNS_PER_LOOP*3+2}
TAIL_ACC_Y_V .req v${COLUMNS_PER_LOOP*3+3}
TAIL_ACC_Y_Q .req q${COLUMNS_PER_LOOP*3+3}
TAIL_ACC_Y_D .req d${COLUMNS_PER_LOOP*3+3}

<% func_name = "zgemv_inc2_n_neon_kernel" %>
${prologue(func_name)}
stp x29, x30, [sp, #-160]!
mov x29, sp
stp  d8,  d9, [sp, #16]
stp d10, d11, [sp, #32]
stp d12, d13, [sp, #48]
stp d14, d15, [sp, #64]
stp x19, x20, [sp, #80]
stp x21, x22, [sp, #96]
stp x23, x24, [sp, #112]
stp x25, x26, [sp, #128]
stp x27, x28, [sp, #144]

// Setup A_ADDR_X values
%for c in range(COLUMNS_PER_LOOP):
	mov x5, ${c * 16}
	madd A_ADDR_${c}, x5, LDA, A
%endfor

// Setup the column increment
sub INTER_COLUMN_STEP_TAIL, LDA, M
lsl INTER_COLUMN_STEP_TAIL, INTER_COLUMN_STEP_TAIL, #4
sub INTER_COLUMN_STEP, LDA, M
mov x5, #${COLUMNS_PER_LOOP-1}
madd INTER_COLUMN_STEP, LDA, x5, INTER_COLUMN_STEP
lsl INTER_COLUMN_STEP, INTER_COLUMN_STEP, #4

// Setup the rounded down versions of M and N
%for unroll in UNROLL_PROGRESSION:
	// INC2: 32 instead of 16
	mov x5, #${unroll * 32}
	udiv  M_ROUND_DOWN_${unroll}, M, x5
	lsl x5, x5, #4
	mul  M_ROUND_DOWN_${unroll}, M_ROUND_DOWN_${unroll}, x5
%endfor
mov x5, #${COLUMNS_PER_LOOP}
udiv N_ROUND_DOWN,   N, x5
mul  N_ROUND_DOWN,   N_ROUND_DOWN,   x5
// Offset M_ROUND_DOWN with Y_ADDR to use for comparisons with pointers
%for unroll in UNROLL_PROGRESSION:
	add M_ROUND_DOWN_${unroll}, M_ROUND_DOWN_${unroll}, Y
%endfor

// Setup M to be the end of the Y vector, so we can use this for comparisons
// INC2: extra ls, 5 instead of 4
lsl M, M, #5
add M, M, Y

// Actual gemv loop(s)
mov COLUMN, #0

mov x5, #6000
cmp M, x5
bgt .Lsize_big

%for size in ["small", "big"]:
	.Lsize_${size}:
	## We want to do this twice, once processing multiple columns at once, then
	## once acting as a 'tail'
	%for is_tail in [False, True]:
		<% tail = "tail" if is_tail else "" %>
		b .Lcolumn_loop_cond_${tail}_${size}
		.Lcolumn_loop_${tail}_${size}:

		// Load the needed Xs for this loop
		%if is_tail:
			ldr X_REG_0_Q, [X]
			//dup X_REG_0_V.2d, X_REG_0_V.d[0]
		%else:
			%for c in range(COLUMNS_PER_LOOP):
				// INC2: multiples of 32 not 16
				ldr X_REG_${c}_Q, [X, ${c * 32}]
				//dup X_REG_${c}_V.2d, X_REG_${c}_V.d[0]
			%endfor
		%endif
		mov Y_ADDR, Y

		## Unrolled version of the loop, similar usage as in sgemv_t.
		<%def name="loop_unrolled(level, limit, is_tail, size)">
			%if is_tail:
				b .Lrow_loop_unrlvl${level}_cond_tail_${size}
				.Lrow_loop_unrlvl${level}_tail_${size}:
				%for l in range(level):
					// INC2: Use single-structure indexed LD2 instructions here.
					ld2 {Y_REG_X_0_V.d, Y_REG_Y_0_V.d}[0], [Y_ADDR]
					add Y_ADDR, Y_ADDR, #32
					ld2 {Y_REG_X_0_V.d, Y_REG_Y_0_V.d}[1], [Y_ADDR]
					sub Y_ADDR, Y_ADDR, #32
					ld2 {A_REG_X_0_V.2d, A_REG_Y_0_V.2d}, [A_ADDR_0], #32
					fmla Y_REG_X_0_V.2d, A_REG_X_0_V.2d, X_REG_0_V.d[0]
					fmls Y_REG_X_0_V.2d, A_REG_Y_0_V.2d, X_REG_0_V.d[1]
					fmla Y_REG_Y_0_V.2d, A_REG_X_0_V.2d, X_REG_0_V.d[1]
					fmla Y_REG_Y_0_V.2d, A_REG_Y_0_V.2d, X_REG_0_V.d[0]
					st2 {Y_REG_X_0_V.d, Y_REG_Y_0_V.d}[0], [Y_ADDR]
					add Y_ADDR, Y_ADDR, #32
					st2 {Y_REG_X_0_V.d, Y_REG_Y_0_V.d}[1], [Y_ADDR]
					add Y_ADDR, Y_ADDR, #32
				%endfor
				.Lrow_loop_unrlvl${level}_cond_tail_${size}:
				cmp Y_ADDR, ${limit}
				blt .Lrow_loop_unrlvl${level}_tail_${size}
			%else:
				b .Lrow_loop_unrlvl${level}_cond_${size}
				.Lrow_loop_unrlvl${level}_${size}:
				// Load all the Ys
				%for l in range(level):
					// INC2: Use single-structure indexed LD2 instructions here.
					ld2 {Y_REG_X_${l}_V.d, Y_REG_Y_${l}_V.d}[0], [Y_ADDR]
					add Y_ADDR, Y_ADDR, #32
					ld2 {Y_REG_X_${l}_V.d, Y_REG_Y_${l}_V.d}[1], [Y_ADDR]
					add Y_ADDR, Y_ADDR, #32
				%endfor
				// Reset Y for storing w/ post-index addressing
				//INC2: 64 not 32
				sub Y_ADDR, Y_ADDR, ${level * 64}
				%for l in range(level):
					// Load from A and fmla into Y
					## Preload A reg
					ld2 {A_REG_X_0_V.2d, A_REG_Y_0_V.2d}, [A_ADDR_0], #32
					%if COLUMNS_PER_LOOP > 1:
						ld2 {A_REG_X_1_V.2d, A_REG_Y_1_V.2d}, [A_ADDR_1], #32
					%endif
					%for c in range(COLUMNS_PER_LOOP):
						%if c < COLUMNS_PER_LOOP-2:
							ld2 {A_REG_X_${c+2}_V.2d, A_REG_Y_${c+2}_V.2d}, [A_ADDR_${c+2}], #32
						%endif
						fmla Y_REG_X_${l}_V.2d, A_REG_X_${c}_V.2d, X_REG_${c}_V.d[0]
						fmls Y_REG_X_${l}_V.2d, A_REG_Y_${c}_V.2d, X_REG_${c}_V.d[1]
						fmla Y_REG_Y_${l}_V.2d, A_REG_X_${c}_V.2d, X_REG_${c}_V.d[1]
						fmla Y_REG_Y_${l}_V.2d, A_REG_Y_${c}_V.2d, X_REG_${c}_V.d[0]
						%if l %2 == 0 and size == "big":
							prfm pldl1keep, [A_ADDR_${c}, #256]
						%endif
					%endfor
				%endfor
				// Store all Ys
				%for l in range(level):
					// INC2: Use single-structure indexed ST2 instructions here.
					st2 {Y_REG_X_${l}_V.d, Y_REG_Y_${l}_V.d}[0], [Y_ADDR]
					add Y_ADDR, Y_ADDR, #32
					st2 {Y_REG_X_${l}_V.d, Y_REG_Y_${l}_V.d}[1], [Y_ADDR]
					add Y_ADDR, Y_ADDR, #32
				%endfor
				.Lrow_loop_unrlvl${level}_cond_${size}:
				cmp Y_ADDR, ${limit}
				blt .Lrow_loop_unrlvl${level}_${size}
			%endif
		</%def>\

		%for unroll in UNROLL_PROGRESSION:
			${loop_unrolled(unroll, "M_ROUND_DOWN_" + str(unroll), is_tail, size)}
		%endfor

		// Finish the tail
		%if is_tail:
			b .Lrow_loop_tail_cond_${tail}_${size}
			.Lrow_loop_tail_${tail}_${size}:
			ldr A_REG_X_0_D, [A_ADDR_0]
			ldr A_REG_Y_0_D, [A_ADDR_0, #8]
			ldr Y_REG_X_0_D, [Y_ADDR]
			ldr Y_REG_Y_0_D, [Y_ADDR, #8]
			fmla Y_REG_X_0_D, A_REG_X_0_D, X_REG_0_V.d[0]
			fmls Y_REG_X_0_D, A_REG_Y_0_D, X_REG_0_V.d[1]
			fmla Y_REG_Y_0_D, A_REG_X_0_D, X_REG_0_V.d[1]
			fmla Y_REG_Y_0_D, A_REG_Y_0_D, X_REG_0_V.d[0]
			str Y_REG_X_0_D, [Y_ADDR]
			str Y_REG_Y_0_D, [Y_ADDR, #8]
			add A_ADDR_0, A_ADDR_0, #16
			//INC2: 32 not 16
			add Y_ADDR, Y_ADDR, #32
			.Lrow_loop_tail_cond_${tail}_${size}:
			cmp Y_ADDR, M
			blt .Lrow_loop_tail_${tail}_${size}
		%else:
			b .Lrow_loop_tail_cond_${tail}_${size}
			.Lrow_loop_tail_${tail}_${size}:
			ldr TAIL_ACC_X_D, [Y_ADDR]
			ldr TAIL_ACC_Y_D, [Y_ADDR, #8]
			%for c in range(COLUMNS_PER_LOOP):
				ldr A_REG_X_${c}_D, [A_ADDR_${c}]
				ldr A_REG_Y_${c}_D, [A_ADDR_${c}, #8]
				fmla TAIL_ACC_X_D, A_REG_X_${c}_D, X_REG_${c}_V.d[0]
				fmls TAIL_ACC_X_D, A_REG_Y_${c}_D, X_REG_${c}_V.d[1]
				fmla TAIL_ACC_Y_D, A_REG_X_${c}_D, X_REG_${c}_V.d[1]
				fmla TAIL_ACC_Y_D, A_REG_Y_${c}_D, X_REG_${c}_V.d[0]
			%endfor
			str TAIL_ACC_X_D, [Y_ADDR]
			str TAIL_ACC_Y_D, [Y_ADDR, #8]
			%for c in range(COLUMNS_PER_LOOP):
				add A_ADDR_${c}, A_ADDR_${c}, #16
			%endfor
			// INC2: 32 not 16
			add Y_ADDR, Y_ADDR, #32
			.Lrow_loop_tail_cond_${tail}_${size}:
			cmp Y_ADDR, M
			blt .Lrow_loop_tail_${tail}_${size}
		%endif

		// INC2: 32 not 16
		%if is_tail:
			add COLUMN, COLUMN, #1
			add X, X, #32
			add A_ADDR_0, A_ADDR_0, INTER_COLUMN_STEP_TAIL
		%else:
			add COLUMN, COLUMN, #${COLUMNS_PER_LOOP}
			add X, X, #${32 * COLUMNS_PER_LOOP}
			%for c in range(COLUMNS_PER_LOOP):
				add A_ADDR_${c}, A_ADDR_${c}, INTER_COLUMN_STEP
			%endfor

		%endif

		.Lcolumn_loop_cond_${tail}_${size}:
		%if is_tail:
			cmp COLUMN, N
		%else:
			cmp COLUMN, N_ROUND_DOWN
		%endif
		blt .Lcolumn_loop_${tail}_${size}

	%endfor // for is_tail in [False, True]

	b .Lcleanup
%endfor // for size in ["big", "small"]:

	.Lcleanup:
	ldp  d8,  d9, [sp, #16]
	ldp d10, d11, [sp, #32]
	ldp d12, d13, [sp, #48]
	ldp d14, d15, [sp, #64]
	ldp x19, x20, [sp, #80]
	ldp x21, x22, [sp, #96]
	ldp x23, x24, [sp, #112]
	ldp x25, x26, [sp, #128]
	ldp x27, x28, [sp, #144]
	ldp x29, x30, [sp], #160
	ret

	${epilogue(func_name)}

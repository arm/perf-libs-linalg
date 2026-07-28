## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

// How many columns do we process per loop?
<% COLUMNS_PER_LOOP = 8 %>
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
N_ROUND_DOWN           .req x13
INTER_COLUMN_STEP      .req x14
INTER_COLUMN_STEP_TAIL .req x15
## Avoid using x18 and x29 when allocating general-purpose registers
## so that our code remains portable
<% gpreg = 15 %>
%for c in range(COLUMNS_PER_LOOP):
	<% gpreg += 1 %>
	%if gpreg == 18 or gpreg == 29:
		<% gpreg += 1 %>
	%endif
	A_ADDR_${c} .req x${gpreg}
%endfor
// .req X registers
%for c in range(COLUMNS_PER_LOOP):
	A_REG_${c}_V .req v${c+2}
	A_REG_${c}_Q .req q${c+2}
	A_REG_${c}_D .req d${c+2}
%endfor
%for c in range(COLUMNS_PER_LOOP):
	X_REG_${c}_V .req v${COLUMNS_PER_LOOP+2+c}
	X_REG_${c}_Q .req q${COLUMNS_PER_LOOP+2+c}
	X_REG_${c}_D .req d${COLUMNS_PER_LOOP+2+c}
%endfor
%for l in range(max(UNROLL_PROGRESSION)):
	Y_REG_${l}_V .req v${COLUMNS_PER_LOOP*2+2+l}
	Y_REG_${l}_Q .req q${COLUMNS_PER_LOOP*2+2+l}
	Y_REG_${l}_D .req d${COLUMNS_PER_LOOP*2+2+l}
%endfor
// .req accumulator for the tail (just Y_REG_0)
TAIL_ACC_V .req v${COLUMNS_PER_LOOP*2+2}
TAIL_ACC_Q .req q${COLUMNS_PER_LOOP*2+2}
TAIL_ACC_D .req d${COLUMNS_PER_LOOP*2+2}

<% func_name = "dgemv_n_neon_kernel" %>
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
	mov x5, ${c * 8}
	madd A_ADDR_${c}, x5, LDA, A
%endfor

// Dup alpha and beta into all lanes of v0 and v1
dup ALPHA.2d, ALPHA.d[0]
dup BETA.2d, BETA.d[0]

// Setup the column increment
sub INTER_COLUMN_STEP_TAIL, LDA, M
lsl INTER_COLUMN_STEP_TAIL, INTER_COLUMN_STEP_TAIL, #3
sub INTER_COLUMN_STEP, LDA, M
mov x5, #${COLUMNS_PER_LOOP-1}
madd INTER_COLUMN_STEP, LDA, x5, INTER_COLUMN_STEP
lsl INTER_COLUMN_STEP, INTER_COLUMN_STEP, #3

// Setup the rounded down versions of M and N
%for unroll in UNROLL_PROGRESSION:
	mov x5, #${unroll * 8}
	udiv  M_ROUND_DOWN_${unroll}, M, x5
	lsl x5, x5, #3 // Mul by 8 because doubles are 8b
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
lsl M, M, #3
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
			ldr d2, [X]
			dup v2.2d, v2.d[0]
		%else:
			%for c in range(COLUMNS_PER_LOOP):
				ldr X_REG_${c}_D, [X, ${c * 8}]
				dup X_REG_${c}_V.2d, X_REG_${c}_V.d[0]
			%endfor
		%endif
		mov Y_ADDR, Y

		## Unrolled version of the loop, similar usage as in sgemv_t.
		<%def name="loop_unrolled(level, limit, is_tail, size)">
			%if is_tail:
				b .Lrow_loop_unrlvl${level}_cond_tail_${size}
				.Lrow_loop_unrlvl${level}_tail_${size}:
				%for l in range(level):
					ldr q3, [A_ADDR_0, #${l * 16}]
					ldr q4, [Y_ADDR, #${l * 16}]
					fmla v4.2d, v3.2d, v2.2d
					str q4, [Y_ADDR, #${l * 16}]
				%endfor
				add A_ADDR_0, A_ADDR_0, #${level * 16}
				add Y_ADDR, Y_ADDR, #${level * 16}
				.Lrow_loop_unrlvl${level}_cond_tail_${size}:
				cmp Y_ADDR, ${limit}
				blt .Lrow_loop_unrlvl${level}_tail_${size}
			%else:
				b .Lrow_loop_unrlvl${level}_cond_${size}
				.Lrow_loop_unrlvl${level}_${size}:
				// Load all the Ys
				%for l in range(level):
					ldr Y_REG_${l}_Q, [Y_ADDR, #${l * 16}]
				%endfor
				%for l in range(level):
					// Load from A and fmla into Y
					## Preload A reg
					ldr A_REG_0_Q, [A_ADDR_0, #${l * 16}]
					% if COLUMNS_PER_LOOP > 1:
						ldr A_REG_1_Q, [A_ADDR_1, #${l * 16}]
					%endif
					%for c in range(COLUMNS_PER_LOOP):
						%if c < COLUMNS_PER_LOOP-2:
							ldr A_REG_${c+2}_Q, [A_ADDR_${c+2}, #${l * 16}]
						%endif
						fmla Y_REG_${l}_V.2d, A_REG_${c}_V.2d, X_REG_${c}_V.2d
						%if l %2 == 0 and size == "big":
							prfm pldl1keep, [A_ADDR_${c}, #128]
						%endif
					%endfor
				%endfor
				// Store all Ys
				%for l in range(level):
					str Y_REG_${l}_Q, [Y_ADDR, #${l * 16}]
				%endfor
				%for c in range(COLUMNS_PER_LOOP):
					add A_ADDR_${c}, A_ADDR_${c}, #${level * 16}
				%endfor
				add Y_ADDR, Y_ADDR, #${level * 16}
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
			ldr d3, [A_ADDR_0]
			ldr d4, [Y_ADDR]
			fmla d4, d3, v2.d[0]
			str d4, [Y_ADDR]
			add A_ADDR_0, A_ADDR_0, #8
			add Y_ADDR, Y_ADDR, #8
			.Lrow_loop_tail_cond_${tail}_${size}:
			cmp Y_ADDR, M
			blt .Lrow_loop_tail_${tail}_${size}
		%else:
			b .Lrow_loop_tail_cond_${tail}_${size}
			.Lrow_loop_tail_${tail}_${size}:
			ldr TAIL_ACC_D, [Y_ADDR]
			%for c in range(COLUMNS_PER_LOOP):
				ldr A_REG_${c}_D, [A_ADDR_${c}]
				fmla TAIL_ACC_D, A_REG_${c}_D, X_REG_${c}_V.d[0]
			%endfor
			str TAIL_ACC_D, [Y_ADDR]
			%for c in range(COLUMNS_PER_LOOP):
				add A_ADDR_${c}, A_ADDR_${c}, #8
			%endfor
			add Y_ADDR, Y_ADDR, #8
			.Lrow_loop_tail_cond_${tail}_${size}:
			cmp Y_ADDR, M
			blt .Lrow_loop_tail_${tail}_${size}
		%endif

		%if is_tail:
			add COLUMN, COLUMN, #1
			add X, X, #8
			add A_ADDR_0, A_ADDR_0, INTER_COLUMN_STEP_TAIL
		%else:
			add COLUMN, COLUMN, #${COLUMNS_PER_LOOP}
			add X, X, #${8 * COLUMNS_PER_LOOP}
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

## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

/* PREFACE: This should be largely identical to dgemv_n, since we don't need ld2
* for loading from x in the non-transposed case. Changes are highlighted by
* comments starting with 'INC2:'. */

## We need this for logarithms when adding across many vectors in a tree
<% import math %>
INC_X          .req x5 // Since we don't care about INC_X and INC_Y here, these are just scratch registers.
INC_Y          .req x7

M              .req x0
N              .req x1
A              .req x2
LDA            .req x3
X              .req x4
Y              .req x6
ALPHA          .req v0
BETA           .req v1
M_ROUND_DOWN_8 .req x9  // M rounded down to the nearest multiple of the vector
// register size * 8, then multiplied by 4 (since floats
// are 4 bytes). Used for loops unrolled 8 times.
M_ROUND_DOWN_4 .req x10 // For loops unrolled 4 times.
M_ROUND_DOWN_2 .req x11 // For loops unrolled 2 times.
M_ROUND_DOWN_1 .req x12 // For loops that aren't unrolled.
N_ROUND_DOWN   .req x13 // Rounding down N since we process multiple columns at once
X_ADDR         .req x14
INTER_COLUMN_STEP .req x15 // The step needed to get from the end of one column to the start of the next
INTER_COLUMN_STEP_TAIL .req x3 // The step needed to get from the end of one column to the start of the next in the tail loop

<%def name="dgemv(is_parallel=False)">

	## This is how many multiplications per load we're doing. If this is 2, we step
	## COLUMN += 2 * LDA, for example, since we're processing 2 columns at once.
	## This means we only load from X once per 2 muls, hence the name.
	<% NUM_COLUMNS = 8 if is_parallel else 14 %>
	<% ORIGINAL_NUM_COLUMNS = NUM_COLUMNS %> ## Used for the tail loop later on
	<% NUM_PRELOAD = NUM_COLUMNS %> ## How many registers to consume for preloading
	<% assert NUM_PRELOAD <= NUM_COLUMNS %>\

	// Each A_ADDR stands for a different address in the A matrix - we loop over
	// columns in the matrix in multiples of ${NUM_COLUMNS} to avoid loading X
	// multiple times.
	A_ADDR_0 .req x2 // A_ADDR_0 aliases A
	## Avoid using x18 and x29 when allocating general-purpose registers
	## so that our code remains portable
	<% gpreg = 15 %>
	%for column in range(1, NUM_COLUMNS):
		<% gpreg += 1 %>
		%if gpreg == 18 or gpreg == 29:
			<% gpreg += 1 %>
		%endif
		A_ADDR_${column} .req x${gpreg}
	%endfor

	// .req the various vector registers in use
	X_REG_V .req v2
	X_REG_Q .req q2
	X_REG_D .req d2
	X_REG_V_ADJ .req v3
	%for column in range(NUM_COLUMNS):
		ACC_REG_${column}_V .req v${4 + NUM_PRELOAD + column}
		ACC_REG_${column}_D .req d${4 + NUM_PRELOAD + column}
	%endfor
	%for pl in range(NUM_PRELOAD):
		A_REG_${pl}_V .req v${4 + pl}
		A_REG_${pl}_Q .req q${4 + pl}
		A_REG_${pl}_D .req d${4 + pl}
	%endfor

	%if is_parallel:
		<% func_name = "dgemv_inc2_t_neon_kernel_parallel" %>
	%else:
		<% func_name = "dgemv_inc2_t_neon_kernel_serial" %>
	%endif
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

	mov x5, #${NUM_COLUMNS-1}
	// Setup inter-column-step and LDA in bytes. Since INTER_COLUMN_STEP_TAIL aliases LDA, set it up as x7 first.
	sub x7, LDA, M
	lsl x7, x7, #3
	sub INTER_COLUMN_STEP, LDA, M
	lsl INTER_COLUMN_STEP, INTER_COLUMN_STEP, #3
	lsl LDA, LDA, #3 // 8b per float
	madd INTER_COLUMN_STEP, LDA, x5, INTER_COLUMN_STEP
	mov x5, #${NUM_COLUMNS}

	// Init N_ROUND_DOWN and M_ROUND_DOWN values
	udiv N_ROUND_DOWN, N, x5
	mul N_ROUND_DOWN, N_ROUND_DOWN, x5
	lsl N, N, #3
	lsl N_ROUND_DOWN, N_ROUND_DOWN, #3
	// INC2: lsl #1 here b/c inc2
	add N, Y, N, lsl #1
	add N_ROUND_DOWN, Y, N_ROUND_DOWN, lsl #1
	asr M_ROUND_DOWN_8, M, #4
	lsl M_ROUND_DOWN_8, M_ROUND_DOWN_8, #7
	asr M_ROUND_DOWN_4, M, #3
	lsl M_ROUND_DOWN_4, M_ROUND_DOWN_4, #6
	asr M_ROUND_DOWN_2, M, #2
	lsl M_ROUND_DOWN_2, M_ROUND_DOWN_2, #5
	asr M_ROUND_DOWN_1, M, #1
	lsl M_ROUND_DOWN_1, M_ROUND_DOWN_1, #4

	// INC2: lsl #1 here b/c inc2
	add M_ROUND_DOWN_8, X, M_ROUND_DOWN_8, lsl #1 // Offset to X, to use as a CMP op
	add M_ROUND_DOWN_4, X, M_ROUND_DOWN_4, lsl #1
	add M_ROUND_DOWN_2, X, M_ROUND_DOWN_2, lsl #1
	add M_ROUND_DOWN_1, X, M_ROUND_DOWN_1, lsl #1

	// Init A_ADDR_X
	%for column in range(1, NUM_COLUMNS):
		mov x5, ${column}
		madd A_ADDR_${column}, LDA, x5, A
	%endfor

	// Actually init INTER_COLUMN_STEP_TAIL.
	mov INTER_COLUMN_STEP_TAIL, x7

	cmp M, 200
	bgt .L${is_parallel}_size_big

	## Big kernels we prefetch, small we don't
	%for size in ["small", "big"]:
		.L${is_parallel}_size_${size}:

		fcmp d1, #0.0
		beq .L${is_parallel}_beta_0_${size}
		fmov d2, #1.0
		fcmp d1, d2
		beq .L${is_parallel}_beta_1_${size}

		%for beta in ["any", 0, 1]:
			.L${is_parallel}_beta_${beta}_${size}:

			// INC2: Extra ls here b/c inc2
			lsl M, M, #4
			add M, M, X

			## Generate one loop like normal, and another to clean up the tail columns
			%for is_tail, tail in [(False, ''), (True, 'tail')]:
				<% NUM_COLUMNS = 1 if is_tail else ORIGINAL_NUM_COLUMNS %>
				// Loop over the columns
				b .L${is_parallel}_loop_cond_${beta}_${size}_${tail}
				.L${is_parallel}_loop_${beta}_${size}_${tail}:

				mov X_ADDR, X

				## Create an unrolled loop with the given unroll level and the given register
				## to use as a limit for X_ADDR (for example, M_ROUND_DOWN). This doesn't
				## handle tails, and just loops X_ADDR until the limit. Make sure limit -
				## X_ADDR is divisible by level * 4 * 4, otherwise this will go over the limit
				## slightly.
				<%def name="loop_unrolled(level, limit, beta, size, is_tail, tail)">
					b .L${is_parallel}_column_loop_unrlvl${level}_cond_${beta}_${size}_${tail}
					.L${is_parallel}_column_loop_unrlvl${level}_${beta}_${size}_${tail}:
					## Keep track of the current reg we're using / preloading
					<% curr_a_reg = 0 %>
					%for l in range(level):
						// INC2: LD2 post-increment here
						ld2 {X_REG_V.2d, X_REG_V_ADJ.2d}, [X_ADDR], #32
						## Since we're preloading registers, initially load them in if this is the
						## first unroll
						%if is_tail:
							ldr A_REG_0_Q, [A_ADDR_0, #${l * 16}]
							fmla ACC_REG_0_V.2d, A_REG_0_V.2d, X_REG_V.2d
							%if l == level - 1:
								add A_ADDR_0, A_ADDR_0, #${level * 16}
							%endif
						%else:
							%if l == 0:
								%for pl in range(NUM_PRELOAD):
									ldr A_REG_${pl}_Q, [A_ADDR_${pl}]
								%endfor
							%endif
							%for column in range(NUM_COLUMNS):
								<% preloading_next = (column >= NUM_COLUMNS - NUM_PRELOAD) %>
								<% curr_preload = (column+NUM_PRELOAD) if not preloading_next else ((column+NUM_PRELOAD)%NUM_COLUMNS) %>
								## Prefetch if this is a multiple of 4
								%if l %2 == 0 and size == "big":
									prfm pldl1keep, [A_ADDR_${column}, #128]
								%endif
								fmla ACC_REG_${column}_V.2d, A_REG_${curr_a_reg}_V.2d, X_REG_V.2d
								## Preload the next register - use different instructions if we're at the end
								## of the current mini-unroll
								## Don't preload if we're at the end of the unrolled section
								%if l < level-1:
									ldr A_REG_${curr_a_reg}_Q, [A_ADDR_${curr_preload}, #${((l+1) if preloading_next else l) * 16}]
									## If we are at the end, increment the A_ADDR values
								%else:
									add A_ADDR_${column}, A_ADDR_${column}, #${level * 16}
								%endif
								## Increment the preload target reg num
								<% curr_a_reg = (curr_a_reg + 1) %NUM_PRELOAD %>
							%endfor
						%endif
					%endfor
					##%for column in range(NUM_COLUMNS):
					##  add A_ADDR_${column}, A_ADDR_${column}, #${level * 16}
					##%endfor
					.L${is_parallel}_column_loop_unrlvl${level}_cond_${beta}_${size}_${tail}:
					cmp X_ADDR, ${limit}
					blt .L${is_parallel}_column_loop_unrlvl${level}_${beta}_${size}_${tail}
				</%def>

				// Init registers for accumulating
				%for column in range(NUM_COLUMNS):
					movi ACC_REG_${column}_V.2d, #0
				%endfor

				// Use a progressively less unrolled loop
				${loop_unrolled(4, "M_ROUND_DOWN_4", beta, size, is_tail, tail)}
				${loop_unrolled(2, "M_ROUND_DOWN_2", beta, size, is_tail, tail)}
				${loop_unrolled(1, "M_ROUND_DOWN_1", beta, size, is_tail, tail)}

				// Finish the loop with scalar instructions
				b .L${is_parallel}_column_loop_tail_cond_${beta}_${size}_${tail}
				.L${is_parallel}_column_loop_tail_${beta}_${size}_${tail}:
				//INC2 post-increment with 16 here instead of adding 8
				ldr X_REG_D, [X_ADDR], #16
				%for column in range(NUM_COLUMNS):
					ldr A_REG_0_D, [A_ADDR_${column}]
					fmla ACC_REG_${column}_V.2d, X_REG_V.2d, A_REG_0_V.2d
					add A_ADDR_${column}, A_ADDR_${column}, #8
				%endfor
				.L${is_parallel}_column_loop_tail_cond_${beta}_${size}_${tail}:
				cmp X_ADDR, M
				blt .L${is_parallel}_column_loop_tail_${beta}_${size}_${tail}

				// Prefetch the next loop
				%for column in range(NUM_COLUMNS):
					%if is_tail:
						prfm pldl1keep, [A_ADDR_${column}, INTER_COLUMN_STEP_TAIL]
					%else:
						prfm pldl1keep, [A_ADDR_${column}, INTER_COLUMN_STEP]
					%endif
				%endfor

				// INC2: This whole next section changed 8 * column to 16 * column
				// Sum across accumulator registers
				%for column in range (NUM_COLUMNS):
					faddp ACC_REG_${column}_V.2d, ACC_REG_${column}_V.2d, ACC_REG_${column}_V.2d
					// Store in Y
					%if beta == 0:
						fmul X_REG_D, ACC_REG_${column}_D, d0
					%elif beta == 1:
						ldr X_REG_D, [Y, #${16 * column}]
						fmla X_REG_D, ACC_REG_${column}_D, ALPHA.d[0]
					%else:
						ldr X_REG_D, [Y, #${16 * column}]
						##fmul X_REG_D, X_REG_D, BETA.d[0]
						fmla X_REG_D, ACC_REG_${column}_D, ALPHA.d[0]
					%endif
					str X_REG_D, [Y, #${16 * column}]
					## Increment A_ADDR_X
					%if is_tail:
						add A_ADDR_${column}, A_ADDR_${column}, INTER_COLUMN_STEP_TAIL
					%else:
						add A_ADDR_${column}, A_ADDR_${column}, INTER_COLUMN_STEP
					%endif
				%endfor
				// INC2: Changed to 16
				add Y, Y, #${16 * NUM_COLUMNS}
				.L${is_parallel}_loop_cond_${beta}_${size}_${tail}:
				%if is_tail:
					cmp Y, N
				%else:
					cmp Y, N_ROUND_DOWN
				%endif
				blt .L${is_parallel}_loop_${beta}_${size}_${tail}

			%endfor // End tail / notail loop
			b .L${is_parallel}_cleanup
		%endfor
	%endfor

	.L${is_parallel}_cleanup:
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

	// Unreq everything here for the other dgemv variant(s)
	.unreq A_ADDR_0
	%for column in range(1, ORIGINAL_NUM_COLUMNS):
		.unreq A_ADDR_${column}
	%endfor

	// .req the various vector registers in use
	.unreq X_REG_V
	.unreq X_REG_Q
	.unreq X_REG_D
	%for column in range(ORIGINAL_NUM_COLUMNS):
		.unreq ACC_REG_${column}_V
		.unreq ACC_REG_${column}_D
	%endfor
	%for pl in range(NUM_PRELOAD):
		.unreq A_REG_${pl}_V
		.unreq A_REG_${pl}_Q
		.unreq A_REG_${pl}_D
	%endfor

	${epilogue(func_name)}
</%def>

${dgemv(is_parallel=False)}
${dgemv(is_parallel=True)}

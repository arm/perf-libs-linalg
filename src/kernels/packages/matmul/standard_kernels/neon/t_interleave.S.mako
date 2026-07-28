## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue, assert_fn"/>
<% import math %>

## # Preface
## There are a couple of tuneable parameters here:
## ## cntg_block
## The naive implementation of t_interleave would result in completely strided
## access over the src matrix, and completely contiguous access over the dst
## matrix. cntg_block is how many rows to n_interleave contiguously in the src
## matrix before striding to the next row - this results in a slightly strided
## access in dst, but may improve performance overall.
## The naive implementation described earlier would be cntg_block = 1
## Ex.
## cntg_block = 2, ir = 4
## This processes 8 items in a row from src, interleaving 2 rows (essentially
## just a copy) before striding.

src_cntg        .req x0
src_strd        .req x1
base_src_ptr    .req x2
orig_src_ld     .req x3
dst_cntg        .req x4
dst_strd        .req x5
dst_ptr_0       .req x6
orig_dst_ld     .req x7
strd_counter    .req x8
cntg_counter    .req x9
cntg_loop_limit .req x10
src_ld          .req x11
dst_ld          .req x12

// x13, x14 disallowed in Arm64EC
iscratch_0 .req x15
iscratch_1 .req x16

scratch_0_s .req s31
scratch_0_d .req d31
scratch_0_q .req q31
scratch_0_v .req v31

/**
 * Zero the buffer until reaching the given limit.
 * @func_name - The function name to use for label scoping
 * @t_bytes - The number of bytes in an element.
 * @t_reg - The register type to use, based on element size - 's', 'd', or 'q'
 * @ir - The number of rows being interleaved
 * @unrolls - The number of strided unrolls on the dst matrix to zero at once
 * @dst_ptr - The start pointer to the destination
 * @cntg_counter - The counter to use
 * @cntg_loop_limit - The limit of the cntg_counter
 *
 * \note
 * The distance between dst_ptr and the end of the buffer should be
 * (cntg_loop_limit - cntg_counter) * ir * t_bytes - in other words, the buffer
 * to zero should be a multiple of ir elements. This function isn't a general
 * 'zero' function - it is specifically meant to be embedded into the loops of
 * this function.
 *
 * \warning This will clobber the scratch_0_v register.
 */
<%def name="zero_until(func_name, t_bytes, t_reg, ir, unrolls, dst_ptr, counter, loop_limit)">
	// TODO optimize with SIMD instructions & unrolling
	eor scratch_0_v.16b, scratch_0_v.16b, scratch_0_v.16b
	cmp ${counter}, ${loop_limit}

	b .L${func_name}_zero_loop_tail_cond
	.L${func_name}_zero_loop_tail:

	sub ${counter}, ${counter}, #1
	cmp ${counter}, ${loop_limit}

	%for ii in range(unrolls):
		%for jj in range(ir):
			str scratch_0_${t_reg}, [${dst_ptr}_${ii}, #${jj * t_bytes}]
		%endfor
		add ${dst_ptr}_${ii}, ${dst_ptr}_${ii}, #${t_bytes * ir}
	%endfor

	.L${func_name}_zero_loop_tail_cond:
	bge .L${func_name}_zero_loop_tail
</%def>

%for t_str, t_reg, t_lsl, t_bytes in [("s", "s", 2, 4), ("d", "d", 3, 8)]:
%for ir in [2, 4, 6, 8, 20]:
%for is_zero_pad, pad_name in [ ( True, "zero_pad_"), (False, "") ]:

	<% func_name = "t_interleave_kernel_" + pad_name + t_str + str(ir) %>

	## The first number is the most important - the remaining numbers just say
	## what tail loops to generate when the matrix size isn't an exact multiple of
	## this number.
	## How much of the src matrix to loop through contiguously before advancing
	## to the next row. Uses additional x registers (to keep track of all the
	## pointers into dst), and incurs some small overhead (when incrementing
	## pointers / processing tails).
	<% cntg_block_progression = [1] %>

	## How much of the src matrix to unroll by in the strd direction. Uses an
	## additional x register per strd unroll, incurs similar overhead to cntg_block.
	<% strd_block_progression = [1] %>

	## Number of registers required to n_interleave one row - because an
	## n_interleave is just a copy, we call this a register block
	<% reg_per_block = int(math.ceil(float(ir * t_bytes) / 16.0)) %>

	## Size of block in bytes
	<% block_size = ir * t_bytes %>

	## Number of spare neon registers available
	<% available_neon_registers = 31 %>

	## A 'register block' is the block of registers required to copy one
	## n_interleave section - for ir = 4 and t_bytes = 4, we need to copy 16 bytes
	## from src to dst per row, so we only need 1 register for this (this means we
	## can afford ~16 reg blocks for unrolling!)
	## We might be able to spare less registers for unrolling if t_bytes and ir
	## are bigger, for example ir = 8, t_bytes = 16 (std::complex<double>) would
	## mean we would need 8 registers per n_interleave, leaving only enough
	## registers spared for a couple extra reg blocks when unrolling.
	<% num_reg_blocks = int(available_neon_registers / reg_per_block) %>

	// Req load registers
	%for ii in range(num_reg_blocks):
		%for jj in range(reg_per_block):
			copy_reg_${jj + reg_per_block*ii}_s .req s${jj + reg_per_block*ii}
			copy_reg_${jj + reg_per_block*ii}_d .req d${jj + reg_per_block*ii}
			copy_reg_${jj + reg_per_block*ii}_q .req q${jj + reg_per_block*ii}
			copy_reg_${jj + reg_per_block*ii}_v .req v${jj + reg_per_block*ii}
		%endfor
	%endfor

	## Create reg block lists in python, for use later when copying
	<%
	reg_blocks = []
	for ii in range(num_reg_blocks):
		block = []
		for jj in range(reg_per_block):
			if jj == reg_per_block - 1 and ir * t_bytes % 16 != 0:
				## Use a non-full-width register for the block 'tail'
				if ir * t_bytes % 16 == 8:
					block.append("copy_reg_" + str(jj + reg_per_block*ii) + "_d")
				elif ir * t_bytes % 16 == 4:
					block.append("copy_reg_" + str(jj + reg_per_block*ii) + "_s")
				else:
					assert False, "Unsupported ir * t_bytes combination: ir = " + str(ir) + ", t_bytes = " + t_bytes
			else:
				block.append("copy_reg_" + str(jj + reg_per_block*ii) + "_q")
		reg_blocks.append(block)
	%>

	// x18, x23, x24, x28 disallowed in Arm64EC
	// the first entry in x_regs is unused because x16 is used for iscratch_1 above
	<% x_regs = [ 16, 17, 19, 20, 21, 22, 25, 26, 27 ] %>
	// Req dst & src ptrs
	%for ii in range(1, max(cntg_block_progression)):
		dst_ptr_${ii} .req x${x_regs[ii]}
	%endfor
	%for ii in range(max(strd_block_progression)):
		src_ptr_${ii} .req x${x_regs[max(cntg_block_progression) + ii]}
	%endfor

	## generate the prologue for the target operating system
	${prologue(func_name)}

	// Setup the stack frame, save all registers
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

	// Setup dst_ptrs
	%for ii in range(1, max(cntg_block_progression)):
		mov iscratch_0, #${ii * t_bytes}
		madd dst_ptr_${ii}, iscratch_0, orig_dst_ld, dst_ptr_0
	%endfor
	// Setup src_ptrs
	%for ii in range(max(strd_block_progression)):
		mov iscratch_0, #${ii * t_bytes}
		madd src_ptr_${ii}, iscratch_0, orig_src_ld, base_src_ptr
	%endfor

	mov strd_counter, src_cntg

	<% orig_func_name_0 = func_name %>
	%for cntg_unrolls in cntg_block_progression:
		<% func_name = orig_func_name_0 + "_" + str(cntg_unrolls) %>

		// Setup dst_ld to jump from the end of one row to the start of
		// the next
		mov iscratch_0, #${ir}
		msub dst_ld, dst_cntg, iscratch_0, orig_dst_ld
		mov iscratch_0, #${(cntg_unrolls - 1)}
		madd dst_ld, orig_dst_ld, iscratch_0, dst_ld
		lsl dst_ld, dst_ld, #${t_lsl}

		/******************************************/
		/*          Strided (outer) loop          */
		b .L${func_name}_strd_loop_cond
		.L${func_name}_strd_loop:

		mov cntg_counter, dst_cntg

		sub cntg_loop_limit, dst_cntg, src_strd
		add cntg_loop_limit, cntg_loop_limit, #${max(strd_block_progression)}

		<% orig_func_name_1 = func_name %>
		%for strd_unrolls_n, strd_unrolls in enumerate(strd_block_progression):
			<% func_name = orig_func_name_1 + "_" + str(strd_unrolls) %>

			// Setup src_ld for this amount of src_ptrs
			mov iscratch_0, #${strd_unrolls * t_bytes}
			mul src_ld, orig_src_ld, iscratch_0


			/******************************************/
			/*         Contiguous (inner) loop        */
			b .L${func_name}_cntg_loop_cond
			.L${func_name}_cntg_loop:

			<% curr_block = 0 %>
			%for ii in range(strd_unrolls):
				%for jj in range(cntg_unrolls):
					## Load the reg block
					%for reg_n, reg in enumerate(reg_blocks[curr_block]):
						ldr ${reg}, [src_ptr_${ii}, #${reg_n * 16 + jj * block_size}]
					%endfor
					## Store the reg block
					%for reg_n, reg in enumerate(reg_blocks[curr_block]):
						str ${reg}, [dst_ptr_${jj}, #${reg_n * 16 + ii * block_size}]
					%endfor
					## Increment curr_block
					<% curr_block = (curr_block + 1) % len(reg_blocks) %>
				%endfor ## for jj in range(cntg_unrolls)
			%endfor ## for jj in range(strd_unrolls)

			// Increment ptrs
			%for ii in range(cntg_unrolls):
				add dst_ptr_${ii}, dst_ptr_${ii}, #${strd_unrolls * block_size}
			%endfor
			%for ii in range(strd_unrolls):
				add src_ptr_${ii}, src_ptr_${ii}, src_ld
			%endfor

			sub cntg_counter, cntg_counter, #${strd_unrolls}

			.L${func_name}_cntg_loop_cond:
			cmp cntg_counter, cntg_loop_limit
			bge .L${func_name}_cntg_loop
			/******************************************/

			%if strd_unrolls_n < len(strd_block_progression) - 1:
				// Modify cntg_loop_limit for the next unroll in the progression
				sub cntg_loop_limit, cntg_loop_limit, #${strd_unrolls - strd_block_progression[strd_unrolls_n + 1]}
			%endif

		%endfor ## for strd_unrolls in [...]
		<% func_name = orig_func_name_1 %>

		sub strd_counter, strd_counter, #${cntg_unrolls * ir}

		// Zero the tail (up to dst_cntg), loop limit = 1 because we started at
		// dst_cntg, not dst_cntg - 1
		${zero_until(func_name, t_bytes, t_reg, ir, cntg_unrolls, 'dst_ptr', 'cntg_counter', '#1')}

		// Setup dst_ptrs for next loop
		%for ii in range(cntg_unrolls):
			add dst_ptr_${ii}, dst_ptr_${ii}, dst_ld
		%endfor
		// Setup src ptrs for next loop
		add base_src_ptr, base_src_ptr, #${block_size * cntg_unrolls}
		%for ii in range(strd_unrolls):
			mov iscratch_0, #${ii}
			madd src_ptr_${ii}, iscratch_0, orig_src_ld, base_src_ptr
		%endfor

		.L${func_name}_strd_loop_cond:
		cmp strd_counter, #${cntg_unrolls * ir}
		bge .L${func_name}_strd_loop
		/******************************************/

	%endfor ## for cntg_unrolls in [...]
	<% func_name = orig_func_name_0 %>

	// Now we need to clean up the bottom, and insert the 0 interleaved rows
	// e.g. for the following matrix and ir of 2:
	// a b c
	// d e f
	// g e h
	// i j k
	// This loop is producing the interleaved row resulting in:
	// c 0 f 0 h 0 k 0
	// We're going to generate multiple loops, for each of the possible
	// remainders, this way we can branch once to the correct loop without
	// suffering the performance impact of branching in the middle of loops.
	cmp strd_counter, #0
	beq .L${func_name}_skip_final_tail_loop
	mov iscratch_0, #${t_bytes}
	mul src_ld, orig_src_ld, iscratch_0
	%for rows_left in range(1, ir):
		cmp strd_counter, #${rows_left}
		beq .L${func_name}_cntg_final_tail_loop_rows_left_${rows_left}
	%endfor
	// This path should be unreachable; use sigtrap for debugging rather than looping.
	${assert_fn()}
	<% orig_func_name_0 = func_name %>
	%for rows_left in range(1, ir):
		<% func_name = orig_func_name_0 + "_" + str(rows_left) %>
		.L${orig_func_name_0}_cntg_final_tail_loop_rows_left_${rows_left}:
		mov cntg_counter, dst_cntg
		sub cntg_loop_limit, dst_cntg, src_strd
		eor scratch_0_v.16b, scratch_0_v.16b, scratch_0_v.16b

		/***************************************/
		/*    Final contiguous cleanup loop    */
		b .L${func_name}_cntg_final_tail_loop_cond
		.L${func_name}_cntg_final_tail_loop:

		%for ii in range(rows_left):
			ldr copy_reg_0_${t_reg}, [src_ptr_0, #${ii * t_bytes}]
			str copy_reg_0_${t_reg}, [dst_ptr_0, #${ii * t_bytes}]
		%endfor

		%if is_zero_pad:
			%for ii in range(rows_left, ir):
				// Zero remaining in this interleaved row
				str scratch_0_${t_reg}, [dst_ptr_0, #${(ii) * t_bytes}]
			%endfor
		%endif

		// Increment pointers
		add dst_ptr_0, dst_ptr_0, #${t_bytes * ir}
		add src_ptr_0, src_ptr_0, src_ld

		sub cntg_counter, cntg_counter, #${1}

		.L${func_name}_cntg_final_tail_loop_cond:
		cmp cntg_counter, cntg_loop_limit
		bgt .L${func_name}_cntg_final_tail_loop
		/***************************************/
		b .L${orig_func_name_0}_end_final_tail_loop
	%endfor ## for rows_left ...
	<% func_name = orig_func_name_0 %>

	.L${func_name}_end_final_tail_loop:
	##// Zero the tail of the cleanup loop (the 0 interleaved row)
	${zero_until(func_name + "_0_interleaved_row", t_bytes, t_reg, ir, 1, 'dst_ptr', 'cntg_counter', '#1')}
	add dst_ptr_0, dst_ptr_0, dst_ld

	// Set proper starting point for strd counter:
	// for (strd_counter = iround(src_cntg, ir);
	//      strd_counter < dst_strd;
	//      strd_counter += ir) {
	//     ...
	// }
	// We don't have to do a full iround since we KNOW src_cntg is not an exact
	// multiple of ir* - we can just do ((strd_counter / ir) * ir) + ir
	//
	// * - If we were on an exact multiple of ir, we'd have branched to
	// .L..._skip_final_tail_loop.
	mov iscratch_0, #${ir}
	udiv strd_counter, src_cntg, iscratch_0
	madd strd_counter, strd_counter, iscratch_0, iscratch_0
	mov iscratch_0, #${ir}
	msub dst_ld, dst_cntg, iscratch_0, orig_dst_ld
	lsl dst_ld, dst_ld, #${t_lsl}
	b .L${func_name}_zero_rows_loop_cond // Start the loop

	.L${func_name}_skip_final_tail_loop:
	// If we skipped the final tail loop, we know src_cntg is on an exact
	// multiple of ir, so iround(src_cntg, ir) is just src_cntg.
	mov strd_counter, src_cntg

	mov iscratch_0, #${ir}
	msub dst_ld, dst_cntg, iscratch_0, orig_dst_ld
	lsl dst_ld, dst_ld, #${t_lsl}

	/********************************************/
	/*          Zero the remaining rows         */
	b .L${func_name}_zero_rows_loop_cond // Start the loop
	.L${func_name}_zero_rows_loop:

	mov cntg_counter, dst_cntg
	${zero_until(func_name + "_zero_rows", t_bytes, t_reg, ir, 1, 'dst_ptr', 'cntg_counter', '#1')}
	add dst_ptr_0, dst_ptr_0, dst_ld

	add strd_counter, strd_counter, #${ir}

	.L${func_name}_zero_rows_loop_cond:
	cmp strd_counter, dst_strd
	blt .L${func_name}_zero_rows_loop
	/********************************************/

	.L${func_name}_cleanup:
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

	// Unreq registers
	%for ii in range(1, max(cntg_block_progression)):
		.unreq dst_ptr_${ii}
	%endfor
	%for ii in range(1, max(strd_block_progression)):
		.unreq src_ptr_${ii}
	%endfor
	%for ii in range(num_reg_blocks):
		%for jj in range(reg_per_block):
			.unreq copy_reg_${jj + reg_per_block*ii}_s
			.unreq copy_reg_${jj + reg_per_block*ii}_d
			.unreq copy_reg_${jj + reg_per_block*ii}_q
			.unreq copy_reg_${jj + reg_per_block*ii}_v
		%endfor
	%endfor

	## generate the epilogue for the target operating system
	${epilogue(func_name)}
%endfor
%endfor
%endfor

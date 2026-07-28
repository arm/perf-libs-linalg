## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue, assert_fn"/>

src_cntg .req x0
src_strd .req x1
src_ptr_0 .req x2
src_ld .req x3
dst_cntg .req x4
dst_strd .req x5
dst_ptr .req x6
dst_ld .req x7

iscratch_0 .req x9
iscratch_1 .req x10

scratch_0_s .req s0
scratch_0_d .req d0
scratch_0_q .req q0
scratch_0_v .req v0

scratch_1_s .req s1
scratch_1_d .req d1
scratch_1_q .req q1
scratch_1_v .req v1

strd_counter .req x11
cntg_counter .req x12

// x13, x14 disallowed in Arm64EC
cntg_loop_limit .req x15

/**
 * Zero the buffer until reaching the given limit.
 * @func_name - The function name to use for label scoping
 * @t_bytes - The number of bytes in an element.
 * @t_reg - The register type to use, based on element size - 's', 'd', or 'q'
 * @ir - The number of rows being interleaved
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
<%def name="zero_until(func_name, t_bytes, t_reg, ir, dst_ptr, counter, loop_limit)">
	// TODO optimize with SIMD instructions & unrolling
	eor scratch_0_v.16b, scratch_0_v.16b, scratch_0_v.16b
	b .L${func_name}_zero_loop_tail_cond
	.L${func_name}_zero_loop_tail:

	%for jj in range(ir):
		str scratch_0_${t_reg}, [${dst_ptr}, #${jj * t_bytes}]
	%endfor

	// Increment pointers
	add ${dst_ptr}, ${dst_ptr}, #${t_bytes * ir}

	add ${counter}, ${counter}, #1

	.L${func_name}_zero_loop_tail_cond:
	cmp ${counter}, ${loop_limit}
	blt .L${func_name}_zero_loop_tail
</%def>

%for t_str, t_reg, t_lsl, t_bytes in [("s", "s", 2, 4), ("d", "d", 3, 8), ("c", "d", 3, 8), ("z", "q", 4, 16)]:
%for ir in [2, 4, 6, 8]:
	## When t_str == 's' we don't have enough immediate offset addressing space
	<% unroll_progression = [2, 1] %>
	## Prefetch distance (in elements). Prefetches occur once for each src ptr,
	## once at the start of each unroll. Prefetches will NOT occur if not
	## unrolling - the overhead isn't worth it.
	<% prefetch_distance = 16 %>

	## Req a reg for each copy, so we can avoid register
	## dependencies when copying elements
	%for ii in range(0, ir):
		ld_reg_${ii}_s .req s${2 + ii * 2}
		ld_reg_${ii}_d .req d${2 + ii * 2}
		ld_reg_${ii}_q .req q${2 + ii * 2}
		ld_reg_${ii}_v .req v${2 + ii * 2}
		st_reg_${ii}_s .req s${3 + ii * 2}
		st_reg_${ii}_d .req d${3 + ii * 2}
		st_reg_${ii}_q .req q${3 + ii * 2}
		st_reg_${ii}_v .req v${3 + ii * 2}
	%endfor

	## Avoid using x13, x14, x18, x23, x24, x28 and x29 when allocating general-purpose registers
	## so that our code remains portable, including on Arm64EC. We also disallow x30,
	## because it can break backtraces and perf profiles.
	<% gpreg = 16 %>
	// Req matrix pointer registers
	%for ii in range(1, ir):
		<% gpreg += 1 %>
		%if gpreg == 18:
			<% gpreg += 1 %>
		%endif
		%if gpreg == 23:
			<% gpreg += 2 %>
		%endif
		<% assert gpreg <= 27, f"gpreg exceeded 27 (got {gpreg})" %>
		src_ptr_${ii} .req x${gpreg}
	%endfor

	<% func_name = "n_interleave_kernel_" + t_str + str(ir) %>
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

	// Setup pointers into src
	%for ii in range(1, ir):
		mov iscratch_0, ${ii * t_bytes}
		madd src_ptr_${ii}, src_ld, iscratch_0, src_ptr_0
	%endfor

	// Setup dst_ld and src_ld to jump from the end of one row to the start of
	// the next
	mov iscratch_0, #${ir}
	msub dst_ld, iscratch_0, dst_cntg, dst_ld
	// src_ld is more complex, since we need to jump to the next row, then skip
	// ir - 1 rows
	mov iscratch_1, src_ld
	sub src_ld, src_ld, src_cntg
	mov iscratch_0, #${ir - 1}
	madd src_ld, iscratch_1, iscratch_0, src_ld

	mov strd_counter, src_strd

	/***************************************/
	/*        Strided (outer) loop         */
	b .L${func_name}_strd_loop_cond
	.L${func_name}_strd_loop:

	mov cntg_counter, #0

	<% orig_func_name = func_name %>
	%for unrolls in unroll_progression:
		<% func_name = orig_func_name + "_" + str(unrolls) %>

		// Figure out the unroll limit
		mov iscratch_0, #${unrolls * int(16 / t_bytes)}
		udiv cntg_loop_limit, src_cntg, iscratch_0
		mul cntg_loop_limit, cntg_loop_limit, iscratch_0

		/***************************************/
		/*       Contiguous (inner) loop       */
		cmp cntg_counter, cntg_loop_limit
		b .L${func_name}_cntg_loop_cond
		.L${func_name}_cntg_loop:

		// Preload all registers
		%for ii in range(ir):
			ldr ld_reg_${ii}_q, [src_ptr_${ii}]
		%endfor


		// Prefetch (only if we're in the main loop, don't bother in the unroll tails)
		%if unrolls > 1 and unrolls == max(unroll_progression):
			%for jj in range(ir):
				prfm pldl1strm, [src_ptr_${jj}, #256]
			%endfor
		%endif

		// 'Sneak' in the increment on cntg counter whilst we wait for preloads and
		// prefetched to resolve
		add cntg_counter, cntg_counter, #${unrolls * int(16 / t_bytes)}
		cmp cntg_counter, cntg_loop_limit

		%for ii in range(unrolls):
			// HOT LOOP
			%if t_str == 'z':
				## If we're double complex, do no zipping, just store & load
				%for jj in range(ir):
					%if unrolls == 1:
						add src_ptr_${jj}, src_ptr_${jj}, #${unrolls * 16}
					%endif
					str ld_reg_${jj}_q, [dst_ptr, #${ii * ir * 16 + jj * 16}]
					%if ii < unrolls - 1:
						ldr ld_reg_${jj}_q, [src_ptr_${jj}, #${((ii+1) % unrolls) * 16}]
					%endif
					%if ii == unrolls - 2 and unrolls != 1:
						add src_ptr_${jj}, src_ptr_${jj}, #${unrolls * 16}
					%endif
				%endfor
			%elif t_str == 's':
				## If we're single precision, we need more complex zipping.
				## Unfortunately there isn't much of a pattern for different values
				## of IR, so they're being hardcoded here.
				<% assert ir in [2, 4, 6, 8], "Not implemented sn_interleave for ir == " + str(ir) %>
				%if unrolls == 1:
					%for jj in range(ir):
						add src_ptr_${jj}, src_ptr_${jj}, #${unrolls * 16}
					%endfor
				%endif
				%if ir == 2:
					zip1 st_reg_0_v.4s, ld_reg_0_v.4s, ld_reg_1_v.4s
					zip2 st_reg_1_v.4s, ld_reg_0_v.4s, ld_reg_1_v.4s
					%if ii < unrolls - 1:
						ldr ld_reg_0_q, [src_ptr_0, #${((ii+1) % unrolls) * 16}]
						ldr ld_reg_1_q, [src_ptr_1, #${((ii+1) % unrolls) * 16}]
					%endif
					str st_reg_0_q, [dst_ptr, #${ii * 8 * t_bytes}]
					str st_reg_1_q, [dst_ptr, #${ii * 8 * t_bytes + 16}]
				%elif ir == 4:
					// Zip everything into scratch / st registers to begin with
					zip1 scratch_0_v.4s,  ld_reg_0_v.4s, ld_reg_1_v.4s
					zip1 st_reg_1_v.4s,    ld_reg_2_v.4s, ld_reg_3_v.4s
					zip2 scratch_1_v.4s,  ld_reg_0_v.4s, ld_reg_1_v.4s
					zip2 st_reg_3_v.4s,    ld_reg_2_v.4s, ld_reg_3_v.4s

					// Preload ASAP (now that we don't need the ld registers)
					%if ii < unrolls - 1:
						%for jj in range(ir):
							ldr ld_reg_${jj}_q, [src_ptr_${jj}, #${((ii+1) % unrolls) * 16}]
						%endfor
					%endif

					// Zip together (double-wise) the registers loaded earlier
					zip1 st_reg_0_v.2d,    scratch_0_v.2d, st_reg_1_v.2d
					zip2 st_reg_1_v.2d,    scratch_0_v.2d, st_reg_1_v.2d
					zip1 st_reg_2_v.2d,    scratch_1_v.2d, st_reg_3_v.2d
					zip2 st_reg_3_v.2d,    scratch_1_v.2d, st_reg_3_v.2d

					// Store
					%for jj in range(ir):
						str st_reg_${jj}_q, [dst_ptr, #${ii * ir * 16 + 16 * jj}]
					%endfor
				%elif ir == 6:
					// Here, X, Y, Z, W, U, V are the st registers in order of storing
					// A, B, C, D, E, F are the ld registers in order of loading
					// Comments after lines indicate that line calculates the
					// final value of the given register - for example:
					// `zip1 something something something // X`
					// The // X comment would indicate this instruction
					// calculates the final value of X, or st_reg_0.
					// This is important, because sometimes the st registers
					// are used as temporaries. In this case, a comment won't
					// be present.
					// Zipping is split into 2 halves, the first half computing
					// X, Y, Z, the second half computing W, U, V. Both halves
					// are basically identical, the only difference being a
					// couple of zip2s instead of zip1s.

					// First half
					zip1 scratch_0_v.4s, ld_reg_0_v.4s, ld_reg_1_v.4s
					zip1 st_reg_2_v.4s,  ld_reg_2_v.4s, ld_reg_3_v.4s
					zip1 scratch_1_v.4s, ld_reg_4_v.4s, ld_reg_5_v.4s
					zip1 st_reg_0_v.2d,  scratch_0_v.2d, st_reg_2_v.2d // X
					ext scratch_0_v.16b, scratch_0_v.16b, scratch_1_v.16b, #8
					zip2 st_reg_2_v.2d,  st_reg_2_v.2d, scratch_1_v.2d // Z
					ext st_reg_1_v.16b, scratch_0_v.16b, scratch_0_v.16b, #8 // Y

					// Second half
					zip2 scratch_0_v.4s, ld_reg_0_v.4s, ld_reg_1_v.4s
					%if ii < unrolls - 1:
						ldr ld_reg_0_q, [src_ptr_0, #${((ii+1) % unrolls) * 16}] // Preload
						ldr ld_reg_1_q, [src_ptr_1, #${((ii+1) % unrolls) * 16}]
					%endif
					zip2 st_reg_5_v.4s,  ld_reg_2_v.4s, ld_reg_3_v.4s
					%if ii < unrolls - 1:
						ldr ld_reg_2_q, [src_ptr_2, #${((ii+1) % unrolls) * 16}] // Preload
						ldr ld_reg_3_q, [src_ptr_3, #${((ii+1) % unrolls) * 16}]
					%endif
					zip2 scratch_1_v.4s, ld_reg_4_v.4s, ld_reg_5_v.4s
					%if ii < unrolls - 1:
						ldr ld_reg_4_q, [src_ptr_4, #${((ii+1) % unrolls) * 16}] // Preload
						ldr ld_reg_5_q, [src_ptr_5, #${((ii+1) % unrolls) * 16}]
					%endif
					zip1 st_reg_3_v.2d,  scratch_0_v.2d, st_reg_5_v.2d // W
					ext scratch_0_v.16b, scratch_0_v.16b, scratch_1_v.16b, #8
					zip2 st_reg_5_v.2d,  st_reg_5_v.2d, scratch_1_v.2d // V
					ext st_reg_4_v.16b, scratch_0_v.16b, scratch_0_v.16b, #8 // U

					// Store
					%for jj in range(ir):
						str st_reg_${jj}_q, [dst_ptr, #${ii * ir * 16 + 16 * jj}]
					%endfor
				%elif ir == 8:
					// See ir == 6 preface, same system applies here
					// First half
					zip1 st_reg_1_v.4s, ld_reg_0_v.4s, ld_reg_1_v.4s
					zip1 st_reg_3_v.4s, ld_reg_2_v.4s, ld_reg_3_v.4s
					zip1 scratch_0_v.4s, ld_reg_4_v.4s, ld_reg_5_v.4s
					zip1 scratch_1_v.4s, ld_reg_6_v.4s, ld_reg_7_v.4s
					zip1 st_reg_0_v.2d, st_reg_1_v.2d, st_reg_3_v.2d // X
					zip2 st_reg_2_v.2d, st_reg_1_v.2d, st_reg_3_v.2d // Z
					zip1 st_reg_1_v.2d, scratch_0_v.2d, scratch_1_v.2d // Y
					zip2 st_reg_3_v.2d, scratch_0_v.2d, scratch_1_v.2d // W

					// Second half
					zip2 st_reg_5_v.4s, ld_reg_0_v.4s, ld_reg_1_v.4s
					zip2 st_reg_7_v.4s, ld_reg_2_v.4s, ld_reg_3_v.4s
					zip2 scratch_0_v.4s, ld_reg_4_v.4s, ld_reg_5_v.4s
					zip2 scratch_1_v.4s, ld_reg_6_v.4s, ld_reg_7_v.4s
					%if ii < unrolls - 1:
						%for jj in range(ir): ## Finished with the ld registers, preload ASAP
							ldr ld_reg_${jj}_q, [src_ptr_${jj}, #${((ii+1) % unrolls) * 16}]
						%endfor
					%endif
					zip1 st_reg_4_v.2d, st_reg_5_v.2d, st_reg_7_v.2d // U
					zip2 st_reg_6_v.2d, st_reg_5_v.2d, st_reg_7_v.2d // V
					zip1 st_reg_5_v.2d, scratch_0_v.2d, scratch_1_v.2d // S
					zip2 st_reg_7_v.2d, scratch_0_v.2d, scratch_1_v.2d // T

					// Store
					%for jj in range(ir):
						str st_reg_${jj}_q, [dst_ptr, #${ii * ir * 16 + 16 * jj}]
					%endfor
				%endif ## if ir == *

				%if ii == unrolls - 2 and unrolls != 1:
					%for jj in range(ir):
						add src_ptr_${jj}, src_ptr_${jj}, #${unrolls * 16}
					%endfor
				%endif
			%else: ## t_str is 'c' or 'd'
				%for jj in range(int(ir/2)):
					%if unrolls == 1:
						add src_ptr_${jj*2}, src_ptr_${jj*2}, #${unrolls * 16}
						add src_ptr_${jj*2+1}, src_ptr_${jj*2+1}, #${unrolls * 16}
					%endif
					zip1 st_reg_${jj}_v.2d, ld_reg_${jj*2}_v.2d, ld_reg_${jj*2+1}_v.2d
					zip2 st_reg_${jj+int(ir/2)}_v.2d, ld_reg_${jj*2}_v.2d, ld_reg_${jj*2+1}_v.2d
					// Preload next unroll
					%if ii < unrolls - 1:
						ldr ld_reg_${jj*2}_q, [src_ptr_${jj*2}, #${((ii+1) % unrolls) * 16}]
						ldr ld_reg_${jj*2+1}_q, [src_ptr_${jj*2+1}, #${((ii+1) % unrolls) * 16}]
					%endif
					%if ii == unrolls - 2 and unrolls != 1:
						add src_ptr_${jj*2}, src_ptr_${jj*2}, #${unrolls * 16}
						add src_ptr_${jj*2+1}, src_ptr_${jj*2+1}, #${unrolls * 16}
					%endif
					// Store into dst
					str st_reg_${jj}_q,      [dst_ptr, #${ii * ir * 16 + jj * 16}]
					str st_reg_${jj+int(ir/2)}_q, [dst_ptr, #${ii * ir * 16 + (jj+int(ir/2)) * 16}]
				%endfor
			%endif ## if t_str == *
		%endfor ## for ii in range(unrolls)

		// Increment pointers
		add dst_ptr, dst_ptr, #${unrolls * ir * 16}

		.L${func_name}_cntg_loop_cond:
		blt .L${func_name}_cntg_loop
		/***************************************/
	%endfor ## for unrolls in ...
	<% func_name = orig_func_name %>

	/***************************************/
	/*    Contiguous (inner) tail loop     */
	mov cntg_loop_limit, src_cntg
	cmp cntg_counter, cntg_loop_limit
	b .L${func_name}_cntg_loop_tail_cond
	.L${func_name}_cntg_loop_tail:

		%for ii in range(ir):
			ldr ld_reg_${ii}_${t_reg}, [src_ptr_${ii}]
			add src_ptr_${ii}, src_ptr_${ii}, #${t_bytes}
		%endfor


		// 'Sneak' in the increment on cntg counter whilst we wait for preloads and
		// prefetched to resolve
		add cntg_counter, cntg_counter, #1
		cmp cntg_counter, cntg_loop_limit

		%for ii in range(ir):
			str ld_reg_${ii}_${t_reg}, [dst_ptr, #${ii * t_bytes}]
		%endfor

		// Increment pointers
		add dst_ptr, dst_ptr, #${t_bytes * ir}

	.L${func_name}_cntg_loop_tail_cond:
	blt .L${func_name}_cntg_loop_tail
	/***************************************/

	// Zero the tail (up to dst_cntg)
	${zero_until(func_name, t_bytes, t_reg, ir, 'dst_ptr', 'cntg_counter', 'dst_cntg')}

	// Increment pointers
	%for ii in range(ir):
		add src_ptr_${ii}, src_ptr_${ii}, src_ld, lsl #${t_lsl}
	%endfor
	add dst_ptr, dst_ptr, dst_ld, lsl #${t_lsl}

	sub strd_counter, strd_counter, ${ir}

	.L${func_name}_strd_loop_cond:
	cmp strd_counter, ${ir}
	bge .L${func_name}_strd_loop
	/***************************************/

	// Now we need to clean up the bottom, and insert the 0 interleaved rows
	// e.g. for the following matrix and ir of 2:
	// a b c d
	// e f g h
	// i j k l
	// This loop is producing the interleaved row resulting in:
	// i 0 j 0 k 0 l 0
	// We're going to generate multiple loops, for each of the possible
	// remainders, this way we can branch once to the correct loop without
	// suffering the performance impact of branching in the middle of loops.
	mov cntg_counter, #0
	cmp strd_counter, #0
	beq .L${func_name}_skip_final_tail_loop
	%for rows_left in range(1, ir):
		cmp strd_counter, #${rows_left}
		beq .L${func_name}_cntg_final_tail_loop_rows_left_${rows_left}
	%endfor
	// This path should be unreachable; use sigtrap for debugging rather than looping.
	${assert_fn()}
	<% orig_func_name = func_name %>
	%for rows_left in range(1, ir):
		.L${orig_func_name}_cntg_final_tail_loop_rows_left_${rows_left}:
		<% func_name = orig_func_name + "_" + str(rows_left) %>
		<% orig_func_name_1 = func_name %>
		%for unrolls in unroll_progression:
			<% func_name = orig_func_name_1 + "_" + str(unrolls) %>

			// Figure out the unroll limit
			mov iscratch_0, #${unrolls}
			udiv cntg_loop_limit, src_cntg, iscratch_0
			mul cntg_loop_limit, cntg_loop_limit, iscratch_0

			eor scratch_1_v.16b, scratch_1_v.16b, scratch_1_v.16b

			/***************************************/
			/*    Final contiguous cleanup loop    */
			b .L${func_name}_cntg_final_tail_loop_cond
			.L${func_name}_cntg_final_tail_loop:

			%for ii in range(unrolls):
				%for jj in range(rows_left):
					ldr scratch_0_${t_reg}, [src_ptr_${jj}, #${ii * t_bytes}]
					str scratch_0_${t_reg}, [dst_ptr, #${(ii * ir + jj) * t_bytes}]
				%endfor
				%for jj in range(rows_left, ir):
					// Zero remaining in this interleaved row
					str scratch_1_${t_reg}, [dst_ptr, #${(ii * ir + jj) * t_bytes}]
				%endfor
			%endfor

			// Increment pointers
			add dst_ptr, dst_ptr, #${t_bytes * unrolls * ir}
			%for jj in range(rows_left):
				add src_ptr_${jj}, src_ptr_${jj}, #${unrolls * t_bytes}
			%endfor

			add cntg_counter, cntg_counter, #${unrolls}

			.L${func_name}_cntg_final_tail_loop_cond:
			cmp cntg_counter, cntg_loop_limit
			blt .L${func_name}_cntg_final_tail_loop
			/***************************************/
		%endfor // for unrolls ...
		<% func_name = orig_func_name_1 %>
		b .L${orig_func_name}_end_final_tail_loop
	%endfor // for rows_left ...
	<% func_name = orig_func_name %>

	.L${func_name}_end_final_tail_loop:
	##// Zero the tail of the cleanup loop (the 0 interleaved row)
	${zero_until(func_name + "_0_interleaved_row", t_bytes, t_reg, ir, 'dst_ptr', 'cntg_counter', 'dst_cntg')}
	add dst_ptr, dst_ptr, dst_ld, lsl #${t_lsl}

	// Set proper starting point for strd counter:
	// for (strd_counter = iround(src_strd, ir);
	//      strd_counter < dst_strd;
	//      strd_counter += ir) {
	//     ...
	// }
	// We don't have to do a full iround since we KNOW src_strd is not an exact
	// multiple of ir* - we can just do ((strd_counter / ir) * ir) + ir
	//
	// * - If we were on an exact multiple of ir, we'd have branched to
	// .L..._skip_final_tail_loop.
	mov iscratch_0, #${ir}
	udiv strd_counter, src_strd, iscratch_0
	madd strd_counter, strd_counter, iscratch_0, iscratch_0
	b .L${func_name}_zero_rows_loop_cond // Start the loop

	.L${func_name}_skip_final_tail_loop:
	// If we skipped the final tail loop, we know src_strd is on an exact
	// multiple of ir, so iround(src_strd, ir) is just src_strd.
	mov strd_counter, src_strd
	mov iscratch_0, #${ir} // Just move for the loop, iscratch_0 should be ir

	/********************************************/
	/*          Zero the remaining rows         */
	b .L${func_name}_zero_rows_loop_cond // Start the loop
	.L${func_name}_zero_rows_loop:

	mov cntg_counter, #0
	${zero_until(func_name + "_zero_rows", t_bytes, t_reg, ir, 'dst_ptr', 'cntg_counter', 'dst_cntg')}
	add dst_ptr, dst_ptr, dst_ld, lsl #${t_lsl}

	add strd_counter, strd_counter, iscratch_0

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

	// Unreq matrix pointer registers
	%for ii in range(1, unrolls * ir):
		.unreq src_ptr_${ii}
	%endfor
	// Unreq copy registers
	%for ii in range(ir):
		.unreq ld_reg_${ii}_s
		.unreq ld_reg_${ii}_d
		.unreq ld_reg_${ii}_q
		.unreq ld_reg_${ii}_v
	%endfor

	${epilogue(func_name)}

%endfor ## for ir ...
%endfor ## for t_str, t_reg ...

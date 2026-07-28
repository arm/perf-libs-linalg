## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue" />
<%
def vreg(name, num):
	return f"""{name}v .req v{num}
{name}h .req h{num}
{name}s .req s{num}
{name}d .req d{num}
{name}q .req q{num} """

def load_lane_increment(options, dst, lane, src):
	return options["load_lane_increment"].format(dst, lane, src)

def multiply_accumulate(options, vd, vn, vm, lane):
	return options["multiply_accumulate"].format(vd, vn, vm, lane)

def multiply_accumulate_elem(options, vd, vn, vm):
	return options["multiply_accumulate_elem"].format(vd, vn, vm)

def load_vector(options, dst, src, offset=0):
	offset_bytes = offset * options["vector_load_bytes"]
	src_str = f"{src}, {offset_bytes}" if offset_bytes != 0 else src
	return options["load_vector"].format(dst, src_str)

def load_element(options, dst, src, offset=0):
	offset_bytes = offset * options["bytes"]
	src_str = f"{src}, {offset_bytes}" if offset_bytes != 0 else src
	return options["load_element"].format(dst, src_str)

def store_vector(options, dst, src, offset=0):
	offset_bytes = offset * options["vector_load_bytes"]
	src_str = f"{src}, {offset_bytes}" if offset_bytes != 0 else src
	return options["store_vector"].format(dst, src_str)

def store_element(options, dst, src, offset=0):
	offset_bytes = offset * options["bytes"]
	src_str = f"{src}, {offset_bytes}" if offset_bytes != 0 else src
	return options["store_element"].format(dst, src_str)

def adjust_scalar(options, register):
	return options["adjust_scalar"].format(register)

r32_options = {
	"shift":               2,
	"bytes":               4,
	"vector_lanes":        4,
	"vector_load_bytes":   4 * 4,
	"load_vector":         "ldr {0}q, [ {1} ]",
	"load_element":        "ldr {0}s, [ {1} ]",
	"load_lane_increment": "ld1 {{ {0}v.s }}[{1}], [ {2} ], #4",
	"store_vector":        "str {0}q, [ {1} ]",
	"store_element":       "str {0}s, [ {1} ]"
}

bf16_to_r32_options = {
	"shift":               1,
	"bytes":               2,
	"vector_lanes":        4,
	"vector_load_bytes":   4 * 2,
	"load_vector":         "ldr {0}d, [ {1} ]\nshll {0}v.4s, {0}v.4h, #16",
	"load_element":        "ldr {0}h, [ {1} ]\n\tshll {0}v.4s, {0}v.4h, #16",
	"load_lane_increment": "mov {0}v.h[{1}*2], wzr\n\tld1 {{ {0}v.h }}[{1}*2+1], [ {2} ], #2",

	"store_vector":        "uzp2 {0}v.8h, {0}v.8h, {0}v.8h\n\tstr {0}d, [ {1} ]",
	"store_element":       "st1 {{ {0}v.h }}[1], [ {1} ]"
}

#
functions={
	"sbgemv_n_fp32_fmla": {
		"in" : bf16_to_r32_options,
		"out": r32_options,
		"multiply_accumulate" :      "fmla {0}v.4s, {1}v.4s, {2}v.s[{3}]",
		"multiply_accumulate_elem" : "fmadd {0}s, {1}s, {2}s, {0}s",
		"adjust_scalar":             ""
	},
	"bgemv_n_fp32_fmla": {
		"in" : bf16_to_r32_options,
		"out": bf16_to_r32_options,
		"multiply_accumulate" :      "fmla {0}v.4s, {1}v.4s, {2}v.s[{3}]",
		"multiply_accumulate_elem" : "fmadd {0}s, {1}s, {2}s, {0}s",
		"adjust_scalar":             "shll {0}v.4s, {0}v.4h, #16"
	}
}
%>


a_strd .req x0
a_cntg .req x1
a_ptr  .req x2
lda    .req x3
b_ptr  .req x4
incx   .req x5
c_ptr  .req x6
incy   .req x7

as_idx      .req x8
as_idx_next .req x9

ac_idx      .req x10
ac_idx_next .req x11

a_ptr_0     .req x12
a_ptr_1     .req x13
a_ptr_2     .req x14
a_ptr_3     .req x15
c_ptr_0     .req x16

${vreg("alpha_", 0)}
${vreg("beta_", 1)}

${vreg("c0", 2)}
${vreg("c00", 3)}

${vreg("a00", 4)}
${vreg("a01", 5)}
${vreg("a02", 6)}
${vreg("a03", 7)}

${vreg("b0", 8)}


%for func_name, opts in functions.items():
${prologue(func_name)}
	stp     x29, x30, [sp, #-16]!
	mov     x29, sp

	stp     x19, x20, [sp, #-16]!
	stp     x21, x22, [sp, #-16]!
	stp     x23, x24, [sp, #-16]!
	stp     x25, x26, [sp, #-16]!
	stp     x27, x28, [sp, #-16]!

	stp     d8,  d9,  [sp, #-16]!
	stp     d10, d11, [sp, #-16]!
	stp     d12, d13, [sp, #-16]!
	stp     d14, d15, [sp, #-16]!

.L${func_name}_ac_loop_preamble:
	${adjust_scalar(opts, "alpha_")}
	${adjust_scalar(opts, "beta_")}
	//branch to test condition, retruns to correct start
	mov ac_idx, 0
	b .L${func_name}_ac_loop4_end

.L${func_name}_ac_loop4_start:
	mov ac_idx, ac_idx_next

.L${func_name}_ac_loop4_calculate_matrix_offsets:
	mov c_ptr_0, c_ptr

	//offsets of each col
	mov a_ptr_0, a_ptr
	add a_ptr_1, a_ptr,   lda, lsl ${opts["in"]["shift"]}
	add a_ptr_2, a_ptr,   lda, lsl ${opts["in"]["shift"]+1}
	add a_ptr_3, a_ptr_1, lda, lsl ${opts["in"]["shift"]+1}
	add a_ptr,   a_ptr,   lda, lsl ${opts["in"]["shift"]+2}

.L${func_name}_ac_loop4_b_load_data:
	${load_vector(opts["in"], "b0", "b_ptr", 0)}
	add b_ptr, b_ptr, ${opts["in"]["vector_load_bytes"]*1}

.L${func_name}_ac_loop4_as_loop_preamble:
	mov as_idx,      0
	b .L${func_name}_ac_loop4_as_loop4_end

.L${func_name}_ac_loop4_as_loop4_start:
	mov as_idx, as_idx_next

.L${func_name}_ac_loop4_as_loop4_zero_accumulators:
	movi c00v.16b, #0

.L${func_name}_ac_loop4_as_loop4_c_load_data:
	${load_vector(opts["out"], "c0", "c_ptr_0")}

.L${func_name}_ac_loop4_as_loop4_a_load_data:
	${load_vector(opts["in"], "a00", "a_ptr_0", 0)}
	add a_ptr_0, a_ptr_0, ${opts["in"]["vector_load_bytes"]*1}

	${load_vector(opts["in"], "a01", "a_ptr_1", 0)}
	add a_ptr_1, a_ptr_1, ${opts["in"]["vector_load_bytes"]*1}

	${load_vector(opts["in"], "a02", "a_ptr_2", 0)}
	add a_ptr_2, a_ptr_2, ${opts["in"]["vector_load_bytes"]*1}

	${load_vector(opts["in"], "a03", "a_ptr_3", 0)}
	add a_ptr_3, a_ptr_3, ${opts["in"]["vector_load_bytes"]*1}

.L${func_name}_ac_loop4_as_loop4_compute:
	${multiply_accumulate(opts, "c00", "a00", "b0", 0)}
	${multiply_accumulate(opts, "c00", "a01", "b0", 1)}
	${multiply_accumulate(opts, "c00", "a02", "b0", 2)}
	${multiply_accumulate(opts, "c00", "a03", "b0", 3)}

.L${func_name}_ac_loop4_as_loop4_shuffle_reduce:
	fmla c0v.4s, c00v.4s, alpha_v.s[0]

.L${func_name}_ac_loop4_as_loop4_c_store_data:
	${store_vector(opts["out"], "c0", "c_ptr_0")}
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]*1}

.L${func_name}_ac_loop4_as_loop4_end:
	add as_idx_next, as_idx, 4
	cmp as_idx_next, a_strd
	bls .L${func_name}_ac_loop4_as_loop4_start

	//Branch to condition of next section
	b .L${func_name}_ac_loop4_as_loop1_end

.L${func_name}_ac_loop4_as_loop1_start:
	mov as_idx, as_idx_next

.L${func_name}_ac_loop4_as_loop1_zero_accumulators:
	movi c00v.16b, #0

.L${func_name}_ac_loop4_as_loop1_c_load_data:
	${load_element(opts["out"], "c0", "c_ptr_0")}

.L${func_name}_ac_loop4_as_loop1_a_load_data:
	${load_lane_increment(opts["in"], "a00", 0, "a_ptr_0")}
	${load_lane_increment(opts["in"], "a01", 0, "a_ptr_1")}
	${load_lane_increment(opts["in"], "a02", 0, "a_ptr_2")}
	${load_lane_increment(opts["in"], "a03", 0, "a_ptr_3")}

.L${func_name}_ac_loop4_as_loop1_compute:
	//either do this as s-reg insturctions or shuffle into one fmla
	${multiply_accumulate(opts, "c00", "a00", "b0", 0)}
	${multiply_accumulate(opts, "c00", "a01", "b0", 1)}
	${multiply_accumulate(opts, "c00", "a02", "b0", 2)}
	${multiply_accumulate(opts, "c00", "a03", "b0", 3)}

.L${func_name}_ac_loop4_as_loop1_shuffle_reduce:
	fmadd c0s, c00s, alpha_s, c0s

.L${func_name}_ac_loop4_as_loop1_c_store_data:
	${store_element(opts["out"], "c0", "c_ptr_0")}
	add c_ptr_0, c_ptr_0, ${opts["out"]["bytes"]}

.L${func_name}_ac_loop4_as_loop1_end:
	add as_idx_next, as_idx, 1
	cmp as_idx_next, a_strd
	bls .L${func_name}_ac_loop4_as_loop1_start


.L${func_name}_ac_loop4_end:
	add ac_idx_next, ac_idx, 4
	cmp ac_idx_next, a_cntg
	bls .L${func_name}_ac_loop4_start

	b .L${func_name}_ac_loop1_end

.L${func_name}_ac_loop1_start:
	mov ac_idx, ac_idx_next

.L${func_name}_ac_loop1_calculate_matrix_offsets:
	mov c_ptr_0, c_ptr

	//offsets of each col
	mov a_ptr_0, a_ptr
	add a_ptr, a_ptr, lda, lsl 1

.L${func_name}_ac_loop1_b_load_data:
	${load_lane_increment(opts["in"], "b0", 0, "b_ptr")}

.L${func_name}_ac_loop1_as_loop_preamble:
	mov as_idx,      0
	b .L${func_name}_ac_loop1_as_loop4_end


.L${func_name}_ac_loop1_as_loop4_start:
	mov as_idx, as_idx_next

.L${func_name}_ac_loop1_as_loop4_zero_accumulators:
	movi c00v.16b, #0

.L${func_name}_ac_loop1_as_loop4_c_load_data:
	${load_vector(opts["out"], "c0", "c_ptr_0")}

.L${func_name}_ac_loop1_as_loop4_a_load_data:
	${load_vector(opts["in"], "a00", "a_ptr_0", 0)}
	add a_ptr_0, a_ptr_0, ${opts["in"]["vector_load_bytes"]}

.L${func_name}_ac_loop1_as_loop4_compute:
	${multiply_accumulate(opts, "c00", "a00", "b0", 0)}

.L${func_name}_ac_loop1_as_loop4_shuffle_reduce:
	fmla c0v.4s, c00v.4s, alpha_v.s[0]

.L${func_name}_ac_loop1_as_loop4_c_store_data:
	${store_vector(opts["out"], "c0", "c_ptr_0")}
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]*1}

.L${func_name}_ac_loop1_as_loop4_end:
	add as_idx_next, as_idx, 4
	cmp as_idx_next, a_strd
	bls .L${func_name}_ac_loop1_as_loop4_start

	//Branch to condition of next section
	b .L${func_name}_ac_loop1_as_loop1_end

.L${func_name}_ac_loop1_as_loop1_start:
	mov as_idx, as_idx_next

.L${func_name}_ac_loop1_as_loop1_zero_accumulators:
	movi c00v.16b, #0

.L${func_name}_ac_loop1_as_loop1_c_load_data:
	${load_element(opts["out"], "c0", "c_ptr_0")}

.L${func_name}_ac_loop1_as_loop1_a_load_data:
	${load_lane_increment(opts["in"], "a00", 0, "a_ptr_0")}

.L${func_name}_ac_loop1_as_loop1_compute:
	${multiply_accumulate_elem(opts, "c00", "a00", "b0")}

.L${func_name}_ac_loop1_as_loop1_shuffle_reduce:
	fmadd c0s, c00s, alpha_s, c0s

.L${func_name}_ac_loop1_as_loop1_c_store_data:
	${store_element(opts["out"], "c0", "c_ptr_0")}
	add c_ptr_0, c_ptr_0, ${opts["out"]["bytes"]}

.L${func_name}_ac_loop1_as_loop1_end:
	add as_idx_next, as_idx, 1
	cmp as_idx_next, a_strd
	bls .L${func_name}_ac_loop1_as_loop1_start

.L${func_name}_ac_loop1_end:
	add ac_idx_next, ac_idx, 1
	cmp ac_idx_next, a_cntg
	bls .L${func_name}_ac_loop1_start

.L${func_name}_epilogue:
	ldp d14, d15, [sp], #16
	ldp d12, d13, [sp], #16
	ldp d10, d11, [sp], #16
	ldp d8,  d9,  [sp], #16

	ldp x27, x28, [sp], #16
	ldp x25, x26, [sp], #16
	ldp x23, x24, [sp], #16
	ldp x21, x22, [sp], #16
	ldp x19, x20, [sp], #16

	ldp x29, x30, [sp], #16

	ret
${epilogue(func_name)}
% endfor

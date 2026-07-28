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

def multiply_accumulate(options, vd, vn, vm):
	return options["multiply_accumulate"].format(vd, vn, vm)

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

bf16_options = {
	"shift":               1,
	"bytes":               2,
	"vector_lanes":        8,
	"vector_load_bytes":   8 * 2,
	"load_vector":         "ldr {0}q, [ {1} ]",
	"load_element":        "ldr {0}h, [ {1} ]",
	"load_lane_increment": "ld1 {{ {0}v.h }}[{1}], [ {2} ], #2",
	"store_vector":        "str {0}q, [ {1} ]",
	"store_element":       "str {0}h, [ {1} ]"
}

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
	"load_lane_increment": "ld1 {{ {0}v.h }}[{1}*2+1], [ {2} ], #2",
	"store_vector":        "uzp2 {0}v.8h, {0}v.8h, {0}v.8h\n\tstr {0}d, [ {1} ]",
	"store_element":       "st1 {{ {0}v.h }}[1], [ {1} ]"
}

functions={
	"bgemv_t_bfdot_4x24": {
		"in": bf16_options,
		"out": bf16_to_r32_options,
		"multiply_accumulate": "bfdot {0}v.4s, {1}v.8h, {2}v.8h",
		"adjust_scalar":       "shll {0}v.4s, {0}v.4h, #16"
	},
	"bgemv_t_fp32_fmla_4x12": {
		"in": bf16_to_r32_options,
		"out": bf16_to_r32_options,
		"multiply_accumulate": "fmla {0}v.4s, {1}v.4s, {2}v.4s",
		"adjust_scalar":       "shll {0}v.4s, {0}v.4h, #16"
	},
	"sbgemv_t_bfdot_4x24": {
		"in": bf16_options,
		"out": r32_options,
		"multiply_accumulate": "bfdot {0}v.4s, {1}v.8h, {2}v.8h",
		"adjust_scalar":       ""
	},
	"sbgemv_t_fp32_fmla_4x12": {
		"in": bf16_to_r32_options,
		"out": r32_options,
		"multiply_accumulate" : "fmla {0}v.4s, {1}v.4s, {2}v.4s",
		"adjust_scalar":       ""
	},
	"sgemv_t_fmla_4x12": {
		"in":  r32_options,
		"out": r32_options,
		"multiply_accumulate" : "fmla {0}v.4s, {1}v.4s, {2}v.4s",
		"adjust_scalar":       ""
	}
}
%>

a_cntg .req x0
a_strd .req x1
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
b_ptr_0     .req x16

${vreg("alpha_", 0)}
${vreg("beta_", 1)}

${vreg("c00", 2)}
${vreg("c01", 3)}
${vreg("c02", 4)}
${vreg("c10", 5)}
${vreg("c11", 6)}
${vreg("c12", 7)}
${vreg("c20", 8)}
${vreg("c21", 9)}
${vreg("c22", 10)}
${vreg("c30", 11)}
${vreg("c31", 12)}
${vreg("c32", 13)}

${vreg("a00", 14)}
${vreg("a01", 15)}
${vreg("a02", 16)}
${vreg("a10", 17)}
${vreg("a11", 18)}
${vreg("a12", 19)}
${vreg("a20", 20)}
${vreg("a21", 21)}
${vreg("a22", 22)}
${vreg("a30", 23)}
${vreg("a31", 24)}
${vreg("a32", 25)}

${vreg("b0", 26)}
${vreg("b1", 27)}
${vreg("b2", 28)}

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

.L${func_name}_as_loop_preamble:
	${adjust_scalar(opts, "alpha_")}
	${adjust_scalar(opts, "beta_")}

	//branch to test condition, retruns to correct start
	mov as_idx, 0
	b .L${func_name}_as_loop4_end

.L${func_name}_as_loop4_start:
	mov as_idx, as_idx_next

.L${func_name}_as_loop4_calculate_matrix_offsets:
	mov b_ptr_0, b_ptr

	mov a_ptr_0, a_ptr
	add a_ptr_1, a_ptr,   lda, lsl  ${opts["in"]["shift"]}
	add a_ptr_2, a_ptr,   lda, lsl  ${opts["in"]["shift"]+1}
	add a_ptr_3, a_ptr_1, lda, lsl  ${opts["in"]["shift"]+1}
	add a_ptr,   a_ptr,   lda, lsl  ${opts["in"]["shift"]+2}

.L${func_name}_as_loop4_zero_accumulators:
	movi c00v.16b, #0
	movi c01v.16b, #0
	movi c02v.16b, #0
	movi c10v.16b, #0
	movi c11v.16b, #0
	movi c12v.16b, #0
	movi c20v.16b, #0
	movi c21v.16b, #0
	movi c22v.16b, #0
	movi c30v.16b, #0
	movi c31v.16b, #0
	movi c32v.16b, #0

.L${func_name}_as_loop4_ac_loop_preamble:
	mov ac_idx, 0

	add ac_idx_next, ac_idx, ${opts["in"]["vector_lanes"]*3}
	cmp ac_idx_next, a_cntg

	bls .L${func_name}_as_loop4_ac_loop24_start

	add ac_idx_next, ac_idx, ${opts["in"]["vector_lanes"]*1}
	cmp ac_idx_next, a_cntg

	bls .L${func_name}_as_loop4_ac_loop8_start

	mov ac_idx_next, ac_idx

	b .L${func_name}_as_loop4_ac_loop1_start

.L${func_name}_as_loop4_ac_loop24_start:
	mov ac_idx, ac_idx_next

.L${func_name}_as_loop4_ac_loop24_b_load_data:
	${load_vector(opts["in"], "b0", "b_ptr_0", 0)}
	${load_vector(opts["in"], "b1", "b_ptr_0", 1)}
	${load_vector(opts["in"], "b2", "b_ptr_0", 2)}
	add b_ptr_0, b_ptr_0, ${opts["in"]["vector_load_bytes"]*3}

.L${func_name}_as_loop4_ac_loop24_a_load_data:
	${load_vector(opts["in"], "a00", "a_ptr_0", 0)}
	${load_vector(opts["in"], "a01", "a_ptr_0", 1)}
	${load_vector(opts["in"], "a02", "a_ptr_0", 2)}
	add a_ptr_0, a_ptr_0, ${opts["in"]["vector_load_bytes"]*3}

	${load_vector(opts["in"], "a10", "a_ptr_1", 0)}
	${load_vector(opts["in"], "a11", "a_ptr_1", 1)}
	${load_vector(opts["in"], "a12", "a_ptr_1", 2)}
	add a_ptr_1, a_ptr_1, ${opts["in"]["vector_load_bytes"]*3}

	${load_vector(opts["in"], "a20", "a_ptr_2", 0)}
	${load_vector(opts["in"], "a21", "a_ptr_2", 1)}
	${load_vector(opts["in"], "a22", "a_ptr_2", 2)}
	add a_ptr_2, a_ptr_2, ${opts["in"]["vector_load_bytes"]*3}

	${load_vector(opts["in"], "a30", "a_ptr_3", 0)}
	${load_vector(opts["in"], "a31", "a_ptr_3", 1)}
	${load_vector(opts["in"], "a32", "a_ptr_3", 2)}
	add a_ptr_3, a_ptr_3, ${opts["in"]["vector_load_bytes"]*3}

.L${func_name}_as_loop4_ac_loop24_compute:
	${multiply_accumulate(opts, "c00", "b0", "a00")}
	${multiply_accumulate(opts, "c01", "b1", "a01")}
	${multiply_accumulate(opts, "c02", "b2", "a02")}

	${multiply_accumulate(opts, "c10", "b0", "a10")}
	${multiply_accumulate(opts, "c11", "b1", "a11")}
	${multiply_accumulate(opts, "c12", "b2", "a12")}

	${multiply_accumulate(opts, "c20", "b0", "a20")}
	${multiply_accumulate(opts, "c21", "b1", "a21")}
	${multiply_accumulate(opts, "c22", "b2", "a22")}

	${multiply_accumulate(opts, "c30", "b0", "a30")}
	${multiply_accumulate(opts, "c31", "b1", "a31")}
	${multiply_accumulate(opts, "c32", "b2", "a32")}

.L${func_name}_as_loop4_ac_loop24_end:
	add ac_idx_next, ac_idx, ${opts["in"]["vector_lanes"]*3}
	cmp ac_idx_next, a_cntg
	bls .L${func_name}_as_loop4_ac_loop24_start

.L${func_name}_as_loop4_ac_loop24_reduce:
	fadd c00v.4s, c00v.4s, c01v.4s
	fadd c10v.4s, c10v.4s, c11v.4s
	fadd c20v.4s, c20v.4s, c21v.4s
	fadd c30v.4s, c30v.4s, c31v.4s

	fadd c00v.4s, c00v.4s, c02v.4s
	fadd c10v.4s, c10v.4s, c12v.4s
	fadd c20v.4s, c20v.4s, c22v.4s
	fadd c30v.4s, c30v.4s, c32v.4s

	b .L${func_name}_as_loop4_ac_loop8_end

.L${func_name}_as_loop4_ac_loop8_start:
	mov ac_idx, ac_idx_next

.L${func_name}_as_loop4_ac_loop8_b_load_data:
	${load_vector(opts["in"], "b0", "b_ptr_0", 0)}
	add b_ptr_0, b_ptr_0, ${opts["in"]["vector_load_bytes"]*1}

.L${func_name}_as_loop4_ac_loop8_a_load_data:
	${load_vector(opts["in"], "a00", "a_ptr_0", 0)}
	add a_ptr_0, a_ptr_0, ${opts["in"]["vector_load_bytes"]*1}

	${load_vector(opts["in"], "a10", "a_ptr_1", 0)}
	add a_ptr_1, a_ptr_1, ${opts["in"]["vector_load_bytes"]*1}

	${load_vector(opts["in"], "a20", "a_ptr_2", 0)}
	add a_ptr_2, a_ptr_2, ${opts["in"]["vector_load_bytes"]*1}

	${load_vector(opts["in"], "a30", "a_ptr_3", 0)}
	add a_ptr_3, a_ptr_3, ${opts["in"]["vector_load_bytes"]*1}

.L${func_name}_as_loop4_ac_loop8_compute:
	${multiply_accumulate(opts, "c00", "b0", "a00")}
	${multiply_accumulate(opts, "c10", "b0", "a10")}
	${multiply_accumulate(opts, "c20", "b0", "a20")}
	${multiply_accumulate(opts, "c30", "b0", "a30")}

.L${func_name}_as_loop4_ac_loop8_end:
	add ac_idx_next, ac_idx, ${opts["in"]["vector_lanes"]*1};
	cmp ac_idx_next, a_cntg
	bls .L${func_name}_as_loop4_ac_loop8_start

	b .L${func_name}_as_loop4_ac_loop1_end

.L${func_name}_as_loop4_ac_loop1_start:
	cmp ac_idx, a_cntg
	bhs .L${func_name}_as_loop4_ac_loop1_end

	movi b0v.8h,  #0
	movi a00v.8h, #0
	movi a10v.8h, #0
	movi a20v.8h, #0
	movi a30v.8h, #0

	add ac_idx, ac_idx, 1
	cmp ac_idx, a_cntg
	${load_lane_increment(opts["in"], "b0",  0, "b_ptr_0")}
	${load_lane_increment(opts["in"], "a00", 0, "a_ptr_0")}
	${load_lane_increment(opts["in"], "a10", 0, "a_ptr_1")}
	${load_lane_increment(opts["in"], "a20", 0, "a_ptr_2")}
	${load_lane_increment(opts["in"], "a30", 0, "a_ptr_3")}
	bhs .L${func_name}_as_loop4_ac_loop1_compute

	add ac_idx, ac_idx, 1
	cmp ac_idx, a_cntg
	${load_lane_increment(opts["in"], "b0",  1, "b_ptr_0")}
	${load_lane_increment(opts["in"], "a00", 1, "a_ptr_0")}
	${load_lane_increment(opts["in"], "a10", 1, "a_ptr_1")}
	${load_lane_increment(opts["in"], "a20", 1, "a_ptr_2")}
	${load_lane_increment(opts["in"], "a30", 1, "a_ptr_3")}
	bhs .L${func_name}_as_loop4_ac_loop1_compute

	add ac_idx, ac_idx, 1
	cmp ac_idx, a_cntg
	${load_lane_increment(opts["in"], "b0",  2, "b_ptr_0")}
	${load_lane_increment(opts["in"], "a00", 2, "a_ptr_0")}
	${load_lane_increment(opts["in"], "a10", 2, "a_ptr_1")}
	${load_lane_increment(opts["in"], "a20", 2, "a_ptr_2")}
	${load_lane_increment(opts["in"], "a30", 2, "a_ptr_3")}
	bhs .L${func_name}_as_loop4_ac_loop1_compute

	add ac_idx, ac_idx, 1
	cmp ac_idx, a_cntg
	${load_lane_increment(opts["in"], "b0",  3, "b_ptr_0")}
	${load_lane_increment(opts["in"], "a00", 3, "a_ptr_0")}
	${load_lane_increment(opts["in"], "a10", 3, "a_ptr_1")}
	${load_lane_increment(opts["in"], "a20", 3, "a_ptr_2")}
	${load_lane_increment(opts["in"], "a30", 3, "a_ptr_3")}
	bhs .L${func_name}_as_loop4_ac_loop1_compute

%if opts["in"]["vector_lanes"] > 4:
	add ac_idx, ac_idx, 1
	cmp ac_idx, a_cntg
	${load_lane_increment(opts["in"], "b0",  4, "b_ptr_0")}
	${load_lane_increment(opts["in"], "a00", 4, "a_ptr_0")}
	${load_lane_increment(opts["in"], "a10", 4, "a_ptr_1")}
	${load_lane_increment(opts["in"], "a20", 4, "a_ptr_2")}
	${load_lane_increment(opts["in"], "a30", 4, "a_ptr_3")}
	bhs .L${func_name}_as_loop4_ac_loop1_compute

	add ac_idx, ac_idx, 1
	cmp ac_idx, a_cntg
	${load_lane_increment(opts["in"], "b0",  5, "b_ptr_0")}
	${load_lane_increment(opts["in"], "a00", 5, "a_ptr_0")}
	${load_lane_increment(opts["in"], "a10", 5, "a_ptr_1")}
	${load_lane_increment(opts["in"], "a20", 5, "a_ptr_2")}
	${load_lane_increment(opts["in"], "a30", 5, "a_ptr_3")}
	bhs .L${func_name}_as_loop4_ac_loop1_compute

	add ac_idx, ac_idx, 1
	cmp ac_idx, a_cntg
	${load_lane_increment(opts["in"], "b0",  6, "b_ptr_0")}
	${load_lane_increment(opts["in"], "a00", 6, "a_ptr_0")}
	${load_lane_increment(opts["in"], "a10", 6, "a_ptr_1")}
	${load_lane_increment(opts["in"], "a20", 6, "a_ptr_2")}
	${load_lane_increment(opts["in"], "a30", 6, "a_ptr_3")}
	bhs .L${func_name}_as_loop4_ac_loop1_compute

	add ac_idx, ac_idx, 1
	${load_lane_increment(opts["in"], "b0",  7, "b_ptr_0")}
	${load_lane_increment(opts["in"], "a00", 7, "a_ptr_0")}
	${load_lane_increment(opts["in"], "a10", 7, "a_ptr_1")}
	${load_lane_increment(opts["in"], "a20", 7, "a_ptr_2")}
	${load_lane_increment(opts["in"], "a30", 7, "a_ptr_3")}
%endif

.L${func_name}_as_loop4_ac_loop1_compute:
	${multiply_accumulate(opts, "c00", "b0", "a00")}
	${multiply_accumulate(opts, "c10", "b0", "a10")}
	${multiply_accumulate(opts, "c20", "b0", "a20")}
	${multiply_accumulate(opts, "c30", "b0", "a30")}

.L${func_name}_as_loop4_ac_loop1_end:
	add ac_idx_next, ac_idx, 1
	cmp ac_idx_next, a_cntg
	bls .L${func_name}_as_loop4_ac_loop1_start

.L${func_name}_as_loop4_ac_loop1_reduce:
	${load_vector(opts["out"], "a00", "c_ptr")}

	faddp c00v.4s, c00v.4s, c00v.4s
	faddp c10v.4s, c10v.4s, c10v.4s
	faddp c20v.4s, c20v.4s, c20v.4s
	faddp c30v.4s, c30v.4s, c30v.4s

	faddp c00s, c00v.2s
	faddp c10s, c10v.2s
	faddp c20s, c20v.2s
	faddp c30s, c30v.2s

	mov c00v.s[1], c10v.s[0]
	mov c00v.s[2], c20v.s[0]
	mov c00v.s[3], c30v.s[0]

	fmul c00v.4s, c00v.4s, alpha_v.s[0]
	fmla c00v.4s, a00v.4s, beta_v.s[0]

	${store_vector(opts["out"], "c00", "c_ptr")}
	add c_ptr, c_ptr, ${opts["out"]["vector_load_bytes"]}

.L${func_name}_as_loop4_end:
	add as_idx_next, as_idx, 4
	cmp as_idx_next, a_strd
	bls .L${func_name}_as_loop4_start

	b .L${func_name}_as_loop1_end


.L${func_name}_as_loop1_start:
	mov as_idx, as_idx_next

.L${func_name}_as_loop1_calculate_matrix_offsets:
	mov b_ptr_0, b_ptr

	mov a_ptr_0, a_ptr
	add a_ptr,   a_ptr,   lda, lsl ${opts["in"]["shift"]}

.L${func_name}_as_loop1_zero_accumulators:
	movi c00v.16b, #0
	movi c01v.16b, #0
	movi c02v.16b, #0

.L${func_name}_as_loop1_ac_loop_preamble:
	mov ac_idx, 0

	add ac_idx_next, ac_idx, ${opts["in"]["vector_lanes"]*3}
	cmp ac_idx_next, a_cntg

	bls .L${func_name}_as_loop1_ac_loop24_start

	add ac_idx_next, ac_idx, ${opts["in"]["vector_lanes"]*1}
	cmp ac_idx_next, a_cntg

	bls .L${func_name}_as_loop1_ac_loop8_start

	mov ac_idx_next, ac_idx

	b .L${func_name}_as_loop1_ac_loop1_start

.L${func_name}_as_loop1_ac_loop24_start:
	mov ac_idx, ac_idx_next

.L${func_name}_as_loop1_ac_loop24_b_load_data:
	${load_vector(opts["in"], "b0", "b_ptr_0", 0)}
	${load_vector(opts["in"], "b1", "b_ptr_0", 1)}
	${load_vector(opts["in"], "b2", "b_ptr_0", 2)}
	add b_ptr_0, b_ptr_0, ${opts["in"]["vector_load_bytes"]*3}

.L${func_name}_as_loop1_ac_loop24_a_load_data:
	${load_vector(opts["in"], "a00", "a_ptr_0", 0)}
	${load_vector(opts["in"], "a01", "a_ptr_0", 1)}
	${load_vector(opts["in"], "a02", "a_ptr_0", 2)}
	add a_ptr_0, a_ptr_0, ${opts["in"]["vector_load_bytes"]*3}

.L${func_name}_as_loop1_ac_loop24_compute:
	${multiply_accumulate(opts, "c00", "b0", "a00")}
	${multiply_accumulate(opts, "c01", "b1", "a01")}
	${multiply_accumulate(opts, "c02", "b2", "a02")}

.L${func_name}_as_loop1_ac_loop24_end:
	add ac_idx_next, ac_idx, ${opts["in"]["vector_lanes"]*3}
	cmp ac_idx_next, a_cntg
	bls .L${func_name}_as_loop1_ac_loop24_start

.L${func_name}_as_loop1_ac_loop24_reduce:
	fadd c00v.4s, c00v.4s, c01v.4s

	fadd c00v.4s, c00v.4s, c02v.4s

	b .L${func_name}_as_loop1_ac_loop8_end

.L${func_name}_as_loop1_ac_loop8_start:
	mov ac_idx, ac_idx_next

.L${func_name}_as_loop1_ac_loop8_b_load_data:
	${load_vector(opts["in"], "b0", "b_ptr_0", 0)}
	add b_ptr_0, b_ptr_0, ${opts["in"]["vector_load_bytes"]*1}

.L${func_name}_as_loop1_ac_loop8_a_load_data:
	${load_vector(opts["in"], "a00", "a_ptr_0", 0)}
	add a_ptr_0, a_ptr_0, ${opts["in"]["vector_load_bytes"]*1}

.L${func_name}_as_loop1_ac_loop8_compute:
	${multiply_accumulate(opts, "c00", "b0", "a00")}

.L${func_name}_as_loop1_ac_loop8_end:
	add ac_idx_next, ac_idx, ${opts["in"]["vector_lanes"]*1}
	cmp ac_idx_next, a_cntg
	bls .L${func_name}_as_loop1_ac_loop8_start

	b .L${func_name}_as_loop1_ac_loop1_end

.L${func_name}_as_loop1_ac_loop1_start:
	cmp ac_idx, a_cntg
	bhs .L${func_name}_as_loop1_ac_loop1_end

	movi b0v.8h,  #0
	movi a00v.8h, #0

	add ac_idx, ac_idx, 1
	cmp ac_idx, a_cntg
	${load_lane_increment(opts["in"], "b0",  0, "b_ptr_0")}
	${load_lane_increment(opts["in"], "a00", 0, "a_ptr_0")}
	bhs .L${func_name}_as_loop1_ac_loop1_compute

	add ac_idx, ac_idx, 1
	cmp ac_idx, a_cntg
	${load_lane_increment(opts["in"], "b0",  1, "b_ptr_0")}
	${load_lane_increment(opts["in"], "a00", 1, "a_ptr_0")}
	bhs .L${func_name}_as_loop1_ac_loop1_compute

	add ac_idx, ac_idx, 1
	cmp ac_idx, a_cntg
	${load_lane_increment(opts["in"], "b0",  2, "b_ptr_0")}
	${load_lane_increment(opts["in"], "a00", 2, "a_ptr_0")}
	bhs .L${func_name}_as_loop1_ac_loop1_compute

	add ac_idx, ac_idx, 1
	cmp ac_idx, a_cntg
	${load_lane_increment(opts["in"], "b0",  3, "b_ptr_0")}
	${load_lane_increment(opts["in"], "a00", 3, "a_ptr_0")}
	bhs .L${func_name}_as_loop1_ac_loop1_compute

%if opts["in"]["vector_lanes"] > 4:
	add ac_idx, ac_idx, 1
	cmp ac_idx, a_cntg
	${load_lane_increment(opts["in"], "b0",  4, "b_ptr_0")}
	${load_lane_increment(opts["in"], "a00", 4, "a_ptr_0")}
	bhs .L${func_name}_as_loop1_ac_loop1_compute

	add ac_idx, ac_idx, 1
	cmp ac_idx, a_cntg
	${load_lane_increment(opts["in"], "b0",  5, "b_ptr_0")}
	${load_lane_increment(opts["in"], "a00", 5, "a_ptr_0")}
	bhs .L${func_name}_as_loop1_ac_loop1_compute

	add ac_idx, ac_idx, 1
	cmp ac_idx, a_cntg
	${load_lane_increment(opts["in"], "b0",  6, "b_ptr_0")}
	${load_lane_increment(opts["in"], "a00", 6, "a_ptr_0")}
	bhs .L${func_name}_as_loop1_ac_loop1_compute

	add ac_idx, ac_idx, 1
	${load_lane_increment(opts["in"], "b0",  7, "b_ptr_0")}
	${load_lane_increment(opts["in"], "a00", 7, "a_ptr_0")}
%endif

.L${func_name}_as_loop1_ac_loop1_compute:
	${multiply_accumulate(opts, "c00", "b0", "a00")}

.L${func_name}_as_loop1_ac_loop1_end:
	add ac_idx_next, ac_idx, 1
	cmp ac_idx_next, a_cntg
	bls .L${func_name}_as_loop1_ac_loop1_start

.L${func_name}_as_loop1_ac_loop1_reduce:
	${load_element(opts["out"], "a00", "c_ptr")}

	faddp c00v.4s, c00v.4s, c00v.4s

	faddp c00s, c00v.2s

	fmul c00s, c00s, alpha_s
	fmadd c00s, a00s, beta_s, c00s

	${store_element(opts["out"], "c00", "c_ptr")}
	add c_ptr, c_ptr, ${opts["out"]["bytes"]}

.L${func_name}_as_loop1_end:
	add as_idx_next, as_idx, 1
	cmp as_idx_next, a_strd
	bls .L${func_name}_as_loop1_start


.L${func_name}_epilogue:
	ldp     d14, d15, [sp], #16
	ldp     d12, d13, [sp], #16
	ldp     d10, d11, [sp], #16
	ldp     d8,  d9,  [sp], #16

	ldp     x27, x28, [sp], #16
	ldp     x25, x26, [sp], #16
	ldp     x23, x24, [sp], #16
	ldp     x21, x22, [sp], #16
	ldp     x19, x20, [sp], #16

	ldp     x29, x30, [sp], #16

	ret
${epilogue(func_name)}
% endfor


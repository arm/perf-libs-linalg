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
	"sbgemv_n_bfmlal_16x8": {
		"in": bf16_options,
		"out": r32_options,
		"adjust_scalar": ""
	},
	"bgemv_n_bfmlal_16x8": {
		"in": bf16_options,
		"out": bf16_to_r32_options,
		"adjust_scalar": "shll {0}v.4s, {0}v.4h, #16"
	}
}

%>

m      .req x0
n      .req x1
a_ptr  .req x2
lda    .req x3
b_ptr  .req x4
incx   .req x5
c_ptr  .req x6
incy   .req x7

a_ptr_0 .req x8
a_ptr_1 .req x9
a_ptr_2 .req x10
a_ptr_3 .req x11
a_ptr_4 .req x12
a_ptr_5 .req x13
a_ptr_6 .req x14
a_ptr_7 .req x15

c_ptr_0    .req x16

m_idx      .req x17
// x18 may be in use for the Platform Register
m_idx_next .req x19

n_idx      .req x20
n_idx_next .req x21


${vreg("alpha_", 0)}
${vreg("beta_", 1)}

//for bfmlalt
${vreg("c_a_0_hi_", 2)}
${vreg("c_b_0_hi_", 3)}
${vreg("c_a_1_hi_", 4)}
${vreg("c_b_1_hi_", 5)}

${vreg("c_a_0_lo_", 6)}
${vreg("c_b_0_lo_", 7)}
${vreg("c_a_1_lo_", 8)}
${vreg("c_b_1_lo_", 9)}

${vreg("b0", 1)}

${vreg("a00", 10)}
${vreg("a01", 11)}
${vreg("a02", 12)}
${vreg("a03", 13)}
${vreg("a04", 14)}
${vreg("a05", 15)}
${vreg("a06", 16)}
${vreg("a07", 17)}

${vreg("a10", 18)}
${vreg("a11", 19)}
${vreg("a12", 20)}
${vreg("a13", 21)}
${vreg("a14", 22)}
${vreg("a15", 23)}
${vreg("a16", 24)}
${vreg("a17", 25)}

${vreg("c0", 26)}
${vreg("c1", 27)}
${vreg("c2", 28)}
${vreg("c3", 29)}

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

.L${func_name}_n_loop_preamble:
	${adjust_scalar(opts, "alpha_")}
	${adjust_scalar(opts, "beta_")}

	//branch to test condition, retruns to correct start
	mov n_idx, 0
	b .L${func_name}_n_loop8_end

.L${func_name}_n_loop8_start:
	mov n_idx, n_idx_next

.L${func_name}_n_loop8_calculate_matrix_offsets:
	mov c_ptr_0, c_ptr

	mov a_ptr_0, a_ptr
	add a_ptr_2, a_ptr,   lda, lsl 2
	add a_ptr_1, a_ptr,   lda, lsl 1
	add a_ptr_4, a_ptr_2, lda, lsl 2
	add a_ptr_3, a_ptr_2, lda, lsl 1
	add a_ptr_6, a_ptr_4, lda, lsl 2
	add a_ptr_5, a_ptr_4, lda, lsl 1
	add a_ptr_7, a_ptr_6, lda, lsl 1

	add a_ptr,   a_ptr,   lda, lsl 4

.L${func_name}_n_loop8_b_load_data:
	ldr b0q,  [ b_ptr ]
	add b_ptr, b_ptr, 16


.L${func_name}_n_loop8_m_loop_preamble:
	mov m_idx,      0
	b .L${func_name}_n_loop8_m_loop16_end

.L${func_name}_n_loop8_m_loop16_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop8_m_loop16_zero_accumulators:
	movi c_a_0_hi_v.16b, #0
	movi c_b_0_hi_v.16b, #0
	movi c_a_1_hi_v.16b, #0
	movi c_b_1_hi_v.16b, #0
	movi c_a_0_lo_v.16b, #0
	movi c_b_0_lo_v.16b, #0
	movi c_a_1_lo_v.16b, #0
	movi c_b_1_lo_v.16b, #0

.L${func_name}_n_loop8_m_loop16_c_load_data:
	${load_vector(opts["out"], "c0", "c_ptr_0", 0)}
	${load_vector(opts["out"], "c1", "c_ptr_0", 1)}
	${load_vector(opts["out"], "c2", "c_ptr_0", 2)}
	${load_vector(opts["out"], "c3", "c_ptr_0", 3)}

.L${func_name}_n_loop8_m_loop16_a_load_data:
	ldr a00q, [ a_ptr_0 ]
	ldr a10q, [ a_ptr_0, 16 ]
	add a_ptr_0, a_ptr_0, 32

	ldr a01q, [ a_ptr_1 ]
	ldr a11q, [ a_ptr_1, 16 ]
	add a_ptr_1, a_ptr_1, 32

	ldr a02q, [ a_ptr_2 ]
	ldr a12q, [ a_ptr_2, 16 ]
	add a_ptr_2, a_ptr_2, 32

	ldr a03q, [ a_ptr_3 ]
	ldr a13q, [ a_ptr_3, 16 ]
	add a_ptr_3, a_ptr_3, 32

	ldr a04q, [ a_ptr_4 ]
	ldr a14q, [ a_ptr_4, 16 ]
	add a_ptr_4, a_ptr_4, 32

	ldr a05q, [ a_ptr_5 ]
	ldr a15q, [ a_ptr_5, 16 ]
	add a_ptr_5, a_ptr_5, 32

	ldr a06q, [ a_ptr_6 ]
	ldr a16q, [ a_ptr_6, 16 ]
	add a_ptr_6, a_ptr_6, 32

	ldr a07q, [ a_ptr_7 ]
	ldr a17q, [ a_ptr_7, 16 ]
	add a_ptr_7, a_ptr_7, 32

.L${func_name}_n_loop8_m_loop16_compute:
	bfmlalt c_a_0_hi_v.4s, a00v.8h, b0v.h[0]
	bfmlalb c_a_0_lo_v.4s, a00v.8h, b0v.h[0]

	bfmlalt c_a_1_hi_v.4s, a10v.8h, b0v.h[0]
	bfmlalb c_a_1_lo_v.4s, a10v.8h, b0v.h[0]

	bfmlalt c_b_0_hi_v.4s, a01v.8h, b0v.h[1]
	bfmlalb c_b_0_lo_v.4s, a01v.8h, b0v.h[1]

	bfmlalt c_b_1_hi_v.4s, a11v.8h, b0v.h[1]
	bfmlalb c_b_1_lo_v.4s, a11v.8h, b0v.h[1]


	bfmlalt c_a_0_hi_v.4s, a02v.8h, b0v.h[2]
	bfmlalb c_a_0_lo_v.4s, a02v.8h, b0v.h[2]

	bfmlalt c_a_1_hi_v.4s, a12v.8h, b0v.h[2]
	bfmlalb c_a_1_lo_v.4s, a12v.8h, b0v.h[2]

	bfmlalt c_b_0_hi_v.4s, a03v.8h, b0v.h[3]
	bfmlalb c_b_0_lo_v.4s, a03v.8h, b0v.h[3]

	bfmlalt c_b_1_hi_v.4s, a13v.8h, b0v.h[3]
	bfmlalb c_b_1_lo_v.4s, a13v.8h, b0v.h[3]

	bfmlalt c_a_0_hi_v.4s, a04v.8h, b0v.h[4]
	bfmlalb c_a_0_lo_v.4s, a04v.8h, b0v.h[4]

	bfmlalt c_a_1_hi_v.4s, a14v.8h, b0v.h[4]
	bfmlalb c_a_1_lo_v.4s, a14v.8h, b0v.h[4]

	bfmlalt c_b_0_hi_v.4s, a05v.8h, b0v.h[5]
	bfmlalb c_b_0_lo_v.4s, a05v.8h, b0v.h[5]

	bfmlalt c_b_1_hi_v.4s, a15v.8h, b0v.h[5]
	bfmlalb c_b_1_lo_v.4s, a15v.8h, b0v.h[5]

	bfmlalt c_a_0_hi_v.4s, a06v.8h, b0v.h[6]
	bfmlalb c_a_0_lo_v.4s, a06v.8h, b0v.h[6]

	bfmlalt c_a_1_hi_v.4s, a16v.8h, b0v.h[6]
	bfmlalb c_a_1_lo_v.4s, a16v.8h, b0v.h[6]

	bfmlalt c_b_0_hi_v.4s, a07v.8h, b0v.h[7]
	bfmlalb c_b_0_lo_v.4s, a07v.8h, b0v.h[7]

	bfmlalt c_b_1_hi_v.4s, a17v.8h, b0v.h[7]
	bfmlalb c_b_1_lo_v.4s, a17v.8h, b0v.h[7]

.L${func_name}_n_loop8_m_loop16_shuffle_reduce:
	zip1 a00v.4s,      c_a_0_lo_v.4s, c_a_0_hi_v.4s
	zip2 c_a_0_hi_v.4s, c_a_0_lo_v.4s, c_a_0_hi_v.4s

	fmla c0v.4s,      a00v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]

	zip1 a10v.4s,      c_a_1_lo_v.4s, c_a_1_hi_v.4s
	zip2 c_a_1_hi_v.4s, c_a_1_lo_v.4s, c_a_1_hi_v.4s

	fmla c2v.4s,      a10v.4s, alpha_v.s[0]
	fmla c3v.4s, c_a_1_hi_v.4s, alpha_v.s[0]

	zip1 a01v.4s,      c_b_0_lo_v.4s, c_b_0_hi_v.4s
	zip2 c_b_0_hi_v.4s, c_b_0_lo_v.4s, c_b_0_hi_v.4s

	fmla c0v.4s,      a01v.4s, alpha_v.s[0]
	fmla c1v.4s, c_b_0_hi_v.4s, alpha_v.s[0]

	zip1 a11v.4s,      c_b_1_lo_v.4s, c_b_1_hi_v.4s
	zip2 c_b_1_hi_v.4s, c_b_1_lo_v.4s, c_b_1_hi_v.4s

	fmla c2v.4s,      a11v.4s, alpha_v.s[0]
	fmla c3v.4s, c_b_1_hi_v.4s, alpha_v.s[0]

.L${func_name}_n_loop8_m_loop16_c_store_data:
	${store_vector(opts["out"], "c0", "c_ptr_0", 0)}
	${store_vector(opts["out"], "c1", "c_ptr_0", 1)}
	${store_vector(opts["out"], "c2", "c_ptr_0", 2)}
	${store_vector(opts["out"], "c3", "c_ptr_0", 3)}
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]*4}

.L${func_name}_n_loop8_m_loop16_end:
	add m_idx_next, m_idx, 16
	cmp m_idx_next, m
	bls .L${func_name}_n_loop8_m_loop16_start

	//Branch to condition of next section
	b .L${func_name}_n_loop8_m_loop8_end


.L${func_name}_n_loop8_m_loop8_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop8_m_loop8_zero_accumulators:
	movi c_a_0_hi_v.16b, #0
	movi c_b_0_hi_v.16b, #0
	movi c_a_0_lo_v.16b, #0
	movi c_b_0_lo_v.16b, #0

.L${func_name}_n_loop8_m_loop8_c_load_data:
	${load_vector(opts["out"], "c0", "c_ptr_0", 0)}
	${load_vector(opts["out"], "c1", "c_ptr_0", 1)}

.L${func_name}_n_loop8_m_loop8_a_load_data:
	ldr a00q, [ a_ptr_0 ]
	add a_ptr_0, a_ptr_0, 16

	ldr a01q, [ a_ptr_1 ]
	add a_ptr_1, a_ptr_1, 16

	ldr a02q, [ a_ptr_2 ]
	add a_ptr_2, a_ptr_2, 16

	ldr a03q, [ a_ptr_3 ]
	add a_ptr_3, a_ptr_3, 16

	ldr a04q, [ a_ptr_4 ]
	add a_ptr_4, a_ptr_4, 16

	ldr a05q, [ a_ptr_5 ]
	add a_ptr_5, a_ptr_5, 16

	ldr a06q, [ a_ptr_6 ]
	add a_ptr_6, a_ptr_6, 16

	ldr a07q, [ a_ptr_7 ]
	add a_ptr_7, a_ptr_7, 16

.L${func_name}_n_loop8_m_loop8_compute:
	bfmlalt c_a_0_hi_v.4s, a00v.8h, b0v.h[0]
	bfmlalb c_a_0_lo_v.4s, a00v.8h, b0v.h[0]

	bfmlalt c_b_0_hi_v.4s, a01v.8h, b0v.h[1]
	bfmlalb c_b_0_lo_v.4s, a01v.8h, b0v.h[1]

	bfmlalt c_a_0_hi_v.4s, a02v.8h, b0v.h[2]
	bfmlalb c_a_0_lo_v.4s, a02v.8h, b0v.h[2]

	bfmlalt c_b_0_hi_v.4s, a03v.8h, b0v.h[3]
	bfmlalb c_b_0_lo_v.4s, a03v.8h, b0v.h[3]

	bfmlalt c_a_0_hi_v.4s, a04v.8h, b0v.h[4]
	bfmlalb c_a_0_lo_v.4s, a04v.8h, b0v.h[4]

	bfmlalt c_b_0_hi_v.4s, a05v.8h, b0v.h[5]
	bfmlalb c_b_0_lo_v.4s, a05v.8h, b0v.h[5]

	bfmlalt c_a_0_hi_v.4s, a06v.8h, b0v.h[6]
	bfmlalb c_a_0_lo_v.4s, a06v.8h, b0v.h[6]

	bfmlalt c_b_0_hi_v.4s, a07v.8h, b0v.h[7]
	bfmlalb c_b_0_lo_v.4s, a07v.8h, b0v.h[7]

.L${func_name}_n_loop8_m_loop8_shuffle_reduce:
	zip1 a00v.4s,      c_a_0_lo_v.4s, c_a_0_hi_v.4s
	zip2 c_a_0_hi_v.4s, c_a_0_lo_v.4s, c_a_0_hi_v.4s

	fmla c0v.4s,      a00v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]

	zip1 a01v.4s,      c_b_0_lo_v.4s, c_b_0_hi_v.4s
	zip2 c_b_0_hi_v.4s, c_b_0_lo_v.4s, c_b_0_hi_v.4s

	fmla c0v.4s,      a01v.4s, alpha_v.s[0]
	fmla c1v.4s, c_b_0_hi_v.4s, alpha_v.s[0]

.L${func_name}_n_loop8_m_loop8_c_store_data:
	${store_vector(opts["out"], "c0", "c_ptr_0", 0)}
	${store_vector(opts["out"], "c1", "c_ptr_0", 1)}
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]*2}

.L${func_name}_n_loop8_m_loop8_end:
	add m_idx_next, m_idx, 8
	cmp m_idx_next, m
	bls .L${func_name}_n_loop8_m_loop8_start

	//Branch to condition of next section
	b .L${func_name}_n_loop8_m_loop1_end


.L${func_name}_n_loop8_m_loop1_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop8_m_loop1_zero_accumulators:
	movi c_a_0_lo_v.16b, #0
	movi c_b_0_lo_v.16b, #0

.L${func_name}_n_loop8_m_loop1_c_load_data:
	${load_element(opts["out"], "c0", "c_ptr_0")}

.L${func_name}_n_loop8_m_loop1_a_load_data:
	ldr a00h, [ a_ptr_0 ]
	add a_ptr_0, a_ptr_0, 2

	ldr a01h, [ a_ptr_1 ]
	add a_ptr_1, a_ptr_1, 2

	ldr a02h, [ a_ptr_2 ]
	add a_ptr_2, a_ptr_2, 2

	ldr a03h, [ a_ptr_3 ]
	add a_ptr_3, a_ptr_3, 2

	ldr a04h, [ a_ptr_4 ]
	add a_ptr_4, a_ptr_4, 2

	ldr a05h, [ a_ptr_5 ]
	add a_ptr_5, a_ptr_5, 2

	ldr a06h, [ a_ptr_6 ]
	add a_ptr_6, a_ptr_6, 2

	ldr a07h, [ a_ptr_7 ]
	add a_ptr_7, a_ptr_7, 2

.L${func_name}_n_loop8_m_loop1_compute:
	bfmlalb c_a_0_lo_v.4s, a00v.8h, b0v.h[0]
	bfmlalb c_b_0_lo_v.4s, a01v.8h, b0v.h[1]
	bfmlalb c_a_0_lo_v.4s, a02v.8h, b0v.h[2]
	bfmlalb c_b_0_lo_v.4s, a03v.8h, b0v.h[3]
	bfmlalb c_a_0_lo_v.4s, a04v.8h, b0v.h[4]
	bfmlalb c_b_0_lo_v.4s, a05v.8h, b0v.h[5]
	bfmlalb c_a_0_lo_v.4s, a06v.8h, b0v.h[6]
	bfmlalb c_b_0_lo_v.4s, a07v.8h, b0v.h[7]

.L${func_name}_n_loop8_m_loop1_shuffle_reduce:
	fadd c_a_0_lo_s, c_a_0_lo_s, c_b_0_lo_s
	fmadd c0s, c_a_0_lo_s, alpha_s, c0s

.L${func_name}_n_loop8_m_loop1_c_store_data:
	${store_element(opts["out"], "c0", "c_ptr_0")}
	add c_ptr_0, c_ptr_0, ${opts["out"]["bytes"]}

.L${func_name}_n_loop8_m_loop1_end:
	add m_idx_next, m_idx, 1
	cmp m_idx_next, m
	bls .L${func_name}_n_loop8_m_loop1_start


.L${func_name}_n_loop8_end:
	add n_idx_next, n_idx, 8
	cmp n_idx_next, n
	bls .L${func_name}_n_loop8_start

	b .L${func_name}_n_loop4_end


.L${func_name}_n_loop4_start:
	mov n_idx, n_idx_next

.L${func_name}_n_loop4_calculate_matrix_offsets:
	mov c_ptr_0, c_ptr

	//offsets of each col
	mov a_ptr_0, a_ptr
	add a_ptr_2, a_ptr,   lda, lsl 2
	add a_ptr_1, a_ptr,   lda, lsl 1
	add a_ptr_4, a_ptr_2, lda, lsl 2
	add a_ptr_3, a_ptr_2, lda, lsl 1

	add a_ptr,   a_ptr_2, lda, lsl 2

.L${func_name}_n_loop4_b_load_data:
	ldr b0d,  [ b_ptr ]
	add b_ptr, b_ptr, 8

.L${func_name}_n_loop4_m_loop_preamble:
	mov m_idx,      0
	b .L${func_name}_n_loop4_m_loop16_end

.L${func_name}_n_loop4_m_loop16_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop4_m_loop16_zero_accumulators:
	movi c_a_0_hi_v.16b, #0
	movi c_b_0_hi_v.16b, #0
	movi c_a_1_hi_v.16b, #0
	movi c_b_1_hi_v.16b, #0
	movi c_a_0_lo_v.16b, #0
	movi c_b_0_lo_v.16b, #0
	movi c_a_1_lo_v.16b, #0
	movi c_b_1_lo_v.16b, #0

.L${func_name}_n_loop4_m_loop16_c_load_data:
	${load_vector(opts["out"], "c0", "c_ptr_0", 0)}
	${load_vector(opts["out"], "c1", "c_ptr_0", 1)}
	${load_vector(opts["out"], "c2", "c_ptr_0", 2)}
	${load_vector(opts["out"], "c3", "c_ptr_0", 3)}

.L${func_name}_n_loop4_m_loop16_a_load_data:
	ldr a00q, [ a_ptr_0 ]
	ldr a10q, [ a_ptr_0, 16 ]
	add a_ptr_0, a_ptr_0, 32

	ldr a01q, [ a_ptr_1 ]
	ldr a11q, [ a_ptr_1, 16 ]
	add a_ptr_1, a_ptr_1, 32

	ldr a02q, [ a_ptr_2 ]
	ldr a12q, [ a_ptr_2, 16 ]
	add a_ptr_2, a_ptr_2, 32

	ldr a03q, [ a_ptr_3 ]
	ldr a13q, [ a_ptr_3, 16 ]
	add a_ptr_3, a_ptr_3, 32

.L${func_name}_n_loop4_m_loop16_compute:
	bfmlalt c_a_0_hi_v.4s, a00v.8h, b0v.h[0]
	bfmlalb c_a_0_lo_v.4s, a00v.8h, b0v.h[0]

	bfmlalt c_a_1_hi_v.4s, a10v.8h, b0v.h[0]
	bfmlalb c_a_1_lo_v.4s, a10v.8h, b0v.h[0]

	bfmlalt c_b_0_hi_v.4s, a01v.8h, b0v.h[1]
	bfmlalb c_b_0_lo_v.4s, a01v.8h, b0v.h[1]

	bfmlalt c_b_1_hi_v.4s, a11v.8h, b0v.h[1]
	bfmlalb c_b_1_lo_v.4s, a11v.8h, b0v.h[1]

	bfmlalt c_a_0_hi_v.4s, a02v.8h, b0v.h[2]
	bfmlalb c_a_0_lo_v.4s, a02v.8h, b0v.h[2]

	bfmlalt c_a_1_hi_v.4s, a12v.8h, b0v.h[2]
	bfmlalb c_a_1_lo_v.4s, a12v.8h, b0v.h[2]

	bfmlalt c_b_0_hi_v.4s, a03v.8h, b0v.h[3]
	bfmlalb c_b_0_lo_v.4s, a03v.8h, b0v.h[3]

	bfmlalt c_b_1_hi_v.4s, a13v.8h, b0v.h[3]
	bfmlalb c_b_1_lo_v.4s, a13v.8h, b0v.h[3]

.L${func_name}_n_loop4_m_loop16_shuffle_reduce:
	zip1 a00v.4s,      c_a_0_lo_v.4s, c_a_0_hi_v.4s
	zip2 c_a_0_hi_v.4s, c_a_0_lo_v.4s, c_a_0_hi_v.4s

	fmla c0v.4s,      a00v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]

	zip1 a10v.4s,      c_a_1_lo_v.4s, c_a_1_hi_v.4s
	zip2 c_a_1_hi_v.4s, c_a_1_lo_v.4s, c_a_1_hi_v.4s

	fmla c2v.4s,      a10v.4s, alpha_v.s[0]
	fmla c3v.4s, c_a_1_hi_v.4s, alpha_v.s[0]

	zip1 a01v.4s,      c_b_0_lo_v.4s, c_b_0_hi_v.4s
	zip2 c_b_0_hi_v.4s, c_b_0_lo_v.4s, c_b_0_hi_v.4s

	fmla c0v.4s,      a01v.4s, alpha_v.s[0]
	fmla c1v.4s, c_b_0_hi_v.4s, alpha_v.s[0]

	zip1 a11v.4s,      c_b_1_lo_v.4s, c_b_1_hi_v.4s
	zip2 c_b_1_hi_v.4s, c_b_1_lo_v.4s, c_b_1_hi_v.4s

	fmla c2v.4s,      a11v.4s, alpha_v.s[0]
	fmla c3v.4s, c_b_1_hi_v.4s, alpha_v.s[0]

.L${func_name}_n_loop4_m_loop16_c_store_data:
	${store_vector(opts["out"], "c0", "c_ptr_0", 0)}
	${store_vector(opts["out"], "c1", "c_ptr_0", 1)}
	${store_vector(opts["out"], "c2", "c_ptr_0", 2)}
	${store_vector(opts["out"], "c3", "c_ptr_0", 3)}
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]*4}

.L${func_name}_n_loop4_m_loop16_end:
	add m_idx_next, m_idx, 16
	cmp m_idx_next, m
	bls .L${func_name}_n_loop4_m_loop16_start

	//Branch to condition of next section
	b .L${func_name}_n_loop4_m_loop8_end


.L${func_name}_n_loop4_m_loop8_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop4_m_loop8_zero_accumulators:
	movi c_a_0_hi_v.16b, #0
	movi c_b_0_hi_v.16b, #0
	movi c_a_0_lo_v.16b, #0
	movi c_b_0_lo_v.16b, #0

.L${func_name}_n_loop4_m_loop8_c_load_data:
	${load_vector(opts["out"], "c0", "c_ptr_0", 0)}
	${load_vector(opts["out"], "c1", "c_ptr_0", 1)}

.L${func_name}_n_loop4_m_loop8_a_load_data:
	ldr a00q, [ a_ptr_0 ]
	add a_ptr_0, a_ptr_0, 16

	ldr a01q, [ a_ptr_1 ]
	add a_ptr_1, a_ptr_1, 16

	ldr a02q, [ a_ptr_2 ]
	add a_ptr_2, a_ptr_2, 16

	ldr a03q, [ a_ptr_3 ]
	add a_ptr_3, a_ptr_3, 16

.L${func_name}_n_loop4_m_loop8_compute:
	bfmlalt c_a_0_hi_v.4s, a00v.8h, b0v.h[0]
	bfmlalb c_a_0_lo_v.4s, a00v.8h, b0v.h[0]

	bfmlalt c_b_0_hi_v.4s, a01v.8h, b0v.h[1]
	bfmlalb c_b_0_lo_v.4s, a01v.8h, b0v.h[1]

	bfmlalt c_a_0_hi_v.4s, a02v.8h, b0v.h[2]
	bfmlalb c_a_0_lo_v.4s, a02v.8h, b0v.h[2]

	bfmlalt c_b_0_hi_v.4s, a03v.8h, b0v.h[3]
	bfmlalb c_b_0_lo_v.4s, a03v.8h, b0v.h[3]

.L${func_name}_n_loop4_m_loop8_shuffle_reduce:
	zip1 a00v.4s,      c_a_0_lo_v.4s, c_a_0_hi_v.4s
	zip2 c_a_0_hi_v.4s, c_a_0_lo_v.4s, c_a_0_hi_v.4s

	fmla c0v.4s,      a00v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]

	zip1 a01v.4s,      c_b_0_lo_v.4s, c_b_0_hi_v.4s
	zip2 c_b_0_hi_v.4s, c_b_0_lo_v.4s, c_b_0_hi_v.4s

	fmla c0v.4s,      a01v.4s, alpha_v.s[0]
	fmla c1v.4s, c_b_0_hi_v.4s, alpha_v.s[0]

.L${func_name}_n_loop4_m_loop8_c_store_data:
	${store_vector(opts["out"], "c0", "c_ptr_0", 0)}
	${store_vector(opts["out"], "c1", "c_ptr_0", 1)}
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]*2}

.L${func_name}_n_loop4_m_loop8_end:
	add m_idx_next, m_idx, 8
	cmp m_idx_next, m
	bls .L${func_name}_n_loop4_m_loop8_start

	//Branch to condition of next section
	b .L${func_name}_n_loop4_m_loop1_end


.L${func_name}_n_loop4_m_loop1_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop4_m_loop1_zero_accumulators:
	movi c_a_0_lo_v.16b, #0
	movi c_b_0_lo_v.16b, #0

.L${func_name}_n_loop4_m_loop1_c_load_data:
	${load_element(opts["out"], "c0", "c_ptr_0")}

.L${func_name}_n_loop4_m_loop1_a_load_data:
	ldr a00h, [ a_ptr_0 ]
	add a_ptr_0, a_ptr_0, 2

	ldr a01h, [ a_ptr_1 ]
	add a_ptr_1, a_ptr_1, 2

	ldr a02h, [ a_ptr_2 ]
	add a_ptr_2, a_ptr_2, 2

	ldr a03h, [ a_ptr_3 ]
	add a_ptr_3, a_ptr_3, 2

.L${func_name}_n_loop4_m_loop1_compute:
	bfmlalb c_a_0_lo_v.4s, a00v.8h, b0v.h[0]
	bfmlalb c_b_0_lo_v.4s, a01v.8h, b0v.h[1]
	bfmlalb c_a_0_lo_v.4s, a02v.8h, b0v.h[2]
	bfmlalb c_b_0_lo_v.4s, a03v.8h, b0v.h[3]

.L${func_name}_n_loop4_m_loop1_shuffle_reduce:
	fadd c_a_0_lo_s, c_a_0_lo_s, c_b_0_lo_s
	fmadd c0s, c_a_0_lo_s, alpha_s, c0s

.L${func_name}_n_loop4_m_loop1_c_store_data:
	${store_element(opts["out"], "c0", "c_ptr_0")}
	add c_ptr_0, c_ptr_0, ${opts["out"]["bytes"]}

.L${func_name}_n_loop4_m_loop1_end:
	add m_idx_next, m_idx, 1
	cmp m_idx_next, m
	bls .L${func_name}_n_loop4_m_loop1_start


.L${func_name}_n_loop4_end:
	add n_idx_next, n_idx, 4
	cmp n_idx_next, n
	bls .L${func_name}_n_loop4_start

	b .L${func_name}_n_loop1_end

.L${func_name}_n_loop1_start:
	mov n_idx, n_idx_next

.L${func_name}_n_loop1_calculate_matrix_offsets:
	mov c_ptr_0, c_ptr

	//offsets of each col
	mov a_ptr_0, a_ptr
	add a_ptr, a_ptr, lda, lsl 1

.L${func_name}_n_loop1_b_load_data:
	ldr b0h,  [ b_ptr ]
	add b_ptr, b_ptr, 2

.L${func_name}_n_loop1_m_loop_preamble:
	mov m_idx,      0
	b .L${func_name}_n_loop1_m_loop16_end

.L${func_name}_n_loop1_m_loop16_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop1_m_loop16_zero_accumulators:
	movi c_a_0_hi_v.16b, #0
	movi c_a_1_hi_v.16b, #0
	movi c_a_0_lo_v.16b, #0
	movi c_a_1_lo_v.16b, #0

.L${func_name}_n_loop1_m_loop16_c_load_data:
	${load_vector(opts["out"], "c0", "c_ptr_0", 0)}
	${load_vector(opts["out"], "c1", "c_ptr_0", 1)}
	${load_vector(opts["out"], "c2", "c_ptr_0", 2)}
	${load_vector(opts["out"], "c3", "c_ptr_0", 3)}

.L${func_name}_n_loop1_m_loop16_a_load_data:
	ldr a00q, [ a_ptr_0 ]
	ldr a10q, [ a_ptr_0, 16 ]
	add a_ptr_0, a_ptr_0, 32

.L${func_name}_n_loop1_m_loop16_compute:
	bfmlalt c_a_0_hi_v.4s, a00v.8h, b0v.h[0]
	bfmlalb c_a_0_lo_v.4s, a00v.8h, b0v.h[0]

	bfmlalt c_a_1_hi_v.4s, a10v.8h, b0v.h[0]
	bfmlalb c_a_1_lo_v.4s, a10v.8h, b0v.h[0]

.L${func_name}_n_loop1_m_loop16_shuffle_reduce:
	zip1 a00v.4s,      c_a_0_lo_v.4s, c_a_0_hi_v.4s
	zip2 c_a_0_hi_v.4s, c_a_0_lo_v.4s, c_a_0_hi_v.4s

	fmla c0v.4s,      a00v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]

	zip1 a10v.4s,      c_a_1_lo_v.4s, c_a_1_hi_v.4s
	zip2 c_a_1_hi_v.4s, c_a_1_lo_v.4s, c_a_1_hi_v.4s

	fmla c2v.4s,      a10v.4s, alpha_v.s[0]
	fmla c3v.4s, c_a_1_hi_v.4s, alpha_v.s[0]

.L${func_name}_n_loop1_m_loop16_c_store_data:
	${store_vector(opts["out"], "c0", "c_ptr_0", 0)}
	${store_vector(opts["out"], "c1", "c_ptr_0", 1)}
	${store_vector(opts["out"], "c2", "c_ptr_0", 2)}
	${store_vector(opts["out"], "c3", "c_ptr_0", 3)}
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]*4}

.L${func_name}_n_loop1_m_loop16_end:
	add m_idx_next, m_idx, 16
	cmp m_idx_next, m
	bls .L${func_name}_n_loop1_m_loop16_start

	//Branch to condition of next section
	b .L${func_name}_n_loop1_m_loop8_end


.L${func_name}_n_loop1_m_loop8_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop1_m_loop8_zero_accumulators:
	movi c_a_0_hi_v.16b, #0
	movi c_a_0_lo_v.16b, #0

.L${func_name}_n_loop1_m_loop8_c_load_data:
	${load_vector(opts["out"], "c0", "c_ptr_0", 0)}
	${load_vector(opts["out"], "c1", "c_ptr_0", 1)}

.L${func_name}_n_loop1_m_loop8_a_load_data:
	ldr a00q, [ a_ptr_0 ]
	add a_ptr_0, a_ptr_0, 16

.L${func_name}_n_loop1_m_loop8_compute:
	bfmlalt c_a_0_hi_v.4s, a00v.8h, b0v.h[0]
	bfmlalb c_a_0_lo_v.4s, a00v.8h, b0v.h[0]

.L${func_name}_n_loop1_m_loop8_shuffle_reduce:
	zip1 a00v.4s,      c_a_0_lo_v.4s, c_a_0_hi_v.4s
	zip2 c_a_0_hi_v.4s, c_a_0_lo_v.4s, c_a_0_hi_v.4s

	fmla c0v.4s,      a00v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]

.L${func_name}_n_loop1_m_loop8_c_store_data:
	${store_vector(opts["out"], "c0", "c_ptr_0", 0)}
	${store_vector(opts["out"], "c1", "c_ptr_0", 1)}
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]*2}

.L${func_name}_n_loop1_m_loop8_end:
	add m_idx_next, m_idx, 8
	cmp m_idx_next, m
	bls .L${func_name}_n_loop1_m_loop8_start

	//Branch to condition of next section
	b .L${func_name}_n_loop1_m_loop1_end

.L${func_name}_n_loop1_m_loop1_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop1_m_loop1_zero_accumulators:
	movi c_a_0_lo_v.16b, #0
	movi c_b_0_lo_v.16b, #0

.L${func_name}_n_loop1_m_loop1_c_load_data:
	${load_element(opts["out"], "c0", "c_ptr_0")}

.L${func_name}_n_loop1_m_loop1_a_load_data:
	ldr a00h, [ a_ptr_0 ]
	add a_ptr_0, a_ptr_0, 2

.L${func_name}_n_loop1_m_loop1_compute:
	bfmlalb c_a_0_lo_v.4s, a00v.8h, b0v.h[0]

.L${func_name}_n_loop1_m_loop1_shuffle_reduce:
	fmadd c0s, c_a_0_lo_s, alpha_s, c0s

.L${func_name}_n_loop1_m_loop1_c_store_data:
	${store_element(opts["out"], "c0", "c_ptr_0")}
	add c_ptr_0, c_ptr_0, ${opts["out"]["bytes"]}

.L${func_name}_n_loop1_m_loop1_end:
	add m_idx_next, m_idx, 1
	cmp m_idx_next, m
	bls .L${func_name}_n_loop1_m_loop1_start


.L${func_name}_n_loop1_end:
	add n_idx_next, n_idx, 1
	cmp n_idx_next, n
	bls .L${func_name}_n_loop1_start


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

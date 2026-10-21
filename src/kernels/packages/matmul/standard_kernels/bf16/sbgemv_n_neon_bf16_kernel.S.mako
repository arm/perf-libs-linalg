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
	"store_vector":        "uzp2 {0}v.8h, {0}v.8h, {0}v.8h\n\tstr {0}d, [ {1} ]",
	"store_element":       "st1 {{ {0}v.h }}[1], [ {1} ]"
}
functions={
	"sbgemv_n_neon_bf16_kernel": {
		"in": bf16_options,
		"out": r32_options,
		"adjust_scalar": ""
	},
	"bgemv_n_neon_bf16_kernel": {
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

// aliases used by the 24-row BFDOT path
${vreg("c4", 3)}
${vreg("c5", 5)}

${vreg("c00", 6)}
${vreg("c10", 2)}
${vreg("c20", 8)}
${vreg("c30", 4)}
${vreg("c40", 7)}
${vreg("c50", 9)}

bf16_mask_v .req v31

%for func_name, opts in functions.items():
<%
is_bgemv_n = func_name == "bgemv_n_neon_bf16_kernel"
is_sbgemv_n = func_name == "sbgemv_n_neon_bf16_kernel"
%>
${prologue(func_name)}

	stp     x29, x30, [sp, #-160]!
	mov     x29, sp

	stp     x19, x20, [sp, #16]
	stp     x21, x22, [sp, #32]
	stp     x23, x24, [sp, #48]
	stp     x25, x26, [sp, #64]
	stp     x27, x28, [sp, #80]

	stp     d8,  d9,  [sp, #96]
	stp     d10, d11, [sp, #112]
	stp     d12, d13, [sp, #128]
	stp     d14, d15, [sp, #144]

.L${func_name}_n_loop_preamble:
	${adjust_scalar(opts, "alpha_")}
	${adjust_scalar(opts, "beta_")}
% if is_bgemv_n:
	movi bf16_mask_v.4s, #0xff, msl #8
% endif

	//branch to test condition, retruns to correct start
	mov n_idx, 0
% if is_bgemv_n:
	b .L${func_name}_n_loop8_end
% elif is_sbgemv_n:
	b .L${func_name}_n_loop16_end

// SBGEMV-only sixteen-column path. Keep C live across the two
// eight-column sums, retaining their separate alpha updates.
a_ptr_8  .req x5
a_ptr_9  .req x22
a_ptr_10 .req x23
a_ptr_11 .req x24
a_ptr_12 .req x25
a_ptr_13 .req x26
a_ptr_14 .req x27
a_ptr_15 .req x28

b_firstv .req v30
b_firstq .req q30
b_nextv  .req v31
b_nextq  .req q31

.L${func_name}_n_loop16_start:
	mov n_idx, n_idx_next

.L${func_name}_n_loop16_calculate_matrix_offsets:
	mov c_ptr_0, c_ptr

	mov a_ptr_0, a_ptr
	add a_ptr_2, a_ptr,   lda, lsl 2
	add a_ptr_1, a_ptr,   lda, lsl 1
	add a_ptr_4, a_ptr_2, lda, lsl 2
	add a_ptr_3, a_ptr_2, lda, lsl 1
	add a_ptr_6, a_ptr_4, lda, lsl 2
	add a_ptr_5, a_ptr_4, lda, lsl 1
	add a_ptr_7, a_ptr_6, lda, lsl 1

	add a_ptr_8,  a_ptr_0, lda, lsl 4
	add a_ptr_9,  a_ptr_1, lda, lsl 4
	add a_ptr_10, a_ptr_2, lda, lsl 4
	add a_ptr_11, a_ptr_3, lda, lsl 4
	add a_ptr_12, a_ptr_4, lda, lsl 4
	add a_ptr_13, a_ptr_5, lda, lsl 4
	add a_ptr_14, a_ptr_6, lda, lsl 4
	add a_ptr_15, a_ptr_7, lda, lsl 4

	add a_ptr,   a_ptr,   lda, lsl 5

.L${func_name}_n_loop16_b_load_data:
	ldp b_firstq, b_nextq, [ b_ptr ]
	add b_ptr, b_ptr, 32

.L${func_name}_n_loop16_m_loop_preamble:
	mov m_idx,      0
	b .L${func_name}_n_loop16_m_loop24_end

.L${func_name}_n_loop16_m_loop24_start:
	mov m_idx, m_idx_next

% for column_start in (0, 8):
	mov b0v.16b, b_${"first" if column_start == 0 else "next"}v.16b

.L${func_name}_n_loop16_m_loop24_zero_accumulators_columns_${column_start}_${column_start + 7}:
	movi c00v.16b, #0
	movi c10v.16b, #0
	movi c20v.16b, #0
	movi c30v.16b, #0
	movi c40v.16b, #0
	movi c50v.16b, #0

.L${func_name}_n_loop16_m_loop24_load_compute_data_columns_${column_start}_${column_start + 7}:
	ldr a14q, [ a_ptr_${column_start+0}, #32 ]
	ldp a00q, a10q, [ a_ptr_${column_start+0} ], #48
	ldr a15q, [ a_ptr_${column_start+1}, #32 ]
	ldp a01q, a11q, [ a_ptr_${column_start+1} ], #48
% if column_start == 0:
	ldp c0q, c1q, [ c_ptr_0 ]
% endif

	zip1 a04v.8h, a14v.8h, a15v.8h
	zip2 a05v.8h, a14v.8h, a15v.8h

	ldr a14q, [ a_ptr_${column_start+2}, #32 ]
	zip1 a02v.8h, a00v.8h, a01v.8h

	ldr a15q, [ a_ptr_${column_start+3}, #32 ]
	zip2 a03v.8h, a00v.8h, a01v.8h

	zip1 a12v.8h, a10v.8h, a11v.8h
	zip2 a13v.8h, a10v.8h, a11v.8h

	bfdot c00v.4s, a02v.8h, b0v.2h[0]
	bfdot c20v.4s, a12v.8h, b0v.2h[0]
	ldp a02q, a12q, [ a_ptr_${column_start+2} ], #48

	bfdot c10v.4s, a03v.8h, b0v.2h[0]
	bfdot c30v.4s, a13v.8h, b0v.2h[0]
	ldp a03q, a13q, [ a_ptr_${column_start+3} ], #48

	bfdot c40v.4s, a04v.8h, b0v.2h[0]
	bfdot c50v.4s, a05v.8h, b0v.2h[0]
% if column_start == 0:
	ldp c2q, c3q, [ c_ptr_0, #32 ]
% endif

	zip1 a00v.8h, a02v.8h, a03v.8h
	zip2 a01v.8h, a02v.8h, a03v.8h
	zip1 a10v.8h, a12v.8h, a13v.8h
	zip2 a11v.8h, a12v.8h, a13v.8h
	zip1 a04v.8h, a14v.8h, a15v.8h
	zip2 a05v.8h, a14v.8h, a15v.8h

	ldr a12q, [ a_ptr_${column_start+4}, #32 ]
	ldr a13q, [ a_ptr_${column_start+5}, #32 ]

	bfdot c00v.4s, a00v.8h, b0v.2h[1]
	bfdot c10v.4s, a01v.8h, b0v.2h[1]
	bfdot c20v.4s, a10v.8h, b0v.2h[1]
	bfdot c30v.4s, a11v.8h, b0v.2h[1]
	bfdot c40v.4s, a04v.8h, b0v.2h[1]
	ldp a04q, a14q, [ a_ptr_${column_start+4} ], #48

	bfdot c50v.4s, a05v.8h, b0v.2h[1]
	ldp a05q, a15q, [ a_ptr_${column_start+5} ], #48
% if column_start == 0:
	ldp c4q, c5q, [ c_ptr_0, #64 ]
% endif

	zip1 a00v.8h, a04v.8h, a05v.8h
	zip2 a01v.8h, a04v.8h, a05v.8h
	zip1 a10v.8h, a14v.8h, a15v.8h
	zip2 a11v.8h, a14v.8h, a15v.8h
	zip1 a02v.8h, a12v.8h, a13v.8h
	zip2 a03v.8h, a12v.8h, a13v.8h

	ldr a14q, [ a_ptr_${column_start+6}, #32 ]
	ldp a06q, a16q, [ a_ptr_${column_start+6} ], #48
	bfdot c00v.4s, a00v.8h, b0v.2h[2]
	bfdot c10v.4s, a01v.8h, b0v.2h[2]

	ldr a15q, [ a_ptr_${column_start+7}, #32 ]
	ldp a07q, a17q, [ a_ptr_${column_start+7} ], #48
	bfdot c20v.4s, a10v.8h, b0v.2h[2]
	bfdot c30v.4s, a11v.8h, b0v.2h[2]

	bfdot c40v.4s, a02v.8h, b0v.2h[2]
	bfdot c50v.4s, a03v.8h, b0v.2h[2]

	zip1 a00v.8h, a06v.8h, a07v.8h
	zip2 a01v.8h, a06v.8h, a07v.8h
	zip1 a10v.8h, a16v.8h, a17v.8h
	zip2 a11v.8h, a16v.8h, a17v.8h
	zip1 a02v.8h, a14v.8h, a15v.8h
	zip2 a03v.8h, a14v.8h, a15v.8h

	bfdot c00v.4s, a00v.8h, b0v.2h[3]
	bfdot c10v.4s, a01v.8h, b0v.2h[3]
	bfdot c20v.4s, a10v.8h, b0v.2h[3]
	bfdot c30v.4s, a11v.8h, b0v.2h[3]
	bfdot c40v.4s, a02v.8h, b0v.2h[3]
	bfdot c50v.4s, a03v.8h, b0v.2h[3]

.L${func_name}_n_loop16_m_loop24_scale_accumulate_columns_${column_start}_${column_start + 7}:
	fmla c0v.4s, c00v.4s, alpha_v.s[0]
	fmla c1v.4s, c10v.4s, alpha_v.s[0]
	fmla c2v.4s, c20v.4s, alpha_v.s[0]
	fmla c3v.4s, c30v.4s, alpha_v.s[0]
	fmla c4v.4s, c40v.4s, alpha_v.s[0]
	fmla c5v.4s, c50v.4s, alpha_v.s[0]

% endfor

.L${func_name}_n_loop16_m_loop24_c_store_data:
	stp c0q, c1q, [ c_ptr_0 ]
	stp c2q, c3q, [ c_ptr_0, #32 ]
	stp c4q, c5q, [ c_ptr_0, #64 ]
	add c_ptr_0, c_ptr_0, 96

.L${func_name}_n_loop16_m_loop24_end:
	add m_idx_next, m_idx, 24
	cmp m_idx_next, m
	bls .L${func_name}_n_loop16_m_loop24_start

	b .L${func_name}_n_loop16_m_loop16_end

.L${func_name}_n_loop16_m_loop16_start:
	mov m_idx, m_idx_next

% for column_start in (0, 8):
	mov b0v.16b, b_${"first" if column_start == 0 else "next"}v.16b

.L${func_name}_n_loop16_m_loop16_zero_accumulators_columns_${column_start}_${column_start + 7}:
	movi c_a_0_hi_v.16b, #0
	movi c_a_1_hi_v.16b, #0
	movi c_a_0_lo_v.16b, #0
	movi c_a_1_lo_v.16b, #0

.L${func_name}_n_loop16_m_loop16_load_compute_data_columns_${column_start}_${column_start + 7}:
	ldp a00q, a10q, [ a_ptr_${column_start+0} ], #32
	ldp a01q, a11q, [ a_ptr_${column_start+1} ], #32
% if column_start == 0:
	ldr c0q, [ c_ptr_0 ]
% endif

	zip1 a02v.8h, a00v.8h, a01v.8h
	zip2 a03v.8h, a00v.8h, a01v.8h
	zip1 a12v.8h, a10v.8h, a11v.8h
	zip2 a13v.8h, a10v.8h, a11v.8h

	bfdot c_a_0_lo_v.4s, a02v.8h, b0v.2h[0]
	bfdot c_a_0_hi_v.4s, a03v.8h, b0v.2h[0]
	bfdot c_a_1_lo_v.4s, a12v.8h, b0v.2h[0]
	bfdot c_a_1_hi_v.4s, a13v.8h, b0v.2h[0]

	ldp a02q, a12q, [ a_ptr_${column_start+2} ], #32
	ldp a03q, a13q, [ a_ptr_${column_start+3} ], #32
% if column_start == 0:
	ldr c1q, [ c_ptr_0, #16 ]
% endif

	zip1 a00v.8h, a02v.8h, a03v.8h
	zip2 a01v.8h, a02v.8h, a03v.8h
	zip1 a10v.8h, a12v.8h, a13v.8h
	zip2 a11v.8h, a12v.8h, a13v.8h

	bfdot c_a_0_lo_v.4s, a00v.8h, b0v.2h[1]
	bfdot c_a_0_hi_v.4s, a01v.8h, b0v.2h[1]
	bfdot c_a_1_lo_v.4s, a10v.8h, b0v.2h[1]
	bfdot c_a_1_hi_v.4s, a11v.8h, b0v.2h[1]

	ldp a04q, a14q, [ a_ptr_${column_start+4} ], #32
	ldp a05q, a15q, [ a_ptr_${column_start+5} ], #32
% if column_start == 0:
	ldr c2q, [ c_ptr_0, #32 ]
% endif

	zip1 a00v.8h, a04v.8h, a05v.8h
	zip2 a01v.8h, a04v.8h, a05v.8h
	zip1 a10v.8h, a14v.8h, a15v.8h
	zip2 a11v.8h, a14v.8h, a15v.8h

	bfdot c_a_0_lo_v.4s, a00v.8h, b0v.2h[2]
	bfdot c_a_0_hi_v.4s, a01v.8h, b0v.2h[2]
	bfdot c_a_1_lo_v.4s, a10v.8h, b0v.2h[2]
	bfdot c_a_1_hi_v.4s, a11v.8h, b0v.2h[2]

	ldp a06q, a16q, [ a_ptr_${column_start+6} ], #32
	ldp a07q, a17q, [ a_ptr_${column_start+7} ], #32
% if column_start == 0:
	ldr c3q, [ c_ptr_0, #48 ]
% endif

	zip1 a00v.8h, a06v.8h, a07v.8h
	zip2 a01v.8h, a06v.8h, a07v.8h
	zip1 a10v.8h, a16v.8h, a17v.8h
	zip2 a11v.8h, a16v.8h, a17v.8h

	bfdot c_a_0_lo_v.4s, a00v.8h, b0v.2h[3]
	bfdot c_a_0_hi_v.4s, a01v.8h, b0v.2h[3]
	bfdot c_a_1_lo_v.4s, a10v.8h, b0v.2h[3]
	bfdot c_a_1_hi_v.4s, a11v.8h, b0v.2h[3]

.L${func_name}_n_loop16_m_loop16_shuffle_reduce_columns_${column_start}_${column_start + 7}:
	fmla c0v.4s, c_a_0_lo_v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]
	fmla c2v.4s, c_a_1_lo_v.4s, alpha_v.s[0]
	fmla c3v.4s, c_a_1_hi_v.4s, alpha_v.s[0]

% endfor

.L${func_name}_n_loop16_m_loop16_c_store_data:
	stp c0q, c1q, [ c_ptr_0 ]
	stp c2q, c3q, [ c_ptr_0, #32 ]
	add c_ptr_0, c_ptr_0, 64

.L${func_name}_n_loop16_m_loop16_end:
	add m_idx_next, m_idx, 16
	cmp m_idx_next, m
	bls .L${func_name}_n_loop16_m_loop16_start

	b .L${func_name}_n_loop16_m_loop8_end

.L${func_name}_n_loop16_m_loop8_start:
	mov m_idx, m_idx_next

% for column_start in (0, 8):
	mov b0v.16b, b_${"first" if column_start == 0 else "next"}v.16b

.L${func_name}_n_loop16_m_loop8_zero_accumulators_columns_${column_start}_${column_start + 7}:
	movi c_a_0_hi_v.16b, #0
	movi c_a_0_lo_v.16b, #0

.L${func_name}_n_loop16_m_loop8_load_compute_data_columns_${column_start}_${column_start + 7}:
	ldr a00q, [ a_ptr_${column_start+0} ], #16
	ldr a01q, [ a_ptr_${column_start+1} ], #16
% if column_start == 0:
	ldr c0q, [ c_ptr_0 ]
% endif

	zip1 a02v.8h, a00v.8h, a01v.8h
	zip2 a03v.8h, a00v.8h, a01v.8h
	bfdot c_a_0_lo_v.4s, a02v.8h, b0v.2h[0]
	bfdot c_a_0_hi_v.4s, a03v.8h, b0v.2h[0]

	ldr a02q, [ a_ptr_${column_start+2} ], #16
	ldr a03q, [ a_ptr_${column_start+3} ], #16
% if column_start == 0:
	ldr c1q, [ c_ptr_0, #16 ]
% endif

	zip1 a00v.8h, a02v.8h, a03v.8h
	zip2 a01v.8h, a02v.8h, a03v.8h
	bfdot c_a_0_lo_v.4s, a00v.8h, b0v.2h[1]
	bfdot c_a_0_hi_v.4s, a01v.8h, b0v.2h[1]

	ldr a04q, [ a_ptr_${column_start+4} ], #16
	ldr a05q, [ a_ptr_${column_start+5} ], #16

	zip1 a00v.8h, a04v.8h, a05v.8h
	zip2 a01v.8h, a04v.8h, a05v.8h
	bfdot c_a_0_lo_v.4s, a00v.8h, b0v.2h[2]
	bfdot c_a_0_hi_v.4s, a01v.8h, b0v.2h[2]

	ldr a06q, [ a_ptr_${column_start+6} ], #16
	ldr a07q, [ a_ptr_${column_start+7} ], #16

	zip1 a00v.8h, a06v.8h, a07v.8h
	zip2 a01v.8h, a06v.8h, a07v.8h
	bfdot c_a_0_lo_v.4s, a00v.8h, b0v.2h[3]
	bfdot c_a_0_hi_v.4s, a01v.8h, b0v.2h[3]

.L${func_name}_n_loop16_m_loop8_scale_accumulate_columns_${column_start}_${column_start + 7}:
	fmla c0v.4s, c_a_0_lo_v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]

% endfor

.L${func_name}_n_loop16_m_loop8_c_store_data:
	stp c0q, c1q, [ c_ptr_0 ]
	add c_ptr_0, c_ptr_0, 32

.L${func_name}_n_loop16_m_loop8_end:
	add m_idx_next, m_idx, 8
	cmp m_idx_next, m
	bls .L${func_name}_n_loop16_m_loop8_start

	b .L${func_name}_n_loop16_m_loop4_end

.L${func_name}_n_loop16_m_loop4_start:
	mov m_idx, m_idx_next

% for column_start in (0, 8):
	mov b0v.16b, b_${"first" if column_start == 0 else "next"}v.16b

.L${func_name}_n_loop16_m_loop4_zero_accumulators_columns_${column_start}_${column_start + 7}:
	movi c_a_0_lo_v.16b, #0
	movi c_b_0_lo_v.16b, #0

.L${func_name}_n_loop16_m_loop4_load_compute_data_columns_${column_start}_${column_start + 7}:
	ldr a00d, [ a_ptr_${column_start+0} ], #8
	ldr a01d, [ a_ptr_${column_start+1} ], #8
% if column_start == 0:
	ldr c0q, [ c_ptr_0 ]
% endif

	zip1 a00v.8h, a00v.8h, a01v.8h
	bfdot c_a_0_lo_v.4s, a00v.8h, b0v.2h[0]

	ldr a02d, [ a_ptr_${column_start+2} ], #8
	ldr a03d, [ a_ptr_${column_start+3} ], #8

	zip1 a02v.8h, a02v.8h, a03v.8h
	bfdot c_b_0_lo_v.4s, a02v.8h, b0v.2h[1]

	ldr a04d, [ a_ptr_${column_start+4} ], #8
	ldr a05d, [ a_ptr_${column_start+5} ], #8

	zip1 a04v.8h, a04v.8h, a05v.8h
	bfdot c_a_0_lo_v.4s, a04v.8h, b0v.2h[2]

	ldr a06d, [ a_ptr_${column_start+6} ], #8
	ldr a07d, [ a_ptr_${column_start+7} ], #8

	zip1 a06v.8h, a06v.8h, a07v.8h
	bfdot c_b_0_lo_v.4s, a06v.8h, b0v.2h[3]

.L${func_name}_n_loop16_m_loop4_shuffle_reduce_columns_${column_start}_${column_start + 7}:
	fadd c_a_0_lo_v.4s, c_a_0_lo_v.4s, c_b_0_lo_v.4s
	fmla c0v.4s, c_a_0_lo_v.4s, alpha_v.s[0]

% endfor

.L${func_name}_n_loop16_m_loop4_c_store_data:
	str c0q, [ c_ptr_0 ]
	add c_ptr_0, c_ptr_0, 16

.L${func_name}_n_loop16_m_loop4_end:
	add m_idx_next, m_idx, 4
	cmp m_idx_next, m
	bls .L${func_name}_n_loop16_m_loop4_start

	b .L${func_name}_n_loop16_m_loop1_end

.L${func_name}_n_loop16_m_loop1_start:
	mov m_idx, m_idx_next

% for column_start in (0, 8):
	mov b0v.16b, b_${"first" if column_start == 0 else "next"}v.16b

.L${func_name}_n_loop16_m_loop1_zero_accumulators_columns_${column_start}_${column_start + 7}:
	movi c_a_0_lo_v.16b, #0
	movi c_b_0_lo_v.16b, #0

.L${func_name}_n_loop16_m_loop1_c_load_data_columns_${column_start}_${column_start + 7}:
% if column_start == 0:
	ldr c0s, [ c_ptr_0 ]
% endif

.L${func_name}_n_loop16_m_loop1_load_compute_data_columns_${column_start}_${column_start + 7}:
	ldr a00h, [ a_ptr_${column_start+0} ], #2
	ldr a01h, [ a_ptr_${column_start+1} ], #2
	zip1 a10v.8h, a00v.8h, a01v.8h
	bfdot c_a_0_lo_v.4s, a10v.8h, b0v.2h[0]

	ldr a02h, [ a_ptr_${column_start+2} ], #2
	ldr a03h, [ a_ptr_${column_start+3} ], #2
	zip1 a11v.8h, a02v.8h, a03v.8h
	bfdot c_b_0_lo_v.4s, a11v.8h, b0v.2h[1]

	ldr a04h, [ a_ptr_${column_start+4} ], #2
	ldr a05h, [ a_ptr_${column_start+5} ], #2
	zip1 a12v.8h, a04v.8h, a05v.8h
	bfdot c_a_0_lo_v.4s, a12v.8h, b0v.2h[2]

	ldr a06h, [ a_ptr_${column_start+6} ], #2
	ldr a07h, [ a_ptr_${column_start+7} ], #2
	zip1 a13v.8h, a06v.8h, a07v.8h
	bfdot c_b_0_lo_v.4s, a13v.8h, b0v.2h[3]

.L${func_name}_n_loop16_m_loop1_shuffle_reduce_columns_${column_start}_${column_start + 7}:
	fadd c_a_0_lo_s, c_a_0_lo_s, c_b_0_lo_s
	fmadd c0s, c_a_0_lo_s, alpha_s, c0s

% endfor

.L${func_name}_n_loop16_m_loop1_c_store_data:
	str c0s, [ c_ptr_0 ], #4

.L${func_name}_n_loop16_m_loop1_end:
	add m_idx_next, m_idx, 1
	cmp m_idx_next, m
	bls .L${func_name}_n_loop16_m_loop1_start

.L${func_name}_n_loop16_end:
	add n_idx_next, n_idx, 16
	cmp n_idx_next, n
	bls .L${func_name}_n_loop16_start

	b .L${func_name}_n_loop8_end
% endif

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
% if is_bgemv_n:
	b .L${func_name}_n_loop8_m_loop16_end
% elif is_sbgemv_n:
	b .L${func_name}_n_loop8_m_loop24_end
% endif

% if is_sbgemv_n:
.L${func_name}_n_loop8_m_loop24_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop8_m_loop24_zero_accumulators:
	movi c00v.16b, #0
	movi c10v.16b, #0
	movi c20v.16b, #0
	movi c30v.16b, #0
	movi c40v.16b, #0
	movi c50v.16b, #0

.L${func_name}_n_loop8_m_loop24_load_compute_data:
	ldr a14q, [ a_ptr_0, #32 ]
	ldp a00q, a10q, [ a_ptr_0 ], #48
	ldr a15q, [ a_ptr_1, #32 ]
	ldp a01q, a11q, [ a_ptr_1 ], #48
	ldp c0q, c1q, [ c_ptr_0 ]

	zip1 a02v.8h, a00v.8h, a01v.8h
	zip2 a03v.8h, a00v.8h, a01v.8h
	zip1 a12v.8h, a10v.8h, a11v.8h
	zip2 a13v.8h, a10v.8h, a11v.8h
	zip1 a04v.8h, a14v.8h, a15v.8h
	zip2 a05v.8h, a14v.8h, a15v.8h

	// Reload each A register as soon as its previous value is consumed.
	ldr a14q, [ a_ptr_2, #32 ]
	ldr a15q, [ a_ptr_3, #32 ]

	bfdot c00v.4s, a02v.8h, b0v.2h[0]
	bfdot c20v.4s, a12v.8h, b0v.2h[0]
	ldp a02q, a12q, [ a_ptr_2 ], #48

	bfdot c10v.4s, a03v.8h, b0v.2h[0]
	bfdot c30v.4s, a13v.8h, b0v.2h[0]
	ldp a03q, a13q, [ a_ptr_3 ], #48

	bfdot c40v.4s, a04v.8h, b0v.2h[0]
	bfdot c50v.4s, a05v.8h, b0v.2h[0]
	ldp c2q, c3q, [ c_ptr_0, #32 ]

	zip1 a00v.8h, a02v.8h, a03v.8h
	zip2 a01v.8h, a02v.8h, a03v.8h
	zip1 a10v.8h, a12v.8h, a13v.8h
	zip2 a11v.8h, a12v.8h, a13v.8h
	zip1 a04v.8h, a14v.8h, a15v.8h
	zip2 a05v.8h, a14v.8h, a15v.8h

	ldr a12q, [ a_ptr_4, #32 ]
	ldr a13q, [ a_ptr_5, #32 ]

	bfdot c00v.4s, a00v.8h, b0v.2h[1]
	bfdot c10v.4s, a01v.8h, b0v.2h[1]
	bfdot c20v.4s, a10v.8h, b0v.2h[1]
	bfdot c30v.4s, a11v.8h, b0v.2h[1]
	bfdot c40v.4s, a04v.8h, b0v.2h[1]
	ldp a04q, a14q, [ a_ptr_4 ], #48

	bfdot c50v.4s, a05v.8h, b0v.2h[1]
	ldp a05q, a15q, [ a_ptr_5 ], #48
	ldp c4q, c5q, [ c_ptr_0, #64 ]

	zip1 a00v.8h, a04v.8h, a05v.8h
	zip2 a01v.8h, a04v.8h, a05v.8h
	zip1 a10v.8h, a14v.8h, a15v.8h
	zip2 a11v.8h, a14v.8h, a15v.8h
	zip1 a02v.8h, a12v.8h, a13v.8h
	zip2 a03v.8h, a12v.8h, a13v.8h

	ldr a14q, [ a_ptr_6, #32 ]
	ldp a06q, a16q, [ a_ptr_6 ], #48
	bfdot c00v.4s, a00v.8h, b0v.2h[2]
	bfdot c10v.4s, a01v.8h, b0v.2h[2]

	ldr a15q, [ a_ptr_7, #32 ]
	ldp a07q, a17q, [ a_ptr_7 ], #48
	bfdot c20v.4s, a10v.8h, b0v.2h[2]
	bfdot c30v.4s, a11v.8h, b0v.2h[2]

	bfdot c40v.4s, a02v.8h, b0v.2h[2]
	bfdot c50v.4s, a03v.8h, b0v.2h[2]

	zip1 a00v.8h, a06v.8h, a07v.8h
	zip2 a01v.8h, a06v.8h, a07v.8h
	zip1 a10v.8h, a16v.8h, a17v.8h
	zip2 a11v.8h, a16v.8h, a17v.8h
	zip1 a02v.8h, a14v.8h, a15v.8h
	zip2 a03v.8h, a14v.8h, a15v.8h

	bfdot c00v.4s, a00v.8h, b0v.2h[3]
	bfdot c10v.4s, a01v.8h, b0v.2h[3]
	bfdot c20v.4s, a10v.8h, b0v.2h[3]
	bfdot c30v.4s, a11v.8h, b0v.2h[3]
	bfdot c40v.4s, a02v.8h, b0v.2h[3]
	bfdot c50v.4s, a03v.8h, b0v.2h[3]

.L${func_name}_n_loop8_m_loop24_scale_accumulate:
	fmla c0v.4s, c00v.4s, alpha_v.s[0]
	fmla c1v.4s, c10v.4s, alpha_v.s[0]
	fmla c2v.4s, c20v.4s, alpha_v.s[0]
	fmla c3v.4s, c30v.4s, alpha_v.s[0]
	fmla c4v.4s, c40v.4s, alpha_v.s[0]
	fmla c5v.4s, c50v.4s, alpha_v.s[0]

.L${func_name}_n_loop8_m_loop24_c_store_data:
	stp c0q, c1q, [ c_ptr_0 ]
	stp c2q, c3q, [ c_ptr_0, #32 ]
	stp c4q, c5q, [ c_ptr_0, #64 ]
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]*6}

.L${func_name}_n_loop8_m_loop24_end:
	add m_idx_next, m_idx, 24
	cmp m_idx_next, m
	bls .L${func_name}_n_loop8_m_loop24_start

	b .L${func_name}_n_loop8_m_loop16_end
% endif

.L${func_name}_n_loop8_m_loop16_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop8_m_loop16_zero_accumulators:
% if is_bgemv_n:
	movi c_a_0_hi_v.16b, #0
	movi c_b_0_hi_v.16b, #0
	movi c_a_1_hi_v.16b, #0
	movi c_b_1_hi_v.16b, #0
	movi c_a_0_lo_v.16b, #0
	movi c_b_0_lo_v.16b, #0
	movi c_a_1_lo_v.16b, #0
	movi c_b_1_lo_v.16b, #0
% elif is_sbgemv_n:
	movi c_a_0_hi_v.16b, #0
	movi c_a_1_hi_v.16b, #0
	movi c_a_0_lo_v.16b, #0
	movi c_a_1_lo_v.16b, #0
% endif

% if is_bgemv_n:
.L${func_name}_n_loop8_m_loop16_load_compute_data:
	ldp a00q, a10q, [ a_ptr_0 ], #32
	ldp a01q, a11q, [ a_ptr_1 ], #32

	ldp c0q, c2q, [ c_ptr_0 ]
	bic c3v.16b, c2v.16b, bf16_mask_v.16b
	bic c1v.16b, c0v.16b, bf16_mask_v.16b
	shl c2v.4s, c2v.4s, #16
	shl c0v.4s, c0v.4s, #16

	bfmlalt c_a_0_hi_v.4s, a00v.8h, b0v.h[0]
	bfmlalb c_a_0_lo_v.4s, a00v.8h, b0v.h[0]
	bfmlalt c_a_1_hi_v.4s, a10v.8h, b0v.h[0]
	bfmlalb c_a_1_lo_v.4s, a10v.8h, b0v.h[0]

	ldp a02q, a12q, [ a_ptr_2 ], #32
	bfmlalt c_b_0_hi_v.4s, a01v.8h, b0v.h[1]
	bfmlalb c_b_0_lo_v.4s, a01v.8h, b0v.h[1]
	bfmlalt c_b_1_hi_v.4s, a11v.8h, b0v.h[1]
	bfmlalb c_b_1_lo_v.4s, a11v.8h, b0v.h[1]

	ldp a03q, a13q, [ a_ptr_3 ], #32
	bfmlalt c_a_0_hi_v.4s, a02v.8h, b0v.h[2]
	bfmlalb c_a_0_lo_v.4s, a02v.8h, b0v.h[2]
	bfmlalt c_a_1_hi_v.4s, a12v.8h, b0v.h[2]
	bfmlalb c_a_1_lo_v.4s, a12v.8h, b0v.h[2]

	ldp a04q, a14q, [ a_ptr_4 ], #32
	bfmlalt c_b_0_hi_v.4s, a03v.8h, b0v.h[3]
	bfmlalb c_b_0_lo_v.4s, a03v.8h, b0v.h[3]
	bfmlalt c_b_1_hi_v.4s, a13v.8h, b0v.h[3]
	bfmlalb c_b_1_lo_v.4s, a13v.8h, b0v.h[3]

	ldp a05q, a15q, [ a_ptr_5 ], #32
	bfmlalt c_a_0_hi_v.4s, a04v.8h, b0v.h[4]
	bfmlalb c_a_0_lo_v.4s, a04v.8h, b0v.h[4]
	bfmlalt c_a_1_hi_v.4s, a14v.8h, b0v.h[4]
	bfmlalb c_a_1_lo_v.4s, a14v.8h, b0v.h[4]

	ldp a06q, a16q, [ a_ptr_6 ], #32
	bfmlalt c_b_0_hi_v.4s, a05v.8h, b0v.h[5]
	bfmlalb c_b_0_lo_v.4s, a05v.8h, b0v.h[5]
	bfmlalt c_b_1_hi_v.4s, a15v.8h, b0v.h[5]
	bfmlalb c_b_1_lo_v.4s, a15v.8h, b0v.h[5]

	ldp a07q, a17q, [ a_ptr_7 ], #32
	bfmlalt c_a_0_hi_v.4s, a06v.8h, b0v.h[6]
	bfmlalb c_a_0_lo_v.4s, a06v.8h, b0v.h[6]
	bfmlalt c_a_1_hi_v.4s, a16v.8h, b0v.h[6]
	bfmlalb c_a_1_lo_v.4s, a16v.8h, b0v.h[6]
	bfmlalt c_b_0_hi_v.4s, a07v.8h, b0v.h[7]
	bfmlalb c_b_0_lo_v.4s, a07v.8h, b0v.h[7]
	bfmlalt c_b_1_hi_v.4s, a17v.8h, b0v.h[7]
	bfmlalb c_b_1_lo_v.4s, a17v.8h, b0v.h[7]
% elif is_sbgemv_n:
.L${func_name}_n_loop8_m_loop16_load_compute_data:
	ldp a00q, a10q, [ a_ptr_0 ], #32
	ldp a01q, a11q, [ a_ptr_1 ], #32
	${load_vector(opts["out"], "c0", "c_ptr_0", 0)}

	zip1 a02v.8h, a00v.8h, a01v.8h
	zip2 a03v.8h, a00v.8h, a01v.8h
	zip1 a12v.8h, a10v.8h, a11v.8h
	zip2 a13v.8h, a10v.8h, a11v.8h

	bfdot c_a_0_lo_v.4s, a02v.8h, b0v.2h[0]
	bfdot c_a_0_hi_v.4s, a03v.8h, b0v.2h[0]
	bfdot c_a_1_lo_v.4s, a12v.8h, b0v.2h[0]
	bfdot c_a_1_hi_v.4s, a13v.8h, b0v.2h[0]

	ldp a02q, a12q, [ a_ptr_2 ], #32
	ldp a03q, a13q, [ a_ptr_3 ], #32
	${load_vector(opts["out"], "c1", "c_ptr_0", 1)}

	zip1 a00v.8h, a02v.8h, a03v.8h
	zip2 a01v.8h, a02v.8h, a03v.8h
	zip1 a10v.8h, a12v.8h, a13v.8h
	zip2 a11v.8h, a12v.8h, a13v.8h

	bfdot c_a_0_lo_v.4s, a00v.8h, b0v.2h[1]
	bfdot c_a_0_hi_v.4s, a01v.8h, b0v.2h[1]
	bfdot c_a_1_lo_v.4s, a10v.8h, b0v.2h[1]
	bfdot c_a_1_hi_v.4s, a11v.8h, b0v.2h[1]

	ldp a04q, a14q, [ a_ptr_4 ], #32
	ldp a05q, a15q, [ a_ptr_5 ], #32
	${load_vector(opts["out"], "c2", "c_ptr_0", 2)}

	zip1 a00v.8h, a04v.8h, a05v.8h
	zip2 a01v.8h, a04v.8h, a05v.8h
	zip1 a10v.8h, a14v.8h, a15v.8h
	zip2 a11v.8h, a14v.8h, a15v.8h

	bfdot c_a_0_lo_v.4s, a00v.8h, b0v.2h[2]
	bfdot c_a_0_hi_v.4s, a01v.8h, b0v.2h[2]
	bfdot c_a_1_lo_v.4s, a10v.8h, b0v.2h[2]
	bfdot c_a_1_hi_v.4s, a11v.8h, b0v.2h[2]

	ldp a06q, a16q, [ a_ptr_6 ], #32
	ldp a07q, a17q, [ a_ptr_7 ], #32
	${load_vector(opts["out"], "c3", "c_ptr_0", 3)}

	zip1 a00v.8h, a06v.8h, a07v.8h
	zip2 a01v.8h, a06v.8h, a07v.8h
	zip1 a10v.8h, a16v.8h, a17v.8h
	zip2 a11v.8h, a16v.8h, a17v.8h

	bfdot c_a_0_lo_v.4s, a00v.8h, b0v.2h[3]
	bfdot c_a_0_hi_v.4s, a01v.8h, b0v.2h[3]
	bfdot c_a_1_lo_v.4s, a10v.8h, b0v.2h[3]
	bfdot c_a_1_hi_v.4s, a11v.8h, b0v.2h[3]
% endif

.L${func_name}_n_loop8_m_loop16_shuffle_reduce:
% if is_bgemv_n:
	fmla c0v.4s, c_a_0_lo_v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]
	fmla c2v.4s, c_a_1_lo_v.4s, alpha_v.s[0]
	fmla c3v.4s, c_a_1_hi_v.4s, alpha_v.s[0]

	fmla c0v.4s, c_b_0_lo_v.4s, alpha_v.s[0]
	fmla c1v.4s, c_b_0_hi_v.4s, alpha_v.s[0]
	fmla c2v.4s, c_b_1_lo_v.4s, alpha_v.s[0]
	fmla c3v.4s, c_b_1_hi_v.4s, alpha_v.s[0]

	trn2 c0v.8h, c0v.8h, c1v.8h
	trn2 c2v.8h, c2v.8h, c3v.8h
% elif is_sbgemv_n:
	fmla c0v.4s, c_a_0_lo_v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]
	fmla c2v.4s, c_a_1_lo_v.4s, alpha_v.s[0]
	fmla c3v.4s, c_a_1_hi_v.4s, alpha_v.s[0]
% endif

.L${func_name}_n_loop8_m_loop16_c_store_data:
% if is_bgemv_n:
	stp c0q, c2q, [ c_ptr_0 ], #${opts["out"]["vector_load_bytes"]*4}
% else:
	stp c0q, c1q, [ c_ptr_0 ]
	stp c2q, c3q, [ c_ptr_0, #32 ]
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]*4}
% endif

.L${func_name}_n_loop8_m_loop16_end:
	add m_idx_next, m_idx, 16
	cmp m_idx_next, m
	bls .L${func_name}_n_loop8_m_loop16_start

	b .L${func_name}_n_loop8_m_loop8_end

.L${func_name}_n_loop8_m_loop8_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop8_m_loop8_zero_accumulators:
	movi c_a_0_hi_v.16b, #0
	movi c_a_0_lo_v.16b, #0
% if is_bgemv_n:
	movi c_b_0_hi_v.16b, #0
	movi c_b_0_lo_v.16b, #0
% endif

% if is_bgemv_n:
.L${func_name}_n_loop8_m_loop8_load_compute_data:
	ldr a00q, [ a_ptr_0 ], #16
	ldr a01q, [ a_ptr_1 ], #16

	ldr c0q, [ c_ptr_0 ]
	bic c1v.16b, c0v.16b, bf16_mask_v.16b
	shl c0v.4s, c0v.4s, #16

	ldr a02q, [ a_ptr_2 ], #16
	bfmlalt c_a_0_hi_v.4s, a00v.8h, b0v.h[0]
	bfmlalb c_a_0_lo_v.4s, a00v.8h, b0v.h[0]

	ldr a03q, [ a_ptr_3 ], #16
	bfmlalt c_b_0_hi_v.4s, a01v.8h, b0v.h[1]
	bfmlalb c_b_0_lo_v.4s, a01v.8h, b0v.h[1]

	ldr a04q, [ a_ptr_4 ], #16
	bfmlalt c_a_0_hi_v.4s, a02v.8h, b0v.h[2]
	bfmlalb c_a_0_lo_v.4s, a02v.8h, b0v.h[2]

	ldr a05q, [ a_ptr_5 ], #16
	bfmlalt c_b_0_hi_v.4s, a03v.8h, b0v.h[3]
	bfmlalb c_b_0_lo_v.4s, a03v.8h, b0v.h[3]

	ldr a06q, [ a_ptr_6 ], #16
	bfmlalt c_a_0_hi_v.4s, a04v.8h, b0v.h[4]
	bfmlalb c_a_0_lo_v.4s, a04v.8h, b0v.h[4]

	ldr a07q, [ a_ptr_7 ], #16
	bfmlalt c_b_0_hi_v.4s, a05v.8h, b0v.h[5]
	bfmlalb c_b_0_lo_v.4s, a05v.8h, b0v.h[5]

	bfmlalt c_a_0_hi_v.4s, a06v.8h, b0v.h[6]
	bfmlalb c_a_0_lo_v.4s, a06v.8h, b0v.h[6]
	bfmlalt c_b_0_hi_v.4s, a07v.8h, b0v.h[7]
	bfmlalb c_b_0_lo_v.4s, a07v.8h, b0v.h[7]
% elif is_sbgemv_n:
.L${func_name}_n_loop8_m_loop8_load_compute_data:
	ldr a00q, [ a_ptr_0 ], #16
	ldr a01q, [ a_ptr_1 ], #16

	zip1 a10v.8h, a00v.8h, a01v.8h
	zip2 a11v.8h, a00v.8h, a01v.8h

	ldr a02q, [ a_ptr_2 ], #16
	ldr a03q, [ a_ptr_3 ], #16
	${load_vector(opts["out"], "c0", "c_ptr_0", 0)}

	bfdot c_a_0_lo_v.4s, a10v.8h, b0v.2h[0]
	bfdot c_a_0_hi_v.4s, a11v.8h, b0v.2h[0]

	zip1 a12v.8h, a02v.8h, a03v.8h
	zip2 a13v.8h, a02v.8h, a03v.8h

	ldr a04q, [ a_ptr_4 ], #16
	ldr a05q, [ a_ptr_5 ], #16
	${load_vector(opts["out"], "c1", "c_ptr_0", 1)}

	bfdot c_a_0_lo_v.4s, a12v.8h, b0v.2h[1]
	bfdot c_a_0_hi_v.4s, a13v.8h, b0v.2h[1]

	zip1 a14v.8h, a04v.8h, a05v.8h
	zip2 a15v.8h, a04v.8h, a05v.8h

	ldr a06q, [ a_ptr_6 ], #16
	ldr a07q, [ a_ptr_7 ], #16

	bfdot c_a_0_lo_v.4s, a14v.8h, b0v.2h[2]
	bfdot c_a_0_hi_v.4s, a15v.8h, b0v.2h[2]

	zip1 a16v.8h, a06v.8h, a07v.8h
	zip2 a17v.8h, a06v.8h, a07v.8h

	bfdot c_a_0_lo_v.4s, a16v.8h, b0v.2h[3]
	bfdot c_a_0_hi_v.4s, a17v.8h, b0v.2h[3]
% endif

.L${func_name}_n_loop8_m_loop8_shuffle_reduce:
% if is_bgemv_n:
	fmla c0v.4s, c_a_0_lo_v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]

	fmla c0v.4s, c_b_0_lo_v.4s, alpha_v.s[0]
	fmla c1v.4s, c_b_0_hi_v.4s, alpha_v.s[0]

	trn2 c0v.8h, c0v.8h, c1v.8h
% elif is_sbgemv_n:
	fmla c0v.4s, c_a_0_lo_v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]
% endif

.L${func_name}_n_loop8_m_loop8_c_store_data:
% if is_bgemv_n:
	str c0q, [ c_ptr_0 ], #${opts["out"]["vector_load_bytes"]*2}
% else:
	stp c0q, c1q, [ c_ptr_0 ]
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]*2}
% endif

.L${func_name}_n_loop8_m_loop8_end:
	add m_idx_next, m_idx, 8
	cmp m_idx_next, m
	bls .L${func_name}_n_loop8_m_loop8_start

% if is_sbgemv_n:
	b .L${func_name}_n_loop8_m_loop4_end

.L${func_name}_n_loop8_m_loop4_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop8_m_loop4_zero_accumulators:
	movi c_a_0_lo_v.16b, #0

.L${func_name}_n_loop8_m_loop4_load_compute_data:
	ldr a00d, [ a_ptr_0 ], #8
	ldr a01d, [ a_ptr_1 ], #8
	zip1 a10v.8h, a00v.8h, a01v.8h

	ldr a02d, [ a_ptr_2 ], #8
	ldr a03d, [ a_ptr_3 ], #8
	${load_vector(opts["out"], "c0", "c_ptr_0", 0)}

	bfdot c_a_0_lo_v.4s, a10v.8h, b0v.2h[0]

	zip1 a11v.8h, a02v.8h, a03v.8h

	ldr a04d, [ a_ptr_4 ], #8
	ldr a05d, [ a_ptr_5 ], #8

	bfdot c_a_0_lo_v.4s, a11v.8h, b0v.2h[1]

	zip1 a12v.8h, a04v.8h, a05v.8h

	ldr a06d, [ a_ptr_6 ], #8
	ldr a07d, [ a_ptr_7 ], #8

	bfdot c_a_0_lo_v.4s, a12v.8h, b0v.2h[2]

	zip1 a13v.8h, a06v.8h, a07v.8h

	bfdot c_a_0_lo_v.4s, a13v.8h, b0v.2h[3]

.L${func_name}_n_loop8_m_loop4_scale_accumulate:
	fmla c0v.4s, c_a_0_lo_v.4s, alpha_v.s[0]

.L${func_name}_n_loop8_m_loop4_c_store_data:
	${store_vector(opts["out"], "c0", "c_ptr_0", 0)}
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]}

.L${func_name}_n_loop8_m_loop4_end:
	add m_idx_next, m_idx, 4
	cmp m_idx_next, m
	bls .L${func_name}_n_loop8_m_loop4_start

% endif
	b .L${func_name}_n_loop8_m_loop1_end

.L${func_name}_n_loop8_m_loop1_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop8_m_loop1_zero_accumulators:
	movi c_a_0_lo_v.16b, #0
	movi c_b_0_lo_v.16b, #0

.L${func_name}_n_loop8_m_loop1_c_load_data:
	${load_element(opts["out"], "c0", "c_ptr_0")}

.L${func_name}_n_loop8_m_loop1_load_compute_data:
	ldr a00h, [ a_ptr_0 ], #2
	ldr a01h, [ a_ptr_1 ], #2
	zip1 a10v.8h, a00v.8h, a01v.8h

	ldr a02h, [ a_ptr_2 ], #2
	ldr a03h, [ a_ptr_3 ], #2
	bfdot c_a_0_lo_v.4s, a10v.8h, b0v.2h[0]

	zip1 a11v.8h, a02v.8h, a03v.8h

	ldr a04h, [ a_ptr_4 ], #2
	ldr a05h, [ a_ptr_5 ], #2
	bfdot c_b_0_lo_v.4s, a11v.8h, b0v.2h[1]

	zip1 a12v.8h, a04v.8h, a05v.8h

	ldr a06h, [ a_ptr_6 ], #2
	ldr a07h, [ a_ptr_7 ], #2
	bfdot c_a_0_lo_v.4s, a12v.8h, b0v.2h[2]

	zip1 a13v.8h, a06v.8h, a07v.8h
	bfdot c_b_0_lo_v.4s, a13v.8h, b0v.2h[3]

.L${func_name}_n_loop8_m_loop1_shuffle_reduce:
	fadd c_a_0_lo_s, c_a_0_lo_s, c_b_0_lo_s
	fmadd c0s, c_a_0_lo_s, alpha_s, c0s

.L${func_name}_n_loop8_m_loop1_c_store_data:
% if is_bgemv_n:
	${store_element(opts["out"], "c0", "c_ptr_0")}
	add c_ptr_0, c_ptr_0, ${opts["out"]["bytes"]}
% else:
	str c0s, [ c_ptr_0 ], #4
% endif

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
	movi c_a_1_hi_v.16b, #0
	movi c_a_0_lo_v.16b, #0
	movi c_a_1_lo_v.16b, #0
% if is_bgemv_n:
	movi c_b_0_hi_v.16b, #0
	movi c_b_1_hi_v.16b, #0
	movi c_b_0_lo_v.16b, #0
	movi c_b_1_lo_v.16b, #0
% endif

% if is_bgemv_n:
.L${func_name}_n_loop4_m_loop16_load_compute_data:
	ldp a00q, a10q, [ a_ptr_0 ], #32
	ldp a01q, a11q, [ a_ptr_1 ], #32

	ldr c0q, [ c_ptr_0 ], #16
	bic c1v.16b, c0v.16b, bf16_mask_v.16b
	shl c0v.4s, c0v.4s, #16

	bfmlalt c_a_0_hi_v.4s, a00v.8h, b0v.h[0]
	bfmlalb c_a_0_lo_v.4s, a00v.8h, b0v.h[0]
	bfmlalt c_a_1_hi_v.4s, a10v.8h, b0v.h[0]
	bfmlalb c_a_1_lo_v.4s, a10v.8h, b0v.h[0]

	ldp a02q, a12q, [ a_ptr_2 ], #32
	ldr c2q, [ c_ptr_0 ]
	bic c3v.16b, c2v.16b, bf16_mask_v.16b
	shl c2v.4s, c2v.4s, #16

	bfmlalt c_b_0_hi_v.4s, a01v.8h, b0v.h[1]
	bfmlalb c_b_0_lo_v.4s, a01v.8h, b0v.h[1]
	bfmlalt c_b_1_hi_v.4s, a11v.8h, b0v.h[1]
	bfmlalb c_b_1_lo_v.4s, a11v.8h, b0v.h[1]

	ldp a03q, a13q, [ a_ptr_3 ], #32
	bfmlalt c_a_0_hi_v.4s, a02v.8h, b0v.h[2]
	bfmlalb c_a_0_lo_v.4s, a02v.8h, b0v.h[2]
	bfmlalt c_a_1_hi_v.4s, a12v.8h, b0v.h[2]
	bfmlalb c_a_1_lo_v.4s, a12v.8h, b0v.h[2]

	bfmlalt c_b_0_hi_v.4s, a03v.8h, b0v.h[3]
	bfmlalb c_b_0_lo_v.4s, a03v.8h, b0v.h[3]
	bfmlalt c_b_1_hi_v.4s, a13v.8h, b0v.h[3]
	bfmlalb c_b_1_lo_v.4s, a13v.8h, b0v.h[3]
% else:
.L${func_name}_n_loop4_m_loop16_load_compute_data:
	${load_vector(opts["out"], "c0", "c_ptr_0", 0)}
	${load_vector(opts["out"], "c1", "c_ptr_0", 1)}
	ldp a00q, a10q, [ a_ptr_0 ], #32
	ldp a01q, a11q, [ a_ptr_1 ], #32

	zip1 a04v.8h, a00v.8h, a01v.8h
	zip2 a05v.8h, a00v.8h, a01v.8h
	zip1 a14v.8h, a10v.8h, a11v.8h
	zip2 a15v.8h, a10v.8h, a11v.8h

	${load_vector(opts["out"], "c2", "c_ptr_0", 2)}
	${load_vector(opts["out"], "c3", "c_ptr_0", 3)}
	ldp a02q, a12q, [ a_ptr_2 ], #32
	ldp a03q, a13q, [ a_ptr_3 ], #32

	bfdot c_a_0_lo_v.4s, a04v.8h, b0v.2h[0]
	bfdot c_a_0_hi_v.4s, a05v.8h, b0v.2h[0]
	bfdot c_a_1_lo_v.4s, a14v.8h, b0v.2h[0]
	bfdot c_a_1_hi_v.4s, a15v.8h, b0v.2h[0]

	zip1 a04v.8h, a02v.8h, a03v.8h
	zip2 a05v.8h, a02v.8h, a03v.8h
	zip1 a14v.8h, a12v.8h, a13v.8h
	zip2 a15v.8h, a12v.8h, a13v.8h

	bfdot c_a_0_lo_v.4s, a04v.8h, b0v.2h[1]
	bfdot c_a_0_hi_v.4s, a05v.8h, b0v.2h[1]
	bfdot c_a_1_lo_v.4s, a14v.8h, b0v.2h[1]
	bfdot c_a_1_hi_v.4s, a15v.8h, b0v.2h[1]
% endif

.L${func_name}_n_loop4_m_loop16_shuffle_reduce:
% if is_bgemv_n:
	fmla c0v.4s, c_a_0_lo_v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]
	fmla c2v.4s, c_a_1_lo_v.4s, alpha_v.s[0]
	fmla c3v.4s, c_a_1_hi_v.4s, alpha_v.s[0]

	fmla c0v.4s, c_b_0_lo_v.4s, alpha_v.s[0]
	fmla c1v.4s, c_b_0_hi_v.4s, alpha_v.s[0]
	fmla c2v.4s, c_b_1_lo_v.4s, alpha_v.s[0]
	fmla c3v.4s, c_b_1_hi_v.4s, alpha_v.s[0]

	trn2 c0v.8h, c0v.8h, c1v.8h
	trn2 c2v.8h, c2v.8h, c3v.8h
% else:
	fmla c0v.4s, c_a_0_lo_v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]
	fmla c2v.4s, c_a_1_lo_v.4s, alpha_v.s[0]
	fmla c3v.4s, c_a_1_hi_v.4s, alpha_v.s[0]
% endif

.L${func_name}_n_loop4_m_loop16_c_store_data:
% if is_bgemv_n:
	stp c0q, c2q, [ c_ptr_0, #-${opts["out"]["vector_load_bytes"]*2} ]
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]*2}
% else:
	stp c0q, c1q, [ c_ptr_0 ]
	stp c2q, c3q, [ c_ptr_0, #32 ]
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]*4}
% endif

.L${func_name}_n_loop4_m_loop16_end:
	add m_idx_next, m_idx, 16
	cmp m_idx_next, m
	bls .L${func_name}_n_loop4_m_loop16_start

	b .L${func_name}_n_loop4_m_loop8_end


.L${func_name}_n_loop4_m_loop8_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop4_m_loop8_zero_accumulators:
	movi c_a_0_hi_v.16b, #0
	movi c_a_0_lo_v.16b, #0
% if is_bgemv_n:
	movi c_b_0_hi_v.16b, #0
	movi c_b_0_lo_v.16b, #0
% endif

.L${func_name}_n_loop4_m_loop8_c_load_data:
% if is_bgemv_n:
	ldr c0q, [ c_ptr_0 ]
	bic c1v.16b, c0v.16b, bf16_mask_v.16b
	shl c0v.4s, c0v.4s, #16
% else:
	${load_vector(opts["out"], "c0", "c_ptr_0", 0)}
	${load_vector(opts["out"], "c1", "c_ptr_0", 1)}
% endif

% if is_bgemv_n:
.L${func_name}_n_loop4_m_loop8_load_compute_data:
	ldr a00q, [ a_ptr_0 ]
	add a_ptr_0, a_ptr_0, 16

	ldr a01q, [ a_ptr_1 ]
	add a_ptr_1, a_ptr_1, 16

	ldr a02q, [ a_ptr_2 ]
	add a_ptr_2, a_ptr_2, 16

	ldr a03q, [ a_ptr_3 ]
	add a_ptr_3, a_ptr_3, 16

	bfmlalt c_a_0_hi_v.4s, a00v.8h, b0v.h[0]
	bfmlalb c_a_0_lo_v.4s, a00v.8h, b0v.h[0]

	bfmlalt c_b_0_hi_v.4s, a01v.8h, b0v.h[1]
	bfmlalb c_b_0_lo_v.4s, a01v.8h, b0v.h[1]

	bfmlalt c_a_0_hi_v.4s, a02v.8h, b0v.h[2]
	bfmlalb c_a_0_lo_v.4s, a02v.8h, b0v.h[2]

	bfmlalt c_b_0_hi_v.4s, a03v.8h, b0v.h[3]
	bfmlalb c_b_0_lo_v.4s, a03v.8h, b0v.h[3]
% else:
.L${func_name}_n_loop4_m_loop8_load_compute_data:
	ldr a00q, [ a_ptr_0 ], #16
	ldr a01q, [ a_ptr_1 ], #16
	zip1 a04v.8h, a00v.8h, a01v.8h
	zip2 a05v.8h, a00v.8h, a01v.8h

	ldr a02q, [ a_ptr_2 ], #16
	ldr a03q, [ a_ptr_3 ], #16

	bfdot c_a_0_lo_v.4s, a04v.8h, b0v.2h[0]
	bfdot c_a_0_hi_v.4s, a05v.8h, b0v.2h[0]

	zip1 a06v.8h, a02v.8h, a03v.8h
	zip2 a07v.8h, a02v.8h, a03v.8h

	bfdot c_a_0_lo_v.4s, a06v.8h, b0v.2h[1]
	bfdot c_a_0_hi_v.4s, a07v.8h, b0v.2h[1]
% endif

.L${func_name}_n_loop4_m_loop8_shuffle_reduce:
% if is_bgemv_n:
	fmla c0v.4s, c_a_0_lo_v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]

	fmla c0v.4s, c_b_0_lo_v.4s, alpha_v.s[0]
	fmla c1v.4s, c_b_0_hi_v.4s, alpha_v.s[0]

	trn2 c0v.8h, c0v.8h, c1v.8h
% else:
	fmla c0v.4s, c_a_0_lo_v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]
% endif

.L${func_name}_n_loop4_m_loop8_c_store_data:
% if is_bgemv_n:
	str c0q, [ c_ptr_0 ]
% else:
	stp c0q, c1q, [ c_ptr_0 ]
% endif
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]*2}

.L${func_name}_n_loop4_m_loop8_end:
	add m_idx_next, m_idx, 8
	cmp m_idx_next, m
	bls .L${func_name}_n_loop4_m_loop8_start

% if is_sbgemv_n:
	b .L${func_name}_n_loop4_m_loop4_end

.L${func_name}_n_loop4_m_loop4_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop4_m_loop4_zero_accumulators:
	movi c_a_0_lo_v.16b, #0

.L${func_name}_n_loop4_m_loop4_load_compute_data:
	ldr a00d, [ a_ptr_0 ], #8
	ldr a01d, [ a_ptr_1 ], #8
	zip1 a10v.8h, a00v.8h, a01v.8h

	${load_vector(opts["out"], "c0", "c_ptr_0", 0)}

	ldr a02d, [ a_ptr_2 ], #8
	ldr a03d, [ a_ptr_3 ], #8

	bfdot c_a_0_lo_v.4s, a10v.8h, b0v.2h[0]

	zip1 a11v.8h, a02v.8h, a03v.8h

	bfdot c_a_0_lo_v.4s, a11v.8h, b0v.2h[1]

.L${func_name}_n_loop4_m_loop4_scale_accumulate:
	fmla c0v.4s, c_a_0_lo_v.4s, alpha_v.s[0]

.L${func_name}_n_loop4_m_loop4_c_store_data:
	${store_vector(opts["out"], "c0", "c_ptr_0", 0)}
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]}

.L${func_name}_n_loop4_m_loop4_end:
	add m_idx_next, m_idx, 4
	cmp m_idx_next, m
	bls .L${func_name}_n_loop4_m_loop4_start

% endif
	b .L${func_name}_n_loop4_m_loop1_end


.L${func_name}_n_loop4_m_loop1_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop4_m_loop1_zero_accumulators:
	movi c_a_0_lo_v.16b, #0
	movi c_b_0_lo_v.16b, #0

.L${func_name}_n_loop4_m_loop1_c_load_data:
	${load_element(opts["out"], "c0", "c_ptr_0")}

.L${func_name}_n_loop4_m_loop1_a_load_data:
% if is_bgemv_n:
	ldr a00h, [ a_ptr_0 ]
	add a_ptr_0, a_ptr_0, 2

	ldr a01h, [ a_ptr_1 ]
	add a_ptr_1, a_ptr_1, 2

	ldr a02h, [ a_ptr_2 ]
	add a_ptr_2, a_ptr_2, 2

	ldr a03h, [ a_ptr_3 ]
	add a_ptr_3, a_ptr_3, 2
% else:
	ld1 { a10v.h }[0], [ a_ptr_0 ], #2
	ld1 { a10v.h }[1], [ a_ptr_1 ], #2
	ld1 { a11v.h }[0], [ a_ptr_2 ], #2
	ld1 { a11v.h }[1], [ a_ptr_3 ], #2
% endif

.L${func_name}_n_loop4_m_loop1_compute:
% if is_bgemv_n:
	bfmlalb c_a_0_lo_v.4s, a00v.8h, b0v.h[0]
	bfmlalb c_b_0_lo_v.4s, a01v.8h, b0v.h[1]
	bfmlalb c_a_0_lo_v.4s, a02v.8h, b0v.h[2]
	bfmlalb c_b_0_lo_v.4s, a03v.8h, b0v.h[3]
% else:
	bfdot c_a_0_lo_v.4s, a10v.8h, b0v.2h[0]
	bfdot c_b_0_lo_v.4s, a11v.8h, b0v.2h[1]
% endif

.L${func_name}_n_loop4_m_loop1_shuffle_reduce:
	fadd c_a_0_lo_s, c_a_0_lo_s, c_b_0_lo_s
	fmadd c0s, c_a_0_lo_s, alpha_s, c0s

.L${func_name}_n_loop4_m_loop1_c_store_data:
% if is_bgemv_n:
	${store_element(opts["out"], "c0", "c_ptr_0")}
	add c_ptr_0, c_ptr_0, ${opts["out"]["bytes"]}
% else:
	str c0s, [ c_ptr_0 ], #4
% endif

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
	movi v30.16b, #0
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
% if is_bgemv_n:
	ldp c0q, c2q, [ c_ptr_0 ]
	bic c3v.16b, c2v.16b, bf16_mask_v.16b
	bic c1v.16b, c0v.16b, bf16_mask_v.16b
	shl c2v.4s, c2v.4s, #16
	shl c0v.4s, c0v.4s, #16
% else:
	${load_vector(opts["out"], "c0", "c_ptr_0", 0)}
	${load_vector(opts["out"], "c1", "c_ptr_0", 1)}
	${load_vector(opts["out"], "c2", "c_ptr_0", 2)}
	${load_vector(opts["out"], "c3", "c_ptr_0", 3)}
% endif

.L${func_name}_n_loop1_m_loop16_a_load_data:
% if is_bgemv_n:
	ldp a00q, a10q, [ a_ptr_0 ], #32
% else:
	ldr a00q, [ a_ptr_0 ]
	ldr a10q, [ a_ptr_0, #16 ]
	add a_ptr_0, a_ptr_0, 32
% endif

.L${func_name}_n_loop1_m_loop16_compute:
	bfmlalt c_a_0_hi_v.4s, a00v.8h, b0v.h[0]
	bfmlalb c_a_0_lo_v.4s, a00v.8h, b0v.h[0]

	bfmlalt c_a_1_hi_v.4s, a10v.8h, b0v.h[0]
	bfmlalb c_a_1_lo_v.4s, a10v.8h, b0v.h[0]

.L${func_name}_n_loop1_m_loop16_shuffle_reduce:
% if is_bgemv_n:
	fmla c0v.4s, c_a_0_lo_v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]
	fmla c2v.4s, c_a_1_lo_v.4s, alpha_v.s[0]
	fmla c3v.4s, c_a_1_hi_v.4s, alpha_v.s[0]

	trn2 c0v.8h, c0v.8h, c1v.8h
	trn2 c2v.8h, c2v.8h, c3v.8h
% else:
	zip1 a00v.4s,      c_a_0_lo_v.4s, c_a_0_hi_v.4s
	zip2 c_a_0_hi_v.4s, c_a_0_lo_v.4s, c_a_0_hi_v.4s

	fmla c0v.4s,      a00v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]

	zip1 a10v.4s,      c_a_1_lo_v.4s, c_a_1_hi_v.4s
	zip2 c_a_1_hi_v.4s, c_a_1_lo_v.4s, c_a_1_hi_v.4s

	fmla c2v.4s,      a10v.4s, alpha_v.s[0]
	fmla c3v.4s, c_a_1_hi_v.4s, alpha_v.s[0]
% endif

.L${func_name}_n_loop1_m_loop16_c_store_data:
% if is_bgemv_n:
	stp c0q, c2q, [ c_ptr_0 ], #${opts["out"]["vector_load_bytes"]*4}
% else:
	stp c0q, c1q, [ c_ptr_0 ]
	stp c2q, c3q, [ c_ptr_0, #32 ]
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]*4}
% endif

.L${func_name}_n_loop1_m_loop16_end:
	add m_idx_next, m_idx, 16
	cmp m_idx_next, m
	bls .L${func_name}_n_loop1_m_loop16_start

	b .L${func_name}_n_loop1_m_loop8_end


.L${func_name}_n_loop1_m_loop8_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop1_m_loop8_zero_accumulators:
	movi c_a_0_hi_v.16b, #0
	movi c_a_0_lo_v.16b, #0

.L${func_name}_n_loop1_m_loop8_c_load_data:
% if is_bgemv_n:
	ldr c0q, [ c_ptr_0 ]
	bic c1v.16b, c0v.16b, bf16_mask_v.16b
	shl c0v.4s, c0v.4s, #16
% else:
	${load_vector(opts["out"], "c0", "c_ptr_0", 0)}
	${load_vector(opts["out"], "c1", "c_ptr_0", 1)}
% endif

.L${func_name}_n_loop1_m_loop8_a_load_data:
	ldr a00q, [ a_ptr_0 ]
	add a_ptr_0, a_ptr_0, 16

.L${func_name}_n_loop1_m_loop8_compute:
	bfmlalt c_a_0_hi_v.4s, a00v.8h, b0v.h[0]
	bfmlalb c_a_0_lo_v.4s, a00v.8h, b0v.h[0]

.L${func_name}_n_loop1_m_loop8_shuffle_reduce:
% if is_bgemv_n:
	fmla c0v.4s, c_a_0_lo_v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]

	trn2 c0v.8h, c0v.8h, c1v.8h
% else:
	zip1 a00v.4s,      c_a_0_lo_v.4s, c_a_0_hi_v.4s
	zip2 c_a_0_hi_v.4s, c_a_0_lo_v.4s, c_a_0_hi_v.4s

	fmla c0v.4s,      a00v.4s, alpha_v.s[0]
	fmla c1v.4s, c_a_0_hi_v.4s, alpha_v.s[0]
% endif

.L${func_name}_n_loop1_m_loop8_c_store_data:
% if is_bgemv_n:
	str c0q, [ c_ptr_0 ], #${opts["out"]["vector_load_bytes"]*2}
% else:
	stp c0q, c1q, [ c_ptr_0 ]
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]*2}
% endif

.L${func_name}_n_loop1_m_loop8_end:
	add m_idx_next, m_idx, 8
	cmp m_idx_next, m
	bls .L${func_name}_n_loop1_m_loop8_start

% if is_sbgemv_n:
	b .L${func_name}_n_loop1_m_loop4_end

.L${func_name}_n_loop1_m_loop4_start:
	mov m_idx, m_idx_next

.L${func_name}_n_loop1_m_loop4_zero_accumulators:
	movi c_a_0_lo_v.16b, #0

.L${func_name}_n_loop1_m_loop4_load_compute_data:
	ldr a00d, [ a_ptr_0 ], #8
	${load_vector(opts["out"], "c0", "c_ptr_0", 0)}

	zip1 a01v.8h, a00v.8h, v30.8h
	bfdot c_a_0_lo_v.4s, a01v.8h, b0v.2h[0]

.L${func_name}_n_loop1_m_loop4_shuffle_reduce:
	fmla c0v.4s, c_a_0_lo_v.4s, alpha_v.s[0]

.L${func_name}_n_loop1_m_loop4_c_store_data:
	${store_vector(opts["out"], "c0", "c_ptr_0", 0)}
	add c_ptr_0, c_ptr_0, ${opts["out"]["vector_load_bytes"]}

.L${func_name}_n_loop1_m_loop4_end:
	add m_idx_next, m_idx, 4
	cmp m_idx_next, m
	bls .L${func_name}_n_loop1_m_loop4_start

% endif
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
	zip1 a01v.8h, a00v.8h, v30.8h
	bfdot c_a_0_lo_v.4s, a01v.8h, b0v.2h[0]

.L${func_name}_n_loop1_m_loop1_shuffle_reduce:
	fmadd c0s, c_a_0_lo_s, alpha_s, c0s

.L${func_name}_n_loop1_m_loop1_c_store_data:
% if is_bgemv_n:
	${store_element(opts["out"], "c0", "c_ptr_0")}
	add c_ptr_0, c_ptr_0, ${opts["out"]["bytes"]}
% else:
	str c0s, [ c_ptr_0 ], #4
% endif

.L${func_name}_n_loop1_m_loop1_end:
	add m_idx_next, m_idx, 1
	cmp m_idx_next, m
	bls .L${func_name}_n_loop1_m_loop1_start

.L${func_name}_n_loop1_end:
	add n_idx_next, n_idx, 1
	cmp n_idx_next, n
	bls .L${func_name}_n_loop1_start

.L${func_name}_epilogue:
	ldp     d14, d15, [sp, #144]
	ldp     d12, d13, [sp, #128]
	ldp     d10, d11, [sp, #112]
	ldp     d8,  d9,  [sp, #96]

	ldp     x27, x28, [sp, #80]
	ldp     x25, x26, [sp, #64]
	ldp     x23, x24, [sp, #48]
	ldp     x21, x22, [sp, #32]
	ldp     x19, x20, [sp, #16]

	ldp     x29, x30, [sp], #160

	ret
${epilogue(func_name)}
% endfor

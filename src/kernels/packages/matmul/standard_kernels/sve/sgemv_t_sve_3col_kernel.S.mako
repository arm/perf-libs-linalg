## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

## FP32 transposed GEMV with three separated output-column streams.
## For q = output_count / 3, each iteration computes columns j, j + q, and j + 2*q.
## The remaining zero, one, or two columns are computed individually.

<%
func_name = "sgemv_t_sve_3col_kernel"
# Fixed at three: register allocation and stream-address setup below assume
# three columns. This constant only generates their repeated instructions.
num_columns = 3
%>

REDUCTION_LEN       .req x0  // a_cntg: shared by A and X.
OUTPUT_COUNT        .req x1  // a_strd: shared by A and Y.
A_ADDR_0            .req x2
LDA                 .req x4  // a_strd_step
X                   .req x5
FULL_REDUCTION_LEN  .req x3  // a_cntg_step is known to be one; reuse its register.
Y_ADDR_0            .req x7
INC_Y               .req x6  // x_cntg_step is known to be one; reuse its register.
ROW                 .req x8
GROUPS              .req x9
Y_ADDR_1            .req x10
Y_ADDR_2            .req x11
A_ADDR_1            .req x12
A_ADDR_2            .req x13
VEC_SIZE            .req x14
REMAINING           .req x15
ISCRATCH            .req x16

ALPHA               .req s0
X_REG               .req z17
BETA                .req s1
Y_VALUE             .req s16
ALL_PRED            .req p0
TAIL_PRED           .req p1

% for c in range(num_columns):
A_REG_${c}          .req z${2 + c}
ACC_REG_${c}_z      .req z${5 + c}
ACC_REG_${c}_s      .req s${5 + c}
% endfor

## Emit the same reduction structure for the three-column group and its
## one-column cleanup. All registers used are caller-saved; x18 is avoided.
<%def name="dot_columns(columns, label, func_name, beta_mode)">
% for c in columns:
  dup ACC_REG_${c}_z.s, #0
% endfor
  mov ROW, #0
  cbz FULL_REDUCTION_LEN, .L${func_name}_${label}_tail

.L${func_name}_${label}_full:
  ld1w X_REG.s, ALL_PRED/z, [X, ROW, lsl #2]
% for c in columns:
  ld1w A_REG_${c}.s, ALL_PRED/z, [A_ADDR_${c}, ROW, lsl #2]
% endfor
% for c in columns:
  fmla ACC_REG_${c}_z.s, ALL_PRED/m, X_REG.s, A_REG_${c}.s
% endfor
  incw ROW
  cmp ROW, FULL_REDUCTION_LEN
  b.lo .L${func_name}_${label}_full

.L${func_name}_${label}_tail:
  cmp ROW, REDUCTION_LEN
  b.eq .L${func_name}_${label}_reduce
  ld1w X_REG.s, TAIL_PRED/z, [X, ROW, lsl #2]
% for c in columns:
  ld1w A_REG_${c}.s, TAIL_PRED/z, [A_ADDR_${c}, ROW, lsl #2]
% endfor
% for c in columns:
  fmla ACC_REG_${c}_z.s, TAIL_PRED/m, X_REG.s, A_REG_${c}.s
% endfor

.L${func_name}_${label}_reduce:
## Inactive tail lanes retain partial sums from the full-vector loop.
## Reduce all lanes, including those inactive in the final tail.
% for c in columns:
  faddv ACC_REG_${c}_s, ALL_PRED, ACC_REG_${c}_z.s
% if beta_mode == "zero":
  // Preserve signed-zero behaviour using beta as the addend; do not read Y.
  fmadd Y_VALUE, ACC_REG_${c}_s, ALPHA, BETA
% else:
  ldr Y_VALUE, [Y_ADDR_${c}]
% if beta_mode == "general":
  // Keep beta scaling separate from FMADD, as in the driver.
  fmul Y_VALUE, BETA, Y_VALUE
% endif
  fmadd Y_VALUE, ACC_REG_${c}_s, ALPHA, Y_VALUE
% endif
  str Y_VALUE, [Y_ADDR_${c}]
  add A_ADDR_${c}, A_ADDR_${c}, LDA
  add Y_ADDR_${c}, Y_ADDR_${c}, INC_Y
% endfor
</%def>

${prologue(func_name)}
  cmp REDUCTION_LEN, #0
  b.le .L${func_name}_return
  cmp OUTPUT_COUNT, #0
  b.le .L${func_name}_return

  ldr INC_Y, [sp]  // y_cntg_step
  // Convert strides to bytes using 64-bit registers.
  lsl LDA, LDA, #2
  lsl INC_Y, INC_Y, #2

  cntw VEC_SIZE
  udiv FULL_REDUCTION_LEN, REDUCTION_LEN, VEC_SIZE
  mul FULL_REDUCTION_LEN, FULL_REDUCTION_LEN, VEC_SIZE
  ptrue ALL_PRED.s
  whilelt TAIL_PRED.s, FULL_REDUCTION_LEN, REDUCTION_LEN

  // GROUPS = q = output_count / 3; REMAINING = output_count % 3.
  mov ISCRATCH, #${num_columns}
  udiv GROUPS, OUTPUT_COUNT, ISCRATCH
  msub REMAINING, GROUPS, ISCRATCH, OUTPUT_COUNT

  // Start the three streams q columns apart and advance each by one column.
  mul ISCRATCH, LDA, GROUPS
  add A_ADDR_1, A_ADDR_0, ISCRATCH
  add A_ADDR_2, A_ADDR_1, ISCRATCH
  mul ISCRATCH, INC_Y, GROUPS
  add Y_ADDR_1, Y_ADDR_0, ISCRATCH
  add Y_ADDR_2, Y_ADDR_1, ISCRATCH
  // Choose a beta path once per invocation, outside the reduction/output loops.
  fcmp BETA, #0.0
  b.eq .L${func_name}_beta0_entry
  fmov Y_VALUE, #1.0
  fcmp BETA, Y_VALUE
  b.ne .L${func_name}_beta_general_entry

% for beta_mode, suffix in (("one", ""), ("zero", "_beta0"), ("general", "_beta_general")):
<% path_name = func_name + suffix %>
.L${path_name}_entry:
  cbz GROUPS, .L${path_name}_remaining_columns

.L${path_name}_group:
${dot_columns(range(num_columns), "group", path_name, beta_mode)}
  subs GROUPS, GROUPS, #1
  b.ne .L${path_name}_group

.L${path_name}_remaining_columns:
  cbz REMAINING, .L${func_name}_return

  // Stream 2 now points at column 3*q (also correct when q == 0).
.L${path_name}_single:
${dot_columns([num_columns - 1], "single", path_name, beta_mode)}
  subs REMAINING, REMAINING, #1
  b.ne .L${path_name}_single
  b .L${func_name}_return

% endfor

.L${func_name}_return:
  ret
${epilogue(func_name)}

<%
aliases = ("REDUCTION_LEN", "OUTPUT_COUNT", "LDA", "X", "FULL_REDUCTION_LEN", "INC_Y", "ROW", "GROUPS",
           "VEC_SIZE", "REMAINING", "ISCRATCH", "ALPHA", "X_REG", "Y_VALUE",
           "ALL_PRED", "TAIL_PRED", "BETA")
%>
% for name in aliases:
.unreq ${name}
% endfor
% for c in range(num_columns):
.unreq A_ADDR_${c}
.unreq Y_ADDR_${c}
.unreq A_REG_${c}
.unreq ACC_REG_${c}_z
.unreq ACC_REG_${c}_s
% endfor

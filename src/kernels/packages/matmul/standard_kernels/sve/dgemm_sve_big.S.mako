## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

// A/B/C/K/M/N/ldc are all passed in
#define A x0
#define B x1
#define C x2
#define K x3
#define M x4
#define N x5
#define ldc x6
// *_idx is the current index into the block,
// outer-loop blocking is handled in C++
#define k_idx x7
#define m_idx x8
#define n_idx x9
// A/B_active_addr keep track of the address to load the
// next element of A/B from in the innermost loop
#define A_active_addr x10
#define B_active_addr x11
// tmp3/4_x are used when loading/storing C to do address
// and predication calculations
#define tmp3_x x12
#define tmp4_x x13

// alpha/beta are broadcast from memory after the inner-
// loop rather than being kept around since this is wasting
// two vectors that we could use for accumulation
#define alpha_z z0
#define beta_z z1

// the vectors to hold C elements in the alpha/beta loop
#define Cb_m0 z2
#define Cb_m1 z3
#define Cb_m2 z4

// the vectors used to hold A elements in the innermost
// loop. note that in order to maximize the distance we
// have to reload new elements we actually use five
// vectors to hold just three vector registers' worth of
// elements. this means we need five iterations of the
// inner loop to return to our original register
// allocation, but since we also need a multiple of three
// iterations to have the original allocation of B
// registers, we end up with lcm(3,5)=15 iterations.
%for i in range(3):
#define A_it${i*5+0}_m0 z0
#define A_it${i*5+0}_m1 z1
#define A_it${i*5+0}_m2 z2
#define A_it${i*5+1}_m0 z3
#define A_it${i*5+1}_m1 z4
#define A_it${i*5+1}_m2 z0
#define A_it${i*5+2}_m0 z1
#define A_it${i*5+2}_m1 z2
#define A_it${i*5+2}_m2 z3
#define A_it${i*5+3}_m0 z4
#define A_it${i*5+3}_m1 z0
#define A_it${i*5+3}_m2 z1
#define A_it${i*5+4}_m0 z2
#define A_it${i*5+4}_m1 z3
#define A_it${i*5+4}_m2 z4
%endfor

// the vectors used to hold B elements in the innermost
// loop. note that in order to maximize the distance we
// have to reload new elements of A we actually use just
// three vectors to hold just four vector registers' worth
// of elements. this means we need three iterations of the
// inner loop to return to our original register
// allocation, but since we also need a multiple of five
// iterations to have the original allocation of A
// registers, we end up with lcm(3,5)=15 iterations.
%for i in range(5):
#define B_it${i*3+0}_n01 z5
#define B_it${i*3+0}_n23 z6
#define B_it${i*3+0}_n45 z7
#define B_it${i*3+0}_n67 z5
#define B_it${i*3+1}_n01 z6
#define B_it${i*3+1}_n23 z7
#define B_it${i*3+1}_n45 z5
#define B_it${i*3+1}_n67 z6
#define B_it${i*3+2}_n01 z7
#define B_it${i*3+2}_n23 z5
#define B_it${i*3+2}_n45 z6
#define B_it${i*3+2}_n67 z7
%endfor

// the C accumulation registers
#define Ca_m0_n0 z8
#define Ca_m0_n1 z9
#define Ca_m0_n2 z10
#define Ca_m0_n3 z11
#define Ca_m0_n4 z12
#define Ca_m0_n5 z13
#define Ca_m0_n6 z14
#define Ca_m0_n7 z15
#define Ca_m1_n0 z16
#define Ca_m1_n1 z17
#define Ca_m1_n2 z18
#define Ca_m1_n3 z19
#define Ca_m1_n4 z20
#define Ca_m1_n5 z21
#define Ca_m1_n6 z22
#define Ca_m1_n7 z23
#define Ca_m2_n0 z24
#define Ca_m2_n1 z25
#define Ca_m2_n2 z26
#define Ca_m2_n3 z27
#define Ca_m2_n4 z28
#define Ca_m2_n5 z29
#define Ca_m2_n6 z30
#define Ca_m2_n7 z31

// since we pack and round up the dimensions of A and B
// into TN format, this is just K.
#define lda K
#define ldb K

// zero a register
#define XZERO(r) eor r, r, r
#define VZERO(r) eor r.16b, r.16b, r.16b
#define ZZERO(r) eor r.d, r.d, r.d

<% func_name = "dgemm_sve_big" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-320]!
	mov x29, sp
	stp x19, x20, [ sp, #224 ]
	stp x21, x22, [ sp, #208 ]
	stp x23, x24, [ sp, #192 ]
	stp x25, x26, [ sp, #176 ]
	stp x27, x28, [ sp, #160 ]
	stp d8, d9, [ sp, #80 ]
	stp d10, d11, [ sp, #64 ]
	stp d12, d13, [ sp, #48 ]
	stp d14, d15, [ sp, #32 ]

	// also save alpha and beta to be used later
	stp d0, d1, [ sp, #16 ]

	// store ourselves a constant-true predicate to be used
	// in the inner loop (which does not need predication)
	ptrue p0.b

	XZERO(n_idx)

	// check if beta is 0.0 and branch to correct loop.
	fcmp d1, #0.0
	beq .Lbzero_n_loop

	// check if beta is 1.0 and branch to correct loop.
	fmov d0, #1.0
	fcmp d1, d0
	beq .Lbone_n_loop

%for beta in ["general", "one", "zero"]:
<% L = ".Lb{}".format(beta) %>\

// loop over n (blocks of 8 elements)
${L}_n_loop:
	cmp n_idx, N
	bge ${L}_n_loop_done

// loop over m (blocks of VL*3 elements)
	XZERO(m_idx)
${L}_m_loop:
	cmp m_idx, M
	bge ${L}_m_loop_done

	// set up A/B pointers based on current m/n idx
	mul A_active_addr, m_idx, lda
	add A_active_addr, A, A_active_addr, lsl #3

	mul B_active_addr, n_idx, ldb
	add B_active_addr, B, B_active_addr, lsl #3

	// zero all accumulation registers and check if K=0
	subs k_idx, K, #0
	ZZERO(Ca_m0_n0)
	ZZERO(Ca_m0_n1)
	ZZERO(Ca_m0_n2)
	ZZERO(Ca_m0_n3)
	ZZERO(Ca_m0_n4)
	ZZERO(Ca_m0_n5)
	ZZERO(Ca_m0_n6)
	ZZERO(Ca_m0_n7)
	ZZERO(Ca_m1_n0)
	ZZERO(Ca_m1_n1)
	ZZERO(Ca_m1_n2)
	ZZERO(Ca_m1_n3)
	ZZERO(Ca_m1_n4)
	ZZERO(Ca_m1_n5)
	ZZERO(Ca_m1_n6)
	ZZERO(Ca_m1_n7)
	ZZERO(Ca_m2_n0)
	ZZERO(Ca_m2_n1)
	ZZERO(Ca_m2_n2)
	ZZERO(Ca_m2_n3)
	ZZERO(Ca_m2_n4)
	ZZERO(Ca_m2_n5)
	ZZERO(Ca_m2_n6)
	ZZERO(Ca_m2_n7)
	// if K=0, nothing to do in inner-loop, just do
	// alpha/beta application.
	beq ${L}_k_loop_done

	subs k_idx, k_idx, #1

	// each iteration expects several registers (everything
	// except B_itx_n67 actually) to already be loaded, so
	// do that before jumping in.

	// TODO: it may be faster to use LDR (vector) here rather
	//       than ld1d, since it avoids needing to inspect
	//       the predicate?
	ld1d { A_it0_m0.d }, p0/z, [A_active_addr]
	ld1d { A_it0_m1.d }, p0/z, [A_active_addr, #1, mul vl]
	ld1d { A_it0_m2.d }, p0/z, [A_active_addr, #2, mul vl]
	addvl A_active_addr, A_active_addr, #3

	ld1rqd { B_it0_n01.d }, p0/z, [B_active_addr]
	ld1rqd { B_it0_n23.d }, p0/z, [B_active_addr, #16]
	ld1rqd { B_it0_n45.d }, p0/z, [B_active_addr, #32]
	add B_active_addr, B_active_addr, #(4 << 4)

	// if K=1, then just jump to the last iteration
	beq ${L}_k_loop_last_0

<%def name="k_loop(it)">
</%def>

// paste 3*5 versions of the above snippet, substituting
// the iteration number as appropriate. when we reach the
// final iteration, jump to the below code which handles
// the last iteration without loading new elements of A/B.
// TODO: we probably do not want/need a beq after each
//       iteration of K. aligned loads of vectors will
//       likely be faster, so we want K % VL = 0 at least.
${L}_k_loop:
%for it in range(15):
<% next = (it+1) % 15 %>\
	fmla Ca_m0_n0.d, A_it${it}_m0.d, B_it${it}_n01.d[0]
	fmla Ca_m0_n1.d, A_it${it}_m0.d, B_it${it}_n01.d[1]
	add B_active_addr, B_active_addr, #(4 << 4)
	fmla Ca_m1_n0.d, A_it${it}_m1.d, B_it${it}_n01.d[0]
	fmla Ca_m1_n1.d, A_it${it}_m1.d, B_it${it}_n01.d[1]
	addvl A_active_addr, A_active_addr, #3
	fmla Ca_m2_n0.d, A_it${it}_m2.d, B_it${it}_n01.d[0]
	fmla Ca_m2_n1.d, A_it${it}_m2.d, B_it${it}_n01.d[1]
	ld1rqd { B_it${it}_n67.d }, p0/z, [B_active_addr, #-80]

	fmla Ca_m0_n2.d, A_it${it}_m0.d, B_it${it}_n23.d[0]
	fmla Ca_m0_n3.d, A_it${it}_m0.d, B_it${it}_n23.d[1]
	subs k_idx, k_idx, #1
	fmla Ca_m1_n2.d, A_it${it}_m1.d, B_it${it}_n23.d[0]
	fmla Ca_m1_n3.d, A_it${it}_m1.d, B_it${it}_n23.d[1]
	ld1d { A_it${next}_m0.d }, p0/z, [A_active_addr, #-3, mul vl]
	fmla Ca_m2_n2.d, A_it${it}_m2.d, B_it${it}_n23.d[0]
	fmla Ca_m2_n3.d, A_it${it}_m2.d, B_it${it}_n23.d[1]
	ld1rqd { B_it${next}_n01.d }, p0/z, [B_active_addr, #-64]

	fmla Ca_m0_n4.d, A_it${it}_m0.d, B_it${it}_n45.d[0]
	fmla Ca_m0_n5.d, A_it${it}_m0.d, B_it${it}_n45.d[1]
	fmla Ca_m1_n4.d, A_it${it}_m1.d, B_it${it}_n45.d[0]
	ld1d { A_it${next}_m1.d }, p0/z, [A_active_addr, #-2, mul vl]
	fmla Ca_m1_n5.d, A_it${it}_m1.d, B_it${it}_n45.d[1]
	fmla Ca_m2_n4.d, A_it${it}_m2.d, B_it${it}_n45.d[0]
	fmla Ca_m2_n5.d, A_it${it}_m2.d, B_it${it}_n45.d[1]
	ld1rqd { B_it${next}_n23.d }, p0/z, [B_active_addr, #-48]

	fmla Ca_m0_n6.d, A_it${it}_m0.d, B_it${it}_n67.d[0]
	fmla Ca_m0_n7.d, A_it${it}_m0.d, B_it${it}_n67.d[1]
	fmla Ca_m1_n6.d, A_it${it}_m1.d, B_it${it}_n67.d[0]
	ld1d { A_it${next}_m2.d }, p0/z, [A_active_addr, #-1, mul vl]
	fmla Ca_m1_n7.d, A_it${it}_m1.d, B_it${it}_n67.d[1]
	fmla Ca_m2_n6.d, A_it${it}_m2.d, B_it${it}_n67.d[0]
	fmla Ca_m2_n7.d, A_it${it}_m2.d, B_it${it}_n67.d[1]
	ld1rqd { B_it${next}_n45.d }, p0/z, [B_active_addr, #-32]
	beq ${L}_k_loop_last_${next}
%endfor
	b ${L}_k_loop

// paste 3*5 versions of the final iteration (so we can
// jump to the version with the correct register
// allocation). We still need to load B_itx_n67, but
// everything else would have been loaded for us in
// advance.
%for it in range(15):
${L}_k_loop_last_${it}:
	fmla Ca_m0_n0.d, A_it${it}_m0.d, B_it${it}_n01.d[0]
	fmla Ca_m0_n1.d, A_it${it}_m0.d, B_it${it}_n01.d[1]
	fmla Ca_m1_n0.d, A_it${it}_m1.d, B_it${it}_n01.d[0]
	fmla Ca_m1_n1.d, A_it${it}_m1.d, B_it${it}_n01.d[1]
	fmla Ca_m2_n0.d, A_it${it}_m2.d, B_it${it}_n01.d[0]
	fmla Ca_m2_n1.d, A_it${it}_m2.d, B_it${it}_n01.d[1]
	ld1rqd { B_it${it}_n67.d }, p0/z, [B_active_addr, #-16]

	fmla Ca_m0_n2.d, A_it${it}_m0.d, B_it${it}_n23.d[0]
	fmla Ca_m0_n3.d, A_it${it}_m0.d, B_it${it}_n23.d[1]
	fmla Ca_m1_n2.d, A_it${it}_m1.d, B_it${it}_n23.d[0]
	fmla Ca_m1_n3.d, A_it${it}_m1.d, B_it${it}_n23.d[1]
	fmla Ca_m2_n2.d, A_it${it}_m2.d, B_it${it}_n23.d[0]
	fmla Ca_m2_n3.d, A_it${it}_m2.d, B_it${it}_n23.d[1]

	fmla Ca_m0_n4.d, A_it${it}_m0.d, B_it${it}_n45.d[0]
	fmla Ca_m0_n5.d, A_it${it}_m0.d, B_it${it}_n45.d[1]
	fmla Ca_m1_n4.d, A_it${it}_m1.d, B_it${it}_n45.d[0]
	fmla Ca_m1_n5.d, A_it${it}_m1.d, B_it${it}_n45.d[1]
	fmla Ca_m2_n4.d, A_it${it}_m2.d, B_it${it}_n45.d[0]
	fmla Ca_m2_n5.d, A_it${it}_m2.d, B_it${it}_n45.d[1]

	fmla Ca_m0_n6.d, A_it${it}_m0.d, B_it${it}_n67.d[0]
	fmla Ca_m0_n7.d, A_it${it}_m0.d, B_it${it}_n67.d[1]
	fmla Ca_m1_n6.d, A_it${it}_m1.d, B_it${it}_n67.d[0]
	fmla Ca_m1_n7.d, A_it${it}_m1.d, B_it${it}_n67.d[1]
	fmla Ca_m2_n6.d, A_it${it}_m2.d, B_it${it}_n67.d[0]
	fmla Ca_m2_n7.d, A_it${it}_m2.d, B_it${it}_n67.d[1]
	b ${L}_k_loop_done
%endfor

${L}_k_loop_done:
	mov tmp3_x, m_idx
	mov tmp4_x, m_idx
	incd tmp3_x, all, mul #1
	incd tmp4_x, all, mul #2

	// load alpha/beta back and broadcast (unindexed FMLAs
	// may be faster than indexed FMLAs, although in the
	// inner loop this shouldn't matter since they will be
	// pipelined regardless).
	ld1rd { alpha_z.d }, p0/z, [ sp, #16 ]
	ld1rd { beta_z.d }, p0/z, [ sp, #24 ]

	// compute an extended predicate over three registers
	// (just an unrolled whilelt m_idx, M)
	whilelt p2.d, m_idx, M
	whilelt p3.d, tmp3_x, M
	whilelt p4.d, tmp4_x, M

	// paste the alpha/beta multiplication step for each
	// of the 8 iterations of n. This simply loads three
	// vectors worth of elements using p2/p3/p4 respectively,
	// multiplies by alpha/beta and stores it back.
%for ofs in range(8):
<% c_reg0 = "Ca_m0_n{}".format(ofs) %>\
<% c_reg1 = "Ca_m1_n{}".format(ofs) %>\
<% c_reg2 = "Ca_m2_n{}".format(ofs) %>\
	add tmp4_x, n_idx, #${ofs}
	cmp tmp4_x, N
	bge 1f
	madd tmp3_x, tmp4_x, ldc, m_idx
	add tmp3_x, C, tmp3_x, lsl #3
%if beta == "one":
	ld1d { Cb_m0.d }, p2/z, [tmp3_x]
	ld1d { Cb_m1.d }, p3/z, [tmp3_x, #1, mul vl]
	ld1d { Cb_m2.d }, p4/z, [tmp3_x, #2, mul vl]
	fmla Cb_m0.d, p0/m, ${c_reg0}.d, alpha_z.d
	fmla Cb_m1.d, p0/m, ${c_reg1}.d, alpha_z.d
	fmla Cb_m2.d, p0/m, ${c_reg2}.d, alpha_z.d
	st1d { Cb_m0.d }, p2, [tmp3_x]
	st1d { Cb_m1.d }, p3, [tmp3_x, #1, mul vl]
	st1d { Cb_m2.d }, p4, [tmp3_x, #2, mul vl]
%elif beta == "zero":
	fmul ${c_reg0}.d, p0/m, ${c_reg0}.d, alpha_z.d
	fmul ${c_reg1}.d, p0/m, ${c_reg1}.d, alpha_z.d
	fmul ${c_reg2}.d, p0/m, ${c_reg2}.d, alpha_z.d
	st1d { ${c_reg0}.d }, p2, [tmp3_x]
	st1d { ${c_reg1}.d }, p3, [tmp3_x, #1, mul vl]
	st1d { ${c_reg2}.d }, p4, [tmp3_x, #2, mul vl]
%else:
	ld1d { Cb_m0.d }, p2/z, [tmp3_x]
	ld1d { Cb_m1.d }, p3/z, [tmp3_x, #1, mul vl]
	ld1d { Cb_m2.d }, p4/z, [tmp3_x, #2, mul vl]
	fmul ${c_reg0}.d, p0/m, ${c_reg0}.d, alpha_z.d
	fmul ${c_reg1}.d, p0/m, ${c_reg1}.d, alpha_z.d
	fmul ${c_reg2}.d, p0/m, ${c_reg2}.d, alpha_z.d
	fmla ${c_reg0}.d, p0/m, Cb_m0.d, beta_z.d
	fmla ${c_reg1}.d, p0/m, Cb_m1.d, beta_z.d
	fmla ${c_reg2}.d, p0/m, Cb_m2.d, beta_z.d
	st1d { ${c_reg0}.d }, p2, [tmp3_x]
	st1d { ${c_reg1}.d }, p3, [tmp3_x, #1, mul vl]
	st1d { ${c_reg2}.d }, p4, [tmp3_x, #2, mul vl]
%endif
1:
%endfor

	incd m_idx, all, mul #3
	b ${L}_m_loop

${L}_m_loop_done:
	add n_idx, n_idx, #8
	b ${L}_n_loop

${L}_n_loop_done:
${L}_epilogue:
	// do not bother to restore alpha/beta since the PCS
	// didn't ask us to. also, we do not need to save/restore
	// any SVE vectors since we did not take any as input.
	ldp x19, x20, [ sp, #224 ]
	ldp x21, x22, [ sp, #208 ]
	ldp x23, x24, [ sp, #192 ]
	ldp x25, x26, [ sp, #176 ]
	ldp x27, x28, [ sp, #160 ]
	ldp d8, d9, [ sp, #80 ]
	ldp d10, d11, [ sp, #64 ]
	ldp d12, d13, [ sp, #48 ]
	ldp d14, d15, [ sp, #32 ]
	ldp x29, x30, [sp], #320
	ret
	${epilogue(func_name)}
%endfor

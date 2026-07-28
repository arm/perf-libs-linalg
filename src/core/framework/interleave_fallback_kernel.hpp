/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifdef PERFLIBS_LINALG_INTERLEAVE_FALLBACK_KERNEL_ACTIVELY_REQUESTED

#ifndef PERFLIBS_LINALG_INTERLEAVE_FALLBACK_KERNEL_HPP
#define PERFLIBS_LINALG_INTERLEAVE_FALLBACK_KERNEL_HPP

#include "linalg_util.hpp"
#include "interleave_fallback_kernel_fwd.hpp"
#include "perflibs_assert.hpp"

namespace perflibs::linalg {

namespace {

/**
 * copy_n copies data from an input buffer to a destination. The items in source should be src_strd_stride
 * apart, which can be one if they are contiguous, and in the destination they will always be contiguous after
 * the copy.
 *
 * Some indices of the input buffer may be "virtual" meaning they should not be accessed. The position these
 * indices correspond to in the destination interleaved matrix can either be set to an actual zero value,
 * or simply skipped to be assigned a value in a future run of this kernel.
 *
 * It can also handle Unit matrices where the diagonal is "virtual" ones that are not present in the source
 * matrix.
 *
 * The result of the copy operation can be padded with zeros up to the InterleaveFactor if required,
 * otherwise those indices may contain junk data.
 *
 * The conjugate of data elements from src can also be stored in dst instead of the actual value read.
 *
 * @tparam FirstNonZeroEl until this index is reached set dst to 0. If Unit and Upper Triangular, the next
 *         index is 1 as it is on the diagonal.
 * @tparam PhysicalElEnd until this index is reached set dst to src[i] (or conj(src[i])) if IsConj. If Unit
 *         and Lower Triangular, this index is 1 as it is on the diagonal.
 * @tparam VirtualZeroEnd until this index is reached set dst to 0
 * @tparam InterleaveFactor if ZeroPad, pad dst up to InterleaveFactor with 0
 * *
 * @tparam Flags bitmask containing interleave_flags
 * @tparam SrcStrdStepType create individual functions if it is able to evaluate cntgstep to a constexpr value.
 * @tparam TSrc value type of input buffer
 * @tparam TDst value type of output buffer
 *
 * @param src pointer to input buffer
 * @param src_strd_stride distance between each element in Src
 * @param dst pointer to output buffer, element assumed to be contiguous
 */

/*
Visual Example on how this code might work:

FirstNonZeroEl   = 2
PhysicalElEnd    = 4
VirtualZeroEnd   = 6
InterleaveFactor = 8

Flags   = 100001
ZeroPad = true
IsConj  = false
DiagVirt  = false
LowerVirt  = false
UpperVirt  = false
VirtZero   = true

Src: --->  Dst:
           [0] [1] [2] [3] [4] [5] [6] [7]
 a          0   0   a   b   0   0  (0) (0)
 b

The result has 4 parts:
 - Prepended Zeroes: 0 <= i < 2
 - Data from src   : 2 <= i < 4
 - Appended Zeroes : 4 <= i < 6
 - Padding Zeroes  : 6 <= i < 8

Another example, for Unit matrices:

Flags   = 01101

DiagVirt  = true
LowerVirt  = true

Src: --->  Dst:
           [0] [1] [2] [3] [4] [5] [6] [7]
 X          0   0   1   b   0   0  (0) (0)
 b

*/
template<kernel_inttype FirstNonZeroEl, kernel_inttype PhysicalElEnd, kernel_inttype VirtualZeroEnd,
         kernel_inttype InterleaveFactor, kernel_inttype Flags, typename SrcStrdStepType, typename TSrc,
         typename TDst>
static PERFLIBS_LINALG_INLINE void copy_n(const TSrc *__restrict src, const SrcStrdStepType src_strd_stride,
                                     TDst *__restrict dst) {
	// i will walk from 0 to the InterleaveFactor, but the value copied at each index depends on the
	// template parameters
	kernel_inttype i = 0;

	if constexpr (FirstNonZeroEl != 0) {
		if constexpr (bool(Flags & interleave_flags::VirtZero)) {
#ifdef __clang__
#pragma unroll(InterleaveFactor)
#else
#pragma GCC unroll(20)
#endif
			// 0 <= i < FirstNonZeroEl : Left Zeroes
			for (; i < FirstNonZeroEl; ++i) {
				dst[i] = zero<TDst>;
			}
		}
		else {
			i = FirstNonZeroEl;
		}
	}

	if constexpr (bool(Flags & interleave_flags::LowerVirt)) {
		// Diag is before data
		if constexpr (bool(Flags & interleave_flags::DiagVirt)) {
			if constexpr (bool(Flags & interleave_flags::DiagUnit)) {
				// Unit Diagonal
				dst[i] = one<TDst>;
			}
			else if constexpr (bool(Flags & interleave_flags::DiagReal)) {
				// Only Imaginary is virtual
				dst[i] = TDst{ real(src[i * src_strd_stride]), 0.0 };

			}
			i++;
		}
	}

	if constexpr (FirstNonZeroEl != PhysicalElEnd) {
#ifdef __clang__
#pragma unroll(InterleaveFactor)
#else
#pragma GCC unroll(20)
#endif
		// FirstNonZeroEl <= i < PhysicalElEnd : Data from src
		for (; i < PhysicalElEnd; ++i) {
			if constexpr (bool(Flags & interleave_flags::IsConj)) {
				dst[i] = conj(src[i * src_strd_stride]);
			}
			else {
				dst[i] = src[i * src_strd_stride];
			}
		}
	}

	if constexpr (bool(Flags & interleave_flags::UpperVirt)) {
		// Diag is after data
		if constexpr (bool(Flags & interleave_flags::DiagVirt)) {
			if constexpr (bool(Flags & interleave_flags::DiagUnit)) {
				// Unit Diagonal
				dst[i] = one<TDst>;
			}
			else if constexpr (bool(Flags & interleave_flags::DiagReal)) {
				// Only Imaginary is virtual
				dst[i] = TDst{ real(src[i * src_strd_stride]), 0.0 };
			}
			i++;
		}
	}

	if constexpr (PhysicalElEnd != VirtualZeroEnd) {
		if constexpr (bool(Flags & interleave_flags::VirtZero)) {
#ifdef __clang__
#pragma unroll(InterleaveFactor)
#else
#pragma GCC unroll(20)
#endif
			// PhysicalElEnd <= i < VirtualZeroEnd : Right Zeroes
			for (; i < VirtualZeroEnd; ++i) {
				dst[i] = zero<TDst>;
			}
		}
		else {
			i = VirtualZeroEnd;
		}
	}

	// Padding happens if enabled, to align the vectors according to interleave factor
	if constexpr (bool(Flags & interleave_flags::ZeroPad)) {

#ifdef __clang__
#pragma unroll(InterleaveFactor)
#else
#pragma GCC unroll(20)
#endif
		// VirtualZeroEnd <= i < InterleaveFactor : Padding Zeroes
		for (; i < InterleaveFactor; ++i) {
			dst[i] = zero<TDst>;
		}
	}
}

/**
 * n_interleave_cntg_loop loops over a src matrix, row by row, invoking a copy operation (copy_n) on
 * each one. This src matrix has InterleaveN columns (strd) and cntg_min rows. The copy places
 * blocks of size InterleaveN from the the non-contiguous dimension into the dst matrix as contiguous elements
 * for vector loading by the optimized kernels.
 *
 * It also stores real zeroes/ones in the dst for indices that are virtual zeroes or ones in the src, and can
 * store the conjugate of src values if required.
 *
 * To handle both upper and lower partial matrices, the matrix is assumed to be a window onto a vertical
 * banded matrix, Such that 0 is always the top left corner of the matrix, cntg first is top of the band of
 * data and cntg_last is the bottom of the band of data.
 * Note that the contiguous dimension goes top to bottom in this diagram.
 *
 *      General Case        Rectangular     Lower Triangle   Partial Upper Tri
 *         X X X X X          X X X X X          X X X X X           X X X X X
 *
 *         X X X X X          X X X X X          X X X X X           X X X X X
 *                                               ---------
 * first-> a X X X X   [-4]-> a X X X X    [0]->|a X X X X|  [-11]-> a X X X X
 *                                              |         |
 *         b c X X X          b c X X X         |b c X X X|          b c X X X
 *                                              |         |
 *         d e f X X          d e f X X         |d e f X X|          d e f X X
 *                                              |         |
 *         g h i j X          g h i j X         |g h i j X|          g h i j X
 *                            ---------         |         |
 *         k l m n o    [0]->|k l m n o|        |k l m n o|          k l m n o
 *                           |         |         ---------
 *         p q r s t         |p q r s t|         p q r s t           p q r s t
 *                           |         |
 *         f g h i j         |f g h i j|         f g h i j           f g h i j
 *                           |         |
 *         a b c d e         |a b c d e|         a b c d e           a b c d e
 *                           |         |
 *  last-> f g h i j    [4]->|f g h i j|   [8]-> f g h i j    [-3]-> f g h i j
 *                            ---------
 *         X k l m n          X k l m n          X k l m n           X k l m n
 *
 *         X X o p q          X X o p q          X X o p q           X X o p q
 *                                                                   ---------
 *         X X X r s          X X X r s          X X X r s     [0]->|X X X r s|
 *                                                                  |         |
 *         X X X X t          X X X X t          X X X X t          |X X X X t|
 *                                                                  |         |
 *         X X X X X          X X X X X          X X X X X          |X X X X X|
 *                                                                  |         |
 *         X X X X X          X X X X X          X X X X X          |X X X X X|
 *                                                                   ---------
 * Therefore the src matrix is interleaved in 5 phases, some of which will apply to
 * zero rows, depending on the matrix structure. Each phase is captured by a loop in
 * this function, which makes calls to copy_n.
 *
 * In some cases the flags passed to copy_n are narrowed to only those that are relevant
 * to that phase, for example IsUnit only impacts the output when processing a diagonal.
 * The flags for each stage are documented below:
 *
 * 1) Interleave columns filled with virtual elements (Top zeros)
 *         X X X X X
 *
 *         X X X X X
 *
 *    - Flags: ZeroPad
 *
 * 2) Interleave columns partially filled with right virtual zeros (Lower Triangle Matrix)
 *         a X X X X
 *
 *         b c X X X
 *
 *         d e f X X
 *
 *         g h i j X
 *
 *    - Flags: All
 *
 * 3) Interleave columns that only contain physical elements (Rectangular Matrix)
 *         k l m n o
 *
 *         p q r s t
 *
 *         u v w y z
 *
 *         a b c d e
 *
 *         f g h i j
 *
 *    - Flags: ZeroPad and IsConj
 *
 * 4) Interleave columns partially filled with left virtual elements (Upper Triangle Matrix)
 *         X k l m n
 *
 *         X X o p q
 *
 *         X X X r s
 *
 *         X X X X t
 *
 *    - Flags: All
 *
 * 5) Interleave columns filled with virtual elements (Right zeros)
 *         X X X X X
 *
 *         X X X X X
 *
 *    - Flags: ZeroPad
 *
 * @tparam InterleaveN number of rows (strd) of src matrix we are trying to interleave
 * @tparam InterleaveMax is the interleave factor. Usually InterleaveN=InterleaveMax except for matrix edges
 * @tparam Flags bitmask containing interleave_flags
 * @tparam SrcCntgStepType   create individual functions if cntgstep to a constexpr value.
 * @tparam SrcStrdStepType   create individual functions if strdstep to a constexpr value.
 * @tparam TSrc value type of input buffer
 * @tparam TDst value type of output buffer
 *
 * @param cntg_min The amount of columns in src
 * @param dst_cntg The actual amount of columns of src that could fit in dst (the actual length of dst is
 * dst_cntg*InterleaveMax)
 * @param src_strd_pos Pointer to start of src block to process
 * @param src_cntg_stride The step between columns
 * @param src_strd_stride The step between rows
 * @param dst_strd_pos Points to start of dst block to assign
 * @param cntg_first Data start diagonal relative to src matrix corner, may be constexpr minimum value
 * @param cntg_last  Data end diagonal relative to src matrix corner, may be constexpr max value
 */

/*
Visual Example on how this code might work:
InterleaveN = 2
InterleaveMax = 2
Flags = 11001
Considering we have a given matrix Src (cntg_min = 2):
    xxabcdeopxx
    xxxefghqrsx
The result would be:
   [0 0] [0 0] [a 0] [b e] [c f] [d g] [e h] [o q] [p r] [0 s] [0 0] (0 0)
Square braces denote interleaved pairs. Brackets show padding zeroes.
All elements are contiguous in the dst buffer.
*/

template<kernel_inttype InterleaveN, kernel_inttype InterleaveMax, kernel_inttype Flags,
         typename SrcCntgStepType, typename SrcStrdStepType, typename TSrc, typename TDst>
static __attribute__((noinline)) void
n_interleave_cntg_loop(kernel_inttype cntg_min, kernel_inttype dst_cntg, const TSrc *__restrict src_strd_pos,
                       SrcCntgStepType src_cntg_stride, SrcStrdStepType src_strd_stride,
                       TDst *__restrict dst_strd_pos, kernel_inttype cntg_first, kernel_inttype cntg_last) {

	kernel_inttype dc = 0; // dc will walk the contiguous dimension and call copy_n with correct params

	// These variables are declared once and reused by each phase as dc walks down the cntg dimension.
	//
	// Tip: When debugging, set a watchpoint on section_end to break between each phase and find out
	//      which indices each phase will affect.
	kernel_inttype section_start, section_end, delta_start;

	if constexpr (bool(Flags & interleave_flags::UpperVirt)) {
		section_end = min(cntg_first, cntg_min);

		// Only loop over fully virtual section if it needs to set to zeroes
		if constexpr (bool(Flags & interleave_flags::VirtZero)) {
			// Top Zeroes
			for (; dc < section_end; ++dc) {
				const auto dst_cntg_pos = dst_strd_pos + dc * InterleaveMax;
				const auto src_cntg_pos = src_strd_pos + dc * src_cntg_stride;
				copy_n<InterleaveN, InterleaveN, InterleaveN, InterleaveMax,
				       Flags &(interleave_flags::ZeroPad | interleave_flags::VirtZero)>(
				    src_cntg_pos, src_strd_stride, dst_cntg_pos);
			}
		}
		else {
			dc = max(dc, section_end);
		}

		// Partial Lower Triangle
		section_start = dc;
		section_end = min(cntg_first + InterleaveN, cntg_min);

		if constexpr (bool(Flags & interleave_flags::DiagVirt))
			delta_start = (cntg_first >= 0 ? 0 : std::abs(cntg_first));
		else
			delta_start = 1 + (cntg_first >= 0 ? 0 : std::abs(cntg_first));

		for (; dc < section_end; ++dc) {
			const auto dst_cntg_pos = dst_strd_pos + dc * InterleaveMax;
			const auto src_cntg_pos = src_strd_pos + dc * src_cntg_stride;
			kernel_inttype delta = delta_start + (dc - section_start);
			switch (delta) {
			default:
				// More new cases below may need to be added if IR is greater than the current max
				// currently we support interleaving up to an interleaving factor == 20,
				// so a maximum of 19 elements can be left over for the maximum interleaving factor
				PERFLIBS_ASSERT(false, "invalid interleave value requested");
				break;

			case 0:
				copy_n<0, 0, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride, dst_cntg_pos);
				break;
			case 1:
				if constexpr (InterleaveN > 0) {
					copy_n<0, 1, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                dst_cntg_pos);
				}
				break;
			case 2:
				if constexpr (InterleaveN > 1) {
					copy_n<0, 2, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                dst_cntg_pos);
				}
				break;
			case 3:
				if constexpr (InterleaveN > 2) {
					copy_n<0, 3, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                dst_cntg_pos);
				}
				break;
			case 4:
				if constexpr (InterleaveN > 3) {
					copy_n<0, 4, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                dst_cntg_pos);
				}
				break;
			case 5:
				if constexpr (InterleaveN > 4) {
					copy_n<0, 5, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                dst_cntg_pos);
				}
				break;
			case 6:
				if constexpr (InterleaveN > 5) {
					copy_n<0, 6, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                dst_cntg_pos);
				}
				break;
			case 7:
				if constexpr (InterleaveN > 6) {
					copy_n<0, 7, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                dst_cntg_pos);
				}
				break;
			case 8:
				if constexpr (InterleaveN > 7) {
					copy_n<0, 8, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                dst_cntg_pos);
				}
				break;
			case 9:
				if constexpr (InterleaveN > 8) {
					copy_n<0, 9, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                dst_cntg_pos);
				}
				break;
			case 10:
				if constexpr (InterleaveN > 9) {
					copy_n<0, 10, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                 dst_cntg_pos);
				}
				break;
			case 11:
				if constexpr (InterleaveN > 10) {
					copy_n<0, 11, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                 dst_cntg_pos);
				}
				break;
			case 12:
				if constexpr (InterleaveN > 11) {
					copy_n<0, 12, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                 dst_cntg_pos);
				}
				break;
			case 13:
				if constexpr (InterleaveN > 12) {
					copy_n<0, 13, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                 dst_cntg_pos);
				}
				break;
			case 14:
				if constexpr (InterleaveN > 13) {
					copy_n<0, 14, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                 dst_cntg_pos);
				}
				break;
			case 15:
				if constexpr (InterleaveN > 14) {
					copy_n<0, 15, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                 dst_cntg_pos);
				}
				break;
			case 16:
				if constexpr (InterleaveN > 15) {
					copy_n<0, 16, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                 dst_cntg_pos);
				}
				break;
			case 17:
				if constexpr (InterleaveN > 16) {
					copy_n<0, 17, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                 dst_cntg_pos);
				}
				break;
			case 18:
				if constexpr (InterleaveN > 17) {
					copy_n<0, 18, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                 dst_cntg_pos);
				}
				break;
			case 19:
				if constexpr (InterleaveN > 18) {
					copy_n<0, 19, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                 dst_cntg_pos);
				}
				break;
			case 20:
				if constexpr (InterleaveN > 19) {
					copy_n<0, 20, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                 dst_cntg_pos);
				}
				break;
				//... until case 20 because that's how big strd can get (so that's how big difference between c
				//- src_strd can get)
			}
		}
	}

	// Rectangle
	if constexpr (bool(Flags & interleave_flags::LowerVirt)) {
		section_end = min(cntg_last, cntg_min);
	}
	else {
		section_end = cntg_min;
	}

	for (; dc < section_end; ++dc) {
		const auto dst_cntg_pos = dst_strd_pos + dc * InterleaveMax;
		const auto src_cntg_pos = src_strd_pos + dc * src_cntg_stride;
		copy_n<0, InterleaveN, InterleaveN, InterleaveMax,
		       Flags &(interleave_flags::ZeroPad | interleave_flags::IsConj)>(src_cntg_pos, src_strd_stride,
		                                                                      dst_cntg_pos);
	}

	if constexpr (bool(Flags & interleave_flags::LowerVirt)) {

		// Partial Upper Triangle Interleave
		section_start = dc;
		section_end = min(cntg_last + (InterleaveN), cntg_min);
		delta_start = (cntg_last >= 0 ? 0 : std::abs(cntg_last));

		for (; dc < section_end; ++dc) {
			const auto dst_cntg_pos = dst_strd_pos + dc * InterleaveMax;
			const auto src_cntg_pos = src_strd_pos + dc * src_cntg_stride;
			kernel_inttype delta = delta_start + (dc - section_start);
			switch (delta) {
			default:
				// More new cases below may need to be added if IR is greater than the current max
				// currently we support interleaving up to an interleaving factor == 20,
				// so a maximum of 19 elements can be left over for the maximum interleaving factor
				PERFLIBS_ASSERT(false, "invalid interleave value requested");
				break;

			case 0:
				copy_n<0, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
				                                                          dst_cntg_pos);
				break;
			case 1:
				if constexpr (InterleaveN > 0) {
					copy_n<1, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                          dst_cntg_pos);
				}
				break;
			case 2:
				if constexpr (InterleaveN > 1) {
					copy_n<2, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                          dst_cntg_pos);
				}
				break;
			case 3:
				if constexpr (InterleaveN > 2) {
					copy_n<3, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                          dst_cntg_pos);
				}
				break;
			case 4:
				if constexpr (InterleaveN > 3) {
					copy_n<4, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                          dst_cntg_pos);
				}
				break;
			case 5:
				if constexpr (InterleaveN > 4) {
					copy_n<5, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                          dst_cntg_pos);
				}
				break;
			case 6:
				if constexpr (InterleaveN > 5) {
					copy_n<6, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                          dst_cntg_pos);
				}
				break;
			case 7:
				if constexpr (InterleaveN > 6) {
					copy_n<7, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                          dst_cntg_pos);
				}
				break;
			case 8:
				if constexpr (InterleaveN > 7) {
					copy_n<8, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                          dst_cntg_pos);
				}
				break;
			case 9:
				if constexpr (InterleaveN > 8) {
					copy_n<9, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                          dst_cntg_pos);
				}
				break;
			case 10:
				if constexpr (InterleaveN > 9) {
					copy_n<10, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                           dst_cntg_pos);
				}
				break;
			case 11:
				if constexpr (InterleaveN > 10) {
					copy_n<11, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                           dst_cntg_pos);
				}
				break;
			case 12:
				if constexpr (InterleaveN > 11) {
					copy_n<12, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                           dst_cntg_pos);
				}
				break;
			case 13:
				if constexpr (InterleaveN > 12) {
					copy_n<13, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                           dst_cntg_pos);
				}
				break;
			case 14:
				if constexpr (InterleaveN > 13) {
					copy_n<14, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                           dst_cntg_pos);
				}
				break;
			case 15:
				if constexpr (InterleaveN > 14) {
					copy_n<15, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                           dst_cntg_pos);
				}
				break;
			case 16:
				if constexpr (InterleaveN > 15) {
					copy_n<16, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                           dst_cntg_pos);
				}
				break;
			case 17:
				if constexpr (InterleaveN > 16) {
					copy_n<17, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                           dst_cntg_pos);
				}
				break;
			case 18:
				if constexpr (InterleaveN > 17) {
					copy_n<18, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                           dst_cntg_pos);
				}
				break;
			case 19:
				if constexpr (InterleaveN > 18) {
					copy_n<19, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                           dst_cntg_pos);
				}
				break;
			case 20:
				if constexpr (InterleaveN > 19) {
					copy_n<20, InterleaveN, InterleaveN, InterleaveMax, Flags>(src_cntg_pos, src_strd_stride,
					                                                           dst_cntg_pos);
				}
				break;

				//... until case 20 because that's how big strd can get (so that's how big difference between c
				//- src_strd can get)
			}
		}

		// Bottom Zeros
		section_end = cntg_min;

		// Only loop over fully virtual section if it needs to set to zeroes
		if constexpr (bool(Flags & interleave_flags::VirtZero)) {

			for (; dc < section_end; ++dc) {
				const auto dst_cntg_pos = dst_strd_pos + dc * InterleaveMax;
				const auto src_cntg_pos = src_strd_pos + dc * src_cntg_stride;
				copy_n<0, 0, InterleaveN, InterleaveMax,
				       Flags &(interleave_flags::ZeroPad | interleave_flags::VirtZero)>(
				    src_cntg_pos, src_strd_stride, dst_cntg_pos);
			}
		}
	}

	for (kernel_inttype dc = cntg_min; dc < dst_cntg; ++dc) {
		const auto dst_cntg_pos = dst_strd_pos + dc * InterleaveMax;
		constexpr kernel_inttype interleave_zero =
		    (Flags & interleave_flags::ZeroPad) ? InterleaveMax : InterleaveN;
		for (kernel_inttype j = 0; j != interleave_zero; ++j) {
			dst_cntg_pos[j] = zero<TDst>;
		}
	}
}

/**
 * Loops over a src matrix, k strides at a time, where k = Interleave and passes to n_interleave_cntg_loop
 * which will copy that data in interleaved format to dst.
 *
 * The amount of loops that gives is floor(src_strd/Interleave).
 *
 * There might be some leftover, which the switch case covers.
 * @tparam Interleave number non-contiguous elements of src matrix we are trying to interleave
 * @tparam Flags bitmask containing interleave_flags
 * @tparam SrcCntgStepType    create individual functions if cntgstep to a constexpr value.
 * @tparam SrcStrdStepType    create individual functions if strdstep to a constexpr value.
 * @tparam TSrc value type of input buffer
 * @tparam TDst value type of output buffer
 *
 * @param src_cntg The amount of columns in src
 * @param src_strd The amount of rows in src
 * @param src Points to the beginning of src
 * @param src_cntg_stride The step between columns in src
 * @param src_strd_stride The step between rows in src
 * @param dst_cntg Amount of columns of src that would fit in dst
 * @param dst_strd Amount of rows in dst
 * @param dst Points to the beginning of dst
 * @param dst_stride The step between rows in dst
 * @param cntg_first Row of first non-virtual element relative to start of "window"
 * @param cntg_last  Row of last non-virtual element relative to start of "window"
 **/

/*
Visual Example on how this code might work:
Interleave = 2
Flags = 00001
Considering we have given a matrix Src:
    a e
    b f
    c g
    d h
    e f
The result would be:
    a b e f
    c d g h
    e 0 f 0
*/
template<kernel_inttype Interleave, kernel_inttype Flags, typename SrcCntgStepType, typename SrcStrdStepType,
         typename TSrc, typename TDst>
static PERFLIBS_LINALG_INLINE void
omni_interleave(kernel_inttype src_cntg, kernel_inttype src_strd, const TSrc *__restrict src,
                SrcCntgStepType src_cntg_stride, SrcStrdStepType src_strd_stride, kernel_inttype dst_cntg,
                kernel_inttype dst_strd, TDst *__restrict dst, kernel_inttype dst_stride,
                kernel_inttype cntg_first, kernel_inttype cntg_last) {

	const kernel_inttype strd_min = min(src_strd, dst_strd);
	//                              Usually src_strd < dst_strd, where dst_strd = no_rows_dst * Interleave
	const kernel_inttype cntg_min = min(src_cntg, dst_cntg);
	//                              Usually src_cntg < dst_cntg, where dst_cntg = no_columns_dst/Interleave

	auto dst_strd_pos = dst;

	kernel_inttype full_strd_end = 0; // Number of rows in src that have been processed

	for (; full_strd_end <= (strd_min - Interleave); full_strd_end += Interleave) {

		// pointer to start of each block of [Interleave] rows from src
		const auto src_strd_pos = src + full_strd_end * src_strd_stride;
		kernel_inttype cntg_first_next;
		if constexpr (bool(Flags & interleave_flags::UpperVirt)) {
			cntg_first_next = cntg_first + full_strd_end;
		}
		else {
			cntg_first_next = cntg_first;
		}
		kernel_inttype cntg_last_next;
		if constexpr (bool(Flags & interleave_flags::LowerVirt)) {
			cntg_last_next = cntg_last + full_strd_end;
		}
		else {
			cntg_last_next = cntg_last;
		}

		n_interleave_cntg_loop<Interleave, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
		                                                      src_cntg_stride, src_strd_stride, dst_strd_pos,
		                                                      cntg_first_next, cntg_last_next);

		dst_strd_pos += dst_stride; // next dst row
	}

	{
		// This case statement deals with any leftover rows that still need processing,
		// and don't divide evenly into a whole interleave block (an Interleave by strd_cntg block).
		const auto src_strd_pos = src + full_strd_end * src_strd_stride;
		kernel_inttype cntg_first_next;
		if constexpr (bool(Flags & interleave_flags::UpperVirt)) {
			cntg_first_next = cntg_first + full_strd_end;
		}
		else {
			cntg_first_next = cntg_first;
		}
		kernel_inttype cntg_last_next;
		if constexpr (bool(Flags & interleave_flags::LowerVirt)) {
			cntg_last_next = cntg_last + full_strd_end;
		}
		else {
			cntg_last_next = cntg_last;
		}

		switch (src_strd - full_strd_end) {
		default:
			// More new cases below may need to be added if IR is greater than the current max
			// currently we support interleaving up to an interleaving factor == 20,
			// so a maximum of 19 elements can be left over for the maximum interleaving factor
			PERFLIBS_ASSERT(false, "invalid interleave value requested");
			break;
		case 0: break; // nothing to do,
		case 1:
			if constexpr (Interleave > 1) {
				n_interleave_cntg_loop<1, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                             src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                             cntg_first_next, cntg_last_next);
			}
			break;
		case 2:
			if constexpr (Interleave > 2) {
				n_interleave_cntg_loop<2, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                             src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                             cntg_first_next, cntg_last_next);
			}
			break;
		case 3:
			if constexpr (Interleave > 3) {
				n_interleave_cntg_loop<3, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                             src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                             cntg_first_next, cntg_last_next);
			}
			break;
		case 4:
			if constexpr (Interleave > 4) {
				n_interleave_cntg_loop<4, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                             src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                             cntg_first_next, cntg_last_next);
			}
			break;
		case 5:
			if constexpr (Interleave > 5) {
				n_interleave_cntg_loop<5, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                             src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                             cntg_first_next, cntg_last_next);
			}
			break;
		case 6:
			if constexpr (Interleave > 6) {
				n_interleave_cntg_loop<6, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                             src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                             cntg_first_next, cntg_last_next);
			}
			break;
		case 7:
			if constexpr (Interleave > 7) {
				n_interleave_cntg_loop<7, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                             src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                             cntg_first_next, cntg_last_next);
			}
			break;
		case 8:
			if constexpr (Interleave > 8) {
				n_interleave_cntg_loop<8, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                             src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                             cntg_first_next, cntg_last_next);
			}
			break;
		case 9:
			if constexpr (Interleave > 9) {
				n_interleave_cntg_loop<9, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                             src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                             cntg_first_next, cntg_last_next);
			}
			break;
		case 10:
			if constexpr (Interleave > 10) {
				n_interleave_cntg_loop<10, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                              src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                              cntg_first_next, cntg_last_next);
			}
			break;
		case 11:
			if constexpr (Interleave > 11) {
				n_interleave_cntg_loop<11, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                              src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                              cntg_first_next, cntg_last_next);
			}
			break;
		case 12:
			if constexpr (Interleave > 12) {
				n_interleave_cntg_loop<12, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                              src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                              cntg_first_next, cntg_last_next);
			}
			break;
		case 13:
			if constexpr (Interleave > 13) {
				n_interleave_cntg_loop<13, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                              src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                              cntg_first_next, cntg_last_next);
			}
			break;
		case 14:
			if constexpr (Interleave > 14) {
				n_interleave_cntg_loop<14, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                              src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                              cntg_first_next, cntg_last_next);
			}
			break;
		case 15:
			if constexpr (Interleave > 15) {
				n_interleave_cntg_loop<15, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                              src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                              cntg_first_next, cntg_last_next);
			}
			break;
		case 16:
			if constexpr (Interleave > 16) {
				n_interleave_cntg_loop<16, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                              src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                              cntg_first_next, cntg_last_next);
			}
			break;
		case 17:
			if constexpr (Interleave > 17) {
				n_interleave_cntg_loop<17, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                              src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                              cntg_first_next, cntg_last_next);
			}
			break;
		case 18:
			if constexpr (Interleave > 18) {
				n_interleave_cntg_loop<18, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                              src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                              cntg_first_next, cntg_last_next);
			}
			break;
		case 19:
			if constexpr (Interleave > 19) {
				n_interleave_cntg_loop<19, Interleave, Flags>(cntg_min, dst_cntg, src_strd_pos,
				                                              src_cntg_stride, src_strd_stride, dst_strd_pos,
				                                              cntg_first_next, cntg_last_next);
			}
			break;
		}
	}
}

} // namespace

// Shim has type of assembly kernels for backwards compat
template<std::size_t Interleave, kernel_inttype Flags, typename TSrc, typename TDst, typename ArchitectureSpec>
void n_cpp_interleave(
	std::size_t src_cntg, std::size_t src_strd, const TSrc *__restrict src, std::size_t src_cntg_stride, std::size_t src_strd_stride,
	std::size_t dst_cntg, std::size_t dst_strd,       TDst *__restrict dst, std::size_t dst_strd_stride,
	kernel_inttype submat_cntg, kernel_inttype submat_strd) {


	PERFLIBS_ASSERT(src_cntg_stride == 1, "cntg stride must be 1 to invoke n_cpp_interleave");

	constexpr step_val_fixed<1> one;

	kernel_inttype cntg_first=0, cntg_last=0;
	if constexpr (bool(Flags & interleave_flags::UpperVirt)) {
		cntg_first = submat_strd - submat_cntg;
		cntg_last  = 0;
	}
	else if constexpr (bool(Flags & interleave_flags::LowerVirt)) {
		cntg_first = 0;
		cntg_last  = submat_strd - submat_cntg;
	}

	omni_interleave<Interleave, Flags>(
		src_cntg, src_strd, src, one, src_strd_stride,
		dst_cntg, dst_strd, dst,      dst_strd_stride,
		cntg_first, cntg_last);
}

template<std::size_t Interleave, kernel_inttype Flags, typename TSrc, typename TDst, typename ArchitectureSpec>
void t_cpp_interleave(
	std::size_t src_cntg, std::size_t src_strd, const TSrc *__restrict src, std::size_t src_cntg_stride, std::size_t src_strd_stride,
	std::size_t dst_cntg, std::size_t dst_strd,       TDst *__restrict dst, std::size_t dst_strd_stride,
	kernel_inttype submat_cntg, kernel_inttype submat_strd) {

	PERFLIBS_ASSERT(src_strd_stride == 1, "strd stride must be 1 to invoke t_cpp_interleave");

	constexpr step_val_fixed<1> one;

	kernel_inttype cntg_first=0, cntg_last=0;
	if constexpr (bool(Flags & interleave_flags::UpperVirt)) {
		cntg_first = submat_strd - submat_cntg;
		cntg_last  = 0;
	}
	else if constexpr (bool(Flags & interleave_flags::LowerVirt)) {
		cntg_first = 0;
		cntg_last  = submat_strd - submat_cntg;
	}

	omni_interleave<Interleave, Flags>(
		src_cntg, src_strd, src, src_cntg_stride, one,
		dst_cntg, dst_strd, dst, dst_strd_stride,
		cntg_first, cntg_last);
}

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_INTERLEAVE_FALLBACK_KERNEL_HPP

#else // NOT PERFLIBS_LINALG_SOLVE_ROUTINE_ACTIVELY_REQUESTED

// This check is here to prevent accidental inclusion of interleave_fallback_kernel.hpp into any src file
// where interleave templates may get silently built by the compiler. The problem with this is that at this
// time optimizing the interleave routines is very slow, in particular pointer association enabled by
// -ftree-pta. Unfortunately -fno-tree-pta incurs a performance penalty.

#error "If you meant to include this header then you need to define " \
       "PERFLIBS_LINALG_INTERLEAVE_FALLBACK_KERNEL_ACTIVELY_REQUESTED=1"

#endif

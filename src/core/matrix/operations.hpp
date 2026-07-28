/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATRIX_OPERATIONS_HPP
#define PERFLIBS_LINALG_MATRIX_OPERATIONS_HPP

#include "perflibs_type_traits.hpp"
#include "framework/copy_kernels.hpp"
#include "framework/ge_kernels.hpp"

#include "matrix_base.hpp"
#include "adaptors.hpp"
#include "type_traits.hpp"

#include "perflibs_float.hpp"

namespace perflibs::linalg {

namespace {
/// Call the interleave_factor function if this matrix_type has one...
template<typename MatType>
PERFLIBS_LINALG_INLINE
auto interleave_factor_impl(const MatType& mat, int) -> decltype(mat.strd_interleave()) {
	return mat.strd_interleave();
}

///..else return one
template<typename MatType>
PERFLIBS_LINALG_INLINE
kernel_inttype interleave_factor_impl(const MatType&, ...)  {
	return 1;
}

} //namespace

/**
 * Returns the number of rows that are interleeaved together
 *
 * If the matrix is not interleaved, then the function returns 1
 */
template<typename MatType>
PERFLIBS_LINALG_INLINE
kernel_inttype interleave_factor(const MatType& mat){
	return interleave_factor_impl(mat, 0);
}

/**
 * the matrix types we have defined allow for either the cntg and strd dimensions
 * to be strided (consecutive access are != 1)
 *
 * however most of the optimsied routines (including helpers, ie interleave) expect
 * one of the dimensions to be contiguous
 *
 * is_cntg_contig predicates if consecutive access on the cntg dim are contiguous in memory
 */
template<typename MatrixType>
PERFLIBS_LINALG_INLINE
bool is_cntg_contig(const MatrixType& a) {
	return a.cntg_step() == 1;
}

/**
 * the matrix types we have defined allow for either the cntg and strd dimensions
 * to be strided (consecutive access are != 1)
 *
 * however most of the optimsied routines (including helpers, ie interleave) expect
 * one of the dimensions to be contiguous
 *
 * is_strd_contig predicates if consecutive access on the strd dim are contiguous in memory
 */
template<typename MatrixType>
PERFLIBS_LINALG_INLINE
bool is_strd_contig(const MatrixType& a) {
	return a.strd_step() == 1;
}

/**
 * Does the matrix have any entries
 */
template<typename MatrixType>
PERFLIBS_LINALG_INLINE
bool empty(const MatrixType& a) {
	return a.cntg() == 0
	    || a.strd() == 0;
}

template<typename MatrixType>
PERFLIBS_LINALG_INLINE
bool is_vector(MatrixType& a) {
	return (a.strd() == 1 && a.cntg() >= 1) || (a.cntg() == 1 && a.strd() >= 1);
}

/**
 * Used to instruct the bounds functions as to how it should
 * handle the diagonals
 *
 * we are not using bools because there may be further flags to be added
 * which have a default value and we do not want the bools to get mixed up
 * or promoted to ints in the overloads
 */
enum class include_diag {
	yes, no
}; //enum class include_diag

template<typename MatrixType>
PERFLIBS_LINALG_INLINE
std::pair<kernel_inttype, kernel_inttype> get_non_virtual_cntg_bounds_for_strd(const MatrixType& mat, kernel_inttype strd, include_diag inc_diag = include_diag::yes) {
	if constexpr(! is_triangular_form_v<MatrixType>) {
		return { 0, mat.cntg() };
	}
	else {
		const auto acntg = mat.absolute_cntg();
		const auto astrd = mat.absolute_strd() + strd;

		if(mat.is_lower()) {
			kernel_inttype diag_is_virtual =  0;

			if(inc_diag == include_diag::no)
				diag_is_virtual = !mat.is_diag_physical();

			return { min(max(astrd - acntg + diag_is_virtual, 0), mat.cntg()), mat.cntg()                         };

		}
		else {
			kernel_inttype diag_is_physical = 1;

			if(inc_diag == include_diag::no)
				diag_is_physical = mat.is_diag_physical();

			return { 0,                    max(min(astrd - acntg + diag_is_physical, mat.cntg()), 0) };
		}
	}
}

template<typename MatrixType>
PERFLIBS_LINALG_INLINE
std::pair<kernel_inttype, kernel_inttype>
get_non_virtual_cntg_bounds_for_strd(const MatrixType& mat, kernel_inttype strd_first,
                                kernel_inttype strd_last, include_diag inc_diag = include_diag::yes) {
	auto [cntg_first0, cntg_last0] = get_non_virtual_cntg_bounds_for_strd(mat, strd_first, inc_diag);
	auto [cntg_first1, cntg_last1] = get_non_virtual_cntg_bounds_for_strd(mat, strd_last, inc_diag);

	return { min(cntg_first0, cntg_first1), max(cntg_last0, cntg_last1) };
}

template<typename MatrixType>
PERFLIBS_LINALG_INLINE
std::pair<kernel_inttype, kernel_inttype> get_non_virtual_cntg_bounds_for_strd(const MatrixType& mat, include_diag inc_diag = include_diag::yes) {
	return get_non_virtual_cntg_bounds_for_strd(mat, 0, mat.strd() - 1, inc_diag);
}

template<typename MatrixType>
PERFLIBS_LINALG_INLINE
std::pair<kernel_inttype, kernel_inttype> get_non_virtual_strd_bounds_for_cntg(const MatrixType& mat, kernel_inttype cntg, include_diag inc_diag = include_diag::yes) {
	if constexpr(! is_triangular_form_v<MatrixType>) {
		return { 0, mat.strd() };
	}
	else {
		const auto astrd = mat.absolute_strd();
		const auto acntg = mat.absolute_cntg() + cntg;

		//please see the comment in adapators.hpp symmetrix_matrix::operator()
		//for any explanation to the storage of the triangle.

		if(mat.is_lower()) {
			kernel_inttype diag_is_physical = 1;

			if(inc_diag == include_diag::no)
				diag_is_physical = mat.is_diag_physical();

			return {
				0, max(min(acntg - astrd + diag_is_physical, mat.strd()), 0)
			};
		}
		else {
			kernel_inttype diag_is_virtual = 0;

			if(inc_diag == include_diag::no) {
				diag_is_virtual = !mat.is_diag_physical();
			}


			return { min(max(acntg - astrd + diag_is_virtual, 0), mat.strd()), mat.strd() };
		}
	}
}

template<typename MatrixType>
PERFLIBS_LINALG_INLINE
std::pair<kernel_inttype, kernel_inttype>
get_non_virtual_strd_bounds_for_cntg(const MatrixType& mat, kernel_inttype cntg_first,
                                kernel_inttype cntg_last, include_diag inc_diag = include_diag::yes) {
	auto [strd_first0, strd_last0] = get_non_virtual_strd_bounds_for_cntg(mat, cntg_first, inc_diag);
	auto [strd_first1, strd_last1] = get_non_virtual_strd_bounds_for_cntg(mat, cntg_last, inc_diag);

	return { min(strd_first0, strd_first1), max(strd_last0, strd_last1) };
}


template<typename MatrixType>
PERFLIBS_LINALG_INLINE
std::pair<kernel_inttype, kernel_inttype> get_non_virtual_strd_bounds_for_cntg(const MatrixType& mat, include_diag inc_diag = include_diag::yes) {
	return get_non_virtual_strd_bounds_for_cntg(mat, 0, mat.cntg() - 1, inc_diag);
}

namespace {

template<typename SrcMatType, typename DstMatType>
PERFLIBS_LINALG_INLINE static
void copy_general_fallback(const SrcMatType& src, DstMatType& dst, bool zero_slack = false) {
	using dst_data_type = typename DstMatType::value_type;

	for(kernel_inttype i = 0; i < min(src.strd(), dst.strd()); ++i) {
		for(kernel_inttype j = 0; j < min(src.cntg(), dst.cntg()); ++j) {
			dst(j, i, write) = src(j, i);
		}
		if(zero_slack) {
			for(kernel_inttype j = src.cntg(); j < dst.cntg(); ++j) {
				dst(j, i, write) = zero<dst_data_type>;
			}
		}
	}

	if(zero_slack) {
		for(kernel_inttype i = src.strd(); i < dst.strd(); ++i) {
			for(kernel_inttype j = src.cntg(); j < dst.cntg(); ++j) {
				dst(j, i, write) = zero<dst_data_type>;
			}
		}
	}
}

}

/*
 * An overload of copy designed to handle copying two general matrices where
 * we can use optimized assembly kernels
 */
template<typename SrcMatType, typename DstMatType>
PERFLIBS_LINALG_INLINE static
void copy_general(const SrcMatType& src, DstMatType& dst, bool zero_slack = false) {

	//if the types are the samne, and they are one of the approve list
	constexpr bool is_optimised =
		value_type_is_one_of_v<SrcMatType, double, float, half> &&
		std::is_same_v<typename SrcMatType::value_type, typename DstMatType::value_type>;

	if constexpr(is_optimised) {
		using value_type = std::remove_cv_t<typename SrcMatType::value_type>;
		/*
		 * gecpy does not yet currently support padding
		 * this could potentially be done in a two stage process, a call to the kernel
		 * followed by clean up -> but for now we'll do it in C++ via the fallback
		 */
		if(!zero_slack) {
			if(src.cntg_step() == 1 && dst.cntg_step() == 1) {
				gecpy_kernel<value_type>(
					min(src.cntg(), dst.cntg()),
					min(src.strd(), dst.strd()),
					src.data(), src.strd_step(),
					dst.data(), dst.strd_step());
				return;
			}
			else if(src.strd_step() == 1 && dst.strd_step() == 1) {
				gecpy_kernel<value_type>(
					min(src.strd(), dst.strd()),
					min(src.cntg(), dst.cntg()),
					src.data(), src.cntg_step(),
					dst.data(), dst.cntg_step());
				return;
			}
		}
	}
	copy_general_fallback(src, dst, zero_slack);
}

/**
 * Copies the contents of one matrix into another matrix
 *
 * @param src [in] the src matrix
 * @param dst [out] the dst matrix
 * @param zero_slack [in] when specified any region of dst that is larger than src will be filled with zeros
 */
//copy from any matrix to any matrix
template<typename SrcMatType, typename DstMatType>
PERFLIBS_LINALG_INLINE
void copy_fallback(const SrcMatType& src, DstMatType& dst, bool zero_slack = false) {
	using dst_data_type = typename DstMatType::value_type;
	const auto strd_last = min(src.strd(), dst.strd());

	for(kernel_inttype i = 0; i < strd_last; ++i) {
		const auto [ dst_cntg_first, dst_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(dst, i);
		const auto cntg_first = dst_cntg_first;
		const auto cntg_last  = min(src.cntg(),  dst_cntg_last);

		for(kernel_inttype j = cntg_first; j < cntg_last; ++j) {
			if constexpr(is_general_matrix_v<DstMatType> || is_triangular_matrix_v<DstMatType>) {
				dst(j, i, write) = src(j, i);
			}
			else {
				dst(j, i) = src(j, i);
			}
		}
		if(zero_slack) {
			//if dst_cnt_last is greater than the the cntg of src, and the user wants to pad the result
			for(kernel_inttype j = cntg_last; j < dst_cntg_last; ++j) {
				if constexpr(is_general_matrix_v<DstMatType> || is_triangular_matrix_v<DstMatType>) {
					dst(j, i, write) = zero<dst_data_type>;
				}
				else {
					dst(j, i) = zero<dst_data_type>;
				}
			}
		}
	}
	if(zero_slack) {
		for(kernel_inttype i = strd_last; i < dst.strd(); ++i) {
			const auto [ dst_cntg_first, dst_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(dst, i);
			for(kernel_inttype j = dst_cntg_first; j < dst_cntg_last; ++j) {
				if constexpr(is_general_matrix_v<DstMatType> || is_triangular_matrix_v<DstMatType>) {
					dst(j, i, write) = zero<dst_data_type>;
				}
				else {
					dst(j, i) = zero<dst_data_type>;
				}
			}
		}
	}
}

namespace {
template<typename SrcMatType, typename DstMatType>
[[maybe_unused]]
void copy_from_tri(const SrcMatType& src, DstMatType& dst, bool zero_slack) {
	using value_type = typename SrcMatType::value_type;

	PERFLIBS_ASSERT(! zero_slack, "not implemented yet");

	const kernel_inttype src_cntg_step = src.cntg_step();
	const kernel_inttype src_strd_step = src.strd_step();
	auto src_ptr = src.data();
	const kernel_inttype dst_cntg_step = dst.cntg_step();
	const kernel_inttype dst_strd_step = dst.strd_step();
	auto dst_ptr = dst.data();

	//TODO: this is not satisfactory. On an SVE machine this will get a neon kernel
	const auto copy_kernel = get_neon_copy_kernel<value_type>(src_cntg_step, dst_cntg_step);

	//this algorithm is somewhat optimized for cntg_step == 1, an example of use is in syrk where we
	//write into the user output matrix in c_copy to ensure we maintain triangular_form

	if(src.is_upper()) {
		auto [ src_strd_first, src_strd_last ] = get_non_virtual_strd_bounds_for_cntg(src, 0);
		auto [ src_cntg_first, src_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(src, src_strd_first);
		PERFLIBS_UNUSED(src_cntg_first);

		const auto strd_last = min(src_strd_last, dst.strd());
		const auto cntg_last = min(src.cntg(), dst.cntg());

		for(kernel_inttype i=src_strd_first; i!=strd_last; ++i) {
			auto dst_ptr_cur = dst_ptr + i * dst_strd_step;
			auto src_ptr_cur = src_ptr + i * src_strd_step;

			copy_kernel(
				src_cntg_last,
				src_ptr_cur, src_cntg_step,
				dst_ptr_cur, dst_cntg_step);

			src_cntg_last = min(src_cntg_last + 1, cntg_last);
		}
	}
	else {
		const auto cntg_last = min(dst.cntg(), src.cntg());

		auto [ src_strd_first, src_strd_last ] = get_non_virtual_strd_bounds_for_cntg(src, 0);
		auto [ src_cntg_first, src_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(src, src_strd_last);

		PERFLIBS_UNUSED(src_cntg_first);
		PERFLIBS_UNUSED(src_strd_first);

		const auto strd_last = min((src_strd_last + cntg_last - 1), min(dst.strd(), src.strd()));

		for(kernel_inttype i = 0; i < src_strd_last; ++i) {
			auto dst_ptr_cur = dst_ptr + i * dst_strd_step;
			auto src_ptr_cur = src_ptr + i * src_strd_step;

			copy_kernel(
				cntg_last,
				src_ptr_cur, src_cntg_step,
				dst_ptr_cur, dst_cntg_step);
		}


		for(kernel_inttype i = src_strd_last; i < strd_last; ++i) {
			auto dst_ptr_cur = dst_ptr + i * dst_strd_step + src_cntg_first * dst_cntg_step;
			auto src_ptr_cur = src_ptr + i * src_strd_step + src_cntg_first * src_cntg_step;

			auto cntg_len = src_cntg_last - src_cntg_first;

			if(cntg_len == 0)
				break;

			copy_kernel(
				cntg_len,
				src_ptr_cur, src_cntg_step,
				dst_ptr_cur, dst_cntg_step);

			++src_cntg_first;
		}
	}
}

} // namespace anon

template<typename SrcMatType, typename DstMatType>
[[maybe_unused]]
static void copy_to_tri(const SrcMatType& src, DstMatType& dst, bool zero_slack) {
	using value_type = typename SrcMatType::value_type;

	PERFLIBS_ASSERT(! zero_slack, "not implemented yet");

	const kernel_inttype src_cntg_step = src.cntg_step();
	const kernel_inttype src_strd_step = src.strd_step();
	auto src_ptr = src.data();
	const kernel_inttype dst_cntg_step = dst.cntg_step();
	const kernel_inttype dst_strd_step = dst.strd_step();
	auto dst_ptr = dst.data();

	//TODO: this is not satisfactory. On an SVE machine this will get a neon kernel
	const auto copy_kernel = get_neon_copy_kernel<value_type>(src_cntg_step, dst_cntg_step);

	//this algorithm is somewhat optimized for cntg_step == 1, an example of use is in syrk where we
	//write into the user output matrix in c_copy to ensure we maintain triangular_form
	if(dst.is_upper()) {
		auto [ dst_strd_first, dst_strd_last ] = get_non_virtual_strd_bounds_for_cntg(dst, 0);
		auto [ dst_cntg_first, dst_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(dst, dst_strd_first);
		PERFLIBS_UNUSED(dst_cntg_first);
		PERFLIBS_UNUSED(dst_strd_first);

		const auto strd_last = min(dst_strd_last, src.strd());
		const auto cntg_last = min(dst.cntg(), src.cntg());

		for(kernel_inttype i=dst_strd_first; i!=strd_last; ++i) {
			auto dst_ptr_cur = dst_ptr + i * dst_strd_step;
			auto src_ptr_cur = src_ptr + i * src_strd_step;

			copy_kernel(
				dst_cntg_last,
				src_ptr_cur, src_cntg_step,
				dst_ptr_cur, dst_cntg_step);

			dst_cntg_last = min(dst_cntg_last + 1, cntg_last);
		}
	}
	else {
		const auto cntg_last = min(dst.cntg(), src.cntg());

		auto [ dst_strd_first, dst_strd_last ] = get_non_virtual_strd_bounds_for_cntg(dst, 0);
		auto [ dst_cntg_first, dst_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(dst, dst_strd_last);
		PERFLIBS_UNUSED(dst_strd_first);

		const auto strd_last = min((dst_strd_last + cntg_last - 1), min(dst.strd(), src.strd()));

		for(kernel_inttype i = 0; i < dst_strd_last; ++i) {
			auto dst_ptr_cur = dst_ptr + i * dst_strd_step;
			auto src_ptr_cur = src_ptr + i * src_strd_step;

			copy_kernel(
				cntg_last,
				src_ptr_cur, src_cntg_step,
				dst_ptr_cur, dst_cntg_step);
		}

		for(kernel_inttype i = dst_strd_last; i < strd_last; ++i) {
			auto dst_ptr_cur = dst_ptr + i * dst_strd_step + dst_cntg_first * dst_cntg_step;
			auto src_ptr_cur = src_ptr + i * src_strd_step + dst_cntg_first * src_cntg_step;

			auto cntg_len = dst_cntg_last - dst_cntg_first;

			if(cntg_len == 0)
				break;

			copy_kernel(
				cntg_len,
				src_ptr_cur, src_cntg_step,
				dst_ptr_cur, dst_cntg_step);

			++dst_cntg_first;
		}
	}
}

template<typename SrcMatType, typename DstMatType>
PERFLIBS_LINALG_INLINE
void copy_adaptor(const SrcMatType& src, DstMatType& dst, bool zero_slack = false) {

	auto gen_dst = to_general_matrix(dst);

	const auto strd_last = min(src.strd(), dst.strd());

	for(kernel_inttype i = 0; i < strd_last; ++i) {
		const auto [ src_cntg_first, src_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(src, i);
		const auto [ dst_cntg_first, dst_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(dst, i);

		const auto cntg_first = max(src_cntg_first, dst_cntg_first);
		const auto cntg_last  = min(src_cntg_last,  dst_cntg_last);

		for(kernel_inttype j = cntg_first; j < cntg_last; ++j) {
			gen_dst(j, i, write) = src(j, i);
		}
		if(zero_slack) {
			//if dst_cnt_last is greater than the the cntg of src, and the user wants to pad the result
			for(kernel_inttype j = cntg_last; j < dst_cntg_last; ++j) {
				gen_dst(j, i, write) = zero<typename DstMatType::value_type>;
			}
		}
	}
	if(zero_slack) {
		for(kernel_inttype i = strd_last; i < dst.strd(); ++i) {
			const auto [ dst_cntg_first, dst_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(dst, i);
			for(kernel_inttype j = dst_cntg_first; j < dst_cntg_last; ++j) {
				gen_dst(j, i, write) = zero<typename DstMatType::value_type>;
			}
		}
	}
}


template<typename SrcMatType, typename MatBaseType1>
PERFLIBS_LINALG_INLINE
void copy_to_herm(const SrcMatType& src, hermitian_matrix<MatBaseType1>& dst, bool zero_slack = false) {
	using dst_data_type = typename MatBaseType1::value_type;

	auto gen_dst = to_general_matrix(dst);
	triangular_matrix<MatBaseType1> tri_dst { dst.uplo(), PERFLIBS_NOUNIT, dst };

	const auto strd_last = min(src.strd(), dst.strd());

	for(kernel_inttype i = 0; i < strd_last; ++i) {
		const auto [ src_cntg_first, src_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(src, i);
		const auto [ dst_cntg_first, dst_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(tri_dst, i);

		const auto cntg_first = max(src_cntg_first, dst_cntg_first);
		const auto cntg_last  = min(src_cntg_last,  dst_cntg_last);

		for(kernel_inttype j = cntg_first; j < cntg_last; ++j) {
			gen_dst(j, i, write) = src(j, i);
		}
		if(zero_slack) {
			//if dst_cnt_last is greater than the the cntg of src, and the user wants to pad the result
			for(kernel_inttype j = cntg_last; j < dst_cntg_last; ++j) {
				gen_dst(j, i, write) = zero<dst_data_type>;
			}
		}
	}
	if(zero_slack) {
		for(kernel_inttype i = strd_last; i < dst.strd(); ++i) {
			const auto [ dst_cntg_first, dst_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(tri_dst, i);
			for(kernel_inttype j = dst_cntg_first; j < dst_cntg_last; ++j) {
				gen_dst(j, i, write) = zero<dst_data_type>;
			}
		}
	}
}

template<typename SrcMatType, typename DstMatType>
PERFLIBS_LINALG_INLINE static
void copy(const SrcMatType& src, DstMatType& dst, bool zero_slack = false) {
	using src_value_type_rcv = std::remove_cv_t<typename SrcMatType::value_type>;
	using dst_value_type_rcv = std::remove_cv_t<typename DstMatType::value_type>;

	//general matrices with the same datatype
	if constexpr(is_general_matrix_v<SrcMatType>
	          && is_general_matrix_v<DstMatType>
	          && std::is_same_v<src_value_type_rcv, dst_value_type_rcv>) {
		return copy_general(src, dst, zero_slack);
	}
	//copy from general to tri or symm where data type matches
	else if constexpr(is_general_matrix_v<SrcMatType> &&
			//( is_triangular_matrix_v<DstMatType> || is_symmetric_matrix_v<DstMatType>)
			( is_triangular_form_v<DstMatType> )
	          && std::is_same_v<src_value_type_rcv, dst_value_type_rcv>) {
		return copy_to_tri(src, dst, zero_slack);
	}
	else if constexpr(is_general_matrix_v<DstMatType> &&
			//( is_triangular_matrix_v<SrcMatType> || is_symmetric_matrix_v<SrcMatType> ||)
			( is_triangular_form_v<SrcMatType> )
	          && std::is_same_v<src_value_type_rcv, dst_value_type_rcv>) {
		return copy_from_tri(src, dst, zero_slack);
	}
	else if constexpr(is_hermitian_matrix_v<DstMatType>) {
		return copy_to_herm(src, dst, zero_slack);
	}

	else if constexpr(is_matrix_adaptor_v<DstMatType>) {
		return copy_adaptor(src, dst, zero_slack);
	}
	else {
		return copy_fallback(src, dst, zero_slack);
	}
}


namespace {

template<typename MatrixType>
constexpr bool set_case_1_v = is_general_matrix_v<MatrixType> &&
	value_type_is_one_of_v<MatrixType, double, float, half>;

template<typename MatrixType>
PERFLIBS_LINALG_INLINE
bool set_optimised(const typename MatrixType::value_type& val, MatrixType& dst) {
	using value_type = std::remove_cv_t<typename MatrixType::value_type>;
	if(is_cntg_contig(dst)) {
		geset_kernel<value_type>(val,
			dst.cntg(), dst.strd(),
			dst.data(), dst.strd_step());
		return  true;
	}
	else if(is_strd_contig(dst)) {
		geset_kernel<value_type>(val,
			dst.strd(), dst.cntg(),
			dst.data(), dst.cntg_step());
		return true;
	}
	return false;
}

template<typename MatrixType>
PERFLIBS_LINALG_INLINE
void set_impl(const typename MatrixType::value_type& val, MatrixType& in) {
	auto gen_in = to_general_matrix(in);

	for(kernel_inttype i = 0; i != in.strd(); ++i) {
		const auto [ cntg_first, cntg_last ] = get_non_virtual_cntg_bounds_for_strd(in, i);

		for(kernel_inttype j = cntg_first; j != cntg_last; ++j) {
			gen_in(j, i, write) = val;
		}
	}
}

} //namsepace

/**
 * Sets every value in a matrix to have the same provided value
 *
 * @param val the value to set every element of the matrix to
 * @param in the matrix to which  the operation is applied
 */
template<typename MatrixType>
PERFLIBS_LINALG_INLINE
auto set(const typename MatrixType::value_type& val, MatrixType& in) -> std::enable_if_t<set_case_1_v<MatrixType>> {
	if(! set_optimised(val, in))
		set_impl(val, in);
}

template<typename MatrixType>
PERFLIBS_LINALG_INLINE
auto set(const typename MatrixType::value_type& val, MatrixType& in) -> std::enable_if_t<!set_case_1_v<MatrixType>> {

	set_impl(val, in);
}


namespace {

template<typename SrcMatType, typename DstMatType>
PERFLIBS_LINALG_INLINE
bool scale_optimised(const typename DstMatType::value_type& val, const SrcMatType& src, DstMatType& dst) {
	using value_type = std::remove_cv_t<typename DstMatType::value_type>;
	if(is_cntg_contig(src) && is_cntg_contig(dst)) {
		gescal_out_of_place_kernel<value_type>(val,
			src.cntg(), src.strd(),
			src.data(), src.strd_step(),
			dst.data(), dst.strd_step());
		return  true;
	}
	else if(is_strd_contig(src) && is_strd_contig(dst)) {
		gescal_out_of_place_kernel<value_type>(val,
			src.strd(), src.cntg(),
			src.data(), src.cntg_step(),
			dst.data(), dst.cntg_step());
		return true;
	}
	return false;
}

} //namsepace


/**
 * Scales every value in a matrix by the provided scalar value
 *  in *= val
 *
 * @param val the value to scale every element of the matrix by
 * @param in the matrix to which  the operation is applied
 */
template<typename MatrixType>
PERFLIBS_LINALG_INLINE
void scale(const typename MatrixType::value_type& val, MatrixType& in) {
	using T1 = typename MatrixType::value_type;

	if(val == zero<T1>) {
		set(zero<T1>, in);
		return;
	}
	else if(val == one<T1>) {
		//as this is an inplace scale, this is no-op
		return;
	}

	auto gen_in = to_general_matrix(in);

	for(kernel_inttype i = 0; i != in.strd(); ++i) {
		const auto [ cntg_first, cntg_last ] = get_non_virtual_cntg_bounds_for_strd(in, i);

		for(kernel_inttype j = cntg_first; j != cntg_last; ++j) {
			gen_in(j, i, write) = in(j, i) *val;
		}
	}
}

/**
 * Out of place scale function
 * 	dst = val * src
 *
 * @param val the value to scale every element of the matrix by
 * @param src the matrix where the initial values are taken
 * @param dst the matrix where the scaled values will be stored
 */
template<typename SrcMatType, typename DstMatType>
PERFLIBS_LINALG_INLINE
auto scale(const typename DstMatType::value_type& val, const SrcMatType& src, DstMatType& dst) -> std::enable_if_t<is_result_of_callable_assignable_v<DstMatType>> {

	using T0 = typename SrcMatType::value_type;
	using T1 = typename DstMatType::value_type;

	if(val == zero<T1>) {
		set(zero<T1>, dst);
		return;
	}
	else if(val == one<T1>) {
		copy(src, dst);
		return;
	}

	if constexpr(all_same_v<double, T0, T1> || all_same_v<float, T0, T1> || all_same_v<half, T0, T1>) {
		// Returns true if the optimized kernel performed the task; may fail for irregular strides.
		if(scale_optimised(val, src, dst))
			return;
	}

	for(kernel_inttype i = 0; i != min(src.strd(), dst.strd()); ++i) {
		const auto [ src_cntg_first, src_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(src, i);
		const auto [ dst_cntg_first, dst_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(dst, i);

		const auto cntg_first = max(src_cntg_first, dst_cntg_first);
		const auto cntg_last  = min(src_cntg_last,  dst_cntg_last);

		for(kernel_inttype j = cntg_first; j < cntg_last; ++j) {
			dst(j, i, write) = src(j, i) * val;
		}
	}
}

/**
 * Overload of out of place scale to handle non-general matrix adaptors
 */
template<typename SrcMatType, typename DstMatType>
PERFLIBS_LINALG_INLINE
auto scale(const typename DstMatType::value_type& val, const SrcMatType& src, DstMatType& dst) -> std::enable_if_t<! is_result_of_callable_assignable_v<DstMatType>> {
	using T1 = typename DstMatType::value_type;

	if(val == zero<T1>) {
		set(zero<T1>, dst);
		return;
	}
	else if(val == one<T1>) {
		copy(src, dst);
		return;
	}

	auto gen_dst = to_general_matrix(dst);

	for(kernel_inttype i = 0; i != min(src.strd(), dst.strd()); ++i) {
		const auto [ src_cntg_first, src_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(src, i);
		const auto [ dst_cntg_first, dst_cntg_last ] = get_non_virtual_cntg_bounds_for_strd(dst, i);

		const auto cntg_first = max(src_cntg_first, dst_cntg_first);
		const auto cntg_last  = min(src_cntg_last,  dst_cntg_last);

		for(kernel_inttype j = cntg_first; j < cntg_last; ++j) {
			gen_dst(j, i, write) = src(j, i) * val;
		}
	}
}

/**
 * returns panel (sub_matrix) of the input matrix which fills the entire cntg dimension for
 * range of strd dimensions
 *
 *     strd
 *    --------
 * c |  |p|   |
 * n |  |a|   |
 * t |  |n|   |
 * g |  |e|   |
 *   |  |l|   |
 *    --------
 *
 * @param in the parent matrix, from which we will get a sub_matrix
 * @param strd_pos the index of the start of the strd range for the panel
 * @param strd_size the size of the panel in the strd dimension
 *
 */
template<typename MatrixType>
PERFLIBS_LINALG_INLINE
MatrixType get_strd_panel(MatrixType& in, kernel_inttype strd_pos, kernel_inttype strd_size) {
 	return in.sub_matrix(0, in.cntg(), strd_pos, strd_size);
}

/**
 * returns panel (sub_matrix) of the input matrix which fills the entire strd dimension for
 * range of cntg dimensions
 *
 *     strd
 *    ------
 * c |      |
 * n |------|
 * t |panel |
 * g |------|
 *   |      |
 *    ------
 *
 * @param in the parent matrix, from which we will get a sub_matrix
 * @param cntg_pos the index of the start of the cntg range for the panel
 * @param cntg_size the size of the panel in the cntg dimension
 *
 */
template<typename MatrixType>
PERFLIBS_LINALG_INLINE
MatrixType get_cntg_panel(MatrixType& in, kernel_inttype cntg_pos, kernel_inttype cntg_size) {
 	return in.sub_matrix(cntg_pos, cntg_size, 0, in.strd());
}

/**
 * Same as get_strd_panel, however if dimension are greater than the size of the matrix then
 * they are cropped and trims to fit.
 *
 * useful for getting panels in a loop which increments by a fixed block size which may not
 * be a multiple of the matrices size
 */
template<typename MatrixType>
PERFLIBS_LINALG_INLINE
MatrixType get_strd_panel_with_clamp(MatrixType& in, kernel_inttype strd_pos, kernel_inttype strd_size) {
 	return in.sub_matrix_with_clamp(0, in.cntg(), strd_pos, strd_size);
}

/**
 * Same as get_cntg_panel, however if dimension are greater than the size of the matrix then
 * they are cropped and trims to fit.
 *
 * useful for getting panels in a loop which increments by a fixed block size which may not
 * be a multiple of the matrices size
 */
template<typename MatrixType>
PERFLIBS_LINALG_INLINE
MatrixType get_cntg_panel_with_clamp(MatrixType& in, kernel_inttype cntg_pos, kernel_inttype cntg_size) {
 	return in.sub_matrix_with_clamp(cntg_pos, cntg_size, 0, in.strd());
}

/**
 * Compares the dimensions and virtual values of the
 * two input matrices.
 *
 * By virtual values, consider Lhs is a triangular matrix,
 * an Rhs is a general matrix which happens to have contain
 * the values that that equate to Lhs, even if the underlying
 * memory is different.
 */
template<typename MatrixTypeLhs, typename MatrixTypeRhs>
PERFLIBS_LINALG_INLINE
bool equal(const MatrixTypeLhs& lhs, const MatrixTypeRhs& rhs,
		perflibs::remove_complex_t<typename MatrixTypeLhs::value_type> tolerance_base=0.0) {

	using lhs_value_type = typename MatrixTypeLhs::value_type;
	using rhs_value_type = typename MatrixTypeRhs::value_type;

	//if the values_types don't match the matrices aren't equal, this
	//could be an option so you could compare float to double
	if constexpr(! std::is_same_v<lhs_value_type, rhs_value_type>) {
		return false;
	}

	//check whether the matrices have the same shape
	if(lhs.cntg() != rhs.cntg() || lhs.strd() != rhs.strd()) {
		return false;
	}

	//check whether the matrices have the same content
	for(kernel_inttype j = 0; j != lhs.strd(); ++j) {
		for(kernel_inttype i = 0; i != lhs.cntg(); ++i) {
			const auto diff  = std::abs(lhs(i, j) - rhs(i, j));
			const auto limit = std::abs(lhs(i, j)) * tolerance_base;

			if(diff > limit) {
				return false;
			}
		}
	}
	return true;
}

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_MATRIX_OPERATIONS_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FRAMEWORK_CONVERT_HPP
#define PERFLIBS_LINALG_FRAMEWORK_CONVERT_HPP

#include "perflibs_assert.hpp"
#include "perflibs_numeric_utils.hpp"
#include "perflibs_util.hpp"
#include "framework/linalg_util.hpp"
#include "matrix/matrix.hpp"
#include "matrix/type_traits.hpp"

namespace perflibs::linalg {

/**
 * Converts the data passed to its operator() from unpacked input data into
 * the interleaved and transposed format as used by the big kernels
 *
 * @tparam T0 the data type of the input matrix
 * @tparam T1 the data type of the output matrix
 * @tparam InParallel is this a team conversion (all threads in a team want the
 *         same data converting in the same order)
 * @tparam InterleaveFunc - Override for SVE implementations / split complex
 * implementations (see framework/interleave_kernels.hpp for defaults)
 */
template<
	typename T0,
	typename T1,
	typename InterleaveFunc
>
class convert {
	///the number of strd that will be interleaved contiguously
	kernel_inttype interleaved_strd_;
	///packed_matrix<T> is made up of multiple panels, the max size of which is the width of the inner k_loop before calling the gemm (in the LINALG there can be multiple K_blocking)
	//this can be set to std::numeric_limits<kernel_inttype>::max() to disable paneling
	kernel_inttype panel_cntg_;
	//ensure that the panel_cntg_ is divisible by this number
	//useful when the kernel is unrolled and needs to process N number of cntg
	kernel_inttype panel_cntg_divis_;
	// Same but in the strided dimension
	kernel_inttype panel_strd_divis_;
	kernel_inttype split_factor_;

	InterleaveFunc interleave_func_;

	matrix_base<T0> last_packed_;
	T1 *last_buffer_;

	/*
	 * When enabled, prevents the reuse of previously packed matrices
	 * which, in some cases, causes incorrect behavior, though improves
	 * performance in most cases
	 */
	bool always_pack_;

	using packed_matrix_type = typename InterleaveFunc::template matrix_type<T1>;
public:
	convert(kernel_inttype interleaved_strd, kernel_inttype panel_cntg, kernel_inttype panel_cntg_divis, kernel_inttype panel_strd_divis, kernel_inttype split_factor, InterleaveFunc interleave_func={}, bool always_pack=true)
	:	interleaved_strd_ { interleaved_strd }
	,	panel_cntg_       { panel_cntg       }
	,	panel_cntg_divis_ { panel_cntg_divis }
	,	panel_strd_divis_ { panel_strd_divis }
	,	split_factor_     { split_factor     }
	,	interleave_func_  { interleave_func  }
	,	last_packed_      {                  }
	,	last_buffer_      { nullptr          }
	,	always_pack_      { always_pack      }
	{	}

	/**
	 * Used to ensure that the packed matrices return from make_packed
	 * has the correct internal settings
	 */
	inline
	kernel_inttype interleave() const { return interleaved_strd_; }

	/*
	 * returns packed matrix with the dimensions of input matrix
	 * This is a preparatory step, it does not actually populate the internal buffer
	 *
	 * This function can either be chained with operator(unpacked, packed)
	 *
	 * or you can call operator(unpacked, buffer) and it will call this function
	 * on your behalf (preferable when applicable)
	 *
	 */
	template<typename MatrixType>
	inline
	packed_matrix_type make_packed(const MatrixType& in, T1 *buffer) const {
		PERFLIBS_ASSERT(buffer != nullptr, "buffer must not be nullptr");

		const kernel_inttype cntg_exten = iround(in.cntg(), panel_cntg_divis_);
		const kernel_inttype strd_step = cntg_exten * interleaved_strd_;

		if constexpr(is_split_complex_matrix_v<packed_matrix_type>) {
			return { buffer, interleaved_strd_, split_factor_, cntg_exten, in.strd(), strd_step };
		}
		else {
			return { buffer, interleaved_strd_, cntg_exten, in.strd(), strd_step };
		}
	}

	/**
	 * Takes an unpacked matrix and a buffer and returns a packed_matrix
	 */
	template<typename MatrixType>
	PERFLIBS_LINALG_INLINE
	packed_matrix_type operator()(const MatrixType& in, T1 *buffer) {
		auto out = make_packed(in, buffer);

		if( ! (!always_pack_ && quick_equal(in, last_packed_) && buffer == last_buffer_)) {
			(*this)(in, out);
		}

		last_packed_ = in;
		last_buffer_ = buffer;

		return out;
	}

	template<typename MatrixTypeSrc, typename MatrixTypeDst>
	PERFLIBS_LINALG_INLINE
	void operator()(const MatrixTypeSrc& in, MatrixTypeDst& out) {
		PERFLIBS_ASSERT(out.interleave() == interleave(),
			"this packed matrix does meet the specification used by this conversion object");

		/*
		 * if the Src type is a general_matrix then we can use the interleave_func
		 * to do an optimized interleave
		 */
		if constexpr ( is_general_matrix_v<MatrixTypeSrc> || is_symmetric_matrix_v<MatrixTypeSrc> || is_hermitian_matrix_v<MatrixTypeSrc>) {
			interleave_func_(in, out);
		}
		else if (is_triangular_matrix_v<MatrixTypeSrc> && !in.is_conj()) {
			interleave_func_(in, out);
		}
		else {
			/*
			 * If it is not, but the region of the virtual matrix adaptor is physical
			 * (such as the "physical half" of a symm matrix) then we can still use
			 * the optimized version
			 *
			 * Otherwise we will use copy(to, from, zero_slack) which uses the op() interface
			 */
			if(in.is_physical()) {
				//as this is_physical it can safely be converted to a general matrix
				general_unpacked_matrix<T0> general_panel { in };

				interleave_func_(general_panel, out);
			}
			else {
				//as this is_physical it can safely be converted to a general matrix
				copy(in, out, true);
			}
		}
	}

	/**
	 * Given a matrix of dimension described by the parameters, return how many elements of
	 * sizeof(T1) you need to pass in as buffer
	 */
	kernel_inttype elements_required(kernel_inttype max_cntg, kernel_inttype max_strd) {
		PERFLIBS_ASSERT(max_cntg >= 0 && max_strd >= 0,
			"dimensions for convert::elements_required must be positive");

		return iround(max_strd, interleaved_strd_ * panel_strd_divis_) * iround(max_cntg, panel_cntg_divis_);
	}
}; //class convert

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_FRAMEWORK_CONVERT_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_COPY_MATRIX_HPP
#define PERFLIBS_LINALG_COPY_MATRIX_HPP

#include "matrix/matrix.hpp"
#include "framework/which.hpp"
#include "framework/compute_position.hpp"
#include "framework/buffer_pool.hpp"

namespace perflibs::linalg {
namespace {

struct general_cntg_contig_generator {
	const bool always_copy { false };

	template<typename MatrixType>
	PERFLIBS_LINALG_INLINE
	bool ignore(const MatrixType& mat) const {
		return mat.is_physical() && is_cntg_contig(mat) && !always_copy;
	}

	template<typename MatrixType, typename T>
	PERFLIBS_LINALG_INLINE
	auto operator()(const MatrixType& src, T *buffer) {
		return general_matrix {
			matrix_base {
				buffer,
				src.cntg(), src.strd(),
				1,          src.cntg()
			},
			src.is_conj()
		};
	}
}; //struct general_cntg_contig_generator

struct general_strd_contig_generator {
	template<typename MatrixType>
	PERFLIBS_LINALG_INLINE
	bool ignore(const MatrixType& mat) const {
		return mat.is_physical() && is_strd_contig(mat);
	}

	template<typename MatrixType, typename T>
	PERFLIBS_LINALG_INLINE
	auto operator()(const MatrixType& src, T *buffer) {

		return general_matrix {
			matrix_base {
				buffer,
				src.cntg(), src.strd(),
				src.strd(), 1
			},
			src.is_conj()
		};
	}
}; //struct general_strd_contig_generator


/**
 * This tools allows the 'next' step to be called using a different data type
 * from the one that is passed to to this level of the stack.
 *
 * For example the user may wish to call a single precision kernel using half precision data
 *
 * This tool copies the values from the input matrix to a new buffer
 * and performs type conversion and then calls the 'next' level, the data is then copied back on completion
 */
template<which_matrix WhichMatrix, typename BufferPool, typename MatrixGenerator, typename Next>
class copy_matrix {
	BufferPool      buffer_;
	MatrixGenerator matrix_generator_;
	Next            next_;

public:
	PERFLIBS_LINALG_INLINE
	copy_matrix(BufferPool buffer, MatrixGenerator matrix_generator, Next next)
	:	buffer_           { std::move(buffer)           }
	,	matrix_generator_ { std::move(matrix_generator) }
	,	next_             { std::move(next)             }
	{	}

	PERFLIBS_LINALG_INLINE
	copy_matrix(which_matrix_constant<WhichMatrix>, BufferPool buffer, MatrixGenerator matrix_generator, Next next)
	:	copy_matrix { std::move(buffer), std::move(matrix_generator), std::move(next) }
	{	}

	template<typename AMatrixType, typename BMatrixType, typename CMatrixType, typename ScalarType, typename... Args>
	PERFLIBS_LINALG_INLINE
	void operator()(const AMatrixType& a, const BMatrixType& b, const CMatrixType& c, compute_position pos, ScalarType alpha, ScalarType beta, Args... args) {
		using dst_value_type = typename BufferPool::value_type;

		if constexpr(WhichMatrix == which_matrix::a) {
			using src_value_type = std::remove_cv_t<typename AMatrixType::value_type>;

			if constexpr(std::is_same_v<dst_value_type, src_value_type>) {
				if(matrix_generator_.ignore(a)) {
					next_(a, b, c, pos, alpha, beta, args...);
					return;
				}
			}

			auto buffer = buffer_.get_buffer(pos.thread_num);
			auto new_a = matrix_generator_(a, buffer);
			copy(a, new_a);
			next_(new_a, b, c, pos, alpha, beta, args...);
			//A is an input matrix, so doesn't need copying back
		}
		if constexpr(WhichMatrix == which_matrix::b) {
			using src_value_type = std::remove_cv_t<typename BMatrixType::value_type>;

			if constexpr(std::is_same_v<dst_value_type, src_value_type>) {
				if(matrix_generator_.ignore(b)) {
					next_(a, b, c, pos, alpha, beta, args...);
					return;
				}
			}

			auto buffer = buffer_.get_buffer(pos.thread_num);
			auto new_b = matrix_generator_(b, buffer);
			copy(b, new_b);
			next_(a, new_b, c, pos, alpha, beta, args...);
			//B is an input matrix, so doesn't need copying back
		}
		else if constexpr(WhichMatrix == which_matrix::c) {
			using src_value_type = std::remove_cv_t<typename CMatrixType::value_type>;

			if constexpr(std::is_same_v<dst_value_type, src_value_type>) {
				if (matrix_generator_.ignore(c)) {
					next_(a, b, c, pos, alpha, beta, args...);
					return;
				}
			}

			auto buffer = buffer_.get_buffer(pos.thread_num);
			auto new_c = matrix_generator_(c, buffer);

			if constexpr(is_general_matrix_v<CMatrixType>) {
				//scale does the optimization for zero and one internally
				scale(beta, c, new_c);
				beta = one<ScalarType>;
			}
			else {
				copy(c, new_c);
			}

			next_(a, b, new_c, pos, alpha, beta, args...);
			copy(new_c, c);
		}
	}
}; //class copy_matrix

} //namespace <anon>
} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_COPY_MATRIX_HPP

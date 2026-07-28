/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_PROBLEM_CONTEXT_BASES_HPP
#define PERFLIBS_LINALG_FACTORIZATION_PROBLEM_CONTEXT_BASES_HPP

#include "matrix/matrix.hpp"

#include <span>

namespace perflibs::linalg::factorization {

template<typename MatrixType, typename IntType>
struct lu_factorization {
	using matrix_type = MatrixType;
	using int_type    = IntType;

	matrix_type         a;
	std::span<int_type> pivot;
	int_type&           info;
}; // struct lu_factorization

template<typename MatrixType, typename IntType>
struct lu_apply_pivot {
	using matrix_type = MatrixType;
	using int_type    = IntType;

	matrix_type a;
	std::span<int_type> pivot;
}; // lu_apply_pivot

/*
* lu_panel_update is used to update
* the panel b, after the LU factorization
* of the panel a.
*/
template<typename MatrixType, typename IntType>
struct lu_panel_update {
	using matrix_type = MatrixType;
	using int_type    = IntType;

	matrix_type a;
	matrix_type b;
	std::span<int_type> pivot;

}; // lu_panel_update

// Cholesky Factorization
template<typename MatrixType, typename IntType>
struct cholesky_factorization {
	using matrix_type = MatrixType;
	using int_type    = IntType;

	matrix_type         a;
	int_type&           info;
}; // struct cholesky_factorization

// QR factorization
template<typename MatrixType>
struct qr_factorization {
	using matrix_type = MatrixType;
	using value_type = typename MatrixType::value_type;

	matrix_type a;
	std::span<value_type> tau;
	std::span<value_type> work;
}; // struct qr_factorization

template<typename MatrixType>
struct qr_panel_update {
	using matrix_type = MatrixType;
	using value_type = typename MatrixType::value_type;

	 // Panel that has been factorized
	matrix_type a;
	 // Scalar factors of elementary reflectors for panel a
	std::span<value_type> tau;
	// Panel to be updated
	matrix_type b;
	std::span<value_type> work;
}; // struct qr_panel_update

// Form triangular factor of block reflector
template<typename MatrixType, typename TriangularMatrixType>
struct form_block_reflector_factor {

	using matrix_type            = MatrixType;
	using triangular_matrix_type = TriangularMatrixType;
	using value_type             = typename MatrixType::value_type;

	perflibs_direct direct;
	perflibs_storev storev;

	// Matrix V contains the elementary reflectors.
	matrix_type v;
	// Vector tau contains scalar factors of elementary reflectors
	std::span<value_type> tau;
	// Triangular factor T represents the block reflector
	triangular_matrix_type t;
};

enum class block_reflector_layout {
	// The first k rows of V form a unit lower-triangular k x k block.
	leading_unit_lower,
	// The last k rows of V form a unit upper-triangular k x k block.
	trailing_unit_upper
};

template<typename VMatrixType, typename TriangularMatrixType>
struct block_reflector {
	using v_matrix_type          = VMatrixType;
	using triangular_matrix_type = TriangularMatrixType;

	// Matrix V contains the compact reflector basis.
	v_matrix_type v;
	// Triangular factor T in the block reflector representation.
	triangular_matrix_type t;
	// Layout of the distinguished unit-triangular block in V.
	block_reflector_layout layout;

	kernel_inttype active_offset(kernel_inttype full_length) const {
		return layout == block_reflector_layout::leading_unit_lower ? 0 : full_length - v.strd();
	}

	auto unit_block() const {
		auto block = v.sub_matrix(active_offset(v.cntg()), v.strd(), 0, v.strd());
		auto base = block.get_matrix_base();
		base.estrange_parent();
		return triangular_matrix {
			layout == block_reflector_layout::leading_unit_lower ? PERFLIBS_LOWER : PERFLIBS_UPPER,
			PERFLIBS_UNIT,
			base,
			block.is_conj()
		};
	}

	auto remainder() const {
		const auto k = v.strd();
		return v.sub_matrix(active_offset(v.cntg()) == 0 ? k : 0, v.cntg() - k, 0, k);
	}
};

template<typename LeftOperand, typename RightOperand>
struct apply_block_reflector;

// Applies H to C, where H = I - V T V^H.
template<typename VMatrixType, typename TriangularMatrixType, typename CMatrixType>
struct apply_block_reflector<block_reflector<VMatrixType, TriangularMatrixType>, CMatrixType> {

	using reflector_type     = block_reflector<VMatrixType, TriangularMatrixType>;
	using c_matrix_type      = CMatrixType;
	using value_type         = typename c_matrix_type::value_type;
	using work_matrix_type   = general_matrix<matrix_base<value_type>>;

	reflector_type reflector;
	c_matrix_type  matrix;
	// Workspace W used while forming and applying the update.
	work_matrix_type work;
};

// Applies H to C from the right, where H = I - V T V^H.
template<typename CMatrixType, typename VMatrixType, typename TriangularMatrixType>
struct apply_block_reflector<CMatrixType, block_reflector<VMatrixType, TriangularMatrixType>> {

	using c_matrix_type      = CMatrixType;
	using reflector_type     = block_reflector<VMatrixType, TriangularMatrixType>;
	using value_type         = typename c_matrix_type::value_type;
	using work_matrix_type   = general_matrix<matrix_base<value_type>>;

	c_matrix_type  matrix;
	reflector_type reflector;
	// Workspace W used while forming and applying the update.
	work_matrix_type work;
};

template<typename VMatrixType, typename TriangularMatrixType, typename CMatrixType>
apply_block_reflector(
	block_reflector<VMatrixType, TriangularMatrixType>,
	CMatrixType,
	general_matrix<matrix_base<typename CMatrixType::value_type>>
) -> apply_block_reflector<block_reflector<VMatrixType, TriangularMatrixType>, CMatrixType>;

template<typename CMatrixType, typename VMatrixType, typename TriangularMatrixType>
apply_block_reflector(
	CMatrixType,
	block_reflector<VMatrixType, TriangularMatrixType>,
	general_matrix<matrix_base<typename CMatrixType::value_type>>
) -> apply_block_reflector<CMatrixType, block_reflector<VMatrixType, TriangularMatrixType>>;

// Householder reflector generator
template<typename MatrixType>
struct generate_reflector {
	using matrix_type = MatrixType;
	using value_type  = typename MatrixType::value_type;

	value_type& alpha;
	matrix_type x;
	value_type& tau;
}; // struct generate_reflector

// Reduction of symmetric matrix to tridiagonal form
template<typename AMatrixType>
struct tridiagonalization {

	using matrix_type         = AMatrixType;
	using value_type          = typename AMatrixType::value_type;
	using real_type           = remove_complex_t<value_type>;
	using general_matrix_type = general_matrix<matrix_base<value_type>>;
	using real_matrix_type    = general_matrix<matrix_base<real_type>>;

	matrix_type         a;
	real_matrix_type    diagonal;
	real_matrix_type    off_diagonal;
	general_matrix_type tau;
	general_matrix_type work;
}; // struct tridiagonalization

template<typename AMatrixType>
struct tridiagonalization_block {
	using matrix_type         = AMatrixType;
	using value_type          = typename AMatrixType::value_type;
	using real_type           = remove_complex_t<value_type>;
	using general_matrix_type = general_matrix<matrix_base<value_type>>;
	using real_matrix_type    = general_matrix<matrix_base<real_type>>;

	matrix_type         a;
	real_matrix_type    off_diagonal;
	general_matrix_type tau;
	general_matrix_type w;
}; // struct tridiagonalization_block

// LARF to applies an elementary reflector to a general rectangular matrix.
template<typename VMatrixType, typename CMatrixType, typename WMatrixType>
struct apply_elementary_reflector {
	using v_matrix_type = VMatrixType;
	using c_matrix_type = CMatrixType;
	using w_matrix_type = WMatrixType;
	using value_type    = typename CMatrixType::value_type;

	perflibs_side side;
	v_matrix_type v;
	c_matrix_type c;
	w_matrix_type work;
	value_type tau;
}; // struct apply_elementary_reflector

// Bidiagonal reduction
template<typename MatrixType>
struct bidiagonalization {

	using matrix_type      = MatrixType;
	using value_type       = typename MatrixType::value_type;
	using real_type        = remove_complex_t<value_type>;
	using real_matrix_type = general_matrix<matrix_base<real_type>>;

	matrix_type      a;
	real_matrix_type d;
	real_matrix_type e;
	matrix_type      tauq;
	matrix_type      taup;
	matrix_type      work;
}; // struct bidiagonalization

template<typename MatrixType>
struct bidiagonalization_block {
	using matrix_type      = MatrixType;
	using value_type       = typename MatrixType::value_type;
	using real_type        = remove_complex_t<value_type>;
	using real_matrix_type = general_matrix<matrix_base<real_type>>;

	matrix_type      a;
	real_matrix_type d;
	real_matrix_type e;
	matrix_type      tauq;
	matrix_type      taup;
	matrix_type      x;
	matrix_type      y;
}; // struct bidiagonalization_block

// Apply Q from QR factorization
template<typename MatrixType>
struct apply_q_from_qr {
	using matrix_type = MatrixType;

	perflibs_side side;
	perflibs_trans trans;

	matrix_type a;
	matrix_type tau;
	matrix_type c;
	matrix_type work;
}; // struct apply_q_from_qr

// Generate Q from QR factorization
template<typename MatrixType>
struct generate_q_from_qr {
	using matrix_type = MatrixType;

	matrix_type a;
	matrix_type tau;
	matrix_type work;
}; // struct generate_q_from_qr

// Apply Q from LQ factorization
template<typename MatrixType>
struct apply_q_from_lq {
	using matrix_type = MatrixType;

	perflibs_side side;
	perflibs_trans trans;

	matrix_type a;
	matrix_type tau;
	matrix_type c;
	matrix_type work;
}; // struct apply_q_from_lq

template<typename MatrixType>
struct generate_q_from_lq {
	using matrix_type = MatrixType;

	matrix_type a;
	matrix_type tau;
	matrix_type work;
}; // struct generate_q_from_lq

template<typename... Args>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const lu_factorization<Args...>& pctx) {
	return min(pctx.a.cntg(), pctx.a.strd());
}

template<typename... Args>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const cholesky_factorization<Args...>& pctx) {
	return min(pctx.a.cntg(), pctx.a.strd());
}

template<typename... Args>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const qr_factorization<Args...>& pctx) {
	return min(pctx.a.cntg(), pctx.a.strd());
}

template<typename... Args>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const bidiagonalization<Args...>& pctx) {
	return min(pctx.a.cntg(), pctx.a.strd());
}

template<typename... Args>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const bidiagonalization_block<Args...>& pctx) {
	return min(pctx.a.cntg(), pctx.a.strd());
}

template<typename... Args>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const tridiagonalization<Args...>& pctx) {
	return pctx.a.cntg();
}

template<typename... Args>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const tridiagonalization_block<Args...>& pctx) {
	return pctx.a.cntg();
}

template<typename... Args>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const apply_q_from_qr<Args...>& pctx) {
	return min(pctx.a.cntg(), pctx.a.strd());
}

template<typename... Args>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const generate_q_from_qr<Args...>& pctx) {
	return min(pctx.a.cntg(), pctx.a.strd());
}

template<typename... Args>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const apply_q_from_lq<Args...>& pctx) {
	return min(pctx.a.cntg(), pctx.a.strd());
}

template<typename... Args>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const generate_q_from_lq<Args...>& pctx) {
	return min(pctx.a.cntg(), pctx.a.strd());
}

} // namespace perflibs::linalg::factorization
#endif // PERFLIBS_LINALG_FACTORIZATION_PROBLEM_CONTEXT_BASES_HPP

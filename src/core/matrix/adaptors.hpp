/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATRIX_ADAPTORS_HPP
#define PERFLIBS_LINALG_MATRIX_ADAPTORS_HPP

#include "matrix/type_traits.hpp"
#include "matrix/write.hpp"

#include "framework/linalg_util.hpp"
#include "framework/linalg_blas_types.hpp"
#include "framework/has_if.hpp"

#include "perflibs_util.hpp"

#ifdef _WIN32
#define NO_UNIQUE_ADDRESS msvc::no_unique_address
#else
#define NO_UNIQUE_ADDRESS no_unique_address
#endif

#include <utility>

namespace perflibs::linalg {

/**
 * This file contains the "adaptors", an adaptor presents a "virtual" view
 * over the underly data storage.
 *
 * The adaptors descend from a storage type which provides member functions
 * to retrieve and manipulate the stored values in memory
 *
 * The adaptors then wrap these underly accesses to provide a coherent view.
 *
 * For example, an symmetric matrix will mirror the contents of a matrix to
 * the other half, regardless of what is stored in the other half.
 *
 * All adaptors as a minimum present the following interface
 *
 * MatrixType:
 * 	cntg()        - cntg dimension in number of elements
 * 	strd()        - strd dimension in number of elements
 * 	sub_matrix()  - returns matrix from the provided coordinates and sizes
 * 	is_physical() - is the matrix physical (ie do elements indexed inside the cntg, strd space related 1to1 with data in memory
 * 	data()        - returns a pointer to the first element
 *
 * if a matrix exposes a "is_lower()" function, we assume that it must have triangular properties
 *     a rectangular matrix need-not (should not) define it
 *
 */


/**
 * Represents a general purpose rectangular matrix with no virtual regions
 *
 * All values inside of the cntg()-strd() region are readable and writable
 * Values are conjugated if `is_conj()` is true
 *
 * operator(i, j, write) returns a reference
 * operator(i, j) returns a value
 */
template<typename MatrixBaseType>
class general_matrix : public MatrixBaseType {
public:
	using base_type        = MatrixBaseType;
	using value_type       = typename base_type::value_type;
	using const_value_type = typename base_type::const_value_type;
	using reference        = typename base_type::reference;
	using const_reference  = typename base_type::const_reference;

#if PERFLIBS_ADAPTOR_AGGREGATE_INIT != 0
public:
	[[NO_UNIQUE_ADDRESS]]
	has_if<is_complex_v<value_type>, false> is_conj_ { false };
#else
private:
	[[NO_UNIQUE_ADDRESS]]
	has_if<is_complex_v<value_type>, false> is_conj_ { false };

public:
	//without these the universal ref ctor set is_conj to false when copy/move constructing
	general_matrix(const general_matrix& other)          = default;
	general_matrix(general_matrix&& other)               = default;
	general_matrix(general_matrix& other)                = default;
	general_matrix& operator=(const general_matrix& rhs) = default;
	general_matrix& operator=(general_matrix&& rhs)      = default;

	template<typename MatrixBaseType2>
	PERFLIBS_LINALG_INLINE
	general_matrix(MatrixBaseType2&& matrix_base, bool is_conj=false)
	:	MatrixBaseType { std::forward<MatrixBaseType2>(matrix_base) }
	,	is_conj_	{ is_conj     }
	{	}
#endif

	PERFLIBS_LINALG_INLINE bool is_conj() const { return is_conj_; }

	PERFLIBS_LINALG_INLINE general_matrix sub_matrix(kernel_inttype cntg_pos, kernel_inttype cntg_size,
	                                            kernel_inttype strd_pos, kernel_inttype strd_size) const {
		return { base_type::sub_matrix(cntg_pos, cntg_size, strd_pos, strd_size), is_conj() };
	}

	PERFLIBS_LINALG_INLINE general_matrix sub_matrix_with_clamp(kernel_inttype cntg_pos, kernel_inttype cntg_size,
	                                                       kernel_inttype strd_pos, kernel_inttype strd_size) const {
		return { base_type::sub_matrix_with_clamp(cntg_pos, cntg_size, strd_pos, strd_size), is_conj() };
	}

	PERFLIBS_LINALG_INLINE general_matrix get_parent() const {
		return { base_type::get_parent(), is_conj() };
	}

	PERFLIBS_LINALG_INLINE general_matrix transpose() const {
		return { base_type::transpose(), is_conj() };
	}

	PERFLIBS_LINALG_INLINE general_matrix reflect() const {
		return { base_type::reflect(), is_conj() };
	}

	PERFLIBS_LINALG_INLINE general_matrix reflect_transpose() const {
		return { base_type::reflect_transpose(), is_conj() };
	}

	// read-only
	PERFLIBS_LINALG_INLINE
	value_type operator()(kernel_inttype cntg, kernel_inttype strd) const {
		if constexpr (is_complex_v<value_type>) {
			return is_conj() ? std::conj( base_type::get_logical_elem(cntg, strd) )
			                 :            base_type::get_logical_elem(cntg, strd)  ;
		}
		else {
			return base_type::get_logical_elem(cntg, strd);
		}
	}

	// write-only
	PERFLIBS_LINALG_INLINE
	reference operator()(kernel_inttype cntg, kernel_inttype strd, write_t) const {
		return base_type::get_logical_elem(cntg, strd);
	}

	PERFLIBS_LINALG_INLINE constexpr bool is_physical() const { return !is_conj(); }

	PERFLIBS_LINALG_INLINE       base_type& get_matrix_base()       { return static_cast<base_type&>(*this); }
	PERFLIBS_LINALG_INLINE const base_type& get_matrix_base() const { return static_cast<const base_type&>(*this); }
}; //class general_matrix

#if !defined(PERFLIBS_ADAPTOR_AGGREGATE_INIT) || (PERFLIBS_ADAPTOR_AGGREGATE_INIT == 0)
template<typename MatrixBaseType2>
general_matrix(MatrixBaseType2&&, bool is_conj) -> general_matrix< std::remove_cvref_t<MatrixBaseType2> >;

template<typename MatrixBaseType2>
general_matrix(MatrixBaseType2&&) -> general_matrix< std::remove_cvref_t<MatrixBaseType2> >;
#endif


/**
 * Represents a triangular unit or non-unit matrix
 *
 * Access to values in "virtual" region will return T{0.0}, regardless of what is stored in the underlying structure
 * Access to the physical elements will return the value of the element at that position
 * Access to the diagonal will return T{1.0} if the triangular matrix is UNIT
 * Values are conjugated if `is_conj()` is true
 */
template<typename MatrixBaseType>
class triangular_matrix : public MatrixBaseType {
	perflibs_uplo uplo_;
	perflibs_diag diag_;
	bool is_conj_;
public:
	using base_type        = MatrixBaseType;
	using value_type       = typename base_type::value_type;
	using const_value_type = typename base_type::const_value_type;
	//this type does not support references, why because half the access space is 'virtual'
	using reference        = value_type;
	using const_reference  = typename base_type::const_reference;


	template<typename... BaseTypeCtorArgs>
	PERFLIBS_LINALG_INLINE
	triangular_matrix(perflibs_uplo uplo, perflibs_diag diag, bool is_conj, BaseTypeCtorArgs&&... base_args)
	:	base_type { std::forward<BaseTypeCtorArgs>(base_args)... }
	,	uplo_  { uplo  }
	,	diag_  { diag  }
	,	is_conj_ { is_conj }
	{	}

	PERFLIBS_LINALG_INLINE
	triangular_matrix(perflibs_uplo uplo, perflibs_diag diag, MatrixBaseType base, bool is_conj=false)
	:	MatrixBaseType { std::move(base) }
	,	uplo_ { uplo }
	,	diag_ { diag }
	,	is_conj_ { is_conj }
	{	}

	PERFLIBS_LINALG_INLINE perflibs_uplo uplo() const { return uplo_; }
	PERFLIBS_LINALG_INLINE perflibs_diag diag() const { return diag_; }
	PERFLIBS_LINALG_INLINE bool is_lower() const { return uplo() == PERFLIBS_LOWER; }
	PERFLIBS_LINALG_INLINE bool is_upper() const { return uplo() == PERFLIBS_UPPER; }
	PERFLIBS_LINALG_INLINE bool is_unit() const { return diag() == PERFLIBS_UNIT; }
	PERFLIBS_LINALG_INLINE bool is_conj() const { return is_conj_; }

	PERFLIBS_LINALG_INLINE triangular_matrix sub_matrix(kernel_inttype cntg_pos, kernel_inttype cntg_size,
	                                               kernel_inttype strd_pos, kernel_inttype strd_size) const {
		return { uplo(), diag(), base_type::sub_matrix(cntg_pos, cntg_size, strd_pos, strd_size), is_conj() };
	}

	PERFLIBS_LINALG_INLINE triangular_matrix sub_matrix_with_clamp(kernel_inttype cntg_pos, kernel_inttype cntg_size,
	                                                          kernel_inttype strd_pos, kernel_inttype strd_size) const {
		return { uplo(), diag(), base_type::sub_matrix_with_clamp(cntg_pos, cntg_size, strd_pos, strd_size), is_conj() };
	}

	PERFLIBS_LINALG_INLINE triangular_matrix get_parent() const {
		return { uplo(), diag(), base_type::get_parent(), is_conj() };
	}

	PERFLIBS_LINALG_INLINE triangular_matrix transpose() const {
		return { lower_flip(uplo_), diag(), base_type::transpose(), is_conj() };
	}

	PERFLIBS_LINALG_INLINE triangular_matrix reflect() const {
		return { uplo(), diag(), base_type::reflect(), is_conj() };
	}

	PERFLIBS_LINALG_INLINE triangular_matrix reflect_transpose() const {
		return { lower_flip(uplo_), diag(), base_type::reflect_transpose(), is_conj() };
	}

	PERFLIBS_LINALG_INLINE value_type operator()(kernel_inttype cntg_idx, kernel_inttype strd_idx) const {
		const auto cntg_pos = cntg_idx + base_type::absolute_cntg();
		const auto strd_pos = strd_idx + base_type::absolute_strd();

		const bool pos_is_lower = cntg_pos > strd_pos;
		const bool is_on_diag = cntg_pos == strd_pos;

		if(is_on_diag && is_unit()) {
			return one<value_type>;
		}
		else if(!is_on_diag && (is_lower() != pos_is_lower)) {
			return zero<value_type>;
		}
		else if constexpr(is_complex_v<value_type>) {
			return is_conj() ? std::conj( base_type::get_logical_elem(cntg_idx, strd_idx) )
			                 :            base_type::get_logical_elem(cntg_idx, strd_idx)  ;
		}
		else {
			return base_type::get_logical_elem(cntg_idx, strd_idx);
		}
	}

	PERFLIBS_LINALG_INLINE constexpr bool is_diag_physical() const { return ! (is_unit() || is_conj()); }

	PERFLIBS_LINALG_INLINE bool is_physical() const {
		//do not include the diagonal if the matrix is_unit() as we virtual the '1' for this position
		const kernel_inttype one_if_not_unit = is_unit() ? 0 : 1;

		if(is_upper()) return ( base_type::cntg() + base_type::absolute_cntg() ) <= ( base_type::absolute_strd() + one_if_not_unit) && !is_conj();
		else           return ( base_type::strd() + base_type::absolute_strd() ) <= ( base_type::absolute_cntg() + one_if_not_unit) && !is_conj();
	}

	PERFLIBS_LINALG_INLINE       base_type& get_matrix_base()       { return static_cast<base_type&>(*this); }
	PERFLIBS_LINALG_INLINE const base_type& get_matrix_base() const { return static_cast<const base_type&>(*this); }
}; //class triangular_matrix

/**
 * Represents a symmetric matrix,
 *
 * access to the physical values will return the value at that position
 * access to the virtual values will return the value at the inverted absolute position
 */
template<typename MatrixBaseType>
class symmetric_matrix : public MatrixBaseType {
	perflibs_uplo uplo_;
public:
	using base_type        = MatrixBaseType;
	using value_type       = typename base_type::value_type;
	using const_value_type = typename base_type::const_value_type;
	using reference        = value_type;
	using const_reference  = typename base_type::const_reference;

	template<typename... BaseTypeCtorArgs>
	PERFLIBS_LINALG_INLINE
	symmetric_matrix(perflibs_uplo uplo, BaseTypeCtorArgs&&... base_args)
	:	base_type { std::forward<BaseTypeCtorArgs>(base_args)... }
	,	uplo_ { uplo }
	{	}

	PERFLIBS_LINALG_INLINE
	symmetric_matrix(perflibs_uplo uplo, MatrixBaseType base)
	:	MatrixBaseType { std::move(base) }
	,	uplo_          { uplo            }
	{	}

	PERFLIBS_LINALG_INLINE perflibs_uplo uplo() const { return uplo_; }
	PERFLIBS_LINALG_INLINE bool is_lower() const { return uplo() == PERFLIBS_LOWER; }
	PERFLIBS_LINALG_INLINE bool is_upper() const { return uplo() == PERFLIBS_UPPER; }

	PERFLIBS_LINALG_INLINE symmetric_matrix sub_matrix(kernel_inttype cntg_pos, kernel_inttype cntg_size,
                                                  kernel_inttype strd_pos, kernel_inttype strd_size) const {
		return { uplo(), base_type::sub_matrix(cntg_pos, cntg_size, strd_pos, strd_size) };
	}

	PERFLIBS_LINALG_INLINE symmetric_matrix sub_matrix_with_clamp(kernel_inttype cntg_pos, kernel_inttype cntg_size,
                                                             kernel_inttype strd_pos, kernel_inttype strd_size) const {
		return { uplo(), base_type::sub_matrix_with_clamp(cntg_pos, cntg_size, strd_pos, strd_size) };
	}

	PERFLIBS_LINALG_INLINE symmetric_matrix get_parent() const {
		return { uplo(), base_type::get_parent() };
	}

	PERFLIBS_LINALG_INLINE symmetric_matrix transpose() const {
		return { lower_flip(uplo_), base_type::transpose() };
	}

	PERFLIBS_LINALG_INLINE symmetric_matrix reflect() const {
		return { uplo(), base_type::reflect() };
	}

    PERFLIBS_LINALG_INLINE symmetric_matrix reflect_transpose() const {
		return { lower_flip(uplo_), base_type::reflect_transpose() };
	}

	PERFLIBS_LINALG_INLINE value_type operator()(kernel_inttype cntg_idx, kernel_inttype strd_idx) const {
		//get our position in the parent space
		auto cntg_pos = cntg_idx + base_type::absolute_cntg();
		auto strd_pos = strd_idx + base_type::absolute_strd();

		const bool pos_is_lower = cntg_pos > strd_pos;

		/*
		 * Our matrix types are storage agnostic and in a GEMM context cntg() is k.
		 *
		 * However in Fortran interface "Lower" means that the physical triangle
		 * 'hugs' the M dimension which is analogous to strd().
		 *
		 * This can be confusing because in a Fortran system M (for matrix A) would
		 * be the contiguous dimension - which is opposite way round.
		 */
		if(is_upper() == pos_is_lower)
			std::swap(cntg_pos, strd_pos);

		return base_type::get_absolute_logical_elem(cntg_pos, strd_pos); // Working value
	}

	PERFLIBS_LINALG_INLINE bool is_physical() const {
		if(is_upper()) return ( base_type::cntg() + base_type::absolute_cntg() ) <= ( base_type::absolute_strd() + 1);
		else           return ( base_type::strd() + base_type::absolute_strd() ) <= ( base_type::absolute_cntg() + 1);
	}

	PERFLIBS_LINALG_INLINE constexpr bool is_diag_physical() const { return true; }
	PERFLIBS_LINALG_INLINE constexpr bool is_conj() const { return false; }

	PERFLIBS_LINALG_INLINE       base_type& get_matrix_base()       { return static_cast<base_type&>(*this); }
	PERFLIBS_LINALG_INLINE const base_type& get_matrix_base() const { return static_cast<const base_type&>(*this); }
}; //class symmetric_matrix

/**
 * Represents a hermitian matrix
 */
template<typename MatrixBaseType>
class hermitian_matrix : public MatrixBaseType {
	perflibs_uplo uplo_;
public:
	using base_type        = MatrixBaseType;
	using value_type       = typename base_type::value_type;
	using const_value_type = typename base_type::const_value_type;
	using reference        = typename base_type::reference;

	template<typename... BaseTypeCtorArgs>
	PERFLIBS_LINALG_INLINE
	hermitian_matrix(perflibs_uplo uplo, BaseTypeCtorArgs&&... base_args)
	:	base_type { std::forward<BaseTypeCtorArgs>(base_args)... }
	,	uplo_ { uplo }
	{	}

	PERFLIBS_LINALG_INLINE
	hermitian_matrix(perflibs_uplo uplo, MatrixBaseType base)
	:	MatrixBaseType { std::move(base) }
	,	uplo_          { uplo            }
	{	}

	PERFLIBS_LINALG_INLINE perflibs_uplo uplo() const { return uplo_; }
	PERFLIBS_LINALG_INLINE bool is_lower() const { return uplo() == PERFLIBS_LOWER; }
	PERFLIBS_LINALG_INLINE bool is_upper() const { return uplo() == PERFLIBS_UPPER; }

	PERFLIBS_LINALG_INLINE hermitian_matrix sub_matrix(kernel_inttype cntg_pos, kernel_inttype cntg_size,
	                                              kernel_inttype strd_pos, kernel_inttype strd_size) const {
		return { uplo(), base_type::sub_matrix(cntg_pos, cntg_size, strd_pos, strd_size) };
	}

	PERFLIBS_LINALG_INLINE hermitian_matrix sub_matrix_with_clamp(kernel_inttype cntg_pos, kernel_inttype cntg_size,
	                                                         kernel_inttype strd_pos, kernel_inttype strd_size) const {
		return { uplo(), base_type::sub_matrix_with_clamp(cntg_pos, cntg_size, strd_pos, strd_size) };
	}

	PERFLIBS_LINALG_INLINE hermitian_matrix get_parent() const {
		return { uplo(), base_type::get_parent() };
	}

	PERFLIBS_LINALG_INLINE hermitian_matrix transpose() const {
		return { lower_flip(uplo_), base_type::transpose() };
	}

	PERFLIBS_LINALG_INLINE hermitian_matrix reflect() const {
		return { uplo(), base_type::reflect() };
	}

	PERFLIBS_LINALG_INLINE hermitian_matrix reflect_transpose() const {
		return { lower_flip(uplo_), base_type::reflect_transpose() };
	}

	PERFLIBS_LINALG_INLINE value_type operator()(kernel_inttype cntg_idx, kernel_inttype strd_idx) const {
		const auto cntg_pos = cntg_idx + base_type::absolute_cntg();
		const auto strd_pos = strd_idx + base_type::absolute_strd();

		if (cntg_pos == strd_pos) {
			//
			// if on the diagonal return just real value
			//
			return { real(base_type::get_logical_elem(cntg_idx, strd_idx)), 0.0 };
		}

		const bool pos_is_lower = cntg_pos > strd_pos;
		if(is_upper() == pos_is_lower)
			return conj(base_type::get_absolute_logical_elem(strd_pos, cntg_pos));
		else
			return base_type::get_absolute_logical_elem(cntg_pos, strd_pos);
	}

	PERFLIBS_LINALG_INLINE bool is_physical() const {
		if(is_upper()) return ( base_type::cntg() + base_type::absolute_cntg() ) <= base_type::absolute_strd();
		else           return ( base_type::strd() + base_type::absolute_strd() ) <= base_type::absolute_cntg();
	}

	PERFLIBS_LINALG_INLINE constexpr bool is_diag_physical() const { return false; }
	PERFLIBS_LINALG_INLINE constexpr bool is_conj() const { return false; }

	PERFLIBS_LINALG_INLINE       base_type& get_matrix_base()       { return static_cast<base_type&>(*this); }
	PERFLIBS_LINALG_INLINE const base_type& get_matrix_base() const { return static_cast<const base_type&>(*this); }
}; //class hermitian_matrix

/**
 * Convert any adaptor into a general_matrix
 * this is useful because you can not write into the value of "virtual" adaptors via op()
 * Note: the resulting matrix has `is_conj() == false`.
 */
template<typename MatrixType>
PERFLIBS_LINALG_INLINE
general_matrix<typename MatrixType::base_type> to_general_matrix(const MatrixType& m) {
	// Because all matrix types descend from a base type then we can pass that
	// in to general_matrix constructor
	if constexpr( is_general_matrix_v<MatrixType> ) {
		return { m.get_matrix_base(), m.is_conj() };
	}
	else {
		return { m.get_matrix_base(), /*is_conj=*/false };
	}
}

template<typename MatrixType>
PERFLIBS_LINALG_INLINE
auto to_const(const MatrixType& mat) {
	return *reinterpret_cast<const to_const_t<MatrixType>*>(&mat);
}

// sets the conj flag to true
template<typename MatrixBase>
PERFLIBS_LINALG_INLINE
auto to_conj(const general_matrix<MatrixBase>& src) {
	return general_matrix { src.get_matrix_base(), true };
}

// Toggles the conjugation flag: sets is_conj to true if it was false,
// or sets it to false if the matrix is already conjugated.
// This implements the mathematical property that conj(conj(z)) = z.
template<typename MatrixBase>
PERFLIBS_LINALG_INLINE
auto toggle_conj(const general_matrix<MatrixBase>& src) {
	return general_matrix { src.get_matrix_base(), !src.is_conj() };
}

template<typename MatrixBase>
PERFLIBS_LINALG_INLINE
auto toggle_conj(const triangular_matrix<MatrixBase>& src) {
	return triangular_matrix { src.uplo(), src.diag(), src.get_matrix_base(), !src.is_conj() };
}

template<typename MatrixBase>
PERFLIBS_LINALG_INLINE
auto adjoint(const general_matrix<MatrixBase>& src) {
	if constexpr (is_complex_v<typename general_matrix<MatrixBase>::value_type>) {
		return toggle_conj(src.transpose());
	}
	else {
		return src.transpose();
	}
}

template<typename MatrixBase>
PERFLIBS_LINALG_INLINE
auto adjoint(const triangular_matrix<MatrixBase>& src) {
	if constexpr (is_complex_v<typename triangular_matrix<MatrixBase>::value_type>) {
		return toggle_conj(src.transpose());
	}
	else {
		return src.transpose();
	}
}

} //namespace perflibs::linalg

#undef NO_UNIQUE_ADDRESS

#endif //PERFLIBS_LINALG_MATRIX_ADAPTORS_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATRIX_OWNING
#define PERFLIBS_LINALG_MATRIX_OWNING

#include "perflibs_util.hpp"

#include <vector>

namespace perflibs::linalg {

/**
 * owning need to initialize its MatrixBaseType with the data pointer
 * This means that we need to calculate the size and init the storage
 * container before, so we've made it a parent class
 */
template<typename T>
class matrix_data_owner {
	std::vector<T> data_;

public:
	matrix_data_owner() = delete;
	matrix_data_owner(matrix_data_owner&&) = default;
	matrix_data_owner(const matrix_data_owner&) = default;
	matrix_data_owner& operator=(matrix_data_owner&&)=default;
	matrix_data_owner& operator=(const matrix_data_owner&)=default;
protected:
	matrix_data_owner(std::size_t sz)
	:	data_ (sz)
	{	}



	      T *data_impl()       { return data_.data(); }
	const T *data_impl() const { return data_.data(); }

	std::vector<T> to_vector() { return data_; }
}; //class matrix_data_owner

/**
 * A wrapper around your favorite MatrixTypes which
 * automatically allocates the memory, where you would have
 * previously specified the data pointer, cntg, strd, cntg_step and
 * strd_step, you now only need to specify the cntg and strd value
 */
template<typename MatrixBaseType>
class owning
	:	private matrix_data_owner<typename MatrixBaseType::value_type>
	,	public MatrixBaseType {

	using data_owner_t = matrix_data_owner<typename MatrixBaseType::value_type>;

public:
	using base_type=MatrixBaseType;
	using value_type=typename base_type::value_type;
	using const_value_type=typename base_type::const_value_type;
	using reference =typename base_type::reference;
	using const_reference =typename base_type::const_reference;

	///constructor, all args pass to base
	template<typename... BaseTypeCtorArgs>
	owning(kernel_inttype cntg, kernel_inttype strd, BaseTypeCtorArgs... base_args)
	:	data_owner_t { static_cast<std::size_t>(cntg * strd) }
	,	base_type(
			data_owner_t::data_impl(),
			cntg,
			strd,
			1,
			cntg,
			std::forward<BaseTypeCtorArgs>(base_args)...
		)
	{	}

	template<typename MatrixTypeRhs>
	owning(const MatrixTypeRhs& rhs)
	:	owning(rhs.cntg(), rhs.strd())
	{
		copy(rhs, static_cast<MatrixBaseType&>(*this));
	}

	using data_owner_t::to_vector;
}; //class owning

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_MATRIX_OWNING

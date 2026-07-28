/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FRAMEWORK_INPLACE_VECTOR_HPP
#define PERFLIBS_LINALG_FRAMEWORK_INPLACE_VECTOR_HPP

#include "perflibs_assert.hpp"
#include <array>
#include <iterator>
#include <algorithm>


namespace perflibs::linalg {

template<typename T, std::size_t N>
class inplace_vector {
public:
	using array_type      = std::array<T, N>;
	using value_type      = typename array_type::value_type;
	using reference       = typename array_type::reference;
	using const_reference = typename array_type::const_reference;
	using size_type       = typename array_type::size_type;
	using iterator        = typename array_type::iterator;
	using const_iterator  = typename array_type::const_iterator;

	constexpr inplace_vector(size_type size = 0, const T& val={})
	:	size_ { size }
	{
		PERFLIBS_ALWAYS_ASSERT(size_ <= N, "inplace_vector over capacity");
		std::fill_n(begin(), size, val);
	}

	template<typename InputIterator>
	constexpr inplace_vector(InputIterator first, InputIterator last)
	{
		size_ = std::distance(begin(), std::copy(first, last, begin()));
		PERFLIBS_ALWAYS_ASSERT(size_ <= N, "inplace_vector over capacity");
	}

	inplace_vector(const std::initializer_list<T>& init)
	:	size_ { init.size() }
	{
		PERFLIBS_ALWAYS_ASSERT(size_ <= N, "inplace_vector over capacity");
		std::copy(init.begin(), init.end(), begin());
	}

	auto capacity() const { return N; }
	auto size() const { return size_; }
	auto data() const { return data_.data(); }
	auto data() { return data_.data(); }
	auto begin() { return data_.begin(); }
	auto end() { return std::next(data_.begin(), size()); }
	auto begin() const { return data_.begin(); }
	auto end() const { return std::next(data_.begin(), size()); }
	auto cbegin() const { return data_.begin(); }
	auto cend() const { return std::next(data_.begin(), size()); }
	auto empty() const { return size_ == 0; }

	auto& front() const { return data_[0]; }
	auto& back() const { return data_[size_  - 1]; }
	auto& front() { return data_[0]; }
	auto& back() { return data_[size_  - 1]; }
	auto& operator[] (size_type pos) const { return data_[pos]; }
	auto& operator[] (size_type pos) { return data_[pos]; }

	iterator insert(const_iterator pos, T value) {
		auto old_end = end();
		auto mut_pos = std::next(begin(), std::distance(cbegin(), pos));

		++size_;
		PERFLIBS_ALWAYS_ASSERT(size_ <= N, "inplace_vector over capacity");

		std::move_backward(mut_pos, old_end, end());

		*mut_pos = std::move(value);
		return mut_pos;
	}

	template<typename InputIterator>
	iterator insert(const_iterator pos, InputIterator first, InputIterator last) {
		const auto dist = std::distance(first, last);
		auto old_end    = end();
		auto mut_pos    = std::next(begin(), std::distance(cbegin(), pos));

		///this might not be correct for certain iterator types like ostream_iterator
		size_ += dist;
		PERFLIBS_ALWAYS_ASSERT(size_ <= N, "inplace_vector over capacity");

		//shuffle tail back one
		std::move_backward(mut_pos, old_end, end());
		std::copy(first, last, mut_pos);

		return mut_pos;
	}

	void push_back(T val) {
		insert(std::move(val), end());
	}

	void pop_back() {
		PERFLIBS_ALWAYS_ASSERT(size_ > 0);
		--size_;
	}

private:
	size_type  size_;
	array_type data_;
}; //class inplace_vector

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_FRAMEWORK_INPLACE_VECTOR_HPP

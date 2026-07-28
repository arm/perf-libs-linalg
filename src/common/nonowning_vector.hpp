/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_NONOWNING_VECTOR_HPP
#define PERFLIBS_NONOWNING_VECTOR_HPP

#include <algorithm>
#include <iterator>
#include "perflibs_assert.hpp"

namespace perflibs {

/**
 * A container which mimics the interface and functionality of std::vector
 * however takes an external buffer in which it'll construct and destruct its
 * contents
 *
 * Note either the vector lifetime must end before the buffer, or clear() must
 * explicitly be called before the buffer is deallocated - this is to ensure
 * that the destructors of the elements are called
 */
template<typename T>
class nonowning_vector {
public:
	using size_type = std::size_t;
	using value_type = T;
	using pointer = T*;
	using const_pointer = const T*;
	using iterator = pointer;
	using const_iterator = const_pointer;
	using reference = T&;
	using const_reference = const T&;
	using reverse_iterator = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;
private:
	///pointer to the data that we do not own the lifetime of
	value_type *data_     { nullptr };
	///maximum size (in elements)
	size_type   capacity_ { 0 };
	///current size
	size_type   size_     { 0 };
public:
	nonowning_vector(value_type *data, size_type capacity)
	:	data_     { data     }
	,	capacity_ { capacity }
	{	}

	nonowning_vector(uint8_t *data, size_type capacity)
	:	nonowning_vector { reinterpret_cast<T*>(data), capacity }
	{	}

	nonowning_vector(value_type *data, size_type capacity, const std::initializer_list<T>& ls)
	:	nonowning_vector { data, capacity, std::begin(ls), std::end(ls) }
	{	}

	nonowning_vector(uint8_t *data, size_type capacity, const std::initializer_list<T>& ls)
	:	nonowning_vector { reinterpret_cast<T*>(data), capacity, ls }
	{	}

	nonowning_vector(value_type *data, size_type capacity, size_type size, const T& value)
	:	nonowning_vector { data, capacity }
	{
		assign(size, value);
	}

	nonowning_vector(uint8_t *data, size_type capacity, size_type size, const T& value)
	:	nonowning_vector { reinterpret_cast<T*>(data), capacity, size, value }
	{	}

	template<typename Iter>
	nonowning_vector(value_type *data, size_type capacity, Iter first, Iter last)
	:	nonowning_vector { data, capacity }
	{
		assign(first, last);
	}

	template<typename Iter>
	nonowning_vector(uint8_t *data, size_type capacity, Iter first, Iter last)
	:	nonowning_vector { reinterpret_cast<T*>(data), first, last }
	{	}

	~nonowning_vector() { clear(); }

	void assign(size_type size, const T& value) {
		const size_type mn = std::min(size, this->size());

		std::fill_n(begin(), mn, value);

		resize(size, value);
	}

	template<typename Iter>
	void assign(Iter first, Iter last) {
		clear();
		std::copy(first, last, std::back_inserter(*this));
	}

//@{ ELEMENT ACCESS
	      reference at(size_type n)       { return data_[n]; }
	const_reference at(size_type n) const { return data_[n]; }

	      reference operator[](size_type n)       { return at(n); }
	const_reference operator[](size_type n) const { return at(n); }

	      reference back()        { return at(size_ - 1); }
	      reference front()       { return at(0); }
	const_reference back()  const { return at(size_ - 1); }
	const_reference front() const { return at(0); }

	      T *data()       { return data_; }
	const T *data() const { return data_; }
///@}
///@{ ITERATORS
	      iterator begin()       { return data_; }
	      iterator end()         { return data_ + size_; }
	const_iterator begin() const { return data_; }
	const_iterator end()   const { return data_ + size_; }
///@}
///@{ CAPACITY
	bool empty() const { return size_ == 0; }
	size_type size() const { return size_; }
	size_type capacity() const { return capacity_; }
	size_type max_size() const { return capacity_; }

///@}
///@{ MODIFIERS
	void clear() {
		while(! empty())
			pop_back();
	}

	iterator insert(const_iterator pos, T value) {
		auto mutable_pos = const_cast<iterator>(pos);

		push_back(std::move(value));
		std::rotate(mutable_pos, std::prev(end()), end());

		return mutable_pos;
	}

	iterator insert(const_iterator pos, size_type count, const T& value) {
		auto mutable_pos = const_cast<iterator>(pos);
		auto old_end = end();

		std::fill_n(std::back_inserter(*this), count, value);

		std::rotate(mutable_pos, old_end, end());

		return mutable_pos;
	}

	iterator erase(const_iterator first, const_iterator last) {
		const size_type new_size = size_ - std::distance(first, last);

		auto mutable_first = const_cast<iterator>(first);
		auto mutable_last = const_cast<iterator>(last);

		std::move(mutable_last, end(), mutable_first);

		resize(new_size);

		return mutable_first;
	}

	iterator erase(const_iterator first) {
		return erase(first, std::next(first));
	}

	void push_back(T value) {
		PERFLIBS_ASSERT(size_ < capacity_);

		new(data_ + size_) T(std::move(value));

		++size_;
	}

	void pop_back() {
		PERFLIBS_ASSERT(size_ > 0);

		--size_;

		data_[ size_ ].~T();
	}

	void resize(size_type count, const value_type& value) {
		while(count < size_) pop_back();
		while(count > size_) push_back(value);
	}

	void resize(size_type count) {
		resize(count, T{});
	}

	void swap(nonowning_vector& rhs) {
		using std::swap;
		swap(data_, rhs.data_);
		swap(capacity_, rhs.capacity_);
		swap(size_, rhs.size_);
	}
///@}
///@{ OTHER
	static size_type require(size_type n) {
		return sizeof(T) * n;
	}
//@}
}; //class nonowning_vector

template<typename T>
bool operator==(const nonowning_vector<T>& lhs,const nonowning_vector<T>& rhs) {
	return std::equal(std::begin(lhs), std::end(lhs), std::begin(rhs), std::end(rhs));
}

template<typename T>
bool operator!=(const nonowning_vector<T>& lhs,const nonowning_vector<T>& rhs) {
	return !(lhs == rhs);
}
template<typename T>
void swap(nonowning_vector<T>& lsh, nonowning_vector<T>& rhs) {
	lsh.swap(rhs);
}

} // namespace perflibs

#endif //PERFLIBS_NONOWNING_VECTOR_HPP

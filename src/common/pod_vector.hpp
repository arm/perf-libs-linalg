/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_COMMON_POD_VECTOR_HPP
#define PERFLIBS_COMMON_POD_VECTOR_HPP

#include "reallocator.hpp"

#include <algorithm>
#include "perflibs_assert.hpp"
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <type_traits>
#include <vector>

namespace perflibs {
template<typename T>
constexpr bool is_input_iterator =
    std::is_base_of_v<std::input_iterator_tag, typename std::iterator_traits<T>::iterator_category>;

/** A vector-like container for value types that will survive not having a constructor called
 *  (e.g. float/double/std::complex). Calling .resize() on this container will leave any newly
 *  allocated data uninitialized, unlike std::vector which will default-initialise it.
 */
template<typename T, typename R = reallocator>
class pod_vector {
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
	static constexpr size_type growth_factor_ { 2 };

	value_type *m_data = nullptr;
	size_type m_capacity = 0;
	size_type m_size = 0;
	R m_realloc;

	template<typename T2, typename R2>
	friend class pod_vector;

	/**
	 * Allocates or reallocates (copying the data) the data_pointer to the
	 * specified size (in number of elements of T), updating the capacity in
	 * the process.
	 */
	void realloc_(size_type size) {
		if (size != m_capacity) {
			m_data = static_cast<T*>(m_realloc.reallocate(m_data, sizeof(T) * size));
			m_capacity = size;
		}
	}

public:
	pod_vector(const R &realloc = R()) : m_realloc{realloc} {}

	pod_vector(const pod_vector<T, R>& other) {
		*this = other;
	}

	pod_vector(pod_vector<T, R>&& other) {
		*this = std::move(other);
	}

	pod_vector(size_type size, const R &realloc = R()) : m_realloc{realloc} {
		resize(size);
	}

	pod_vector(size_type size, const T& value, const R &realloc = R())
	: m_realloc{realloc} {
		assign(size, value);
	}

	template<typename It1, typename = std::enable_if_t<is_input_iterator<It1>>>
	pod_vector(It1 first, It1 last, const R &realloc = R()) : m_realloc{realloc} {
		assign(first, last);
	}

	~pod_vector() {
		PERFLIBS_ASSERT(m_size <= m_capacity);
		PERFLIBS_ASSERT(m_data || m_capacity == 0);
		if (m_data) m_realloc.deallocate(m_data);
	}

	pod_vector<T, R>& operator=(pod_vector<T, R>&& other) {
		swap(other);
		return *this;
	}

	pod_vector<T, R>& operator=(const pod_vector<T, R>& other) {
		if(&other == this) {
			return *this;
		}
		realloc_(other.m_capacity);
		m_size = other.m_size;
		std::memcpy((void *)m_data, (const void *)other.m_data, sizeof(T) * m_size);
		m_realloc = other.m_realloc;
		return *this;
	}

	void assign(size_type size, const T& value) {
		resize(size);
		std::fill_n(m_data, size, value);
	}

	template<typename It1, typename = std::enable_if_t<is_input_iterator<It1>>>
	void assign(It1 first, It1 last) {
		resize(std::distance(first, last));
		std::copy(first, last, m_data);
	}

	reference at(size_type pos) {
		PERFLIBS_ASSERT(pos >= 0);
		PERFLIBS_ASSERT(pos < m_size);
		PERFLIBS_ASSERT(m_data);
		return m_data[pos];
	}

	const_reference at(size_type pos) const {
		PERFLIBS_ASSERT(pos >= 0);
		PERFLIBS_ASSERT(pos < m_size);
		PERFLIBS_ASSERT(m_data);
		return m_data[pos];
	}

	reference operator[](size_type pos) {
		return m_data[pos];
	}
	const_reference operator[](size_type pos) const {
		return m_data[pos];
	}

	reference front() {
		return at(0);
	}
	const_reference front() const {
		return at(0);
	}

	reference back() {
		return at(m_size - 1);
	}
	const_reference back() const {
		return at(m_size - 1);
	}

	pointer data() {
		return m_data;
	}
	const_pointer data() const {
		return m_data;
	}

	iterator begin() {
		return m_data;
	}
	const_iterator begin() const {
		return m_data;
	}
	const_iterator cbegin() const {
		return m_data;
	}

	iterator end() {
		return m_data + m_size;
	}
	const_iterator end() const {
		return m_data + m_size;
	}
	const_iterator cend() const {
		return m_data + m_size;
	}

	reverse_iterator rbegin() {
		return { end() };
	}
	const_reverse_iterator rbegin() const {
		return { end() };
	}
	const_reverse_iterator crbegin() const {
		return { cend() };
	}

	reverse_iterator rend() {
		return { begin() };
	}
	const_reverse_iterator rend() const {
		return { begin() };
	}
	const_reverse_iterator crend() const {
		return { cbegin() };
	}

	bool empty() const {
		return m_size == 0;
	}
	size_type size() const {
		return m_size;
	}
	size_type capacity() const {
		return m_capacity;
	}

	void reserve(size_type new_cap) {
		if (new_cap <= m_capacity) return;
		realloc_(new_cap);
		PERFLIBS_ASSERT(m_data);
	}

	void shrink_to_fit() {
		realloc_(m_size);
		PERFLIBS_ASSERT(m_data || m_size == 0);
	}

	void resize(size_type count) {
		realloc_(count);
		PERFLIBS_ASSERT(m_data || count == 0);
		m_size = count;
	}

	void resize(size_type count, const value_type& value) {
		realloc_(count);
		PERFLIBS_ASSERT(m_data || count == 0);
		if(count > m_size) std::fill(m_data + m_size, m_data + count, value);
		m_size = count;
	}

	void clear() {
		resize(0);
	}

	void swap(pod_vector<T, R>& other) {
		std::swap(m_data, other.m_data);
		std::swap(m_capacity, other.m_capacity);
		std::swap(m_size, other.m_size);
	}

	void push_back(T value) {
		if (m_size == m_capacity) {
			size_type new_capacity = m_capacity == 0 ? 1 : m_capacity * growth_factor_;
			realloc_(new_capacity);
		}
		m_data[m_size++] = std::move(value);
	}
};

/** A wrapper around void* equivalent roughly to void&.
 *  void& is disallowed by the language, but if all we want
 *  is to take the address of it there's no problem. */
class void_ref {
	void *ptr;

public:
	void_ref(void *ptr) : ptr(ptr) {
	}

	void *operator&() {
		return ptr;
	}

	const void *operator&() const {
		return ptr;
	}
};

/** A vector-like container for unknown value types (roughly a
 *  pod_vector<std::any>). This allows us to keep the memory management bits
 *  that we like from pod_vector while not caring about the exact underlying
 *  type.
 */
template<typename R>
class pod_vector<void, R> {
public:
	using size_type = std::size_t;
	using pointer = void*;
	using const_pointer = const void*;
	using reference = void_ref;
	using const_reference = const void_ref;

private:
	void *m_data = nullptr;
	size_type m_capacity = 0;
	size_type m_size = 0;
	R m_realloc;

	/** In addition to the parameters from a standard pod_vector, we must also
	 *  keep the size of each element, since we don't know it otherwise. */
	size_type m_elem_size = 0;

public:
	pod_vector() = default;

	template<typename T>
	pod_vector(const pod_vector<T, R>& other) {
		*this = other;
	}

	template<typename T>
	pod_vector(pod_vector<T, R>&& other) {
		*this = std::move(other);
	}

	pod_vector(const pod_vector<void, R>& other) {
		*this = other;
	}

	pod_vector(pod_vector<void, R>&& other) {
		*this = std::move(other);
	}

	/** We can move into this from any other pod_vector, since we can know the
	 *  element size by doing sizeof(T) */
	template<typename T>
	pod_vector<void, R>& operator=(pod_vector<T, R>&& other) {
		if ((void*) this == (void*) &other) {
			return *this;
		}
		if (m_data) m_realloc.deallocate(m_data);
		m_data = other.m_data;
		m_capacity = other.m_capacity;
		m_size = other.m_size;
		m_elem_size = sizeof(T);
		other.m_data = nullptr;
		other.m_capacity = other.m_size = 0;
		return *this;
	}

	template<typename T>
	pod_vector<void, R>& operator=(const pod_vector<T, R>& other) {
		if ((void*) this == (void*) &other) {
			return *this;
		}
		if (other.m_capacity * sizeof(T) != m_capacity * m_elem_size) {
			m_data = m_realloc.reallocate(m_data, other.m_capacity * sizeof(T));
		}
                m_capacity = other.m_capacity;
		m_size = other.m_size;
		m_elem_size = sizeof(T);
		std::memcpy(m_data, other.m_data, m_elem_size*m_size);
		return *this;
	}

	/** We can move into this from any other pod_vector<void, R>, since we already
	 *  have the (original) element size. */
	pod_vector<void, R>& operator=(pod_vector<void, R>&& other) {
		if (this == &other) {
			return *this;
		}
		if (m_data) m_realloc.deallocate(m_data);
		m_data = other.m_data;
		m_capacity = other.m_capacity;
		m_size = other.m_size;
		m_elem_size = other.m_elem_size;
		other.m_data = nullptr;
		other.m_capacity = other.m_size = other.m_elem_size = 0;
		return *this;
	}

	pod_vector<void, R>& operator=(const pod_vector<void, R>& other) {
		if (&other == this) {
			return *this;
		}
		if (other.m_capacity * other.m_elem_size != m_capacity * m_elem_size) {
			m_data = m_realloc.reallocate(m_data, other.m_capacity * other.m_elem_size);
		}
                m_capacity = other.m_capacity;
		m_size = other.m_size;
		m_elem_size = other.m_elem_size;
		std::memcpy(m_data, other.m_data, m_elem_size*m_size);
		return *this;
	}

	~pod_vector() {
		PERFLIBS_ASSERT(m_size <= m_capacity);
		PERFLIBS_ASSERT(m_data || m_capacity == 0);
		if (m_data) m_realloc.deallocate(m_data);
	}

	reference at(size_type pos) {
		PERFLIBS_ASSERT(pos >= 0);
		PERFLIBS_ASSERT(pos < m_size);
		PERFLIBS_ASSERT(m_data);
		uintptr_t data = reinterpret_cast<uintptr_t>(m_data);
		return reinterpret_cast<void*>(data + pos*m_elem_size);
	}

	const_reference at(size_type pos) const {
		PERFLIBS_ASSERT(pos >= 0);
		PERFLIBS_ASSERT(pos < m_size);
		PERFLIBS_ASSERT(m_data);
		uintptr_t data = reinterpret_cast<uintptr_t>(m_data);
		return reinterpret_cast<void*>(data + pos*m_elem_size);
	}

	reference operator[](size_type pos) {
		uintptr_t data = reinterpret_cast<uintptr_t>(m_data);
		return reinterpret_cast<void*>(data + pos*m_elem_size);
	}

	const_reference operator[](size_type pos) const {
		uintptr_t data = reinterpret_cast<uintptr_t>(m_data);
		return reinterpret_cast<void*>(data + pos*m_elem_size);
	}

	reference front() {
		return at(0);
	}
	const_reference front() const {
		return at(0);
	}

	reference back() {
		return at(m_size - 1);
	}
	const_reference back() const {
		return at(m_size - 1);
	}

	pointer data() {
		return m_data;
	}
	const_pointer data() const {
		return m_data;
	}

	bool empty() const {
		return m_size == 0;
	}
	size_type size() const {
		return m_size;
	}
	size_type capacity() const {
		return m_capacity;
	}
};
template<typename V>
struct is_vec_type : std::false_type {};

template<typename T>
struct is_vec_type<std::vector<T>> : std::true_type {};

template<typename T, typename R>
struct is_vec_type<perflibs::pod_vector<T, R>> : std::true_type {};

/**
 * Utility to help check that a template parameter is a vector type
 * (either std::vector or @ref perflibs::pod_vector for now)
 */
template<typename V>
constexpr bool is_vec_type_v = is_vec_type<V>::value;

template<typename T>
struct has_complex_value_type : std::false_type {};

template<typename T>
struct has_complex_value_type<std::vector<std::complex<T>>> : std::true_type {};

template<typename T, typename R>
struct has_complex_value_type<perflibs::pod_vector<std::complex<T>, R>> : std::true_type {};

template<typename T>
constexpr bool has_complex_value_type_v = has_complex_value_type<T>::value;

} // namespace perflibs

#endif

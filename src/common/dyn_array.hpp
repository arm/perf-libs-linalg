/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_DYN_ARRAY_HPP
#define PERFLIBS_DYN_ARRAY_HPP

#include "perflibs_type_traits.hpp"

#include <type_traits>
#include <vector>
#include <array>
#include <variant>
#include <algorithm>

namespace perflibs {

//namespace {
	/**
	 * We are using this wrapper class to stop std::variant from initializing
	 * std::array when emplacing.
	 */
	template<typename T, std::size_t N>
	struct no_init_array {
		std::array<T, N> array;

		no_init_array() { }

		template<typename... Args>
		inline
		no_init_array(Args&&... args)
		:	array { std::forward<Args>(args)... }
		{	}
	}; //struct no_init_array
//}

/**
 * Basically a std::array, but where the size can be set at runtime like
 * a std::vector however:
 * 		- once set the size is fixed unless assigned to
 * 		- and if the size is lt MaxStaticCapacity then the elements are stored on the stack, if larger they stored on the heap
 * 			* useful  when you want to optimize for a particular capacity, but also need it to work for other cases
 */
template<typename T, std::size_t MaxStaticCapacity>
class dyn_array {
public:
	using value_type             = T;
	using size_type              = std::size_t;
	using difference_type        = std::ptrdiff_t;
	using reference              = value_type&;
	using const_reference        = const value_type&;
	using pointer                = value_type*;
	using const_pointer          = const value_type*;
	using iterator               = pointer;
	using const_iterator         = const_pointer;
	using reverse_iterator       = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;
private:
	using static_storage_type    = no_init_array<T, MaxStaticCapacity>;
	using dynamic_storage_type   = std::vector<T>;
	using storage_type           = std::variant<static_storage_type, dynamic_storage_type>;

	constexpr static size_type static_capacity_ = MaxStaticCapacity;

	size_type    size_;    //number of elements
	storage_type storage_; //the actual data
	T           *data_;    //a cached pointer to the start of data

	inline       static_storage_type&  get_static()        { return std::get<static_storage_type>(storage_); }
	inline const static_storage_type&  get_static()  const { return std::get<static_storage_type>(storage_); }
	inline       dynamic_storage_type& get_dynamic()       { return std::get<dynamic_storage_type>(storage_); }
	inline const dynamic_storage_type& get_dynamic() const { return std::get<dynamic_storage_type>(storage_); }

	///from store, as an oppose to from cached
	inline
	T *get_data_from_store() {
		return is_static()
		     ? get_static().array.data()
		     : get_dynamic().data();
	}

	inline
	void reset_cached_data() {
		data_ = get_data_from_store();
	}
public:
	///init to a specified size
	explicit dyn_array(size_type size)
	:	size_ { size }
	{
		//initialize the storage type and cache the pointer
		data_ = is_static()
		      ? storage_.template emplace<static_storage_type>().array.data()
		      : storage_.template emplace<dynamic_storage_type>(size).data();
	}

	template<typename IntType>
	dyn_array(std::initializer_list<IntType> init)
	:	dyn_array( init.size() )
	{
		std::move( init.begin(), init.end(), begin() );
	}

	///init list ctor
	template<typename... Args>
	dyn_array(Args&&... args, std::enable_if_t<all_same_v<T, Args...>>* x=0)
	:	size_ { sizeof...(Args) }
	{
		//initialize the storage type and cache the pointer
		data_ = is_static()
		      ? storage_.template emplace<static_storage_type>( args... ).array.data()
			  : storage_.template emplace<dynamic_storage_type>( args... ).data();
	}

	///copy ctor
	dyn_array(const dyn_array& rhs)
	:	size_    { rhs.size_             }
	,	storage_ { rhs.storage_          }
	,   data_    { get_data_from_store() }
	{	}

	///copy ctor
	dyn_array(dyn_array&& rhs)
	:	size_    { rhs.size_               }
	,	storage_ { std::move(rhs.storage_) }
	,   data_    { get_data_from_store()   }
	{
		rhs.size_ = 0;
	}

	//default ctor, will do whatever std::array does in this case, empty() -> true
	dyn_array() : dyn_array(0) { };

	inline dyn_array& operator=(dyn_array&& rhs) {
		swap(rhs);
		return *this;
	}

	inline dyn_array& operator=(const dyn_array& rhs) {
		size_    = rhs.size_;
		storage_ = rhs.storage_;
		reset_cached_data();
		return *this;
	}

	inline iterator insert(const_iterator it, T val) {
		//this is not optimized, but is design to allow the JSON library to work
		//this simply makes a new dyn_array that has a capacity a single element
		//larger and then copies the data across the the inserted element
		dyn_array other ( size() + 1 );
		auto other_it = std::move(cbegin(), it, other.begin());
		const auto dist = std::distance(other.begin(), other_it);
		*other_it = std::move(val);
		std::move(it, cend(), std::next(other_it));
		swap(other);
		return std::next(begin(), dist);
	}

	inline void push_back(T val) {
		insert(end(), std::move(val));
	}

	void swap(dyn_array& rhs) {
		using std::swap;
		swap(size_, rhs.size_);
		swap(storage_, rhs.storage_);

		reset_cached_data();
		rhs.reset_cached_data();
	}

	inline reference       operator[](const size_type pos)       { return data()[pos]; }
	inline const_reference operator[](const size_type pos) const { return data()[pos]; }
	inline reference       at        (const size_type pos)       { return data()[pos]; }
	inline const_reference at        (const size_type pos) const { return data()[pos]; }
	inline reference       front()                               { return *data_; }
	inline const_reference front()                         const { return *data_; }
	inline reference       back()                                { return data_[size_-1]; }
	inline const_reference back()                          const { return data_[size_-1]; }
	inline       T *       data()                                { return data_; }
	inline const T *       data()                          const { return data_; }
	const_iterator         cbegin()                        const { return data_; }
	const_iterator         begin()                         const { return data_; }
	      iterator         begin()                               { return data_; }
	const_iterator         cend()                          const { return data_ + size_; }
	const_iterator         end()                           const { return data_ + size_; }
	      iterator         end()                                 { return data_ + size_; }

	inline bool            is_static()                     const { return size_ <= static_capacity_; }
	constexpr size_type    static_capacity()               const { return static_capacity_; }
	inline size_type       max_size()                      const { return get_dynamic().max_size(); }
	inline size_type       size()                          const { return size_; }
	inline bool            empty()                         const { return size_ == 0; }
}; //class dyn_array

template<typename T, std::size_t MaxN>
void swap(dyn_array<T, MaxN>& lhs, dyn_array<T, MaxN>& rhs) {
	lhs.swap(rhs);
}

///inline tests
static_assert(std::is_default_constructible_v<dyn_array<int, 4>>, "batch_base_value is not default constructible");
static_assert(std::is_copy_constructible_v<dyn_array<int, 4>>,    "batch_base_value is not copy constructible");
static_assert(std::is_move_constructible_v<dyn_array<int, 4>>,    "batch_base_value is not move constructible");
static_assert(std::is_copy_assignable_v<dyn_array<int, 4>>,       "batch_base_value is not copy constructible");
static_assert(std::is_move_assignable_v<dyn_array<int, 4>>,       "batch_base_value is not move constructible");

} //namespace perflibs

#endif //PERFLIBS_DYN_ARRAY_HPP

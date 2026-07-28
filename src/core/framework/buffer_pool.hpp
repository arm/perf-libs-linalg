/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BUFFER_POOL_HPP
#define PERFLIBS_LINALG_BUFFER_POOL_HPP

#include "linalg_util.hpp"
#include "perflibs_assert.hpp"

#include <cstdint>
#include <limits>

namespace perflibs::linalg {
namespace {
/**
 * A tag type to indicate to the buffer_pool constructor
 * that we using the unit of bytes and not elements.
 */
struct bytes {
	kernel_inttype value;
};


/**
 * This object divides up a resource amongst different workers
 */
template<typename T>
class buffer_pool {
public:
	using value_type = T;
private:
	//a pointer to our buffer, in bytes
	std::uint8_t  *buffer_;
	kernel_inttype nbuffers_;
	kernel_inttype buffer_stride_bytes_;
public:
	/**
	 * construct the buffer pool specifying the strid ein number of elements of T
	 */
	buffer_pool(value_type *buffer, kernel_inttype nbuffers, kernel_inttype buffer_stride_elems)
	:	buffer_              { reinterpret_cast<std::uint8_t*>(buffer) }
	,	nbuffers_            { nbuffers }
	,	buffer_stride_bytes_ { buffer_stride_elems * static_cast<kernel_inttype>(sizeof(T)) }
	{	}

	/**
	 * construct the buffer pool specifying the stride in number of bytes
	 */
	buffer_pool(value_type *buffer, kernel_inttype nbuffers, bytes buffer_stride_bytes)
	:	buffer_              { reinterpret_cast<std::uint8_t*>(buffer) }
	,	nbuffers_            { nbuffers }
	,	buffer_stride_bytes_ { buffer_stride_bytes.value }
	{	}

	/**
	 * This is a compatibility constructor for using operators designed for a
	 * multithreaded environment in sequential code.
	 */
	buffer_pool(value_type *buffer)
	:	buffer_pool { buffer, std::numeric_limits<kernel_inttype>::max(), 0 }
	{	}

	/**
	 * Get the buffer at a specified index, usually a thread or group identifier
	 */
	inline T *get_buffer(kernel_inttype buffer_index) {
		// Correct for the the special case that BLAS routine is called inside
		// an OpenMP task that is being executed by not thread 0.
		if (buffer_index > 0 && nbuffers_ == 1) {
			buffer_index = 0;
		}
		PERFLIBS_ASSERT(buffer_index < nbuffers_, "Buffer out of range");

		return reinterpret_cast<T*>(buffer_ + buffer_stride_bytes_ * buffer_index);
	}

	/**
	 * get the total number of buffers
	 */
	inline kernel_inttype get_nbuffers() const {
		return nbuffers_;
	}

	/**
	 * Get the number of elements between separate buffers
	 * note can lead to truncation if buffer_stride is not a
	 * multiple of sizeof(T) which may happen with alignment
	 */
	inline kernel_inttype get_buffer_stride() const {
		return buffer_stride_bytes_ / sizeof(T);
	}

	/**
	 * Get the number of bytes between separate buffers
	 */
	inline kernel_inttype get_buffer_stride_bytes() const {
		return buffer_stride_bytes_;
	}
}; //class buffer_pool

} // namespace anon
} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BUFFER_POOL_HPP

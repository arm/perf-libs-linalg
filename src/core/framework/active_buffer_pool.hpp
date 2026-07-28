/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_ACTIVE_BUFFER_POOL_HPP
#define PERFLIBS_LINALG_ACTIVE_BUFFER_POOL_HPP

#include "framework/alloc.hpp"
#include "dyn_array.hpp"
#include "framework/buffer_pool.hpp" //bytes
#include "perflibs_assert.hpp"
#include <mutex>
#include <vector>

namespace perflibs::linalg {

template<typename T, memory_bank Bank = memory_bank::main>
class basic_buffer_pool_allocator {
	kernel_inttype size_bytes_;
	kernel_inttype align_bytes_;
	kernel_inttype disalign_bytes_;

public:
	using value_type = T;

	basic_buffer_pool_allocator(bytes size_bytes, bytes align_bytes, bytes disalign_bytes)
	:	size_bytes_     { size_bytes.value     }
	,	align_bytes_    { align_bytes.value    }
	,	disalign_bytes_ { disalign_bytes.value }
	{	}

	basic_buffer_pool_allocator(kernel_inttype size_elems, bytes align_bytes, bytes disalign_bytes)
	:	size_bytes_     { size_elems * static_cast<kernel_inttype>(sizeof(T)) }
	,	align_bytes_    { align_bytes.value      }
	,	disalign_bytes_ { disalign_bytes.value   }
	{	}

	auto *allocate(kernel_inttype thread_id) const {
		PERFLIBS_UNUSED(thread_id);
		return reinterpret_cast<T*>(get_memory<std::uint8_t, Bank>(size_bytes_));
	}

	void deallocate(T *ptr) {
		return return_memory<std::uint8_t, Bank>(ptr);
	}
}; // class basic_buffer_pool_allocator


template<typename Allocator>
class active_buffer_pool {
public:
	using value_type = typename Allocator::value_type;

private:
	Allocator                  allocator_;
	//TODO replace with dyn_array
	std::vector<std::pair<std::once_flag, value_type*>> buffers_;
	kernel_inttype             team_size_;

public:
	active_buffer_pool(kernel_inttype nthreads, kernel_inttype team_size, Allocator allocator)
	:	allocator_ { std::move(allocator) }
	,	buffers_   ( static_cast<std::size_t>( nthreads / team_size ) )
	,	team_size_ { team_size            }
	{	}

	active_buffer_pool(const active_buffer_pool&) = delete;
	active_buffer_pool(active_buffer_pool&&)      = default;

	value_type *get_buffer(kernel_inttype thread_idx) {
		const auto buffer_idx = thread_idx / team_size_;

		PERFLIBS_ASSERT(static_cast<kernel_inttype>(buffers_.size()) > buffer_idx, "buffer id out of range");
		auto& [flag, buffer] = buffers_[buffer_idx];
		if(team_size_ > 1) {
			std::call_once(flag, [this, buffer_idx] (auto& b) {
				b = allocator_.allocate(buffer_idx);
			}, buffer);
		}
		else {
			if(buffer == nullptr) {
				buffer = allocator_.allocate(buffer_idx);
			}
		}
		PERFLIBS_ASSERT(buffer != nullptr, "buf is NULL");

		return buffer;
	}
}; //class active_buffer_pool

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_ACTIVE_BUFFER_POOL_HPP

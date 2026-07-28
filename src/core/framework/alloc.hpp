/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_ALLOC
#define PERFLIBS_LINALG_ALLOC

#include "pod_vector.hpp"

#include <cstdint>
#include <cstdlib>

namespace perflibs::linalg {
namespace {

using buffer_type = pod_vector<std::uint8_t>;

enum class memory_strategies {
	hoard, release
}; //enum class memory_strategies


/**
 * There can be many memory banks, banks are useful when there maybe a need
 * to allocate from multiple nesting routines.
 *
 * For example,  TBMV maybe need to allocate for its own implementation,
 * however that implementation may in turn invoke GEMV which allocate.
 *
 * In this use case it is critical that they have their own memory banks
 */
enum class memory_bank {
	///the default memory bank, use when there is no fear of recursive usage
	main,
	gemv,
	outer_product,

	level2,

	/*
	 * Triangular Solve uses compute(matmul3) itself, so to be safe it has its own bank
	 */
	triangular_solve,


	//the main buffers used in L3 blas operations for A, B & C
	main_a,
	main_b,
	main_c,

	//the team record each team of threads needs
	bcms_team_record
}; //enum class memory_bank

///access the memory strategy
inline memory_strategies& get_memory_strategy() {
	static memory_strategies memory_strategy { memory_strategies::hoard };
	return memory_strategy;
}

///thread local singleton to get the buffer associated with a specified bank
template<memory_bank Bank>
inline buffer_type& get_buffer() {
	thread_local static buffer_type buffer;
	return buffer;
}

///allocation function
template<typename T, memory_bank Bank = memory_bank::main>
inline T *get_memory(std::size_t i)  {
	//reference to thread local buffer
	auto& buffer = get_buffer<Bank>();

	i *= sizeof(T);

	if(std::size(buffer) < i) { //assign rather than resize as we want to maintain the alignment - shouldn't be called often
		buffer = buffer_type( i );
	}
	return reinterpret_cast<T*>(std::data(buffer));
}

///deallocation function, note if the strategy is to hoard, then this will be a no-op
template<typename T, memory_bank Bank = memory_bank::main>
inline void return_memory(const T *unaligned_data) {
	if(get_memory_strategy() == memory_strategies::release) {
		get_buffer<Bank>().clear();
	}
}

}
} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_ALLOC

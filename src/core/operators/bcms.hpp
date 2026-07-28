/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_OPERATORS_BCMS_HPP
#define PERFLIBS_LINALG_OPERATORS_BCMS_HPP

#include "matrix/matrix.hpp"

#include "framework/parallel.hpp"
#include "framework/buffer_pool.hpp"
#include "framework/synchronization.hpp"

#include "perflibs_util.hpp"

namespace perflibs::linalg {

/**
 * There will be one of these records per team of threads
 */
template<typename T>
struct bcms_team_record {
	using value_type = T;
	/// pointer to the buffer that the packed data will be stored into
	T               *buffer;
	/// the synchronization object used as a barrier for all of the threads in the team
	synchronization *synchro;
}; // struct bcms_team_record


/**
 * Each thread will have its own bcms_record, this record from the bcms operator
 * via a buffer_pool.
 *
 * Each record contains an ID which after the first access will be updated to contain
 * the threads ID in its thread group. The record also has a pointer to this group
 * shared buffer and this buffers shared synchronisatoin object
 */
template<typename T>
struct bcms_thread_record {
	using value_type = T;
	/// the ID of the thread in the team (0 < team_size)
	int                  thread_id;
	/// pointer to the team record shared between all of the threads in the team
	bcms_team_record<T> *team_record;
}; //struct bcms_record


/**
 * There is one of these structs allocated per team of threads, this stores
 * all of the book keeping data and required for that team
 */
template<typename T, memory_bank Bank>
class bcms_team_buffer_pool_allocator {
	/// number of threads working together in a single team
	kernel_inttype team_size_;
	/// alignment of synchronization object in bytes
	kernel_inttype syncro_align_bytes_;
	/// size of the buffers in bytes
	kernel_inttype buffer_size_bytes_;
	/// alignment of buffer in bytes
	kernel_inttype buffer_align_bytes_;
	/// disalignment of buffer in bytes
	kernel_inttype buffer_disalign_bytes_;

public:
	using value_type = bcms_team_record<T>;

	PERFLIBS_LINALG_INLINE
	bcms_team_buffer_pool_allocator(kernel_inttype team_size, bytes syncro_align_bytes, bytes buffer_size_bytes, bytes buffer_align_bytes, bytes buffer_disalign_bytes)
	:	team_size_             { team_size }
	,	syncro_align_bytes_    { syncro_align_bytes.value }
	,	buffer_size_bytes_     { buffer_size_bytes.value }
	,	buffer_align_bytes_    { buffer_align_bytes.value }
	,	buffer_disalign_bytes_ { buffer_disalign_bytes.value }
	{	}

	PERFLIBS_LINALG_INLINE
	bcms_team_buffer_pool_allocator(kernel_inttype team_size, bytes syncro_align_bytes, kernel_inttype buffer_size_elems, bytes buffer_align_bytes, bytes buffer_disalign_bytes)
	:	team_size_             { team_size }
	,	syncro_align_bytes_    { syncro_align_bytes.value }
	,	buffer_size_bytes_     { buffer_size_elems * static_cast<kernel_inttype>(sizeof(T)) }
	,	buffer_align_bytes_    { buffer_align_bytes.value }
	,	buffer_disalign_bytes_ { buffer_disalign_bytes.value }
	{	}

	PERFLIBS_LINALG_INLINE
	bcms_team_record<T> *allocate(kernel_inttype thread_id) const {

		const kernel_inttype size = sizeof(bcms_team_record<T>)
		                          + sizeof(synchronization) + 2*syncro_align_bytes_ //2x because align_to may also apply a disalign step
		                          + buffer_size_bytes_      + 2*buffer_align_bytes_;

		// allocate, partition & align
		auto bcms_ptr_bytes = get_memory<std::uint8_t, Bank>(size);
		auto sync_ptr_bytes = align_to(bcms_ptr_bytes + sizeof(bcms_team_record<T>), syncro_align_bytes_);
		auto buff_ptr_bytes = align_to(sync_ptr_bytes + sizeof(synchronization),     buffer_align_bytes_);

		// construct the bcms_team_record inplace at the allocated point
		return new(bcms_ptr_bytes) bcms_team_record<T> {
			reinterpret_cast<T*>(buff_ptr_bytes),
			// construct the sync obj with the team_size
			new(sync_ptr_bytes) synchronization { team_size_ }
		};
	}
}; // class bcms_team_buffer_pool_allocator


template<typename TeamBufferPool, memory_bank Bank>
class bcms_buffer_pool_allocator {
public:
	using buffer_value_type = typename TeamBufferPool::value_type::value_type;
	using value_type = bcms_thread_record<buffer_value_type>;

private:
	TeamBufferPool team_buffer_pool_;

public:
	PERFLIBS_LINALG_INLINE
	bcms_buffer_pool_allocator(TeamBufferPool team_buffer_pool)
	:	team_buffer_pool_ { std::move(team_buffer_pool) }
	{	}

	PERFLIBS_LINALG_INLINE
	value_type *allocate(kernel_inttype thread_id) {
		return new(get_memory<value_type, Bank>(1)) value_type {
			/*
			 * Initialize the thread ID to negative so that it is set
			 * on the first run through the BCMS
			 */
			-1_ki,
			// get the team record, first thread in team will allocate
			team_buffer_pool_.get_buffer(thread_id)
		};
	}

	PERFLIBS_LINALG_INLINE
	void deallocate(value_type *ptr) {

	}
}; // class bcms_buffer_pool_allocator

template<typename T>
PERFLIBS_LINALG_INLINE
kernel_inttype bcms_total_size(kernel_inttype teams, kernel_inttype threads_per_team,  kernel_inttype buffer_size_bytes, kernel_inttype align) {
	const auto synchro_align_bytes           = align;
	const auto buffer_align_bytes            = align;
	const auto thread_record_align_bytes     = align;
	const auto team_record_align_bytes       = align;

	const auto bcms_team_record_size_bytes   = sizeof(bcms_team_record<T>) + 2 * team_record_align_bytes
	                                         + sizeof(synchronization)     + 2 * synchro_align_bytes
	                                         + buffer_size_bytes           + 2 * buffer_align_bytes;

	const auto total_team_size_bytes         = bcms_team_record_size_bytes * teams;
	const auto total_thread_size_bytes       = ( sizeof(bcms_thread_record<T>) + 2 * thread_record_align_bytes )  * teams * threads_per_team;

	return total_team_size_bytes + total_thread_size_bytes;
}

template<typename T>
PERFLIBS_LINALG_INLINE
auto setup_static_bcms(kernel_inttype teams, kernel_inttype threads_per_team, std::uint8_t *ptr, kernel_inttype buffer_size_bytes, kernel_inttype align) {
	const auto synchro_align_bytes           = align;
	const auto buffer_align_bytes            = align;
	const auto team_record_align_bytes       = align;

	const auto bcms_team_record_size_bytes   = sizeof(bcms_team_record<T>);
	const auto synchro_size_bytes            = sizeof(synchronization);
	const auto bcms_thread_record_size_bytes = sizeof(bcms_thread_record<T>);

	const auto bcms_team_size_bytes   = bcms_team_record_size_bytes + 2 * team_record_align_bytes
	                                  + synchro_size_bytes          + 2 * synchro_align_bytes
	                                  + buffer_size_bytes           + 2 * buffer_align_bytes;

	const auto bcms_size_bytes        = bcms_team_size_bytes * teams;

	auto bcms_team_record_ptr_bytes   = ptr;
	auto bcms_thread_record_ptr_bytes = bcms_team_record_ptr_bytes + bcms_size_bytes;

	kernel_inttype thread_counters = 0;

	for(kernel_inttype team = 0; team != teams; ++team) {
		auto bcms_team_ptr_bytes      = align_to(bcms_team_record_ptr_bytes + team * bcms_team_size_bytes, team_record_align_bytes);
		auto sync_ptr_bytes           = align_to(bcms_team_ptr_bytes + sizeof(bcms_team_record<T>),        synchro_align_bytes);
		auto buff_ptr_bytes           = align_to(sync_ptr_bytes      + sizeof(synchronization),            buffer_align_bytes);

		auto bcms_team_ptr = new (bcms_team_ptr_bytes) bcms_team_record<T> {
			reinterpret_cast<T*>(buff_ptr_bytes),
			new(sync_ptr_bytes) synchronization { threads_per_team }
		};

		for(kernel_inttype thread = 0; thread < threads_per_team; ++thread) {
			new(bcms_thread_record_ptr_bytes + bcms_thread_record_size_bytes * thread_counters++) bcms_thread_record<T> { -1, bcms_team_ptr };
		}
	}
	return reinterpret_cast<bcms_thread_record<T>*>(bcms_thread_record_ptr_bytes);
}

/**
 * Block Compute Memory Shuffle
 */
template<which_matrix WhichMatrix, typename BufferPool, typename Convert, typename Next>
class bcms {
	BufferPool record_pool_;
	Convert    convert_;
	Next       next_;

public:
	PERFLIBS_LINALG_INLINE
	bcms(BufferPool record_pool, Convert convert, Next next)
	:	record_pool_ { std::move(record_pool) }
	,	convert_     { std::move(convert)     }
	,	next_        { std::move(next)        }
	{	}

	PERFLIBS_LINALG_INLINE
	bcms(which_matrix_constant<WhichMatrix>, BufferPool record_pool, Convert convert, Next next)
	:	bcms(std::move(record_pool), std::move(convert), std::move(next)) { }

	template<typename AMatrixType, typename BMatrixType, typename CMatrixType, typename... Args>
	PERFLIBS_LINALG_INLINE
	void operator()(const AMatrixType& a, const BMatrixType& b, const CMatrixType& c, compute_position pos, Args... args) {
		auto record = record_pool_.get_buffer(pos.thread_num);

		auto& thread_id = record->thread_id;
		auto& synchro   = *record->team_record->synchro;
		auto  buffer    = record->team_record->buffer;

		//synchronize and set an id if not set
		if(auto t = synchro(); thread_id < 0)
			thread_id = t;

		if constexpr (WhichMatrix == which_matrix::a) {
			auto split = make_parallel_split(a.strd(), convert_.strd_interleave(), synchro.nthreads(), a.strd());
			auto packed_a = convert_.make_packed(a, buffer);
			if (thread_id < split.threads) {
				//TODO should also include strd unroll
				auto [ start, chunk_size ] = work_distribution(thread_id, split, convert_.strd_interleave());
				if (chunk_size + start > a.strd()) chunk_size = a.strd() - start;
				auto thread_a_panel        = get_strd_panel(a,        start, chunk_size);
				auto thread_packed_a_panel = get_strd_panel(packed_a, start, chunk_size);
				convert_.pack(thread_a_panel, thread_packed_a_panel);
			}
			synchro(); //barrier
			next_(packed_a, b, c, pos, args...);
		}
		else {
			auto split = make_parallel_split(b.strd(), convert_.strd_interleave(), synchro.nthreads(), b.strd());
			auto packed_b = convert_.make_packed(b, buffer);
			if (thread_id < split.threads) {
				auto [ start, chunk_size ] = work_distribution(thread_id, split, convert_.strd_interleave());
				if (chunk_size + start > b.strd()) chunk_size = b.strd() - start;
				auto thread_b_panel        = get_strd_panel(b,        start, chunk_size);
				auto thread_packed_b_panel = get_strd_panel(packed_b, start, chunk_size);
				convert_.pack(thread_b_panel, thread_packed_b_panel);
			}
			synchro(); //barrier
			next_(a, packed_b, c, pos, args...);
		}
	}
}; //class bcms

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_OPERATORS_BCMS_HPP


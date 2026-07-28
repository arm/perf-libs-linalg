/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SYNCHRONISATION
#define PERFLIBS_LINALG_SYNCHRONISATION

#include "perflibs_util.hpp"

#include <atomic>

namespace perflibs::linalg {
namespace {

/**
 * A barrier implemented using std::atomic to align an arbietary number of threads
 */
class synchronization {
public:
	using counter_type = kernel_inttype;

	counter_type nthreads() const { return last_thread_id_ + 1; }

	synchronization(counter_type nthreads)
	:	last_thread_id_ { nthreads - 1 }
	{	}

	synchronization(const synchronization& other)
	:	last_thread_id_ { other.last_thread_id_ }
	{	}

	synchronization& operator=(const synchronization& other) {
		last_thread_id_ = other.last_thread_id_;
		bar = 0;
		passed = 0;
		return *this;
	}

	/**
	 * Update the number of threads needed to pass this barrier. If bar != 0,
	 * callers must ensure no threads are currently waiting on this barrier.
	 *
	 * This isn't threadsafe, but should be fine if all the threads are calling
	 * this with the same value & no other computations are happening with this
	 * object in the meantime.
	 */
	void update_threads_needed(kernel_inttype needed) {
		last_thread_id_ = needed - 1;
	}

	counter_type operator()() {
		counter_type passed_old = passed.load(std::memory_order_relaxed);

		counter_type thread = bar.fetch_add(1);

		// The last thread, faced barrier.
		if(thread == last_thread_id_) {
			bar = 0;
			// Synchronize and store in one operation.
			passed.store(passed_old + 1, std::memory_order_release);
		}
		// Not the last thread. Wait others.
		else {
			while(passed.load(std::memory_order_relaxed) == passed_old) {};
			// Need to synchronize cache with other threads, passed barrier.
			std::atomic_thread_fence(std::memory_order_acquire);
		}
		return thread;
	}
private:
	counter_type last_thread_id_;
	std::atomic<counter_type> bar    { 0 }; // Counter of threads, faced barrier.
	std::atomic<counter_type> passed { 0 }; // Number of barriers, passed by all threads.
}; // class synchronization

}
};

#endif //PERFLIBS_LINALG_SYNCHRONISATION

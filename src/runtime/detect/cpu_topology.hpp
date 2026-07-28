/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#pragma once

#include <cstdint>

namespace perflibs::cpu_topology {

extern int64_t core_threads;
extern int64_t total_threads;
extern int64_t total_cores;

/** Get the number of threads (SMT ...) for this machine.
 *
 * @note This returns the cached value in `core_threads`.
 * On OpenMP Linux builds, that value is initialized once from
 * `/sys/devices/system/cpu/cpu0/topology/thread_siblings`.
 *
 * The number of threads for cpu0 is read and returned - it is assumed that this
 * number is the same as all the other cores.
 */
inline int64_t get_core_threads() {
	return core_threads;
}

/** Get the number of threads, irrespective of OMP_NUM_THREADS. */
inline int64_t get_total_threads() {
	return total_threads;
}

/** Get the number of cores, irrespective of OMP_NUM_THREADS. */
inline int64_t get_total_cores() {
	return total_cores;
}

}
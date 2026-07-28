/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "cpu_topology.hpp"

#ifdef _OPENMP

#if defined(_WIN32)

#include <Windows.h>
#include <Sysinfoapi.h>

static int64_t init_total_cores() {
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	return sysinfo.dwNumberOfProcessors;
}

namespace perflibs::cpu_topology {
// Assume no hyper threading.
// If we ever need to detect thread per core in the future,
// we could probably use GetLogicalProcessorInformationEx
// function.
int64_t core_threads = 1;
int64_t total_cores = init_total_cores();
int64_t total_threads = core_threads * total_cores;
}

#else

#include <unistd.h>
#include <cassert>
#include <cctype>
#include <fstream>
#include <iterator>
#include <string>

static int64_t init_total_threads() {
	return sysconf(_SC_NPROCESSORS_ONLN);
}

static int64_t init_core_threads() {
	std::ifstream t("/sys/devices/system/cpu/cpu0/topology/thread_siblings");
	if (!t) {
		return 1;
	}
	std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
	// Str is something like "00010001,0000000,01000000,00000000" where each 1
	// represents a hyperthread.
	// However, this is actually a hex string, so "00000003" is a valid value,
	// which would represent 2 threads for this core.
	// Commas should be ignored.
	int64_t core_threads = 0;
	for (const auto c : str) {
		if (!isxdigit(c)) { continue; }
		int val = (c >= '0' && c <= '9') ?
			  (c - '0') : ((c >= 'a' && c <= 'f') ?
			  (c - 'a' + 10) : (c - 'A' + 10));
		int lookup[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
		core_threads += lookup[val];
	}
	assert(core_threads >= 1);
	return core_threads;
}

namespace perflibs::cpu_topology {
int64_t core_threads = init_core_threads();
int64_t total_threads = init_total_threads();
int64_t total_cores = total_threads/core_threads;
}

#endif

#else

namespace perflibs::cpu_topology {
int64_t core_threads = 1;
int64_t total_threads = 1;
int64_t total_cores = 1;
}

#endif

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SPEC_GENERIC_MACHINE_SPEC_HPP
#define PERFLIBS_LINALG_SPEC_GENERIC_MACHINE_SPEC_HPP

#include "framework/linalg_util.hpp"
#include <type_traits>

#include "perflibs_util.hpp"
#include "detect/numa.hpp"

namespace perflibs::linalg::spec {



template<typename T, typename MachineSpec>
constexpr kernel_inttype l1_data_cache_size_elements =
	MachineSpec::l1_data_cache_size_bytes / sizeof(T);

/**
 * This is a MachineSpec trait which denotes whether or not a
 * target has sve or not. By default this is false, sve targets
 * need to specialize this trait for their MachineSpec
 */
template<typename MachineSpec>
struct is_sve : std::false_type { };

template<typename MachineSpec>
constexpr bool is_sve_v = is_sve<MachineSpec>::value;

template<typename MachineSpec>
struct is_fp16 : std::false_type { };

template<typename MachineSpec>
constexpr bool is_fp16_v = is_fp16<MachineSpec>::value;

/*
 * On systems with no FEAT_FP16 support, we promote the
 * input types and perform the compute in FP32 instead.
 */
template<typename T, typename MachineSpec>
using promote_fp16 = std::conditional_t<is_fp16_v<MachineSpec>, T, std::common_type_t<T, float>>;

/**
 * Estimates the minimum number of NUMA nodes (based on runtime NUMA detection) required
 * to provide the requested number of threads (pinning 1 thread per CPU).
 * Overload based on MachineSpec to specialize.
 *
 * @param requested_threads - the number of threads available
 * @param MachineSpec an instantiation of the MachineSpec
 *
 * @returns the number of numa nodes
 */
template<typename MachineSpec>
PERFLIBS_LINALG_INLINE
const kernel_inttype numa_nodes(kernel_inttype requested_threads, const MachineSpec&) {
	const auto& numa = machine::get_numa_topology();
	// Work out the number of nodes required to provide requested_threads, assuming
	// that we pin 1 thread per CPU in consecutive node order starting from node 0
	for (size_t node = 0; node < numa.size(); node++) {
		requested_threads -= numa[node];
		if (requested_threads <= 0) {
			return node + 1;
		}
	}
	return numa.size();
}

} // namespace perflibs::linalg::spec

#endif //PERFLIBS_LINALG_SPEC_GENERIC_MACHINE_SPEC_HPP

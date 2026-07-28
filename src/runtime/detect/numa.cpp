/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "numa.hpp"
#include "cpu_info.hpp"

#include "os.hpp"
#include "pl_linalg_int.h"

#include <fstream>
#include <string>
#include <vector>

namespace perflibs::machine {

const std::vector<int>& get_numa_topology() {
	static const auto topology = [] {
		//For linux builds we'll try and read /sys/devices for numa info
		std::vector<int> cpus_per_node;

		if constexpr (linalg::os::linux) {
			int n = 0;
			std::string path { "/sys/devices/system/node/node" };

			while (std::ifstream stream { path + std::to_string(n++) + "/cpulist" }) {
				uint32_t start, end; char delim;
				if ((stream >> start >> delim >> end) && delim == '-' && end >= start) {
					// Parse a cpulist string such as '0-31' -> 32 cpus
					// TODO: parse a wider range of cpulist formats (e.g. '0-15,32-47')
					cpus_per_node.push_back(static_cast<int>(end - start + 1));
				}
				else {
					// We've failed to parse, so just record this as an empty node
					cpus_per_node.push_back(0);
				}
			}
		}

		if (cpus_per_node.empty()) {
			cpus_per_node.push_back(cpu_info::get_num_cores());
		}

		return cpus_per_node;
	} ();

	return topology;
}

} // namespace perflibs::machine

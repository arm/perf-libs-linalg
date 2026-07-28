/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "system.hpp"

#include "os.hpp"
#include "dmi.hpp"
#include "cpu_info.hpp"
#include "micro_arch.hpp"
#include "numa.hpp"
#include "sysctl.hpp"

#include <utility>

namespace perflibs::machine {

namespace {
constexpr std::pair<system, std::string_view> names[]{
	{ system::unknown,           "Unknown"                                       },

	{ system::taishan,           "Cortex A72 (generic)"                          },

	{ system::n1_c6g,            "Neoverse N1 (AWS)"                             },
	{ system::n1_c6g_arm64ec,    "Neoverse N1 (Arm64EC ABI)"                     },
	{ system::n1_altra_x80,      "Neoverse N1 (80-core/socket Ampere Altra)"     },
	{ system::n1_altra_max_x128, "Neoverse N1 (128-core/socket Ampere Altra Max)"},

	{ system::v1_c7g,            "Neoverse V1 (AWS, c7g)"                        },

	{ system::n2_cobalt100,      "Neoverse N2 (Microsoft Azure Cobalt 100)"      },
	{ system::n2_y710,           "Neoverse N2 (Yitian 710)"                      },

	{ system::n3_axion,          "Neoverse N3 (GCP, Axion)"                      },

	{ system::v2_axion,          "Neoverse V2 (GCP, Axion)"                      },
	{ system::v2_c8g,            "Neoverse V2 (AWS, c8g)"                        },
	{ system::v2_grace,          "Neoverse V2 (Grace)"                           },

	{ system::v3_c9g,            "Neoverse V3 (AWS, c9g)"                        },
	{ system::v3_cobalt200,      "Neoverse V3 (Microsoft Azure Cobalt 200)"      },

	{ system::ampere_one_gcp,    "Ampere One (GCP)"                              },

	{ system::apple_m1,          "Apple M1 (generic)"                            },
	{ system::apple_m2,          "Apple M2 (generic)"                            },
	{ system::apple_m4,          "Apple M4 (generic)"                            },

	{ system::fujitsu_monaka,    "Fujitsu MONAKA (generic)"                      },
};
}

system get_system_mac() {
	if (const auto sysctl = get_sysctl_info()) {
		if (sysctl->brand.starts_with("Apple M1"))
			return system::apple_m1;

		// Detect both Apple M4 and M5 systems as M4.
		// The SME codepath is enabled for this system, so we can't
		// use this as a fall-through.
		if (sysctl->brand.starts_with("Apple M4") || sysctl->brand.starts_with("Apple M5"))
			return system::apple_m4;
	}
	return system::apple_m2;
}

/*
 * Heuristic system detection used to select tuned system specs.
 *
 * The goal is to map the current machine to the closest system profile
 * available in this project and use the corresponding tuned settings.
 *
 * This selection logic has been validated on a range of tested systems, but
 * there is no guarantee it will always select correctly on every platform.
 *
 * It is implementation guidance for this project, not an official or canonical
 * method for determining system identity.
 */
system get_system_impl() {
	if constexpr (linalg::os::mac) {
		return get_system_mac();
	}

	// First layer of distinction is micro-arch
	const auto uarch = detect_micro_arch();
	switch (uarch) {

	case micro_arch::neoverse_n1: {
		const auto& numa = get_numa_topology();
		switch (numa[0]) {
		case 80: return system::n1_altra_x80;
		case 128: return system::n1_altra_max_x128;
		default: return system::n1_c6g;
		}
	}

	case micro_arch::generic_sve:
	case micro_arch::neoverse_v1: return system::v1_c7g;

	// Neoverse N3 uses N2 settings for now
	case micro_arch::neoverse_n2:
	case micro_arch::neoverse_n3: {
		const auto implementer = cpu_info::get_cpu_implementer();
		const auto& dmi = get_dmi_info();
		// Microsoft implementer or DMI info -> assume Microsoft Azure Cobalt 100
		if (implementer == 'm' || (dmi && dmi->sys_vendor == "Microsoft Corporation")) {
			return system::n2_cobalt100;
		}
		// Fall back to Yitian 710
		return system::n2_y710;
	}

	case micro_arch::neoverse_v2: {
		const auto& dmi = get_dmi_info();
		if (dmi && dmi->sys_vendor == "Amazon EC2") {
			return system::v2_c8g;
		}
		if (dmi && dmi->sys_vendor == "Google") {
			return system::v2_axion;
		}
		return system::v2_grace;
	}

	case micro_arch::neoverse_v3: {
		const auto& dmi = get_dmi_info();
		// DMI info -> assume Microsoft Azure Cobalt 200
		if (dmi && dmi->sys_vendor == "Microsoft Corporation") {
			return system::v3_cobalt200;
		}
		// Fall back to AWS Graviton5
		return system::v3_c9g;
	}

	case micro_arch::ampere_one: return system::ampere_one_gcp;

	case micro_arch::fujitsu_monaka: return system::fujitsu_monaka;

	default: return system::taishan;
	}
}

const system global_system = get_system_impl();

system get_system() {
	static const auto sys = get_system_impl();
	return sys;
}

system get_system_unsafe() {
	return global_system;
}

std::string_view get_system_str() {
	auto sys = get_system();
	for (const auto& [s, name] : names)
		if (s == sys) return name;
	return "Unknown";
}

} // namespace perflibs::machine

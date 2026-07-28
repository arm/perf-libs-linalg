/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "micro_arch.hpp"
#include "cpu_info.hpp"

#include <cstdlib>
#include <optional>
#include <string_view>

namespace perflibs::machine {
namespace {

struct micro_arch_entry {
	micro_arch arch;
	std::string_view env_name;         // PL_LINALG_TARGET value that selects this arch
	long implementer;                  // MIDR_EL1[31:24] implementer IDs
	long part;                         // MIDR_EL1[15:4] part number
	bool requires_sve;                 // whether this micro-architecture assumes SVE support
};

// Table of recognized micro-architectures, used for both environment-variable overrides
// (PL_LINALG_TARGET) and MIDR-based hardware detection.
constexpr micro_arch_entry micro_arch_table[] = {
	{ micro_arch::neoverse_n1,    "PL_LINALG_NEOVERSE_N1",    0x41, 0xd0c, false },
	{ micro_arch::neoverse_v1,    "PL_LINALG_NEOVERSE_V1",    0x41, 0xd40,  true },
	{ micro_arch::neoverse_n2,    "PL_LINALG_NEOVERSE_N2",    0x41, 0xd49,  true },
	{ micro_arch::neoverse_n3,    "PL_LINALG_NEOVERSE_N3",    0x41, 0xd8e,  true },
	{ micro_arch::neoverse_v2,    "PL_LINALG_NEOVERSE_V2",    0x41, 0xd4f,  true },
	{ micro_arch::neoverse_v3,    "PL_LINALG_NEOVERSE_V3",    0x41, 0xd83,  true }, // V3AE
	{ micro_arch::neoverse_v3,    "PL_LINALG_NEOVERSE_V3",    0x41, 0xd84,  true }, // V3
	{ micro_arch::ampere_one,     "PL_LINALG_AMPERE_ONE",     0xc0, 0xac3, false },
	{ micro_arch::fujitsu_monaka, "PL_LINALG_FUJITSU_MONAKA", 0x46, 0x003,  true },
};

// Looks up the micro-architecture from the PL_LINALG_TARGET environment variable, if set.
// Returns nullopt if the variable is unset or contains an unrecognized value
std::optional<micro_arch> micro_arch_from_env() {
	const char *env = std::getenv("PL_LINALG_TARGET");
	if (!env) {
		return std::nullopt;
	}

	const std::string_view which { env };
	for (const auto &entry : micro_arch_table) {
		// If the environment variable matches this entry,
		// return the corresponding micro-architecture.
		if (which == entry.env_name) {
			return entry.arch;
		}
	}
	if (which == "PL_LINALG_GENERIC_SVE") {
		return micro_arch::generic_sve;
	}
	if (which == "PL_LINALG_GENERIC") {
		return micro_arch::generic;
	}

	// Unrecognized value (or PL_LINALG_DETERMINE): ignore and fall back to probing.
	return std::nullopt;
}

// Looks up the micro-architecture for a given implementer/part pair.
// Returns nullopt if no entry matches, allowing the caller to apply a fallback..
// SVE-gated entries (requires_sve) return nullopt when SVE is unavailable; the caller applies fallback.
std::optional<micro_arch> micro_arch_from_midr(long implementer, const long part, const bool sve) {
	// See Microsoft Azure Cobalt 100 MIDR implementer handling:
	// https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=fb091ff394792c018527b3211bbdfae93ea4ac02
	if (implementer == 0x6d) implementer = 0x41;

	// Look for a matching entry in the table.
	for (const auto &entry : micro_arch_table) {
		if ((entry.implementer == implementer) && (entry.part == part)) {
			// If this micro-arch requires SVE but it is not present, return nullopt.
			// Should be resolved by the caller.
			if (entry.requires_sve && !sve) {
				return std::nullopt;
			}
			return entry.arch;
		}
	}

	return std::nullopt;
}

} // namespace

micro_arch detect_micro_arch() {
	// Get the micro-architecture from the environment variable, if set.
	if (const auto from_env = micro_arch_from_env()) {
		return *from_env;
	}

	// `implementer` and `part` are values parsed from MIDR fields exposed via
	// `/proc/cpuinfo` (or equivalent):
	//   - implementer ~= MIDR_EL1[31:24] (e.g. 0x41 Arm, 0x6d Microsoft)
	//   - part        ~= MIDR_EL1[15:4]  (e.g. 0xd0c N1, 0xd40 V1, ...)
	// These hex IDs are architecture-defined CPU identifiers.
	const bool sve = cpu_info::get_cpu_features().sve;
	const auto fallback_arch = sve ? micro_arch::generic_sve : micro_arch::generic;
	auto implementer = cpu_info::get_cpu_implementer();
	const auto part = cpu_info::get_cpu_part();

	// Use MIDR values to detect the micro-architecture.
	const auto arch = micro_arch_from_midr(implementer, part, sve);
	if (arch) {
		return *arch;
	}

	// Fall back to generic if we don't recognize the implementer/part pair.
	return fallback_arch;
}

} // namespace perflibs::machine

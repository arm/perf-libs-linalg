/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "cpu_info.hpp"

#include "registry.hpp"
#include "sysctl.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#if !defined(__APPLE__) && !defined(_WIN32)
#include <sys/auxv.h>
#endif
#include <thread>

#if defined(_WIN32)
// Embed the link requirement for advapi32.lib into the object
#pragma comment(lib, "advapi32.lib")
#endif

static perflibs::machine::cpu_info::cpu_features parse_features_from_string(std::string str, const char *src) {
	perflibs::machine::cpu_info::cpu_features ret;
	ret.source = src;
	for (size_t pos=0; pos < str.size(); ++pos) {
		size_t pos2 = str.find(' ', pos);
		if (pos2 == std::string::npos) {
			pos2 = str.size();
		}
		auto word = str.substr(pos, pos2-pos);
		if (word == "fp") ret.fp = true;
		else if (word == "asimd") ret.asimd = true;
		else if (word == "evtstrm") ret.evtstrm = true;
		else if (word == "aes") ret.aes = true;
		else if (word == "pmull") ret.pmull = true;
		else if (word == "sha1") ret.sha1 = true;
		else if (word == "sha2") ret.sha2 = true;
		else if (word == "crc32") ret.crc32 = true;
		else if (word == "atomics") ret.atomics = true;
		else if (word == "fphp") ret.fphp = true;
		else if (word == "asimdhp") ret.asimdhp = true;
		else if (word == "cpuid") ret.cpuid = true;
		else if (word == "asimdrdm") ret.asimdrdm = true;
		else if (word == "jscvt") ret.jscvt = true;
		else if (word == "fcma") ret.fcma = true;
		else if (word == "lrcpc") ret.lrcpc = true;
		else if (word == "dcpop") ret.dcpop = true;
		else if (word == "sha3") ret.sha3 = true;
		else if (word == "sm3") ret.sm3 = true;
		else if (word == "sm4") ret.sm4 = true;
		else if (word == "asimddp") ret.asimddp = true;
		else if (word == "sha512") ret.sha512 = true;
		else if (word == "sve") ret.sve = true;
		else if (word == "asimdfhm") ret.asimdfhm = true;
		else if (word == "dit") ret.dit = true;
		else if (word == "uscat") ret.uscat = true;
		else if (word == "ilrcpc") ret.ilrcpc = true;
		else if (word == "flagm") ret.flagm = true;
		else if (word == "ssbs") ret.ssbs = true;
		else if (word == "sb") ret.sb = true;
		else if (word == "paca") ret.paca = true;
		else if (word == "pacg") ret.pacg = true;
		else if (word == "bf16") ret.neon_bf16 = true;
		else if (word == "svebf16") ret.sve_bf16 = true;
		pos = pos2;
	}
	return ret;
}

#if !defined(__APPLE__) && !defined(_WIN32)
static perflibs::machine::cpu_info::cpu_features parse_features_from_hwcap(unsigned long x0, unsigned long x1, const char *src) {
	perflibs::machine::cpu_info::cpu_features ret;
	ret.source = src;
	ret.fp = (x0 & (1ul << 0)) != 0ul;
	ret.asimd = (x0 & (1ul << 1)) != 0ul;
	ret.evtstrm = (x0 & (1ul << 2)) != 0ul;
	ret.aes = (x0 & (1ul << 3)) != 0ul;
	ret.pmull = (x0 & (1ul << 4)) != 0ul;
	ret.sha1 = (x0 & (1ul << 5)) != 0ul;
	ret.sha2 = (x0 & (1ul << 6)) != 0ul;
	ret.crc32 = (x0 & (1ul << 7)) != 0ul;
	ret.atomics = (x0 & (1ul << 8)) != 0ul;
	ret.fphp = (x0 & (1ul << 9)) != 0ul;
	ret.asimdhp = (x0 & (1ul << 10)) != 0ul;
	ret.cpuid = (x0 & (1ul << 11)) != 0ul;
	ret.asimdrdm = (x0 & (1ul << 12)) != 0ul;
	ret.jscvt = (x0 & (1ul << 13)) != 0ul;
	ret.fcma = (x0 & (1ul << 14)) != 0ul;
	ret.lrcpc = (x0 & (1ul << 15)) != 0ul;
	ret.dcpop = (x0 & (1ul << 16)) != 0ul;
	ret.sha3 = (x0 & (1ul << 17)) != 0ul;
	ret.sm3 = (x0 & (1ul << 18)) != 0ul;
	ret.sm4 = (x0 & (1ul << 19)) != 0ul;
	ret.asimddp = (x0 & (1ul << 20)) != 0ul;
	ret.sha512 = (x0 & (1ul << 21)) != 0ul;
	ret.sve = (x0 & (1ul << 22)) != 0ul;
	ret.asimdfhm = (x0 & (1ul << 23)) != 0ul;
	ret.dit = (x0 & (1ul << 24)) != 0ul;
	ret.uscat = (x0 & (1ul << 25)) != 0ul;
	ret.ilrcpc = (x0 & (1ul << 26)) != 0ul;
	ret.flagm = (x0 & (1ul << 27)) != 0ul;
	ret.ssbs = (x0 & (1ul << 28)) != 0ul;
	ret.sb = (x0 & (1ul << 29)) != 0ul;
	ret.paca = (x0 & (1ul << 30)) != 0ul;
	ret.pacg = (x0 & (1ul << 31)) != 0ul;

	//AT_HWCAP2
	ret.neon_bf16 = (x1 & (1 << 14))  != 0ul; //HWCAP2_BF16
	ret.sve_bf16 = (x1 & (1 << 12))  != 0ul; //HWCAP2_SVEBF16

	return ret;
}
#endif

#if defined(__APPLE__)
static perflibs::machine::cpu_info::cpu_features parse_features_from_sysctl(const char *src) {
	perflibs::machine::cpu_info::cpu_features ret;
	ret.source = src;

	if (const auto sysctl = perflibs::machine::get_sysctl_info()) {
		/* sysctlbyname() is not necessarily cheap, so we only read in
		 * features that are needed by the library. Further entries
		 * should be added here as needed. */
		ret.fphp = sysctl->fphp;
		ret.neon_bf16 = sysctl->neon_bf16;
	}

	return ret;
}
#endif

struct cpuinfo {
	int ncores;
	long implementer;
	long variant;
	long part;
	long revision;
	perflibs::machine::cpu_info::cpu_features features;
};

static cpuinfo parse_proc_cpuinfo() {
	static cpuinfo ret;
	if (ret.ncores) {
		return ret;
	}

	#if defined(_WIN32)
	if (const auto registry = perflibs::machine::get_registry_info()) {
		const auto& pname = registry->processor_name;
		const auto is_ampere = pname.find("Ampere") != std::string::npos
		                    && pname.find("Altra")  != std::string::npos;
		// TODO: this will probably be folded into subtarget detection eventually
		if (is_ampere) {
			// Neoverse N1
			ret.implementer = 0x41;
			ret.part = 0xd0c;
		}
	}
	#elif defined(__linux__)
	std::ifstream input("/proc/cpuinfo");
	assert(input);
	for (std::string line; getline(input, line); ) {
		if (line.rfind("processor") == 0) {
			++ret.ncores;
		}
		else if (line.rfind("CPU implementer") == 0) {
			line = line.substr(line.find(':')+2);
			ret.implementer = strtol(line.c_str(), nullptr, 0);
		}
		else if (line.rfind("CPU variant") == 0) {
			line = line.substr(line.find(':')+2);
			ret.variant = strtol(line.c_str(), nullptr, 0);
		}
		else if (line.rfind("CPU part") == 0) {
			line = line.substr(line.find(':')+2);
			ret.part = strtol(line.c_str(), nullptr, 0);
		}
		else if (line.rfind("CPU revision") == 0) {
			line = line.substr(line.find(':')+2);
			ret.revision = strtol(line.c_str(), nullptr, 0);
		}
		else if (line.rfind("Features") == 0) {
			line = line.substr(line.find(':')+2);
			ret.features = parse_features_from_string(std::move(line), "/proc/cpuinfo");
		}
	}
	#endif

	if (!ret.ncores) {
		ret.ncores = std::max(std::thread::hardware_concurrency(), 1U);
	}

	return ret;
}

int perflibs::machine::cpu_info::get_num_cores() {
	return parse_proc_cpuinfo().ncores;
}

long perflibs::machine::cpu_info::get_cpu_implementer() {
	return parse_proc_cpuinfo().implementer;
}

long perflibs::machine::cpu_info::get_cpu_variant() {
	return parse_proc_cpuinfo().variant;
}

long perflibs::machine::cpu_info::get_cpu_part() {
	return parse_proc_cpuinfo().part;
}

long perflibs::machine::cpu_info::get_cpu_revision() {
	return parse_proc_cpuinfo().revision;
}

perflibs::machine::cpu_info::cpu_features perflibs::machine::cpu_info::get_cpu_features() {
	static cpu_features features;
	if (!features.source) {
		const char *env_features = getenv("PL_LINALG_FEATURES");
		if (env_features) {
			// Start by trying to read the feature set from an environment
			// variable, in case the user has decided to explicitly specify the
			// environment.
			features = parse_features_from_string(env_features, "environment");
		}
#if !defined(__APPLE__) && !defined(_WIN32)
		else if (unsigned long cap1 = getauxval(AT_HWCAP)) {
			unsigned long cap2 = getauxval(AT_HWCAP2);
			// Else try to read the feature set from getauxval, since QEMU
			// and other emulators will augment this vector with the
			// set of emulated features but won't replace /proc/cpuinfo.
			features = parse_features_from_hwcap(cap1, cap2, "getauxval(AT_HWCAP); getauxval(AT_HWCAP2);");
		}
		else {
			// In some cases (like when running in a model, this may return zero),
			// in which case we do the next best thing and parse /proc/cpuinfo.
			features = parse_proc_cpuinfo().features;
		}
#elif defined(__APPLE__)
		else {
			// On macOS we can attempt to read CPU features in via sysctl
			features = parse_features_from_sysctl("sysctl");
		}
#endif
	}
	return features;
}

std::string perflibs::machine::cpu_info::get_cpu_features_str() {
	auto features = get_cpu_features();
	std::ostringstream sstm;
	if (features.fp) sstm << " fp";
	if (features.asimd) sstm << " asimd";
	if (features.evtstrm) sstm << " evtstrm";
	if (features.aes) sstm << " aes";
	if (features.pmull) sstm << " pmull";
	if (features.sha1) sstm << " sha1";
	if (features.sha2) sstm << " sha2";
	if (features.crc32) sstm << " crc32";
	if (features.atomics) sstm << " atomics";
	if (features.fphp) sstm << " fphp";
	if (features.asimdhp) sstm << " asimdhp";
	if (features.cpuid) sstm << " cpuid";
	if (features.asimdrdm) sstm << " asimdrdm";
	if (features.jscvt) sstm << " jscvt";
	if (features.fcma) sstm << " fcma";
	if (features.lrcpc) sstm << " lrcpc";
	if (features.dcpop) sstm << " dcpop";
	if (features.sha3) sstm << " sha3";
	if (features.sm3) sstm << " sm3";
	if (features.sm4) sstm << " sm4";
	if (features.asimddp) sstm << " asimddp";
	if (features.sha512) sstm << " sha512";
	if (features.sve) sstm << " sve";
	if (features.asimdfhm) sstm << " asimdfhm";
	if (features.dit) sstm << " dit";
	if (features.uscat) sstm << " uscat";
	if (features.ilrcpc) sstm << " ilrcpc";
	if (features.flagm) sstm << " flagm";
	if (features.ssbs) sstm << " ssbs";
	if (features.sb) sstm << " sb";
	if (features.paca) sstm << " paca";
	if (features.pacg) sstm << " pacg";
	if (features.neon_bf16) sstm << " bf16";
	if (features.sve_bf16) sstm << " svebf16";

	auto ret = std::move(sstm).str();
	assert(!ret.empty());
	return std::move(ret).substr(1);
}

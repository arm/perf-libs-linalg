/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_DETECT_CPU_INFO_HPP
#define PERFLIBS_DETECT_CPU_INFO_HPP

#include <string>

namespace perflibs::machine::cpu_info {
/// get number of cores at runtime
int get_num_cores();

/// get cpu implementer
long get_cpu_implementer();

/// get cpu variant
long get_cpu_variant();

/// get cpu part
long get_cpu_part();

/// get cpu revision
long get_cpu_revision();

/** supported features on the current cpu; not all of these
 *  are useful, but they are included for completeness. */
struct cpu_features {
	const char *source = nullptr; ///< Where we got the feature set from.
	bool fp = false; ///< Floating point is supported. Mandatory.
	bool asimd = false; ///< ASIMD (Neon) is supported. Mandatory.
	bool evtstrm = false; ///< Event Stream
	bool aes = false; ///< AES instructions (AESE, etc).
	bool pmull = false; ///< Polynomial Multiply Long instructions (PMULL/PMULL2).
	bool sha1 = false; ///< SHA-1 instructions (SHA1C, etc).
	bool sha2 = false; ///< SHA-2 instructions (SHA256H, etc).
	bool crc32 = false; ///< CRC32/CRC32C instructions.
	bool atomics = false; ///< Large System Extensions, Load/Store exclusives.
	bool fphp = false; ///< Half-precision floating-point (FP16).
	bool asimdhp = false; ///< Half-precision ASIMD (FP16).
	bool cpuid = false; ///< Expose CPU registers by emulation.
	bool asimdrdm = false; ///< Rounding Double Multiply Accumulate/Subtract (SQRDMLAH/SQRDMLSH).
	bool jscvt = false; ///< Javascript-style double->int convert (FJCVTZS).
	bool fcma = false; ///< Complex-number instructions.
	bool lrcpc = false; ///< Weaker release consistency (LDAPR, etc).
	bool dcpop = false; ///< Data cache clean to Point of Persistence (DC CVAP).
	bool sha3 = false; ///< SHA-3 instructions (EOR3, RAXI, XAR, BCAX).
	bool sm3 = false; ///< SM3 instructions (crypto).
	bool sm4 = false; ///< SM4 instructions (crypto).
	bool asimddp = false; ///< SIMD Dot Product.
	bool sha512 = false; ///< SHA512 instructions.
	bool sve = false; ///< Scalable Vector Extension (SVE).
	bool asimdfhm = false; ///< FMLAL{2}/FMLSL{2}.
	bool dit = false; ///< Data-Independent Instruction Timings.
	bool uscat = false; ///< Unaligned/Single-copy atomic instructions.
	bool ilrcpc = false; ///< LDAPR and STLR instructions with immediate offsets.
	bool flagm = false; ///< Flag manipulation instructions.
	bool ssbs = false; ///< Speculative Store Barrier.
	bool sb = false; ///< Speculation Barrier.
	bool paca = false; ///< Pointer Authentication.
	bool pacg = false; ///< Pointer Authentication.
	bool neon_bf16 = false; ///< Neon bfloat16
	bool sve_bf16 = false; ///< SVE bfloat16
};

cpu_features get_cpu_features();
std::string get_cpu_features_str();
}

#endif // PERFLIBS_DETECT_CPU_INFO_HPP

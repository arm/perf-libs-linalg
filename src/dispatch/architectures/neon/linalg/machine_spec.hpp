/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_NEON_ARCHITECTURE_SPEC_HPP
#define PERFLIBS_NEON_ARCHITECTURE_SPEC_HPP

#include "spec/config.hpp" //get_accelerator_count()
#include "spec/generic_machine_spec.hpp" //is_sve

#include "framework/linalg_util.hpp"

#include "detect/system.hpp" //system

#include <type_traits> //std::true_type

namespace perflibs::linalg::spec {

/**
 * In the LINALG spec system we use a MachineSpec where we want
 * routines to have the specific target on for which the code
 * was compiled "baked" into a symbol name. This way the compiler
 * and linker will try and remove different versions of the
 * same function.
 *
 * This is a struct rather than an enum as because in future we
 * we may wish to add fields to this struct.
 *
 * This particular struct is the MachineSpec for the generic_aarch64
 * target
 *
 * We need to make at least the generic case always available
 * so that code can always be built that calls into the generic
 * implementations
 */
struct neon_architecture_spec {
	constexpr static kernel_inttype cacheline_len            { 128 };
	constexpr static kernel_inttype l1_data_cache_size_bytes { 48 * 1024 };
	machine::system system                                   { machine::system::unknown };

	auto get_accelerator_count() const {
		const auto accelerator_count = spec::get_accelerator_count();

		if(accelerator_count == accelerator_count_not_set_value) {
			if(system == machine::system::apple_m4) {
				return 2_ki;
			}
			return 0_ki;
		}

		return accelerator_count;
	}
};

} // namespace perflibs::linalg::spec

#endif //PERFLIBS_NEON_ARCHITECTURE_SPEC_HPP

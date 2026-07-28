/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_SVE_ARCHITECTURE_SPEC_HPP
#define PERFLIBS_SVE_ARCHITECTURE_SPEC_HPP

#include "detect/system.hpp" //system
#include "spec/generic_machine_spec.hpp" //is_sve
#include "framework/linalg_util.hpp"

#include <type_traits> //std::true_type

namespace perflibs::linalg::spec {

struct sve_architecture_spec {
	constexpr static kernel_inttype cacheline_len            { 64 };
	constexpr static kernel_inttype l1_data_cache_size_bytes { 64 * 1024 };
	machine::system system                                   { machine::system::unknown };

	auto get_accelerator_count() const {
		return 0_ki;
	}
};

template<>
struct is_sve<sve_architecture_spec> : std::true_type { };

template<>
struct is_fp16<sve_architecture_spec> : std::true_type { };

} // namespace perflibs::linalg::spec

#endif //PERFLIBS_SVE_ARCHITECTURE_SPEC_HPP

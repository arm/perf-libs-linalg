/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_GEMM_SMALL_IMPL_HPP
#define PERFLIBS_LINALG_GEMM_SMALL_IMPL_HPP

#include "detect/os.hpp"
#include "detect/omp.hpp"
#include "perflibs_util.hpp"

#include "gemm_small_framework.hpp"
#include "gemm_small_vanilla.hpp"
#include "spec/problem_context_helpers.hpp"

namespace perflibs::gemm {

void sgemm_small_generic_aarch64(const kernel_inttype max_threads, const perflibs_trans transa,
                                 const perflibs_trans transb, kernel_inttype m, kernel_inttype n,
                                 kernel_inttype k, float alpha, const float *a, kernel_inttype lda,
                                 const float *b, kernel_inttype ldb, float beta, float *c,
                                 kernel_inttype ldc);
}

namespace perflibs::linalg {

template<typename T, typename>
struct gemm_small_impl {
	constexpr static gemm::gemm_small_f<T> get_kernel() {
		return nullptr;
	}

	template<typename ProblemContext>
	constexpr static bool can_compute(const ProblemContext&) {
		return false;
	}
};

template<typename ArchitectureSpec>
struct gemm_small_impl<std::complex<float>, ArchitectureSpec> {
	using value_type = std::complex<float>;

	constexpr static auto get_kernel() {
		return &gemm::cgemm_small_vanilla;
	}

	template<typename ProblemContext>
	static bool can_compute(const ProblemContext& pctx) {
		const int nt = omp::get_max_cores();
		const kernel_inttype prob_size = gemm_m(pctx) * gemm_n(pctx) * gemm_k(pctx);
		perflibs_trans transa = c_to_trans(gemm_transa(pctx));
		perflibs_trans transb = c_to_trans(gemm_transb(pctx));

		// We don't have a valid optimized small CGEMM kernel for SVE or Arm64EC
		if constexpr (os::windows_arm64ec || spec::is_sve_v<ArchitectureSpec>) {
			return false;
		}

		kernel_inttype limit;
		if (!is_trans(transa)) {
			if (!is_trans(transb)) {
				limit = 525; // NN
			}
			else {
				limit = 300; // NT
			}
		}
		else {
			if (!is_trans(transb)) {
				limit = 1450; // TN
			}
			else {
				limit = 825; // TT
			}
		}
		return prob_size / nt < (limit * limit * limit);
	}
};

template<typename ArchitectureSpec>
struct gemm_small_impl<float, ArchitectureSpec> {
	using value_type = float;

	constexpr static auto get_kernel() {
		return &gemm::sgemm_small_generic_aarch64;
	}

	template<typename ProblemContext>
	static bool can_compute(const ProblemContext& pctx) {
		const kernel_inttype m = gemm_m(pctx);
		const kernel_inttype n = gemm_n(pctx);
		const kernel_inttype k = gemm_k(pctx);
		perflibs_trans transa = c_to_trans(gemm_transa(pctx));
		perflibs_trans transb = c_to_trans(gemm_transb(pctx));

		// We don't have a valid optimized small SGEMM kernel for Arm64EC or SVE
		if constexpr (os::windows_arm64ec || spec::is_sve_v<ArchitectureSpec>) {
			return false;
		}

		if (!is_trans(transa)) {
			if (!is_trans(transb)) {
				// NN
				return (((k <= 4 || n <= 10) && m >= 60) || (n < 4 && m > 4) ||
				        (n <= 4 && k > 500 && m >= 5 && m <= 30));
			}
			else {
				// NT
				return ((n <= 4 && m > 4) || (n <= 4 && k > 60 && m > 60) ||
				        (m >= 50 && (n <= 10 || k <= 4)));
			}
		}
		else {
			if (!is_trans(transb)) {
				// TN
				return ((n <= 3) || (k <= 3 && m >= 100) || (n <= 4 && k >= 100 && m >= 5 && m <= 500) ||
				        (n > 4 && k >= 200 && m >= 5 && m <= 15));
			}
			else {
				// TT
				return (((n == 4) && (k >= 60)) || ((k == 4) && (m <= 4) && (n >= 60)) ||
				        ((m <= 5) && (n >= 60)) || ((m <= 10) && (k >= 100) && (n < 1000)));
			}
		}
	}
};

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_GEMM_SMALL_IMPL_HPP

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_DOT_KERNELS_HPP
#define PERFLIBS_LINALG_DOT_KERNELS_HPP

#include "framework/linalg_util.hpp" //PERFLIBS_LINALG_INLINE

#include "spec/problem_context_helpers.hpp"

#include "perflibs_complex.hpp"

#include "detect/cpu_info.hpp" //features.neon_bf16

template<typename AType, typename BType=AType, typename CType=AType>
using dot_kernel_t = CType (kernel_inttype n, const AType *x, const BType *y,
                            const kernel_inttype incx, const kernel_inttype incy);

template<typename T1, typename T2>
using dot_kernel_2t = dot_kernel_t<T1, T1, T2>;


extern "C" {
#pragma GCC diagnostic push
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wreturn-type-c-linkage"
#endif
dot_kernel_2t<perflibs::bf16, perflibs::bf16> bdot_kernel;
dot_kernel_2t<perflibs::bf16, perflibs::r32> sbdot_kernel;
dot_kernel_2t<float, float> sdot_kernel;
dot_kernel_2t<float, double> dsdot_kernel;
dot_kernel_2t<double, double> ddot_kernel;
dot_kernel_2t<complex_float, complex_float> cdot_kernel;
dot_kernel_2t<complex_float, complex_float> cdot_conj_kernel;
dot_kernel_2t<complex_double, complex_double> zdot_kernel;
dot_kernel_2t<complex_double, complex_double> zdot_conj_kernel;

dot_kernel_2t<float,float> sdot_sve_kernel;
dot_kernel_2t<float, float> sdot_sve_sg_kernel;
dot_kernel_2t<float, double> dsdot_sve_kernel;
dot_kernel_2t<float, double> dsdot_sve_sg_kernel;
dot_kernel_2t<double, double> ddot_sve_kernel;
dot_kernel_2t<double, double> ddot_sve_sg_kernel;
dot_kernel_2t<complex_float, complex_float> cdotu_sve_kernel;
dot_kernel_2t<complex_float, complex_float> cdotu_sve_kernel_fcmla;
dot_kernel_2t<complex_float, complex_float> cdotu_sve_sg_kernel;
dot_kernel_2t<complex_float, complex_float> cdotu_sve_sg_kernel_fcmla;
dot_kernel_2t<complex_double, complex_double> zdotu_sve_kernel;
dot_kernel_2t<complex_double, complex_double> zdotu_sve_kernel_fcmla;
dot_kernel_2t<complex_double, complex_double> zdotu_sve_sg_kernel;
dot_kernel_2t<complex_double, complex_double> zdotu_sve_sg_kernel_fcmla;
dot_kernel_2t<complex_float, complex_float> cdotc_sve_kernel;
dot_kernel_2t<complex_float, complex_float> cdotc_sve_kernel_fcmla;
dot_kernel_2t<complex_float, complex_float> cdotc_sve_sg_kernel;
dot_kernel_2t<complex_float, complex_float> cdotc_sve_sg_kernel_fcmla;
dot_kernel_2t<complex_double, complex_double> zdotc_sve_kernel;
dot_kernel_2t<complex_double, complex_double> zdotc_sve_kernel_fcmla;
dot_kernel_2t<complex_double, complex_double> zdotc_sve_sg_kernel;
dot_kernel_2t<complex_double, complex_double> zdotc_sve_sg_kernel_fcmla;
#pragma GCC diagnostic pop
} //extern C

namespace perflibs::linalg {
namespace {

template <typename T1, typename T2>
[[maybe_unused]]
T2 dotu_fallback(kernel_inttype n, const T1 *x, const T1 *y, kernel_inttype incx, kernel_inttype incy){
	T2 result = zero<T2>;
	kernel_inttype ix = 0;
	kernel_inttype iy = 0;
	for (kernel_inttype i = 0; i < n; ++i, ix += incx, iy += incy) {
		result+= T2(y[iy]) * T2(x[ix]);
	}
	return result;
}

template <typename T1, typename T2>
[[maybe_unused]]
T2 dotc_fallback(kernel_inttype n, const T1 *x, const T1 *y, kernel_inttype incx, kernel_inttype incy){
	T2 result = zero<T2>;
	kernel_inttype ix = 0;
	kernel_inttype iy = 0;
	for (kernel_inttype i = 0; i < n; ++i, ix += incx, iy += incy) {
		result+= T2(y[iy]) * conj(T2(x[ix]));
	}
	return result;
}

} // namespace <anon>


template<typename T1, typename T2>
dot_kernel_2t<T1, T2> *dot_kernel { nullptr };

template<> [[maybe_unused]]
inline auto dot_kernel<bf16, bf16> = bdot_kernel;
template<> [[maybe_unused]]
inline auto dot_kernel<bf16, r32> = sbdot_kernel;
template<> [[maybe_unused]]
inline auto dot_kernel<float, float> = sdot_kernel;
template<> [[maybe_unused]]
inline auto dot_kernel<double, double> = ddot_kernel;
template<> [[maybe_unused]]
inline auto dot_kernel<float, double> = dsdot_kernel;
template<> [[maybe_unused]]
inline auto dot_kernel<complex_float, complex_float> = cdot_kernel;
template<> [[maybe_unused]]
inline auto dot_kernel<complex_double, complex_double> = zdot_kernel;




template<typename T1, typename T2>
dot_kernel_2t<T1, T2> *dot_conj_kernel;
template<> [[maybe_unused]]
inline auto dot_conj_kernel<complex_float, complex_float> = cdot_conj_kernel;
template<> [[maybe_unused]]
inline auto dot_conj_kernel<complex_double, complex_double> = zdot_conj_kernel;

template<typename T1, typename T2>
dot_kernel_2t<T1, T2> *dot_sve_kernel { nullptr };
template<> [[maybe_unused]]
inline auto dot_sve_kernel<float, float> = sdot_sve_kernel;
template<> [[maybe_unused]]
inline auto dot_sve_kernel<double, double> = ddot_sve_kernel;
template<> [[maybe_unused]]
inline auto dot_sve_kernel<float, double> = dsdot_sve_kernel;
template<> [[maybe_unused]]
inline auto dot_sve_kernel<complex_float, complex_float> = cdotu_sve_kernel_fcmla;
template<> [[maybe_unused]]
inline auto dot_sve_kernel<complex_double, complex_double> = zdotu_sve_kernel_fcmla;

template<typename T1, typename T2>
dot_kernel_2t<T1, T2> *dot_sve_sg_kernel;
template<> [[maybe_unused]]
inline auto dot_sve_sg_kernel<float, float> = sdot_sve_sg_kernel;
template<> [[maybe_unused]]
inline auto dot_sve_sg_kernel<double, double> = ddot_sve_sg_kernel;
template<> [[maybe_unused]]
inline auto dot_sve_sg_kernel<float, double> = dsdot_sve_sg_kernel;
template<> [[maybe_unused]]
inline auto dot_sve_sg_kernel<complex_float, complex_float> = cdotu_sve_sg_kernel;
template<> [[maybe_unused]]
inline auto dot_sve_sg_kernel<complex_double, complex_double> = zdotu_sve_sg_kernel;

template<typename T1, typename T2>
dot_kernel_2t<T1, T2> *dot_sve_kernel_fcmla = dot_sve_kernel<T1, T2>;
template<> [[maybe_unused]]
inline auto dot_sve_kernel_fcmla<complex_float, complex_float> = cdotu_sve_kernel_fcmla;
template<> [[maybe_unused]]
inline auto dot_sve_kernel_fcmla<complex_double, complex_double> = zdotu_sve_kernel_fcmla;

template<typename T1, typename T2>
dot_kernel_2t<T1, T2> *dot_sve_sg_kernel_fcmla = dot_sve_sg_kernel<T1, T2>;
template<> [[maybe_unused]]
inline auto dot_sve_sg_kernel_fcmla<complex_float, complex_float> = cdotu_sve_sg_kernel_fcmla;
template<> [[maybe_unused]]
inline auto dot_sve_sg_kernel_fcmla<complex_double, complex_double> = zdotu_sve_sg_kernel_fcmla;


template<typename T1, typename T2>
dot_kernel_2t<T1, T2> *dot_conj_sve_kernel;
template<> [[maybe_unused]]
inline auto dot_conj_sve_kernel<complex_float, complex_float> = cdotc_sve_kernel;
template<> [[maybe_unused]]
inline auto dot_conj_sve_kernel<complex_double, complex_double> = zdotc_sve_kernel;

template<typename T1, typename T2>
dot_kernel_2t<T1, T2> *dot_conj_sve_sg_kernel;
template<> [[maybe_unused]]
inline auto dot_conj_sve_sg_kernel<complex_float, complex_float> = cdotc_sve_sg_kernel;
template<> [[maybe_unused]]
inline auto dot_conj_sve_sg_kernel<complex_double, complex_double> = zdotc_sve_sg_kernel;

template<typename T1, typename T2>
dot_kernel_2t<T1, T2> *dot_conj_sve_kernel_fcmla;
template<> [[maybe_unused]]
inline auto dot_conj_sve_kernel_fcmla<complex_float, complex_float> = cdotc_sve_kernel_fcmla;
template<> [[maybe_unused]]
inline auto dot_conj_sve_kernel_fcmla<complex_double, complex_double> = zdotc_sve_kernel_fcmla;

template<typename T1, typename T2>
dot_kernel_2t<T1, T2> *dot_conj_sve_sg_kernel_fcmla;
template<> [[maybe_unused]]
inline auto dot_conj_sve_sg_kernel_fcmla<complex_float, complex_float> = cdotc_sve_sg_kernel_fcmla;
template<> [[maybe_unused]]
inline auto dot_conj_sve_sg_kernel_fcmla<complex_double, complex_double> = zdotc_sve_sg_kernel_fcmla;

namespace spec {

template<typename ProblemContext>
PERFLIBS_LINALG_INLINE
auto get_neon_dot_kernel(const ProblemContext& pctx) {
	using T1 = std::remove_cv_t<typename ProblemContext::a_matrix_type::value_type>;
	using T2 =                  typename ProblemContext::c_matrix_type::value_type ;

	const auto is_conj = is_complex_v<T1> && dot_is_conj(pctx);

	if (is_conj) {
		return dot_conj_kernel<T1, T2>;
	}

	if constexpr (std::is_same_v<T1, bf16>) {
		if(! perflibs::machine::cpu_info::get_cpu_features().neon_bf16) {
			return dotu_fallback<T1, T2>;
		}
	}
	return dot_kernel<T1, T2>;
}

template<typename ProblemContextBase, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
auto get_neon_dot_kernel(const spec::problem_context<ProblemContextBase, ArchitectureSpec>& pctx) {
	using T1 = std::remove_cv_t<typename ProblemContextBase::a_matrix_type::value_type>;
	using T2 =                  spec::compute_precision_t<spec::problem_context<ProblemContextBase, ArchitectureSpec>>;

	const auto is_conj = is_complex_v<T1> && dot_is_conj(pctx);

	if (is_conj) {
		return dot_conj_kernel<T1, T2>;
	}

	if constexpr (std::is_same_v<T1, bf16>) {
		const auto features = perflibs::machine::cpu_info::get_cpu_features();

		if(! features.neon_bf16) {
			return dotu_fallback<T1, T2>;
		}
	}
	return dot_kernel<T1, T2>;
}

template<typename ProblemContext>
PERFLIBS_LINALG_INLINE
auto get_sve_dot_kernel(const ProblemContext& pctx) {
	using T1 = std::remove_cv_t<typename ProblemContext::a_matrix_type::value_type>;
	using T2 =                  typename ProblemContext::c_matrix_type::value_type ;

	const auto incx = dot_incx(pctx);
	const auto incy = dot_incy(pctx);
	const auto is_conj = is_complex_v<T1> && dot_is_conj(pctx);

	if (incx == 1 && incy == 1) {
		if (is_conj) {
			if(auto k = dot_conj_sve_kernel_fcmla<T1, T2>; k != nullptr)
				return k;
		}
		else {
			if(auto k = dot_sve_kernel_fcmla<T1, T2>; k != nullptr)
				return k;
		}
	}

	return get_neon_dot_kernel(pctx);
}

} // namespace spec
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_DOT_KERNELS_HPP

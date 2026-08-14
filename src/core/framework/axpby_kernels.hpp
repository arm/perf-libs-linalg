/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_AXPBY_KERNELS_HPP
#define PERFLIBS_LINALG_AXPBY_KERNELS_HPP

#include "perflibs_complex.hpp"
#include "perflibs_assert.hpp"

#include "framework/linalg_util.hpp"
#include "framework/copy_kernels.hpp"
#include "framework/scal_kernels.hpp"
#include "framework/vecadd_kernels.hpp"

namespace perflibs::linalg {

template<typename StrategyTag>
struct axpby_kernel_tag {
	axpby_kernel_tag(const StrategyTag&) { }

	template<typename ProblemContext>
	auto incx(const ProblemContext& pctx) const {
		return pctx.a.strd_step();
	}

	template<typename ProblemContext>
	auto incy(const ProblemContext& pctx) const {
		return pctx.c.cntg_step();
	}

	template<typename ProblemContext>
	auto alpha(const ProblemContext& pctx) const {
		return pctx.alpha;
	}

	template<typename ProblemContext>
	auto beta(const ProblemContext& pctx) const {
		return pctx.beta;
	}

	template<typename ProblemContext>
	auto is_conj(const ProblemContext& pctx) const {
		return pctx.a.is_conj();
	}

	template<typename XType, typename YType, typename ScalarType, typename ProblemContext>
	constexpr static bool match() {
		return std::is_same_v<XType,      std::remove_const_t<typename ProblemContext::a_matrix_type::value_type>>
		    && std::is_same_v<YType,      typename ProblemContext::c_matrix_type::value_type>
		    && std::is_same_v<ScalarType, typename ProblemContext::scalar_type>;
	}
};

template<typename AType, typename BType=AType, typename CType=AType,
         typename ScalarType = promote_t<AType, BType, CType>>
using axpy_kernel_t = void (kernel_inttype n, ScalarType alpha, const AType *x, CType *y,
                            const kernel_inttype incx, const kernel_inttype incy);

template<typename AType, typename BType=AType, typename CType=AType,
         typename ScalarType = promote_t<AType, BType, CType>>
using axpby_kernel_t = void(kernel_inttype n, ScalarType alpha, const AType *x, ScalarType beta, CType *y,
                            const kernel_inttype incx, const kernel_inttype incy);

} //namespace perflibs::linalg

extern "C" {

using bf16=perflibs::bf16;
using r32 =perflibs::r32;

//Neon - axpy kernels
perflibs::linalg::axpy_kernel_t<float>                 saxpy_kernel;
perflibs::linalg::axpy_kernel_t<double>                daxpy_kernel;
perflibs::linalg::axpy_kernel_t<std::complex<float>>   caxpy_kernel;
perflibs::linalg::axpy_kernel_t<std::complex<double>>  zaxpy_kernel;
perflibs::linalg::axpy_kernel_t<std::complex<float>>   caxpy_conj_kernel;
perflibs::linalg::axpy_kernel_t<std::complex<double>>  zaxpy_conj_kernel;
perflibs::linalg::axpy_kernel_t<float>                 saxpy_no_prefetch_kernel;
perflibs::linalg::axpy_kernel_t<double>                daxpy_no_prefetch_kernel;
perflibs::linalg::axpy_kernel_t<std::complex<float>>   caxpy_no_prefetch_kernel;
perflibs::linalg::axpy_kernel_t<std::complex<double>>  zaxpy_no_prefetch_kernel;

//Neon - axpby kernels
perflibs::linalg::axpby_kernel_t<bf16, bf16, bf16>     baxpby_kernel;
perflibs::linalg::axpby_kernel_t<bf16, bf16,  r32>     sbaxpby_kernel;
perflibs::linalg::axpby_kernel_t<float>                saxpby_kernel;
perflibs::linalg::axpby_kernel_t<double>               daxpby_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<float>>  caxpby_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<double>> zaxpby_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<float>>  caxpby_conj_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<double>> zaxpby_conj_kernel;
perflibs::linalg::axpby_kernel_t<float>                sscal_out_of_place_kernel;
perflibs::linalg::axpby_kernel_t<double>               dscal_out_of_place_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<float>>  cscal_out_of_place_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<double>> zscal_out_of_place_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<float>>  cscal_out_of_place_conj_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<double>> zscal_out_of_place_conj_kernel;

//SVE - axpy kernels
perflibs::linalg::axpy_kernel_t<float>                 saxpy_sve_kernel;
perflibs::linalg::axpy_kernel_t<double>                daxpy_sve_kernel;
perflibs::linalg::axpy_kernel_t<std::complex<float>>   caxpy_sve_kernel;
perflibs::linalg::axpy_kernel_t<std::complex<float>>   caxpy_sve_conj_kernel;
perflibs::linalg::axpy_kernel_t<std::complex<float>>   caxpy_sve_kernel_fcmla;
perflibs::linalg::axpy_kernel_t<std::complex<float>>   caxpy_sve_conj_kernel_fcmla;
perflibs::linalg::axpy_kernel_t<std::complex<double>>  zaxpy_sve_kernel;
perflibs::linalg::axpy_kernel_t<std::complex<double>>  zaxpy_sve_conj_kernel;
perflibs::linalg::axpy_kernel_t<std::complex<double>>  zaxpy_sve_kernel_fcmla;
perflibs::linalg::axpy_kernel_t<std::complex<double>>  zaxpy_sve_conj_kernel_fcmla;

//SVE - axpby kernels
perflibs::linalg::axpby_kernel_t<float>                saxpby_sve_kernel;
perflibs::linalg::axpby_kernel_t<double>               daxpby_sve_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<float>>  caxpby_sve_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<float>>  caxpby_sve_conj_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<float>>  caxpby_sve_kernel_fcmla;
perflibs::linalg::axpby_kernel_t<std::complex<float>>  caxpby_sve_conj_kernel_fcmla;
perflibs::linalg::axpby_kernel_t<std::complex<double>> zaxpby_sve_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<double>> zaxpby_sve_conj_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<double>> zaxpby_sve_kernel_fcmla;
perflibs::linalg::axpby_kernel_t<std::complex<double>> zaxpby_sve_conj_kernel_fcmla;
perflibs::linalg::axpby_kernel_t<float>                sscal_out_of_place_sve_kernel;
perflibs::linalg::axpby_kernel_t<double>               dscal_out_of_place_sve_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<float>>  cscal_out_of_place_sve_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<float>>  cscal_out_of_place_sve_conj_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<float>>  cscal_out_of_place_sve_kernel_fcmla;
perflibs::linalg::axpby_kernel_t<std::complex<float>>  cscal_out_of_place_sve_conj_kernel_fcmla;
perflibs::linalg::axpby_kernel_t<std::complex<double>> zscal_out_of_place_sve_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<double>> zscal_out_of_place_sve_conj_kernel;
perflibs::linalg::axpby_kernel_t<std::complex<double>> zscal_out_of_place_sve_kernel_fcmla;
perflibs::linalg::axpby_kernel_t<std::complex<double>> zscal_out_of_place_sve_conj_kernel_fcmla;

}

namespace perflibs::linalg {
namespace {

/*
 * @tparam T the data type of the computation
 * @tparam VecAddKernelFunc the vecadd kernel we are shimming
 */
template<typename T, vecadd_kernel_t<T> *VecAddKernelFunc>
PERFLIBS_LINALG_INLINE
void vecadd_axpby_shim(kernel_inttype n, T alpha, const T *x, T beta, T *y, kernel_inttype incx, kernel_inttype incy) {
	PERFLIBS_ASSERT(alpha == one<T>, "alpha must be 1.0 to shim to use vecadd");
	PERFLIBS_ASSERT(beta == one<T>, "beta must be 1.0 to shim to use vecadd");

	return VecAddKernelFunc(n, /*alpha removed*/ x, incx, /*beta removed*/ y, incy);
}
/*
 * @tparam T the data type of the computation
 * @tparam CopyKernelFunc the copy kernel we are shimming
 */
template<typename T, copy_kernel_t<T> *CopyKernelFunc>
PERFLIBS_LINALG_INLINE
void copy_axpby_shim(kernel_inttype n, T alpha, const T *x, T beta, T *y, kernel_inttype incx, kernel_inttype incy) {
	PERFLIBS_ASSERT(alpha == one<T>,  "alpha must be 1.0 to shim to use copy");
	PERFLIBS_ASSERT(beta  == zero<T>, "beta must be 0.0 to shim to use copy");

	return CopyKernelFunc(n, /*alpha removed*/ x, incx, /*beta removed*/ y, incy);
}

/*
 * @tparam T the data type of the computation
 * @tparam ScalarType data type of the kernel's scalar arguments
 * @tparam ScalKernelFunc the scal kernel we are shimming
 */
template<typename T, typename ScalarType, scal_kernel_t<T, T, T, ScalarType> *ScalKernelFunc>
PERFLIBS_LINALG_INLINE
void scal_axpby_shim(kernel_inttype n, T alpha, const T *x, T beta, T *y, kernel_inttype incx, kernel_inttype incy) {
	PERFLIBS_ASSERT(alpha == zero<T>, "alpha must be 0.0 to shim to use scal");

	ScalarType beta_shim;
	if constexpr (is_complex_v<T> && !is_complex_v<ScalarType>) {
		// We need to pass in complex beta as real if the kernel expects real scalars
		PERFLIBS_ASSERT(beta.imag() == zero<remove_complex_t<T>>);
		beta_shim = perflibs::real(beta);
	}
	else {
		beta_shim = beta;
	}

	return ScalKernelFunc(n, /*alpha is removed*/ beta_shim, /*x is removed*/ y, incy /*incx is removed*/);
}

/*
 * @tparam T the data type of the computation
 * @tparam AxpyKernelFunc the axpy kernel we are shimming
 */
template<typename T, axpy_kernel_t<T> *AxpyKernelFunc>
PERFLIBS_LINALG_INLINE
void axpy_axpby_shim(kernel_inttype n, T alpha, const T *x, T beta, T *y, kernel_inttype incx, kernel_inttype incy) {
	PERFLIBS_ASSERT(beta == one<T>, "beta must be 1.0 to shim to use axpy");

	return AxpyKernelFunc(n, alpha, x, y, /*beta is removed*/ incx, incy);
}

constexpr static inline auto saxpy_kernel_shim                = axpy_axpby_shim<float,                saxpy_kernel>;
constexpr static inline auto daxpy_kernel_shim                = axpy_axpby_shim<double,               daxpy_kernel>;
constexpr static inline auto caxpy_kernel_shim                = axpy_axpby_shim<std::complex<float>,  caxpy_kernel>;
constexpr static inline auto zaxpy_kernel_shim                = axpy_axpby_shim<std::complex<double>, zaxpy_kernel>;
constexpr static inline auto caxpy_conj_kernel_shim           = axpy_axpby_shim<std::complex<float>,  caxpy_conj_kernel>;
constexpr static inline auto zaxpy_conj_kernel_shim           = axpy_axpby_shim<std::complex<double>, zaxpy_conj_kernel>;
constexpr static inline auto saxpy_kernel_no_prefetch_shim    = axpy_axpby_shim<float,                saxpy_no_prefetch_kernel>;
constexpr static inline auto daxpy_kernel_no_prefetch_shim    = axpy_axpby_shim<double,               daxpy_no_prefetch_kernel>;
constexpr static inline auto caxpy_kernel_no_prefetch_shim    = axpy_axpby_shim<std::complex<float>,  caxpy_no_prefetch_kernel>;
constexpr static inline auto zaxpy_kernel_no_prefetch_shim    = axpy_axpby_shim<std::complex<double>, zaxpy_no_prefetch_kernel>;

constexpr static inline auto saxpy_sve_kernel_shim            = axpy_axpby_shim<float,                saxpy_sve_kernel>;
constexpr static inline auto daxpy_sve_kernel_shim            = axpy_axpby_shim<double,               daxpy_sve_kernel>;
constexpr static inline auto caxpy_sve_kernel_shim            = axpy_axpby_shim<std::complex<float>,  caxpy_sve_kernel>;
constexpr static inline auto caxpy_sve_conj_kernel_shim       = axpy_axpby_shim<std::complex<float>,  caxpy_sve_conj_kernel>;
constexpr static inline auto caxpy_sve_kernel_fcmla_shim      = axpy_axpby_shim<std::complex<float>,  caxpy_sve_kernel_fcmla>;
constexpr static inline auto caxpy_sve_conj_kernel_fcmla_shim = axpy_axpby_shim<std::complex<float>,  caxpy_sve_conj_kernel_fcmla>;
constexpr static inline auto zaxpy_sve_kernel_shim            = axpy_axpby_shim<std::complex<double>, zaxpy_sve_kernel>;
constexpr static inline auto zaxpy_sve_conj_kernel_shim       = axpy_axpby_shim<std::complex<double>, zaxpy_sve_conj_kernel>;
constexpr static inline auto zaxpy_sve_kernel_fcmla_shim      = axpy_axpby_shim<std::complex<double>, zaxpy_sve_kernel_fcmla>;
constexpr static inline auto zaxpy_sve_conj_kernel_fcmla_shim = axpy_axpby_shim<std::complex<double>, zaxpy_sve_conj_kernel_fcmla>;

constexpr static inline auto sscal_kernel_shim                = scal_axpby_shim<float, float,         sscal_kernel>;
constexpr static inline auto dscal_kernel_shim                = scal_axpby_shim<double, double,       dscal_kernel>;
constexpr static inline auto cscal_kernel_shim                = scal_axpby_shim<std::complex<float>,  std::complex<float>,  cscal_kernel>;
constexpr static inline auto sscal_real_cplx_kernel_shim      = scal_axpby_shim<std::complex<float>,  float,                sscal_real_cplx_kernel>;
constexpr static inline auto zscal_kernel_shim                = scal_axpby_shim<std::complex<double>, std::complex<double>, zscal_kernel>;
constexpr static inline auto dscal_real_cplx_kernel_shim      = scal_axpby_shim<std::complex<double>, double,               dscal_real_cplx_kernel>;

constexpr static inline auto sscal_sve_kernel_shim            = scal_axpby_shim<float, float,         sscal_sve_kernel>;
constexpr static inline auto dscal_sve_kernel_shim            = scal_axpby_shim<double, double,       dscal_sve_kernel>;
constexpr static inline auto cscal_sve_kernel_shim            = scal_axpby_shim<std::complex<float>,  std::complex<float>,  cscal_sve_kernel>;
constexpr static inline auto cscal_sve_kernel_fcmla_shim      = scal_axpby_shim<std::complex<float>,  std::complex<float>,  cscal_sve_kernel_fcmla>;
constexpr static inline auto sscal_real_cplx_sve_kernel_shim  = scal_axpby_shim<std::complex<float>,  float,                sscal_real_cplx_sve_kernel>;
constexpr static inline auto zscal_sve_kernel_shim            = scal_axpby_shim<std::complex<double>, std::complex<double>, zscal_sve_kernel>;
constexpr static inline auto zscal_sve_kernel_fcmla_shim      = scal_axpby_shim<std::complex<double>, std::complex<double>, zscal_sve_kernel_fcmla>;
constexpr static inline auto dscal_real_cplx_sve_kernel_shim  = scal_axpby_shim<std::complex<double>, double,               dscal_real_cplx_sve_kernel>;

constexpr static inline auto scopy_kernel_shim                = copy_axpby_shim<float,                scopy_kernel>;
constexpr static inline auto dcopy_kernel_shim                = copy_axpby_shim<double,               dcopy_kernel>;
constexpr static inline auto ccopy_kernel_shim                = copy_axpby_shim<std::complex<float>,  ccopy_kernel_generic>;
constexpr static inline auto zcopy_kernel_shim                = copy_axpby_shim<std::complex<double>, zcopy_kernel_generic>;
constexpr static inline auto zcopy_kernel_with_inc_shim       = copy_axpby_shim<std::complex<double>, zcopy_kernel_with_inc>;

constexpr static inline auto scopy_sve_kernel_shim            = copy_axpby_shim<float,                scopy_sve_kernel>;
constexpr static inline auto dcopy_sve_kernel_shim            = copy_axpby_shim<double,               dcopy_sve_kernel>;
constexpr static inline auto ccopy_sve_kernel_shim            = copy_axpby_shim<std::complex<float>,  ccopy_sve_kernel_generic>;
constexpr static inline auto zcopy_sve_kernel_shim            = copy_axpby_shim<std::complex<double>, zcopy_sve_kernel_generic>;
constexpr static inline auto zcopy_sve_kernel_with_inc_shim   = copy_axpby_shim<std::complex<double>, zcopy_sve_kernel_with_inc>;

constexpr static inline auto svecadd_kernel_shim              = vecadd_axpby_shim<float,                svecadd_kernel>;
constexpr static inline auto dvecadd_kernel_shim              = vecadd_axpby_shim<double,               dvecadd_kernel>;
constexpr static inline auto cvecadd_kernel_shim              = vecadd_axpby_shim<std::complex<float>,  cvecadd_kernel>;
constexpr static inline auto zvecadd_kernel_shim              = vecadd_axpby_shim<std::complex<double>, zvecadd_kernel>;
constexpr static inline auto cvecadd_conj_kernel_shim         = vecadd_axpby_shim<std::complex<float>,  cvecadd_conj_kernel>;
constexpr static inline auto zvecadd_conj_kernel_shim         = vecadd_axpby_shim<std::complex<double>, zvecadd_conj_kernel>;

constexpr static inline auto svecadd_sve_kernel_shim          = vecadd_axpby_shim<float,                svecadd_sve_kernel>;
constexpr static inline auto dvecadd_sve_kernel_shim          = vecadd_axpby_shim<double,               dvecadd_sve_kernel>;
constexpr static inline auto cvecadd_sve_kernel_shim          = vecadd_axpby_shim<std::complex<float>,  cvecadd_sve_kernel>;
constexpr static inline auto zvecadd_sve_kernel_shim          = vecadd_axpby_shim<std::complex<double>, zvecadd_sve_kernel>;
constexpr static inline auto cvecadd_conj_sve_kernel_shim     = vecadd_axpby_shim<std::complex<float>,  cvecadd_conj_sve_kernel>;
constexpr static inline auto zvecadd_conj_sve_kernel_shim     = vecadd_axpby_shim<std::complex<double>, zvecadd_conj_sve_kernel>;
constexpr static inline auto cvecadd_sve_kernel_fcmla_shim    = vecadd_axpby_shim<std::complex<float>,  cvecadd_sve_kernel_fcmla>;
constexpr static inline auto zvecadd_sve_kernel_fcmla_shim    = vecadd_axpby_shim<std::complex<double>, zvecadd_sve_kernel_fcmla>;

} //namespace <anon>
} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_AXPBY_KERNELS_HPP

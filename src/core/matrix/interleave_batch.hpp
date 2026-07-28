/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_INTERLEAVE_BATCH
#define PERFLIBS_LINALG_INTERLEAVE_BATCH

#include "framework/linalg_util.hpp"

#include "detect/simd.hpp"
#include "perflibs_assert.hpp"
#include "perflibs_util.hpp"
#include "dyn_array.hpp"

#include <complex>
#include <type_traits>

#include <arm_neon.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif


namespace perflibs::linalg {

constexpr kernel_inttype max_interleave_size = 64;

template<typename T0, typename T1>
constexpr bool is_same_rem_cv_v = std::is_same_v<std::remove_cv_t<T0>, std::remove_cv_t<T1>>;

/// Test to see where T has value_type, if true, descends from true_type, else false_type
template<typename T, typename = void> struct has_value_type                                         : std::false_type {};
template<typename T>                  struct has_value_type<T, std::void_t<typename T::value_type>> : std::true_type {};

///convenience wrapper around has_value_type, extracts the bool value from true_type or false type
template<typename T> constexpr bool has_value_type_v = has_value_type<T>::value;


/// Test to see where T has get_as, if true, descends from true_type, else false_type
template<typename T, typename = void> struct has_get_as                                                                                       : std::false_type {};
template<typename T>                  struct has_get_as<T, std::void_t<decltype(std::declval<T>().template get_as<typename T::value_type>(std::declval<kernel_inttype>()))>> : std::true_type {};

///convenience wrapper around has_get_as, extracts the bool value from true_type or false type
template<typename T> constexpr bool has_get_as_v = has_get_as<T>::value;

template<typename T>
constexpr bool is_interleave_batch_expr_v = has_get_as_v<T> && has_value_type_v<T>;

template<typename T>
constexpr void assert_interleave_batch_expr(const T&) {
	static_assert(has_value_type_v<T>, "Type expected to be InterleaveBatchExpr but does not have value_type");
	static_assert(has_get_as_v<T>, "Type expected to be InterleaveBatchExpr but does not have get_as<T>()");
};

/*
 * This namespace contains generic helpers to handle SIMD vectors
 */
namespace simd {
	template<typename PtrType, typename SimdVec>
	PERFLIBS_LINALG_INLINE
	void store(PtrType *ptr, SimdVec v) {
		*reinterpret_cast<SimdVec*>(ptr) = v;
	}

	template<typename SimdVec, typename PtrType>
	PERFLIBS_LINALG_INLINE
	SimdVec load(PtrType *ptr) {
		return *reinterpret_cast<const SimdVec*>(ptr);
	}

	PERFLIBS_LINALG_INLINE float64x2x2_t load_batch2(const double *ptr) { return vld1q_f64_x2(ptr); }
	PERFLIBS_LINALG_INLINE float32x4x2_t load_batch2(const float *ptr) { return vld1q_f32_x2(ptr); }

	PERFLIBS_LINALG_INLINE float64x2x2_t load2(double *ptr) { return vld2q_f64(ptr); }
	PERFLIBS_LINALG_INLINE float32x4x2_t load2(float *ptr) { return vld2q_f32(ptr); }

	PERFLIBS_LINALG_INLINE void store2(double *ptr, float64x2x2_t val) { return vst2q_f64(ptr, val); }
	PERFLIBS_LINALG_INLINE void store2(float *ptr, float32x4x2_t val) { return vst2q_f32(ptr, val); }

	//TODO extend these are required
	PERFLIBS_LINALG_INLINE double add(double v0, double v1) { return v0 + v1; }
	PERFLIBS_LINALG_INLINE double sub(double v0, double v1) { return v0 - v1; }
	PERFLIBS_LINALG_INLINE double div(double v0, double v1) { return v0 / v1; }
	PERFLIBS_LINALG_INLINE double mul(double v0, double v1) { return v0 * v1; }
	PERFLIBS_LINALG_INLINE double mla(double& vd, double v0, double v1) { vd += v0 * v1; return vd; }
	PERFLIBS_LINALG_INLINE double mls(double& vd, double v0, double v1) { vd -= v0 * v1; return vd; }

	PERFLIBS_LINALG_INLINE float64x2_t add(float64x2_t v0, float64x2_t v1) { return vaddq_f64(v0, v1); }
	PERFLIBS_LINALG_INLINE float64x2_t sub(float64x2_t v0, float64x2_t v1) { return vsubq_f64(v0, v1); }
	PERFLIBS_LINALG_INLINE float64x2_t div(float64x2_t v0, float64x2_t v1) { return vdivq_f64(v0, v1); }

	PERFLIBS_LINALG_INLINE float64x2_t mul(float64x2_t v0, float64x2_t v1) { return vmulq_f64(v0, v1); }
	PERFLIBS_LINALG_INLINE float32x4_t mul(float32x4_t v0, float32x4_t v1) { return vmulq_f32(v0, v1); }

	PERFLIBS_LINALG_INLINE float64x2_t mla(float64x2_t& vd, float64x2_t v0, float64x2_t v1) { vd = vfmaq_f64(vd, v0, v1); return vd; }
	PERFLIBS_LINALG_INLINE float32x4_t mla(float32x4_t& vd, float32x4_t v0, float32x4_t v1) { vd = vfmaq_f32(vd, v0, v1); return vd; }

	PERFLIBS_LINALG_INLINE float64x2_t mls(float64x2_t& vd, float64x2_t v0, float64x2_t v1) { vd = vfmsq_f64(vd, v0, v1); return vd; }
	PERFLIBS_LINALG_INLINE float32x4_t mls(float32x4_t& vd, float32x4_t v0, float32x4_t v1) { vd = vfmsq_f32(vd, v0, v1); return vd; }

	template<typename return_type = float64x2_t> return_type dup(float64_t v0){ return vdupq_n_f64(v0); };
	template<typename return_type = float32x4_t> return_type dup(float32_t v0){ return vdupq_n_f32(v0); };
#ifdef __ARM_FEATURE_SVE
	template<> PERFLIBS_LINALG_INLINE svfloat64_t dup(float64_t v0) { return svdup_f64(v0); }
	template<> PERFLIBS_LINALG_INLINE svfloat32_t dup(float32_t v0) { return svdup_f32(v0); }
#endif

	PERFLIBS_LINALG_INLINE float64x2_t abs(float64x2_t v0) { return vabsq_f64(v0); }
	PERFLIBS_LINALG_INLINE float32x4_t abs(float32x4_t v0) { return vabsq_f32(v0); }

	PERFLIBS_LINALG_INLINE uint64x2_t cmgt(float64x2_t v0, float64x2_t v1) { return vcgtq_f64(v0, v1); }
	PERFLIBS_LINALG_INLINE uint32x4_t cmgt(float32x4_t v0, float32x4_t v1) { return vcgtq_f32(v0, v1); }

	PERFLIBS_LINALG_INLINE uint64x2_t cmlt(float64x2_t v0, float64x2_t v1) { return vcltq_f64(v0, v1); }
	PERFLIBS_LINALG_INLINE uint32x4_t cmlt(float32x4_t v0, float32x4_t v1) { return vcltq_f32(v0, v1); }

	PERFLIBS_LINALG_INLINE uint64x2_t cmeqz(float64x2_t v0) { return vceqzq_f64(v0); }
	PERFLIBS_LINALG_INLINE uint32x4_t cmeqz(float32x4_t v0) { return vceqzq_f32(v0); }

	PERFLIBS_LINALG_INLINE uint64x2_t orr(uint64x2_t v0, uint64x2_t v1) { return vorrq_u64(v0, v1); }
	PERFLIBS_LINALG_INLINE uint32x4_t orr(uint32x4_t v0, uint32x4_t v1) { return vorrq_u32(v0, v1); }

	PERFLIBS_LINALG_INLINE float64x2_t apply_mask(uint64x2_t m0, float64x2_t v1) { return vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(v1), m0)); }
	PERFLIBS_LINALG_INLINE float32x4_t apply_mask(uint32x4_t m0, float32x4_t v1) { return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(v1), m0)); }

	PERFLIBS_LINALG_INLINE float64x2_t bit_clear(float64x2_t v0, uint64x2_t m1) { return vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(v0), m1)); }
	PERFLIBS_LINALG_INLINE float32x4_t bit_clear(float32x4_t v0, uint32x4_t m1) { return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(v0), m1)); }
	PERFLIBS_LINALG_INLINE uint64x2_t bit_clear(uint64x2_t v0, uint64x2_t m1) { return vbicq_u64(v0, m1); }
	PERFLIBS_LINALG_INLINE uint32x4_t bit_clear(uint32x4_t v0, uint32x4_t m1) { return vbicq_u32(v0, m1); }

	PERFLIBS_LINALG_INLINE bool is_zero(uint64x2_t v0) { return vmaxvq_f64(vreinterpretq_f64_u64(v0)) == 0; }
	PERFLIBS_LINALG_INLINE bool is_zero(uint32x4_t v0) { return vmaxvq_u32(v0) == 0; }

	PERFLIBS_LINALG_INLINE float64_t sum(float64x2_t v0) { return vpaddd_f64(v0); }
	PERFLIBS_LINALG_INLINE float32_t sum(float32x4_t v0) {
		return vaddvq_f32(v0);
	}

#ifdef __ARM_FEATURE_SVE
	PERFLIBS_LINALG_INLINE svfloat64_t add(svfloat64_t v0, svfloat64_t v1) { return svadd_x(svptrue_b64(), v0, v1); }
	PERFLIBS_LINALG_INLINE svfloat64_t sub(svfloat64_t v0, svfloat64_t v1) { return svsub_x(svptrue_b64(), v0, v1); }
	PERFLIBS_LINALG_INLINE svfloat64_t div(svfloat64_t v0, svfloat64_t v1) { return svdiv_x(svptrue_b64(), v0, v1); }
	PERFLIBS_LINALG_INLINE svfloat64_t mul(svfloat64_t v0, svfloat64_t v1) { return svmul_x(svptrue_b64(), v0, v1); }
	PERFLIBS_LINALG_INLINE svfloat64_t mla(svfloat64_t& vd, svfloat64_t v0, svfloat64_t v1) { vd = svmla_x(svptrue_b64(), vd, v0, v1); return vd; }
	PERFLIBS_LINALG_INLINE svfloat64_t mls(svfloat64_t& vd, svfloat64_t v0, svfloat64_t v1) { vd = svmls_x(svptrue_b64(), vd, v0, v1); return vd; }
#endif

	PERFLIBS_LINALG_INLINE float64x2x2_t add(float64x2x2_t v0, float64x2x2_t v1) {
		float64x2x2_t ret;
		ret.val[0] = vaddq_f64(v0.val[0], v1.val[0]);
		ret.val[1] = vaddq_f64(v0.val[1], v1.val[1]);
		return ret;
	}
	PERFLIBS_LINALG_INLINE float64x2x2_t sub(float64x2x2_t v0, float64x2x2_t v1) {
		float64x2x2_t ret;
		ret.val[0] = vsubq_f64(v0.val[0], v1.val[0]);
		ret.val[1] = vsubq_f64(v0.val[1], v1.val[1]);
		return ret;
	}
	PERFLIBS_LINALG_INLINE float64x2x2_t div(float64x2x2_t v0, float64x2x2_t v1) {
		float64x2x2_t ret;
		ret.val[0] = vdivq_f64(v0.val[0], v1.val[0]);
		ret.val[1] = vdivq_f64(v0.val[1], v1.val[1]);
		return ret;
	}
	PERFLIBS_LINALG_INLINE float64x2x2_t mul(float64x2x2_t v0, float64x2x2_t v1) {
		float64x2x2_t ret;
		ret.val[0] = vmulq_f64(v0.val[0], v1.val[0]);
		ret.val[1] = vmulq_f64(v0.val[1], v1.val[1]);
		return ret;
	}
} //namespace simd

/**
 * This expr bind single scalar value and minics a batch
 * where every element has the same value
 */
template<typename T>
class scalar {
	T value_;
public:
	using value_type = T;

	///ctor
	PERFLIBS_LINALG_INLINE
	scalar(T value = zero<T>)
	:	value_ { value }
	{	}

	template<typename SimdVec>
	PERFLIBS_LINALG_INLINE
	SimdVec get_as(kernel_inttype) const {
		//TODO static_assert that SimdVec is vec of T so we aren't doing some casting on the fly
		if      constexpr (is_same_rem_cv_v<SimdVec, T>)             { return value_; }
#ifdef __ARM_FEATURE_SVE
		else if constexpr (is_same_rem_cv_v<SimdVec, svfloat64_t>)   { return svdup_f64(value_); }
#endif
		else if constexpr (is_same_rem_cv_v<SimdVec, float64x2_t>)   { return vdupq_n_f64(value_); }
		else if constexpr (is_same_rem_cv_v<SimdVec, float64x2x2_t>) {
			float64x2x2_t ret;
			ret.val[0] = vdupq_n_f64(value_);
			ret.val[1] = vdupq_n_f64(value_);
			return ret;
		}
	}

	PERFLIBS_LINALG_INLINE value_type operator()(kernel_inttype cntg) const { return value_; }
	PERFLIBS_LINALG_INLINE value_type operator[](kernel_inttype cntg) const { return value_; }

	PERFLIBS_LINALG_INLINE value_type value() const { return value_; }
}; //struct scalar

//inline compile time test
static_assert(is_interleave_batch_expr_v<scalar<float>>, "scalar failed to meet requirements of InterleaveBatchExpr");
static_assert(is_interleave_batch_expr_v<scalar<double>>, "scalar failed to meet requirements of InterleaveBatchExpr");
static_assert(is_interleave_batch_expr_v<scalar<std::complex<float>>>, "scalar failed to meet requirements of InterleaveBatchExpr");
static_assert(is_interleave_batch_expr_v<scalar<std::complex<double>>>, "scalar failed to meet requirements of InterleaveBatchExpr");

static_assert(std::is_default_constructible_v<scalar<float>>, "scalar is not default constructible");
static_assert(std::is_copy_constructible_v<scalar<float>>,    "scalar is not copy constructible");
static_assert(std::is_move_constructible_v<scalar<float>>,    "scalar is not move constructible");
static_assert(std::is_copy_assignable_v<scalar<float>>,       "scalar is not copy constructible");
static_assert(std::is_move_assignable_v<scalar<float>>,       "scalar is not move constructible");



/**
 * Note that for batched operations this IS the value_type
 */
template<typename T>
class batch_base_owning {
	dyn_array<T, max_interleave_size> data_;
public:
	using value_type = T;

	batch_base_owning()=default;

	PERFLIBS_LINALG_INLINE
	batch_base_owning(kernel_inttype cntg)
	:	data_ { static_cast<std::size_t>(cntg) }
	{	}

	PERFLIBS_LINALG_INLINE       T *data()       { return data_.data(); }
	PERFLIBS_LINALG_INLINE const T *data() const { return data_.data(); }

	PERFLIBS_LINALG_INLINE kernel_inttype cntg() const { return data_.size(); }
}; //class owning_batch

/**
 *  This class provides a fixed interleave_batch width for use with temporary values
 */
template<typename T>
class batch_base_fixed {
	std::array<T, max_interleave_size> data_;
public:
	using value_type = T;

	batch_base_fixed()
	{}

	PERFLIBS_LINALG_INLINE
	batch_base_fixed(kernel_inttype cntg) {
		PERFLIBS_ASSERT(cntg < size(data_), "cntg out of range of data_");
	}

	PERFLIBS_LINALG_INLINE
	batch_base_fixed(T val) {
		data_.fill(val);
	}

	PERFLIBS_LINALG_INLINE       T *data()       { return data_.data(); }
	PERFLIBS_LINALG_INLINE const T *data() const { return data_.data(); }

	PERFLIBS_LINALG_INLINE kernel_inttype cntg() const { return 0; } //data_.size(); }
}; //class batch_base_fixed

///inline tests
static_assert(std::is_default_constructible_v<batch_base_owning<float>>, "batch_base_owning is not default constructible");
static_assert(std::is_copy_constructible_v<batch_base_owning<float>>,    "batch_base_owning is not copy constructible");
static_assert(std::is_move_constructible_v<batch_base_owning<float>>,    "batch_base_owning is not move constructible");
static_assert(std::is_copy_assignable_v<batch_base_owning<float>>,       "batch_base_owning is not copy constructible");
static_assert(std::is_move_assignable_v<batch_base_owning<float>>,       "batch_base_owning is not move constructible");

/*
 * When combined interleave_batch, this forms the InterleaveBatchExpression
 * which provides access onto existing data (ie a user provided matrix...)
 */
template<typename T>
class batch_base_reference {
	T *data_;
	kernel_inttype cntg_;
public:
	using value_type = T;

	PERFLIBS_LINALG_INLINE
	batch_base_reference(T *data, kernel_inttype cntg)
	:	data_ { data }
	,	cntg_ { cntg }
	{	}

	PERFLIBS_LINALG_INLINE const T       *data() const { return data_; }
	PERFLIBS_LINALG_INLINE       T       *data()       { return data_; }
	PERFLIBS_LINALG_INLINE kernel_inttype cntg() const { return cntg_; }
}; //class value_batch

/**
 * The basic operators we support for batch types
 */
enum class optype {
	add, sub, div, mul
}; //enum class optype


/**
 * This class provides the mechanisms to extra batch data
 * to and from a pointer to value_type.
 *
 * This is then complete with either value (which owns its own batch data)
 * or a reference which is imbued with external data
 */
template<typename BatchBase>
class interleave_batch : public BatchBase {
public:
	using value_type = typename BatchBase::value_type;
private:
	/**
	 * as +=, -=, /=, *= are effectively the same operation
	 * with only one instruction different then this function
	 * implements those functions, but has a template parameter
	 * which details which operations we want to perform
	 *
	 * @tparam OP the operation we are performing
	 * @tparam ExprRhs the type of the Expression on the right hand side of operation (ie  this += rhs;)
	 *
	 * @param [in] the rhs expr
	 */
	template<optype OP, typename ExprRhs>
	PERFLIBS_LINALG_INLINE
	void assign(const ExprRhs& rhs) {
		const auto cntg = max(BatchBase::cntg(), rhs.cntg());
		auto data = BatchBase::data();

/*
	The loads below are not predicated. Disable for now.
	We will need to be able to pass predicates through to the get_as function in the ExprRhs type.
*/
#if 0
//#ifdef __ARM_FEATURE_SVE
		if constexpr(is_same_rem_cv_v<value_type, double> && simd::is_sve) {
			kernel_inttype i = 0;
			svbool_t pg = svwhilelt_b64(i, cntg);
			do {
				svfloat64_t a =              get_as<svfloat64_t>(i);
				svfloat64_t b = rhs.template get_as<svfloat64_t>(i);
				if constexpr(OP == optype::add) {
					svst1(pg, &data[i], svadd_x(pg, a, b));
				}
				else if constexpr(OP == optype::sub) {
					svst1(pg, &data[i], svsub_x(pg, a, b));
				}
				else if constexpr(OP == optype::div) {
					svst1(pg, &data[i], svdiv_x(pg, a, b));
				}
				else if constexpr(OP == optype::mul) {
					svst1(pg, &data[i], svmul_x(pg, a, b));
				}
				i+=svcntd();
				pg = svwhilelt_b64(i, cntg);
			} while (svptest_any(svptrue_b64(), pg));
			return;
		}
#endif
		if constexpr(is_same_rem_cv_v<value_type, double> && simd::is_neon) {
			switch(cntg) {
			case 8_ki: {
				auto a0 =              get_as<float64x2x2_t>(0);
				auto b0 = rhs.template get_as<float64x2x2_t>(0);
				auto a1 =              get_as<float64x2x2_t>(4);
				auto b1 = rhs.template get_as<float64x2x2_t>(4);

				if constexpr(OP == optype::add) {
					b0.val[0] = vaddq_f64(a0.val[0], b0.val[0]);
					b0.val[1] = vaddq_f64(a0.val[1], b0.val[1]);
					b1.val[0] = vaddq_f64(a1.val[0], b1.val[0]);
					b1.val[1] = vaddq_f64(a1.val[1], b1.val[1]);
				}
				else if constexpr(OP == optype::sub) {
					b0.val[0] = vsubq_f64(a0.val[0], b0.val[0]);
					b0.val[1] = vsubq_f64(a0.val[1], b0.val[1]);
					b1.val[0] = vsubq_f64(a1.val[0], b1.val[0]);
					b1.val[1] = vsubq_f64(a1.val[1], b1.val[1]);
				}
				else if constexpr(OP == optype::div) {
					b0.val[0] = vdivq_f64(a0.val[0], b0.val[0]);
					b0.val[1] = vdivq_f64(a0.val[1], b0.val[1]);
					b1.val[0] = vdivq_f64(a1.val[0], b1.val[0]);
					b1.val[1] = vdivq_f64(a1.val[1], b1.val[1]);
				}
				else if constexpr(OP == optype::mul) {
					b0.val[0] = vmulq_f64(a0.val[0], b0.val[0]);
					b0.val[1] = vmulq_f64(a0.val[1], b0.val[1]);
					b1.val[0] = vmulq_f64(a1.val[0], b1.val[0]);
					b1.val[1] = vmulq_f64(a1.val[1], b1.val[1]);
				}

				simd::store(data,   b0);
				simd::store(data+4, b1);
				return;
			}
			case 4_ki: {
				auto [ a0, a1 ] =              get_as<float64x2x2_t>(0).val;
				auto [ b0, b1 ] = rhs.template get_as<float64x2x2_t>(0).val;

				float64x2x2_t ret;

				if constexpr(OP == optype::add) {
					ret.val[0] = vaddq_f64(a0, b0);
					ret.val[1] = vaddq_f64(a1, b1);
				}
				else if constexpr(OP == optype::sub) {
					ret.val[0] = vsubq_f64(a0, b0);
					ret.val[1] = vsubq_f64(a1, b1);
				}
				else if constexpr(OP == optype::div) {
					ret.val[0] = vdivq_f64(a0, b0);
					ret.val[1] = vdivq_f64(a1, b1);
				}
				else if constexpr(OP == optype::mul) {
					ret.val[0] = vmulq_f64(a0, b0);
					ret.val[1] = vmulq_f64(a1, b1);
				}

				simd::store(data, ret);
				return;
			}
			case 2_ki: {
				auto a0 =              get_as<float64x2_t>(0);
				auto b0 = rhs.template get_as<float64x2_t>(0);

				if      constexpr(OP == optype::add) { b0 = vaddq_f64(a0, b0); }
				else if constexpr(OP == optype::sub) { b0 = vsubq_f64(a0, b0); }
				else if constexpr(OP == optype::div) { b0 = vdivq_f64(a0, b0); }
				else if constexpr(OP == optype::mul) { b0 = vmulq_f64(a0, b0); }

				simd::store(data, b0);
				return;
			}
			//we don't efficiently support this case
			default: break;
			}
		}

		for(kernel_inttype i = 0_ki; i!=cntg; ++i) {
			if constexpr(OP == optype::add) { data[i] += rhs[i]; }
			if constexpr(OP == optype::sub) { data[i] -= rhs[i]; }
			if constexpr(OP == optype::div) { data[i] /= rhs[i]; }
			if constexpr(OP == optype::mul) { data[i] *= rhs[i]; }
		}
	}
public:
	/**
	 * ctor
	 * @param data [in] a pointer to the first element of the batch
	 * @param cntg [in] the number of contiguous elements in the batch
	 */
	template<typename... Args>
	PERFLIBS_LINALG_INLINE
	interleave_batch(Args&&... args)
	:	BatchBase{ std::forward<Args>(args)... }
	{	}

	/**
	 * Binary operators
	 * These operators all apply operator they overload to each of the
	 * elements in the batch IF there is a SIMD implementation then
	 * it will use that
	 */
	template<typename ExprRhs>
	PERFLIBS_LINALG_INLINE
	std::enable_if_t<is_interleave_batch_expr_v<ExprRhs>, interleave_batch&>
	operator+=(const ExprRhs& rhs) {
		assign<optype::add>(rhs);
		return *this;
	}
	template<typename ExprRhs>
	PERFLIBS_LINALG_INLINE
	std::enable_if_t<is_interleave_batch_expr_v<ExprRhs>, interleave_batch&>
	operator-=(const ExprRhs& rhs) {
		assign<optype::sub>(rhs);
		return *this;
	}
	template<typename ExprRhs>
	PERFLIBS_LINALG_INLINE
	std::enable_if_t<is_interleave_batch_expr_v<ExprRhs>, interleave_batch&>
	operator/=(const ExprRhs& rhs) {
		assign<optype::div>(rhs);
		return *this;
	}
	template<typename ExprRhs>
	PERFLIBS_LINALG_INLINE
	std::enable_if_t<is_interleave_batch_expr_v<ExprRhs>, interleave_batch&>
	operator*=(const ExprRhs& rhs) {
		assign<optype::mul>(rhs);
		return *this;
	}
	template<typename ExprRhs>
	PERFLIBS_LINALG_INLINE
	std::enable_if_t<is_interleave_batch_expr_v<ExprRhs>, interleave_batch&>
	operator=(const ExprRhs& rhs) {
		auto data = BatchBase::data();

		kernel_inttype cntg;
		if constexpr(std::is_same_v<ExprRhs, scalar<value_type>>) {
			cntg = BatchBase::cntg();
		}
		else {
			cntg = max(BatchBase::cntg(), rhs.cntg());
		}
/*
	The loads below are not predicated. Disable for now.
	We will need to be able to pass predicates through to the get_as function in the ExprRhs type.
*/
//#ifdef __ARM_FEATURE_SVE
#if 0
		if constexpr(is_same_rem_cv_v<value_type, double> && simd::is_sve) {
			kernel_inttype i = 0;
			svbool_t pg = svwhilelt_b64(i, cntg);
			do {
				svst1(pg, &data[i], rhs.template get_as<svfloat64_t>(i));
				i+=svcntd();
				pg = svwhilelt_b64(i, cntg);
			} while (svptest_any(svptrue_b64(), pg));
			return *this;
		}
#endif
		if constexpr(is_same_rem_cv_v<value_type, double> && simd::is_neon) {
			switch(cntg) {
			case 8_ki:
				simd::store(data,   rhs.template get_as<float64x2x2_t>(0));
				simd::store(data+4, rhs.template get_as<float64x2x2_t>(4));
				return *this;
			case 4_ki:
				simd::store(data,   rhs.template get_as<float64x2x2_t>(0));
				return *this;
			case 2_ki:
				simd::store(data,   rhs.template get_as<float64x2_t>(0));
				return *this;
			//we don't efficiently support this case
			default: break;
			}
		}
		for(kernel_inttype i = 0_ki; i!=cntg; ++i) {
			data[i] = rhs[i];
		}
		return *this;
	}
	/**
	 * Element Access, the following operators allow extraction
	 * of individual elements from the batch.
	 *
	 * We implement op() to remain consistent with the linalg
	 * (which needs multi index access for matrices) and op[]
	 * for consistency with the rest of C++
	 */
	PERFLIBS_LINALG_INLINE
	const value_type& operator[](kernel_inttype cntg) const {
		return BatchBase::data()[cntg];
	}
	PERFLIBS_LINALG_INLINE
	value_type& operator[](kernel_inttype cntg) {
		return BatchBase::data()[cntg];
	}
	PERFLIBS_LINALG_INLINE
	const value_type& operator()(kernel_inttype cntg) const {
		return BatchBase::data()[cntg];
	}
	PERFLIBS_LINALG_INLINE
	value_type& operator()(kernel_inttype cntg) {
		return BatchBase::data()[cntg];
	}

	/**
	 * @tparam SimdVec is simd type we want to receive, also supports 'T'
	 * @param the offset in the batch from which to start then population of the SimdVec
	 * @returns a SimdVec populate with the elements starting at data() + cntg
	 */
	template<typename SimdVec>
	PERFLIBS_LINALG_INLINE
	SimdVec get_as(kernel_inttype cntg) const {
		return simd::load<SimdVec>(BatchBase::data() + cntg);
	}
}; //class interleave_batch

template<typename T> using interleave_batch_ref = interleave_batch<batch_base_reference<T>>;
template<typename T> using interleave_batch_own = interleave_batch<batch_base_owning<T>>;
template<typename T> using interleave_batch_val = interleave_batch<batch_base_fixed<T>>;

///inline tests
static_assert(has_value_type_v<interleave_batch_ref<float>>, "types needs value_type member typedef");
static_assert(has_value_type_v<interleave_batch_own<float>>, "types needs value_type member typedef");


/**
 * This class encapsulate a batch value without the need to refer specific to
 * some own memory. unlike `interleave_batch_own` it does not need to know its
 * batch size on initialization, this can be inferred later
 *
 * The type is implemented as a a variant of `interleave_batch_own` and `scalar`,
 * when it is initialized without a size and just a scalar value (ie t = 1.0) then
 * the variant is set to scalar<T>{1.0}; if no scalar value is provided it defaults
 * to T{0.0}. (ie default initialization)
 */
//template<typename T>
//class interleave_batch_val {
//public:
//	using value_type = T;
//
//private:
//	using scalar_type = scalar<T>;
//	using owning_type = interleave_batch_own<T>;
//
//	std::variant<owning_type, scalar_type> data_;
//
//	PERFLIBS_LINALG_INLINE
//	value_type get(kernel_inttype cntg) const {
//		if(auto p = std::get_if<owning_type>(&data_))
//			return (*p)[cntg];
//
//		return std::get_if<scalar_type>(&data_)->value();
//	}
//
//public:
//	/**
//	 * We have a size, so lets init out batch to it
//	 */
//	PERFLIBS_LINALG_INLINE
//	interleave_batch_val(kernel_inttype cntg)
//	:	data_ { owning_type{ cntg } }
//	{	}
//	/**
//	 * We have a batch and a size
//	 */
//	PERFLIBS_LINALG_INLINE
//	interleave_batch_val(kernel_inttype cntg, T scal)
//	{
//		auto& data =data_.template emplace<owning_type>(cntg);
//		data = scalar{ scal };
//	}
//
//	/**
//	 * We only have a scalar value so we init to that, no size yet
//	 */
//	PERFLIBS_LINALG_INLINE
//	interleave_batch_val(const scalar<T>& scal)
//	:	data_ { scal }
//	{	}
//
//	/**
//	 * We only have a scalar value so we init to that, no size yet
//	 */
//	PERFLIBS_LINALG_INLINE
//	interleave_batch_val(T scal = T{0.0})
//	:	data_ { scalar_type{ scal } }
//	{	}
//
//
//	/**
//	 * Initialize this batch to the same value as another batch expression; this will evaluate
//	 * that expression and copy its data
//	 *
//	 * @param rhs [in] the expression from which the value will be taken
//	 * @param unused SFINAE guard to require ExprRhs to be an InterleaveBatchExpression and avoid nested template errors
//	 */
//	template<typename ExprRhs>
//	PERFLIBS_LINALG_INLINE
//	interleave_batch_val(const ExprRhs& rhs, std::enable_if_t<is_interleave_batch_expr_v<ExprRhs>>* =nullptr)
//	{
//		auto& data = data_.template emplace<owning_type>(rhs.cntg());
//		data = rhs;
//	}
//
//	/**
//	 * @returns the size of the batch
//	 */
//	PERFLIBS_LINALG_INLINE
//	kernel_inttype cntg() const {
//		if(auto p = std::get_if<owning_type>(&data_))
//			return p->cntg();
//		return 0;
//	}
//	/**
//	 * @returns the value(s) from the expression start from the specified position in the batch
//	 * @tparam SimdVec the vector type you want to receive
//	 * @param cntg [in] position in the batch
//	 */
//	template<typename SimdVec>
//	PERFLIBS_LINALG_INLINE
//	SimdVec get_as(kernel_inttype cntg) const {
//		if(auto p = std::get_if<owning_type>(&data_))
//			return p->template get_as<SimdVec>(cntg);
//		return std::get_if<scalar_type>(&data_)->template get_as<SimdVec>(cntg);
//	}
//
//	/**
//	 * @returns Scalar value at the specified position
//	 */
//	PERFLIBS_LINALG_INLINE value_type operator[](kernel_inttype cntg) const { return get(cntg); }
//	PERFLIBS_LINALG_INLINE value_type operator()(kernel_inttype cntg) const { return get(cntg); }
//
//	PERFLIBS_LINALG_INLINE
//	interleave_batch_val& operator=(const scalar<T>& rhs) {
//		auto op = std::get_if<owning_type>(&data_);
//		if(op == nullptr) {
//			data_ = rhs;
//			return *this;
//		}
//		(*op) = rhs;
//		return *this;
//	}
//	PERFLIBS_LINALG_INLINE
//	interleave_batch_val& operator+=(const scalar<T>& rhs) {
//		auto op = std::get_if<owning_type>(&data_);
//		if(op == nullptr) {
//			auto& scal = std::get<scalar_type>(&data_)->value();
//			scal += rhs.value();
//			return *this;
//		}
//		(*op) += rhs;
//		return *this;
//	}
//	PERFLIBS_LINALG_INLINE
//	interleave_batch_val& operator-=(const scalar<T>& rhs) {
//		auto op = std::get_if<owning_type>(&data_);
//		if(op == nullptr) {
//			auto& scal = std::get<scalar_type>(&data_)->value();
//			scal -= rhs.value();
//			return *this;
//		}
//		(*op) -= rhs;
//		return *this;
//	}
//	PERFLIBS_LINALG_INLINE
//	interleave_batch_val& operator/=(const scalar<T>& rhs) {
//		auto op = std::get_if<owning_type>(&data_);
//		if(op == nullptr) {
//			auto& scal = std::get<scalar_type>(&data_)->value();
//			scal /= rhs.value();
//			return *this;
//		}
//		(*op) /= rhs;
//		return *this;
//	}
//	PERFLIBS_LINALG_INLINE
//	interleave_batch_val& operator*=(const scalar<T>& rhs) {
//		auto op = std::get_if<owning_type>(&data_);
//		if(op == nullptr) {
//			auto& scal = std::get<scalar_type>(&data_)->value();
//			scal *= rhs.value();
//			return *this;
//		}
//		(*op) *= rhs;
//		return *this;
//	}
//
//
//	PERFLIBS_LINALG_INLINE interleave_batch_val& operator=(T scal)  { return *this = scalar { scal }; } PERFLIBS_LINALG_INLINE interleave_batch_val& operator+=(T scal) { return *this += scalar { scal }; }
//	PERFLIBS_LINALG_INLINE interleave_batch_val& operator-=(T scal) { return *this -= scalar { scal }; }
//	PERFLIBS_LINALG_INLINE interleave_batch_val& operator/=(T scal) { return *this /= scalar { scal }; }
//	PERFLIBS_LINALG_INLINE interleave_batch_val& operator*=(T scal) { return *this *= scalar { scal }; }
//
//	template<typename ExprRhs>
//	PERFLIBS_LINALG_INLINE
//	std::enable_if_t<is_interleave_batch_expr_v<ExprRhs>, interleave_batch_val&>
//	operator=(const ExprRhs& rhs) {
//		auto op = std::get_if<owning_type>(&data_);
//		if(op == nullptr) {
//			op = &data_.template emplace<owning_type>(rhs.cntg());
//		}
//		(*op) = rhs;
//		return *this;
//	}
//
//	template<typename ExprRhs>
//	PERFLIBS_LINALG_INLINE
//	std::enable_if_t<is_interleave_batch_expr_v<ExprRhs>, interleave_batch_val&>
//	operator+=(const ExprRhs& rhs) {
//		auto op = std::get_if<owning_type>(&data_);
//		if(op == nullptr) {
//			auto scal = std::get_if<scalar_type>(&data_)->value();
//			op = &data_.template emplace<owning_type>(rhs.cntg());
//			*op = scalar{ scal };
//		}
//		(*op) += rhs;
//		return *this;
//	}
//
//	template<typename ExprRhs>
//	PERFLIBS_LINALG_INLINE
//	std::enable_if_t<is_interleave_batch_expr_v<ExprRhs>, interleave_batch_val&>
//	operator-=(const ExprRhs& rhs) {
//		auto op = std::get_if<owning_type>(&data_);
//		if(op == nullptr) {
//			auto scal = std::get_if<scalar_type>(&data_)->value();
//			op = &data_.template emplace<owning_type>(rhs.cntg());
//			*op = scalar{ scal };
//		}
//		(*op) -= rhs;
//	}
//	template<typename ExprRhs>
//	PERFLIBS_LINALG_INLINE
//	std::enable_if_t<is_interleave_batch_expr_v<ExprRhs>, interleave_batch_val&>
//	operator/=(const ExprRhs& rhs) {
//		auto op = std::get_if<owning_type>(&data_);
//		if(op == nullptr) {
//			auto scal = std::get_if<scalar_type>(&data_)->value();
//			op = &data_.template emplace<owning_type>(rhs.cntg());
//			*op = scalar{ scal };
//		}
//		(*op) /= rhs;
//		return *this;
//	}
//	template<typename ExprRhs>
//	PERFLIBS_LINALG_INLINE
//	std::enable_if_t<is_interleave_batch_expr_v<ExprRhs>, interleave_batch_val&>
//	operator*=(const ExprRhs& rhs) {
//		auto op = std::get_if<owning_type>(&data_);
//		if(op == nullptr) {
//			auto scal = std::get_if<scalar_type>(&data_)->value();
//			op = &data_.template emplace<owning_type>(rhs.cntg());
//			*op = scalar{ scal };
//		}
//		(*op) *= rhs;
//		return *this;
//	}
//}; //interleave_batch_val
//
//inline tests
static_assert(std::is_default_constructible_v<interleave_batch_val<float>>, "interleave_batch_val is not default constructible");
static_assert(std::is_copy_constructible_v<interleave_batch_val<float>>,    "interleave_batch_val is not copy constructible");
static_assert(std::is_move_constructible_v<interleave_batch_val<float>>,    "interleave_batch_val is not move constructible");
static_assert(std::is_copy_assignable_v<interleave_batch_val<float>>,       "interleave_batch_val is not copy constructible");
static_assert(std::is_move_assignable_v<interleave_batch_val<float>>,       "interleave_batch_val is not move constructible");


/**
 * @returns true if every element of rhs matches every element in lhs
 */
template<typename BatchBase>
bool operator==(const interleave_batch<BatchBase>& lhs, const interleave_batch<BatchBase>& rhs) {
	PERFLIBS_ASSERT(rhs.cntg() == lhs.cntg(), "interleave_batches A & C do not share dimensions");

	for(kernel_inttype i = 0; i!=lhs.cntg(); ++i) {
		if(lhs(i) != rhs(i))
			return false;
	}

	return true;
}

/**
 * The following allow use to compose together sequence of operator expressions
 * building proxy objects in procress, only when we want to access the final
 * result do we actually compute any of the values.
 *
 * This is necessary because when process batches of variable length we have
 * nowhere to store the temporary results produce. By doing it this where we
 * can return the temporary partial results on the stack
 *
 * All expression types implement the following:
 *
 * value_type - typedef denoting the type of the elements in the batch
 * cntg() - returns the length of the batch
 * get_as<SimdVec>(cntg) - returns a SimdVec populated with some number of element starting at cntg
 * op[cntg] - returns the value at position cntg
 * op(cntg) - see above, included for uniformity with the linalg
 *
 * The basic types are:
 *
 * interleave_batch - real batch data
 * scalar - represents a single value which broadcast into an entire batch
 * binder - binds together two other expression (including nesting binders themselves) and applies an operation to the result of both
 */


/**
 * Binds together two ExpressionTypes and applies and lazily (to avoid temporaries)
 * evaluates them when get_as<>() is called, function applied is defined by OP
 *
 * @tparam Op the operation type to apply: ie this->get_as() returns e0.get_as() <op> e1.get_as()
 * @ExprType0 the First expression to bind
 * @ExprType0 the Second expression to bind
 */
template<optype Op, typename ExprType0, typename ExprType1>
class binder {
public:
	constexpr static optype operation = Op;
	using value_type                  = typename ExprType0::value_type;

	/**
	 * Constructor
	 * @param expr0 [in] first expression
	 * @param expr1 [in] second expression
	 */
	PERFLIBS_LINALG_INLINE
	binder(const ExprType0& expr0, const ExprType1& expr1)
	:	expr0_ { expr0 }
	,	expr1_ { expr1 }
	{
		//simply to stop any errors at this point
		assert_interleave_batch_expr(expr0);
		assert_interleave_batch_expr(expr1);
	}

	template<typename SimdVec>
	PERFLIBS_LINALG_INLINE
	SimdVec get_as(kernel_inttype cntg) const {
		auto val0 = expr0_.template get_as<SimdVec>(cntg);
		auto val1 = expr1_.template get_as<SimdVec>(cntg);

		if      constexpr (operation == optype::add) { return simd::add(val0, val1); }
		else if constexpr (operation == optype::sub) { return simd::sub(val0, val1); }
		else if constexpr (operation == optype::div) { return simd::div(val0, val1); }
		else if constexpr (operation == optype::mul) { return simd::mul(val0, val1); }
	}

	PERFLIBS_LINALG_INLINE
	kernel_inttype cntg() const {
		if constexpr(! std::is_same_v<ExprType0, scalar<value_type>> &&
		             ! std::is_same_v<ExprType1, scalar<value_type>>) {
			return max(expr0_.cntg(), expr1_.cntg());
		}
		else if constexpr(! std::is_same_v<ExprType0, scalar<value_type>>) {
			return expr0_.cntg();
		}
		else if constexpr(! std::is_same_v<ExprType1, scalar<value_type>>) {
			return expr1_.cntg();
		}
		return 0;
	}

	PERFLIBS_LINALG_INLINE value_type operator()(kernel_inttype cntg) const { return get_as<value_type>(cntg); }
	PERFLIBS_LINALG_INLINE value_type operator[](kernel_inttype cntg) const { return get_as<value_type>(cntg); }
private:
	ExprType0 expr0_;
	ExprType1 expr1_;
}; //class binder

//inline compile time test
static_assert(is_interleave_batch_expr_v<binder<optype::add, interleave_batch_val<float>, scalar<float>>>, "binder failed to meet requirements of InterleaveBatchExpr");
static_assert(is_interleave_batch_expr_v<binder<optype::sub, scalar<double>, interleave_batch_ref<double>>>, "binder failed to meet requirements of InterleaveBatchExpr");
static_assert(is_interleave_batch_expr_v<binder<optype::div, scalar<float>, binder<optype::mul, interleave_batch_ref<float>, interleave_batch_val<float>>>>, "binder failed to meet requirements of InterleaveBatchExpr");

/**
 *
 * The following binary operators take to interleave_batch expressions and combine them
 * together in  a binder type fused with the appropriate optype for the operator
 *
 * So for example:
 *
 *      expr0 + expr1
 *
 * will result in:
 *
 *     binder<add, ExprType0, ExprType1>
 *
 * All of the operators also suppose a batched operator being bound with a scalar value
 * to allow the programmer to write their code more naturally, ie:
 *
 *      expr0 + 2.0
 *
 * in this case, the operator will produce a `scalar` expression, and form a binder like so:
 *
 *      binder<add, ExprType0, scalar<double>>
 */

/// Operator+ overloads
template<typename ExprType0, typename ExprType1>
PERFLIBS_LINALG_INLINE
std::enable_if_t<is_interleave_batch_expr_v<ExprType0>  //REQUIRES
              && is_interleave_batch_expr_v<ExprType1>,
binder<optype::add, ExprType0, ExprType1>> //return type
operator+(const ExprType0& expr0, const ExprType1& expr1)
{ return { expr0, expr1 }; }

template<typename ExprType0, typename T1>
PERFLIBS_LINALG_INLINE
std::enable_if_t<is_interleave_batch_expr_v<ExprType0>  //REQUIRES
             && !is_interleave_batch_expr_v<T1>,
binder<optype::add, ExprType0, scalar<T1>>>
operator+(const ExprType0& expr0, const T1& expr1)
{ return { expr0, expr1 }; }

template<typename T0, typename ExprType1>
PERFLIBS_LINALG_INLINE
std::enable_if_t<!is_interleave_batch_expr_v<T0>  //REQUIRES
               && is_interleave_batch_expr_v<ExprType1>,
binder<optype::add, scalar<T0>, ExprType1>>
operator+(const T0& expr0, const ExprType1& expr1)
{ return { expr0, expr1 }; }

/// Operator- overloads
template<typename ExprType0, typename ExprType1>
PERFLIBS_LINALG_INLINE
std::enable_if_t<is_interleave_batch_expr_v<ExprType0>  //REQUIRES
              && is_interleave_batch_expr_v<ExprType1>,
binder<optype::sub, ExprType0, ExprType1>> //return type
operator-(const ExprType0& expr0, const ExprType1& expr1)
{ return { expr0, expr1 }; }

template<typename ExprType0, typename T1>
PERFLIBS_LINALG_INLINE
std::enable_if_t<is_interleave_batch_expr_v<ExprType0>  //REQUIRES
             && !is_interleave_batch_expr_v<T1>,
binder<optype::sub, ExprType0, scalar<T1>>>
operator-(const ExprType0& expr0, const T1& expr1)
{ return { expr0, expr1 }; }

template<typename T0, typename ExprType1>
PERFLIBS_LINALG_INLINE
std::enable_if_t<!is_interleave_batch_expr_v<T0>  //REQUIRES
               && is_interleave_batch_expr_v<ExprType1>,
binder<optype::sub, scalar<T0>, ExprType1>>
operator-(const T0& expr0, const ExprType1& expr1)
{ return { expr0, expr1 }; }

/// Operator/ overloads
template<typename ExprType0, typename ExprType1>
PERFLIBS_LINALG_INLINE
std::enable_if_t<is_interleave_batch_expr_v<ExprType0>  //REQUIRES
              && is_interleave_batch_expr_v<ExprType1>,
binder<optype::div, ExprType0, ExprType1>> //return type
operator/(const ExprType0& expr0, const ExprType1& expr1)
{ return { expr0, expr1 }; }

template<typename ExprType0, typename T1>
PERFLIBS_LINALG_INLINE
std::enable_if_t<is_interleave_batch_expr_v<ExprType0>  //REQUIRES
             && !is_interleave_batch_expr_v<T1>,
binder<optype::div, ExprType0, scalar<T1>>>
operator/(const ExprType0& expr0, const T1& expr1)
{ return { expr0, expr1 }; }

template<typename T0, typename ExprType1>
PERFLIBS_LINALG_INLINE
std::enable_if_t<!is_interleave_batch_expr_v<T0>  //REQUIRES
               && is_interleave_batch_expr_v<ExprType1>,
binder<optype::div, scalar<T0>, ExprType1>>
operator/(const T0& expr0, const ExprType1& expr1)
{ return { expr0, expr1 }; }

/// Operator* overloads
template<typename ExprType0, typename ExprType1>
PERFLIBS_LINALG_INLINE
std::enable_if_t<is_interleave_batch_expr_v<ExprType0>  //REQUIRES
              && is_interleave_batch_expr_v<ExprType1>,
binder<optype::mul, ExprType0, ExprType1>> //return type
operator*(const ExprType0& expr0, const ExprType1& expr1)
{ return { expr0, expr1 }; }

template<typename ExprType0, typename T1>
PERFLIBS_LINALG_INLINE
std::enable_if_t<is_interleave_batch_expr_v<ExprType0>  //REQUIRES
             && !is_interleave_batch_expr_v<T1>,
binder<optype::mul, ExprType0, scalar<T1>>>
operator*(const ExprType0& expr0, const T1& expr1)
{ return { expr0, expr1 }; }

template<typename T0, typename ExprType1>
PERFLIBS_LINALG_INLINE
std::enable_if_t<!is_interleave_batch_expr_v<T0>  //REQUIRES
               && is_interleave_batch_expr_v<ExprType1>,
binder<optype::mul, scalar<T0>, ExprType1>>
operator*(const T0& expr0, const ExprType1& expr1)
{ return { expr0, expr1 }; }

/**
 * Bind a unary function provided as a template parameter
 * into the expression structure
 *
 * When get_as<>() is invoked then the value is retrieved
 * from Expr0 and then the function is invoked with that value
 * that is the return
 */
template<typename Func, typename ExprType0>
class binder_unary {
public:
	using value_type                  = typename ExprType0::value_type;
	/**
	 * Constructor
	 * @param expr0 [in] first expression
	 */
	PERFLIBS_LINALG_INLINE
	binder_unary(const ExprType0& expr0)
	:	func_  {       }
	,	expr0_ { expr0 }
	{
		//simply to stop any errors at this point
		assert_interleave_batch_expr(expr0);
	}

	/**
	 * Constructor
	 * @param func [in] the functor we want to apply in this unary op
	 * @param expr0 [in] first expression
	 */
	PERFLIBS_LINALG_INLINE
	binder_unary(const Func& func, const ExprType0& expr0)
	:	func_  { func  }
	,	expr0_ { expr0 }
	{
		//simply to stop any errors at this point
		assert_interleave_batch_expr(expr0);
	}

	template<typename SimdVec>
	PERFLIBS_LINALG_INLINE
	SimdVec get_as(kernel_inttype cntg) const {
		auto val0 = expr0_.template get_as<SimdVec>(cntg);

		return func_ ( val0 );
	}

	PERFLIBS_LINALG_INLINE
	kernel_inttype cntg() const {
		if constexpr(! std::is_same_v<ExprType0, scalar<value_type>>) {
			return expr0_.cntg();
		}
		return 0;
	}

	PERFLIBS_LINALG_INLINE value_type operator()(kernel_inttype cntg) const { return get_as<value_type>(cntg); }
	PERFLIBS_LINALG_INLINE value_type operator[](kernel_inttype cntg) const { return get_as<value_type>(cntg); }
private:
	Func func_;
	ExprType0 expr0_;
}; //class binder_unary

namespace simd {

/**
 * Computes sqrt for simd types
 */
struct sqrt {
	PERFLIBS_LINALG_INLINE double        operator()(double v0) const { return std::sqrt(v0); }
	PERFLIBS_LINALG_INLINE float64x2_t   operator()(float64x2_t v0) const { return vsqrtq_f64(v0); }
	PERFLIBS_LINALG_INLINE float64x2x2_t operator()(float64x2x2_t v0) const {
		float64x2x2_t ret;
		ret.val[0] = vsqrtq_f64(v0.val[0]);
		ret.val[1] = vsqrtq_f64(v0.val[1]);
		return ret;

	}
#ifdef __ARM_FEATURE_SVE
	PERFLIBS_LINALG_INLINE svfloat64_t   operator()(svfloat64_t v0) const { return svsqrt_f64_x(svptrue_b64(), v0); }
#endif
}; //struct sqrt

} //namespace simd

template<typename ExprType0>
PERFLIBS_LINALG_INLINE
std::enable_if_t<is_interleave_batch_expr_v<ExprType0>,  //REQUIRES
binder_unary<simd::sqrt, ExprType0>>
sqrt(const ExprType0& expr0) {
	return { expr0 };
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_INTERLEAVE_BATCH

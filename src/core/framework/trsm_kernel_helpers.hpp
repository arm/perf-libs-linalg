/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_TRSM_KERNEL_HELPERS_HPP
#define PERFLIBS_LINALG_TRSM_KERNEL_HELPERS_HPP

#include "perflibs_complex.hpp"
#include <arm_neon.h>

namespace perflibs::linalg {
namespace {

// simd helper methods for use in trsm_kernel_impl

template<typename VecType, typename ScalarType>
PERFLIBS_LINALG_INLINE
VecType load(const ScalarType *);

template<>
PERFLIBS_LINALG_INLINE
float32x4_t load(const float * p) {
	// load 4 floats
	return vld1q_f32(p);
}

template<>
PERFLIBS_LINALG_INLINE
float64x2x2_t load(const double * p) {
	// load 4 doubles
	float64x2x2_t ret;
	ret.val[0] = vld1q_f64(p + 0);
	ret.val[1] = vld1q_f64(p + 2);
	return ret;
}

template<>
PERFLIBS_LINALG_INLINE
float32x4x2_t load(const complex_float * p) {
	// load 4 complex_floats, deinterleaved as RRRR IIII
	return vld2q_f32(reinterpret_cast<const float *>(p));
}

template<>
PERFLIBS_LINALG_INLINE
float64x2x4_t load(const complex_double * p) {
	// load 4 complex_doubles, deinterleaved as RR RR II II
	float64x2x4_t ret;
	auto dp = reinterpret_cast<const double *>(p);

	auto a0 = vld2q_f64(dp    );
	auto a1 = vld2q_f64(dp + 4);
	ret.val[0] = a0.val[0];
	ret.val[1] = a1.val[0];
	ret.val[2] = a0.val[1];
	ret.val[3] = a1.val[1];

	return ret;
}


template<typename ScalarType, typename VecType>
PERFLIBS_LINALG_INLINE
void store(ScalarType *, VecType);

template<>
PERFLIBS_LINALG_INLINE
void store(float * p, float32x4_t v) {
	// store 4 floats
	vst1q_f32(p, v);
}

template<>
PERFLIBS_LINALG_INLINE
void store(double * p, float64x2x2_t v) {
	// store 4 doubles
	vst1q_f64(p + 0, v.val[0]);
	vst1q_f64(p + 2, v.val[1]);
}

template<>
PERFLIBS_LINALG_INLINE
void store(complex_float * p, float32x4x2_t v) {
	// Given 4 complex floats de-interleaved as RRRR IIII, interleave as
	// RIRIRIRI and store to memory
	vst2q_f32(reinterpret_cast<float *>(p), v);
}

template<>
PERFLIBS_LINALG_INLINE
void store(complex_double * p, float64x2x4_t v) {
	// Given 4 complex doubles de-interleaved as RR RR II II, interleave as
	// RIRIRIRI and store to memory
	float64x2x2_t a;
	auto dp = reinterpret_cast<double *>(p);

	a.val[0] = v.val[0];
	a.val[1] = v.val[2];
	vst2q_f64(dp, a);
	a.val[0] = v.val[1];
	a.val[1] = v.val[3];
	vst2q_f64(dp + 4, a);
}


template<const int Lane, bool Conj=false, typename VecType>
PERFLIBS_LINALG_INLINE
VecType mul_lane(VecType& a, VecType b);

template<const int Lane, bool Conj=false>
PERFLIBS_LINALG_INLINE
float32x4_t mul_lane(float32x4_t& a, float32x4_t b) {
	// multiply 4 floats in a by float in given lane of b
	a = vmulq_laneq_f32(a, b, Lane);
	return a;
}

template<const int Lane, bool Conj=false>
PERFLIBS_LINALG_INLINE
float64x2x2_t mul_lane(float64x2x2_t& a, float64x2x2_t b) {
	// multiply 4 doubles in a by double in given lane of b
	a.val[0] = vmulq_laneq_f64(a.val[0], b.val[Lane >> 1], Lane & 1);
	a.val[1] = vmulq_laneq_f64(a.val[1], b.val[Lane >> 1], Lane & 1);
	return a;
}

template<const int Lane, bool Conj=false>
PERFLIBS_LINALG_INLINE
float32x4x2_t mul_lane(float32x4x2_t& a, float32x4x2_t b) {
	// multiply 4 complex floats in a by complex_float in given lane of b,
	// where a and b are both deinterleaved as RR RR II II

	// (a.re[i] + i * a.im[i]) = (a.re[i] + i * a.im[i]) * (b.re[lane] + i * b.im[lane])
	// where a.re := a.val[0], a.im := a.val[1] etc.
	//
	// so a.re[i] = a.re[i] * b.re[lane] - a.im[i] * b.im[lane]
	//    a.im[i] = a.im[i] * b.re[lane] + a.re[i] * b.im[lane]

	float32x4_t re, im;

	re     = vmulq_laneq_f32(    a.val[0], b.val[0], Lane);
	im     = vmulq_laneq_f32(    a.val[1], b.val[0], Lane);
	if constexpr (Conj) {
	    re = vfmaq_laneq_f32(re, a.val[1], b.val[1], Lane);
	    im = vfmsq_laneq_f32(im, a.val[0], b.val[1], Lane);
	}
	else {
	    re = vfmsq_laneq_f32(re, a.val[1], b.val[1], Lane);
	    im = vfmaq_laneq_f32(im, a.val[0], b.val[1], Lane);
	}
	a.val[0] = re;
	a.val[1] = im;

	return a;
}

template<const int Lane, bool Conj=false>
PERFLIBS_LINALG_INLINE
float64x2x4_t mul_lane(float64x2x4_t& a, float64x2x4_t b) {
	// multiply 4 complex doubles in a by complex_double in given lane of b,
	// where a and b are both deinterleaved as RR RR II II

	float64x2_t re, im;

	re     = vmulq_laneq_f64(    a.val[0], b.val[     Lane >> 1 ], Lane & 1);
	im     = vmulq_laneq_f64(    a.val[2], b.val[     Lane >> 1 ], Lane & 1);
	if constexpr (Conj) {
	    re = vfmaq_laneq_f64(re, a.val[2], b.val[2 + (Lane >> 1)], Lane & 1);
	    im = vfmsq_laneq_f64(im, a.val[0], b.val[2 + (Lane >> 1)], Lane & 1);
	}
	else {
	    re = vfmsq_laneq_f64(re, a.val[2], b.val[2 + (Lane >> 1)], Lane & 1);
	    im = vfmaq_laneq_f64(im, a.val[0], b.val[2 + (Lane >> 1)], Lane & 1);
	}
	a.val[0] = re;
	a.val[2] = im;

	re     = vmulq_laneq_f64(    a.val[1], b.val[     Lane >> 1 ], Lane & 1);
	im     = vmulq_laneq_f64(    a.val[3], b.val[     Lane >> 1 ], Lane & 1);
	if constexpr (Conj) {
	    re = vfmaq_laneq_f64(re, a.val[3], b.val[2 + (Lane >> 1)], Lane & 1);
	    im = vfmsq_laneq_f64(im, a.val[1], b.val[2 + (Lane >> 1)], Lane & 1);
	}
	else {
	    re = vfmsq_laneq_f64(re, a.val[3], b.val[2 + (Lane >> 1)], Lane & 1);
	    im = vfmaq_laneq_f64(im, a.val[1], b.val[2 + (Lane >> 1)], Lane & 1);
	}
	a.val[1] = re;
	a.val[3] = im;

	return a;
}


template<const int Lane, bool Conj=false, typename VecType>
PERFLIBS_LINALG_INLINE
VecType fms_lane(VecType& a, VecType b, VecType c);

template<const int Lane, bool conj=false>
PERFLIBS_LINALG_INLINE
float32x4_t fms_lane(float32x4_t& a, float32x4_t b, float32x4_t c) {
	a = vfmsq_laneq_f32(a, b, c, Lane);
	return a;
}

template<const int Lane, bool Conj=false>
PERFLIBS_LINALG_INLINE
float64x2x2_t fms_lane(float64x2x2_t& a, float64x2x2_t b, float64x2x2_t c) {
	a.val[0] = vfmsq_laneq_f64(a.val[0], b.val[0], c.val[Lane >> 1], Lane & 1);
	a.val[1] = vfmsq_laneq_f64(a.val[1], b.val[1], c.val[Lane >> 1], Lane & 1);
	return a;
}

template<const int Lane, bool Conj=false>
PERFLIBS_LINALG_INLINE
float32x4x2_t fms_lane(float32x4x2_t& a, float32x4x2_t b, float32x4x2_t c) {
	// complex fused-multiply subtract, 4 at a time.
	float32x4_t re, im;

	re     = vmulq_laneq_f32(    b.val[0], c.val[0], Lane);
	im     = vmulq_laneq_f32(    b.val[1], c.val[0], Lane);
	if constexpr (Conj) {
	    re = vfmaq_laneq_f32(re, b.val[1], c.val[1], Lane);
	    im = vfmsq_laneq_f32(im, b.val[0], c.val[1], Lane);
	}
	else {
	    re = vfmsq_laneq_f32(re, b.val[1], c.val[1], Lane);
	    im = vfmaq_laneq_f32(im, b.val[0], c.val[1], Lane);
	}
	a.val[0] = vsubq_f32(a.val[0], re);
	a.val[1] = vsubq_f32(a.val[1], im);

	return a;
}

template<const int Lane, bool Conj=false>
PERFLIBS_LINALG_INLINE
float64x2x4_t fms_lane(float64x2x4_t& a, float64x2x4_t b, float64x2x4_t c) {
	// complex fused-multiply subtract, 4 at a time.
	float64x2_t re, im;

	re     = vmulq_laneq_f64(    b.val[0], c.val[     Lane >> 1 ], Lane & 1);
	im     = vmulq_laneq_f64(    b.val[2], c.val[     Lane >> 1 ], Lane & 1);
	if constexpr (Conj) {
	    re = vfmaq_laneq_f64(re, b.val[2], c.val[2 + (Lane >> 1)], Lane & 1);
	    im = vfmsq_laneq_f64(im, b.val[0], c.val[2 + (Lane >> 1)], Lane & 1);
	}
	else {
	    re = vfmsq_laneq_f64(re, b.val[2], c.val[2 + (Lane >> 1)], Lane & 1);
	    im = vfmaq_laneq_f64(im, b.val[0], c.val[2 + (Lane >> 1)], Lane & 1);
	}
	a.val[0] = vsubq_f64(a.val[0], re);
	a.val[2] = vsubq_f64(a.val[2], im);

	re     = vmulq_laneq_f64(    b.val[1], c.val[     Lane >> 1 ], Lane & 1);
	im     = vmulq_laneq_f64(    b.val[3], c.val[     Lane >> 1 ], Lane & 1);
	if constexpr (Conj) {
	    re = vfmaq_laneq_f64(re, b.val[3], c.val[2 + (Lane >> 1)], Lane & 1);
	    im = vfmsq_laneq_f64(im, b.val[1], c.val[2 + (Lane >> 1)], Lane & 1);
	}
	else {
	    re = vfmsq_laneq_f64(re, b.val[3], c.val[2 + (Lane >> 1)], Lane & 1);
	    im = vfmaq_laneq_f64(im, b.val[1], c.val[2 + (Lane >> 1)], Lane & 1);
	}
	a.val[1] = vsubq_f64(a.val[1], re);
	a.val[3] = vsubq_f64(a.val[3], im);

	return a;
}


template<typename VecType>
PERFLIBS_LINALG_INLINE
float32x4_t div(VecType&a, VecType b);

template<>
PERFLIBS_LINALG_INLINE
float32x4_t div(float32x4_t& a, float32x4_t b) {
	// 4 floats in a by corresponding 4 floats in b
	a = vdivq_f32(a, b);
	return a;
}


template<const int Lane, typename VecType>
PERFLIBS_LINALG_INLINE
float32x4_t dup_lane(VecType a);

template<const int Lane>
PERFLIBS_LINALG_INLINE
float32x4_t dup_lane(float32x4_t a) {
	// duplicate lane of a x4
	return vdupq_laneq_f32(a, Lane);
}


template<typename VecType>
PERFLIBS_LINALG_INLINE
void transpose(VecType&, VecType&, VecType&, VecType&);

template<>
PERFLIBS_LINALG_INLINE
void transpose(float32x4_t& a0, float32x4_t& a1, float32x4_t& a2, float32x4_t& a3) {
	// transpose values in registers a[0-3]
	auto t0 = vuzp1q_f32(a0, a2);
	auto t1 = vuzp1q_f32(a1, a3);
	auto t2 = vuzp2q_f32(a0, a2);
	auto t3 = vuzp2q_f32(a1, a3);
	a0 = vtrn1q_f32(t0, t1);
	a1 = vtrn1q_f32(t2, t3);
	a2 = vtrn2q_f32(t0, t1);
	a3 = vtrn2q_f32(t2, t3);
}

template<>
PERFLIBS_LINALG_INLINE
void transpose(float64x2x2_t& a0, float64x2x2_t& a1, float64x2x2_t& a2, float64x2x2_t& a3) {
	// transpose values in registers a[0-3]
	float64x2_t t0, t1, t2, t3;

	t0 = vtrn1q_f64(a0.val[0], a1.val[0]);
	t1 = vtrn2q_f64(a0.val[0], a1.val[0]);
	t2 = vtrn1q_f64(a2.val[1], a3.val[1]);
	t3 = vtrn2q_f64(a2.val[1], a3.val[1]);

	a0.val[0] = t0;
	a1.val[0] = t1;
	a2.val[1] = t2;
	a3.val[1] = t3;

	t0 = vtrn1q_f64(a2.val[0], a3.val[0]);
	t1 = vtrn2q_f64(a2.val[0], a3.val[0]);
	t2 = vtrn1q_f64(a0.val[1], a1.val[1]);
	t3 = vtrn2q_f64(a0.val[1], a1.val[1]);

	a0.val[1] = t0;
	a1.val[1] = t1;
	a2.val[0] = t2;
	a3.val[0] = t3;
}

template<>
PERFLIBS_LINALG_INLINE
void transpose(float32x4x2_t& a0, float32x4x2_t& a1, float32x4x2_t& a2, float32x4x2_t& a3) {
	// transpose values in registers a[0-3], real and imaginary part separately
	transpose(a0.val[0], a1.val[0], a2.val[0], a3.val[0]);
	transpose(a0.val[1], a1.val[1], a2.val[1], a3.val[1]);
}

template<>
PERFLIBS_LINALG_INLINE
void transpose(float64x2x4_t& a0, float64x2x4_t& a1, float64x2x4_t& a2, float64x2x4_t& a3) {
	// transpose values in registers a[0-3], real and imaginary part separately
	float64x2x2_t t0, t1, t2, t3;

	t0.val[0] = a0.val[0]; t0.val[1] = a0.val[1];
	t1.val[0] = a1.val[0]; t1.val[1] = a1.val[1];
	t2.val[0] = a2.val[0]; t2.val[1] = a2.val[1];
	t3.val[0] = a3.val[0]; t3.val[1] = a3.val[1];
	transpose(t0, t1, t2, t3);
	a0.val[0] = t0.val[0]; a0.val[1] = t0.val[1];
	a1.val[0] = t1.val[0]; a1.val[1] = t1.val[1];
	a2.val[0] = t2.val[0]; a2.val[1] = t2.val[1];
	a3.val[0] = t3.val[0]; a3.val[1] = t3.val[1];

	t0.val[0] = a0.val[2]; t0.val[1] = a0.val[3];
	t1.val[0] = a1.val[2]; t1.val[1] = a1.val[3];
	t2.val[0] = a2.val[2]; t2.val[1] = a2.val[3];
	t3.val[0] = a3.val[2]; t3.val[1] = a3.val[3];
	transpose(t0, t1, t2, t3);
	a0.val[2] = t0.val[0]; a0.val[3] = t0.val[1];
	a1.val[2] = t1.val[0]; a1.val[3] = t1.val[1];
	a2.val[2] = t2.val[0]; a2.val[3] = t2.val[1];
	a3.val[2] = t3.val[0]; a3.val[3] = t3.val[1];
}

template<typename VecType>
PERFLIBS_LINALG_INLINE
void recip_diagonal(VecType&, VecType&, VecType&, VecType&);

template<>
PERFLIBS_LINALG_INLINE
void recip_diagonal(float32x4_t& a0, float32x4_t& a1, float32x4_t& a2, float32x4_t& a3) {
	// take reciprocal of diagonal elements of a
	auto r0 = float{1} / vgetq_lane_f32(a0, 0);
	auto r1 = float{1} / vgetq_lane_f32(a1, 1);
	auto r2 = float{1} / vgetq_lane_f32(a2, 2);
	auto r3 = float{1} / vgetq_lane_f32(a3, 3);
	a0 = vsetq_lane_f32(r0, a0, 0);
	a1 = vsetq_lane_f32(r1, a1, 1);
	a2 = vsetq_lane_f32(r2, a2, 2);
	a3 = vsetq_lane_f32(r3, a3, 3);
}

template<>
PERFLIBS_LINALG_INLINE
void recip_diagonal(float64x2x2_t& a0, float64x2x2_t& a1, float64x2x2_t& a2, float64x2x2_t& a3) {
	// take reciprocal of diagonal elements of a
	auto r0 = double{1} / vgetq_lane_f64(a0.val[0], 0);
	auto r1 = double{1} / vgetq_lane_f64(a1.val[0], 1);
	auto r2 = double{1} / vgetq_lane_f64(a2.val[1], 0);
	auto r3 = double{1} / vgetq_lane_f64(a3.val[1], 1);
	a0.val[0] = vsetq_lane_f64(r0, a0.val[0], 0);
	a1.val[0] = vsetq_lane_f64(r1, a1.val[0], 1);
	a2.val[1] = vsetq_lane_f64(r2, a2.val[1], 0);
	a3.val[1] = vsetq_lane_f64(r3, a3.val[1], 1);
}

template<>
PERFLIBS_LINALG_INLINE
void recip_diagonal(float32x4x2_t& a0, float32x4x2_t& a1, float32x4x2_t& a2, float32x4x2_t& a3) {
	// take reciprocal of diagonal elements of a
	auto r0 = complex_float{1} / complex_float{vgetq_lane_f32(a0.val[0], 0), vgetq_lane_f32(a0.val[1], 0)};
	auto r1 = complex_float{1} / complex_float{vgetq_lane_f32(a1.val[0], 1), vgetq_lane_f32(a1.val[1], 1)};
	auto r2 = complex_float{1} / complex_float{vgetq_lane_f32(a2.val[0], 2), vgetq_lane_f32(a2.val[1], 2)};
	auto r3 = complex_float{1} / complex_float{vgetq_lane_f32(a3.val[0], 3), vgetq_lane_f32(a3.val[1], 3)};

	// set real parts
	a0.val[0] = vsetq_lane_f32(r0.real(), a0.val[0], 0);
	a1.val[0] = vsetq_lane_f32(r1.real(), a1.val[0], 1);
	a2.val[0] = vsetq_lane_f32(r2.real(), a2.val[0], 2);
	a3.val[0] = vsetq_lane_f32(r3.real(), a3.val[0], 3);

	// set imaginary parts
	a0.val[1] = vsetq_lane_f32(r0.imag(), a0.val[1], 0);
	a1.val[1] = vsetq_lane_f32(r1.imag(), a1.val[1], 1);
	a2.val[1] = vsetq_lane_f32(r2.imag(), a2.val[1], 2);
	a3.val[1] = vsetq_lane_f32(r3.imag(), a3.val[1], 3);
}

template<>
PERFLIBS_LINALG_INLINE
void recip_diagonal(float64x2x4_t& a0, float64x2x4_t& a1, float64x2x4_t& a2, float64x2x4_t& a3) {
	// take reciprocal of diagonal elements of a
	auto r0 = complex_double{1} / complex_double{vgetq_lane_f64(a0.val[0], 0), vgetq_lane_f64(a0.val[2], 0)};
	auto r1 = complex_double{1} / complex_double{vgetq_lane_f64(a1.val[0], 1), vgetq_lane_f64(a1.val[2], 1)};
	auto r2 = complex_double{1} / complex_double{vgetq_lane_f64(a2.val[1], 0), vgetq_lane_f64(a2.val[3], 0)};
	auto r3 = complex_double{1} / complex_double{vgetq_lane_f64(a3.val[1], 1), vgetq_lane_f64(a3.val[3], 1)};

	// set real parts
	a0.val[0] = vsetq_lane_f64(r0.real(), a0.val[0], 0);
	a1.val[0] = vsetq_lane_f64(r1.real(), a1.val[0], 1);
	a2.val[1] = vsetq_lane_f64(r2.real(), a2.val[1], 0);
	a3.val[1] = vsetq_lane_f64(r3.real(), a3.val[1], 1);

	// set imaginary parts
	a0.val[2] = vsetq_lane_f64(r0.imag(), a0.val[2], 0);
	a1.val[2] = vsetq_lane_f64(r1.imag(), a1.val[2], 1);
	a2.val[3] = vsetq_lane_f64(r2.imag(), a2.val[3], 0);
	a3.val[3] = vsetq_lane_f64(r3.imag(), a3.val[3], 1);
}

} // anonymous namespace
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_TRSM_KERNEL_HELPERS_HPP

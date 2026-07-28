/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_ROT_KERNELS_HPP
#define PERFLIBS_LINALG_ROT_KERNELS_HPP

#include "perflibs_complex.hpp"
#include "perflibs_util.hpp"
#include "framework/rot_fallback_kernel.hpp"
#include "framework/rot_neon_kernel.hpp"
#include "framework/rot_sve_kernel.hpp"
#include "framework/linalg_util.hpp"
#include "spec/generic_machine_spec.hpp"

#include <climits>
#include <cmath>
#include <cstdlib>

namespace perflibs::linalg {
namespace {

template<typename T1, typename T2>
using rot_kernel_t = void(kernel_inttype n, T1 *x, T1 *y, const kernel_inttype incx,
                          const kernel_inttype incy, const remove_complex_t<T1> c, const T2 s);

template<typename T>
using rotg_kernel_t = void(T *a, T *b, perflibs::remove_complex_t<T> *c, T *s);

template<typename T>
using rotm_kernel_t = void (
	kernel_inttype n,
	T *x, kernel_inttype incx,
	T *y, kernel_inttype incy,
	const T *param);

template<typename T>
using rotmg_kernel_t = void(T *d1, T *d2, T *x, const T *y, T *param);

// Scaling constants calculated from https://doi.org/10.1145/3061665
template<typename T>
struct rotg_scale_constants;

template<>
struct rotg_scale_constants<float> {
	constexpr static float SAF_MIN = 0x1p-126;
	constexpr static float SAF_MAX = 0x1p+126;
	constexpr static float RT_MIN = 0x1.cp-63; // sqrt(SAF_MIN)
};

template<>
struct rotg_scale_constants<double> {
	constexpr static double SAF_MIN = 0x1p-1022;
	constexpr static double SAF_MAX = 0x1p+1022;
	constexpr static double RT_MIN =  0x1.fp-511; //std::sqrt(SAF_MIN);
};

template<typename T>
inline static void rotg_kernel(T *a, T *b, perflibs::remove_complex_t<T> *c, T *s) {
	using sc = rotg_scale_constants<remove_complex_t<T>>;

	// check if value is complex
	if constexpr (perflibs::is_complex_v<T>) {
		T f = *a;
		T g = *b;

		T r;

		if (g == zero<T>) {
			*c = 1.0;
			*s = 0.0;
			r = f;
		}
		else if (f == zero<T>) {
			*c = 0.0;
			if (g.real() == 0.0) {
				r = std::abs(g.imag());
				*s = std::conj(g) / r;
			}
			else if (g.imag() == 0.0) {
				r = std::abs(g.real());
				*s = std::conj(g) / r;
			}
			else {
				auto g_max = std::max(std::abs(g.real()), std::abs(g.imag()));
				auto rtmax = std::sqrt(sc::SAF_MAX / 2);
				if (g_max > sc::RT_MIN && g_max < rtmax) { // well scaled
					r = std::abs(g);
					*s = std::conj(g) / r;
				}
				else { // use scaled algorithm
					auto u = std::min(sc::SAF_MAX, std::max(sc::SAF_MIN, g_max));
					auto g_s = g / u;
					auto d = std::abs(g_s);
					*s = std::conj(g_s) / d;
					r = (is_real_nan(g_s) || is_imag_nan(g_s))
					        ? std::numeric_limits<remove_complex_t<T>>::quiet_NaN()
					        : d * u;
				}
			}
		} // f ! = zero
		else {
			auto f_max = std::max(std::abs(f.real()), std::abs(f.imag()));
			auto g_max = std::max(std::abs(g.real()), std::abs(g.imag()));
			auto rtmax = std::sqrt(sc::SAF_MAX / 4);
			if (f_max > sc::RT_MIN && f_max < rtmax && g_max > sc::RT_MIN && g_max < rtmax) { // well scaled

				// Use unscaled allgorithm
				auto f2 = f.real() * f.real() + f.imag() * f.imag();
				auto g2 = g.real() * g.real() + g.imag() * g.imag();
				auto h2 = f2 + g2;
				// safmin <= f2 <= h2 <= safmax
				if (f2 >= h2 * sc::SAF_MIN) {
					// safmin <= f2/h2 <= 1, and h2/f2 is finite
					*c = std::sqrt(f2 / h2);
					r = f / *c;
					rtmax = rtmax * 2;
					if (f2 > sc::RT_MIN && h2 < rtmax) {
						// safmin <= sqrt( f2*h2 ) <= safmax
						*s = std::conj(g) * (f / std::sqrt(f2 * h2));
					}
					else {
						*s = std::conj(g) * (r / h2);
					}
				}
				else {
					// f2/h2 <= safmin may be subnormal, and h2/f2 may overflow.
					// Moreover,
					// safmin <= f2*f2 * safmax < f2 * h2 < h2*h2 * safmin <= safmax,
					// sqrt(safmin) <= sqrt(f2 * h2) <= sqrt(safmax).
					// Also, g2 >> f2, which means that h2 = g2.
					auto d = std::sqrt(f2 * h2);
					*c = f2 / d;
					if (*c >= sc::SAF_MIN) {
						r = f / *c;
					}
					else {
						// f2 / sqrt(f2 * h2) < safmin, then
						// sqrt(safmin) <= f2 * sqrt(safmax) <= h2 / sqrt(f2 * h2) <= h2 * (safmin / f2) <= h2
						// <= safmax
						r = f * (h2 / d);
					}
					*s = std::conj(g) * (f / d);
				}
			}
			else {

				// Use scaled algorithm
				auto u = std::min(sc::SAF_MAX, std::max({ sc::SAF_MIN, f_max, g_max }));
				auto g_s = g / u;
				auto g2 = g_s.real() * g_s.real() + g_s.imag() * g_s.imag();

				remove_complex_t<T> w, f2, h2;
				T f_s;
				if (f_max / u < sc::RT_MIN) {
					auto v = std::min(sc::SAF_MAX, std::max(sc::SAF_MIN, f_max));
					w = v / u;
					f_s = f / v;
					f2 = f_s.real() * f_s.real() + f_s.imag() * f_s.imag();
					h2 = f2 * w * w + g2;
				}
				else {
					w = 1;
					f_s = f / u;
					f2 = f_s.real() * f_s.real() + f_s.imag() * f_s.imag();
					h2 = f2 + g2;
				}
				// safmin <= f2 <= h2 <= safmax
				if (f2 >= h2 * sc::SAF_MIN) {
					// safmin <= f2/h2 <= 1, and h2/f2 is finite
					*c = std::sqrt(f2 / h2);
					r = f_s / *c;
					rtmax = rtmax * 2;
					if (f2 > sc::RT_MIN && h2 < rtmax) {
						// safmin <= sqrt( f2*h2 ) <= safmax
						*s = std::conj(g_s) * (f_s / std::sqrt(f2 * h2));
					}
					else {
						*s = std::conj(g_s) * (r / h2);
					}
				}
				else {
					// f2/h2 <= safmin may be subnormal, and h2/f2 may overflow.
					// Moreover,
					// safmin <= f2*f2 * safmax < f2 * h2 < h2*h2 * safmin <= safmax,
					// sqrt(safmin) <= sqrt(f2 * h2) <= sqrt(safmax).
					// Also,
					// g2 >> f2, which means that h2 = g2.
					auto d = std::sqrt(f2 * h2);
					*c = f2 / d;
					if (*c >= sc::SAF_MIN) {
						r = f_s / *c;
					}
					else {
						// f2 / sqrt(f2 * h2) < safmin, then
						// sqrt(safmin) <= f2 * sqrt(safmax) <= h2 / sqrt(f2 * h2) <= h2 * (safmin / f2) <= h2
						// <= safmax
						r = f_s * (h2 / d);
					}
					*s = std::conj(g_s) * (f_s / d);
				}
				// Rescale c and r
				*c = *c * w;
				r = r * u;
			}
		}
		*a = r;
	}
	else { // Real case
		if (*b == zero<T>) {
			*c = 1.0;
			*s = 0.0;
			*b = 0.0;
		}
		else if (*a == zero<T>) {
			*c = 0.0;
			*s = 1.0;
			*a = *b;
			*b = 1.0;
		}
		else {
			T scale = std::min(sc::SAF_MAX, std::max({ sc::SAF_MIN, std::abs(*a), std::abs(*b) }));
			T roe = std::abs(*a) > std::abs(*b) ? *a : *b;

			T r = scale * std::sqrt(std::pow(*a / scale, 2) + std::pow(*b / scale, 2));
			r = copysign(1.0, roe) * r;
			*c = *a / r;
			*s = *b / r;
			T z = std::abs(*a) > std::abs(*b) ? *s : T(1.0);
			z = std::abs(*b) >= std::abs(*a) && *c != 0 ? T(1.0) / *c : z;
			*a = r;
			*b = z;
		}
	}
}

template<typename T>
PERFLIBS_LINALG_INLINE
void rotm_kernel(kernel_inttype n, T *x, kernel_inttype incx, T *y, kernel_inttype incy, const T *param) {
	const auto flag = param[0];
	const auto sh11 = param[1];
	const auto sh12 = param[3];
	const auto sh21 = param[2];
	const auto sh22 = param[4];
	if (n <= 0 || (flag + 2.0) == 0) {
		return;
	}
	if (incx == incy && incx > 0) {
		int_type nsteps = n * incx;
		if (flag < 0) {
			for (int i = 0; i < nsteps; i += incx) {
				T w = x[i];
				T z = y[i];
				x[i] = w * sh11 + z * sh12;
				y[i] = w * sh21 + z * sh22;
			}
		}
		else if (flag == 0) {
			for (int i = 0; i < nsteps; i += incx) {
				T w = x[i];
				T z = y[i];
				x[i] = w + z * sh12;
				y[i] = w * sh21 + z;
			}
		}
		else {
			for (int i = 0; i < nsteps; i += incx) {
				T w = x[i];
				T z = y[i];
				x[i] = w * sh11 + z;
				y[i] = -w + z * sh22;
			}
		}
	}
	else {
		int_type kx = incx < 0 ? (1 - n) * incx : 0;
		int_type ky = incy < 0 ? (1 - n) * incy : 0;

		if (flag < 0) {
			for (int_type i = 0; i < n; i++) {
				T w = x[kx];
				T z = y[ky];
				x[kx] = w * sh11 + z * sh12;
				y[ky] = w * sh21 + z * sh22;
				kx += incx;
				ky += incy;
			}
		}
		else if (flag == 0) {
			for (int_type i = 0; i < n; i++) {
				T w = x[kx];
				T z = y[ky];
				x[kx] = w + z * sh12;
				y[ky] = w * sh21 + z;
				kx += incx;
				ky += incy;
			}
		}
		else {
			for (int_type i = 0; i < n; i++) {
				T w = x[kx];
				T z = y[ky];
				x[kx] = w * sh11 + z;
				y[ky] = -w + z * sh22;
				kx += incx;
				ky += incy;
			}
		}
	}
}

template<typename T>
static void rotmg_kernel(T *d1, T *d2, T *x, const T *y, T *param) {
	const auto gam = 4096;
	const auto gamsq = 1.67772e7;
	const auto rgamsq = 5.96046e-8;
	T sflag = -1;
	T sh11 = 0;
	T sh12 = 0;
	T sh21 = 0;
	T sh22 = 0;
	if (*d1 < 0) {
		*d1 = 0;
		*d2 = 0;
		*x = 0;
	}
	else {
		T p2 = *d2 * *y;
		if (p2 == 0) {
			sflag = -2;
			param[0] = sflag;
			return;
		}
		T p1 = *d1 * *x;
		T q2 = p2 * *y;
		T q1 = p1 * *x;
		if (std::abs(q1) > std::abs(q2)) {
			sh21 = -*y / *x;
			sh12 = p2 / p1;

			T su = 1 - (sh12 * sh21);
			if (su > 0) {
				sflag = 0;
				*d1 = *d1 / su;
				*d2 = *d2 / su;
				*x = *x * su;
			}
		}
		else {
			if (q2 < 0) {
				sflag = -1;
				sh11 = 0;
				sh12 = 0;
				sh21 = 0;
				sh22 = 0;

				*d1 = 0;
				*d2 = 0;
				*x = 0;
			}
			else {
				sflag = 1;
				sh11 = p1 / p2;
				sh22 = *x / *y;
				T su = 1 + sh11 * sh22;
				T stemp = *d2 / su;
				*d2 = *d1 / su;
				*d1 = stemp;
				*x = *y * su;
			}
		}
		if (*d1 != 0) {
			while (*d1 <= rgamsq || *d1 >= gamsq) {
				if (sflag == 0) {
					sh11 = 1;
					sh22 = 1;
					sflag = -1;
				}
				else {
					sh21 = -1;
					sh12 = 1;
					sflag = -1;
				}
				if (*d1 <= rgamsq) {
					*d1 = *d1 * (gam * gam);
					*x = *x / gam;
					sh11 = sh11 / gam;
					sh12 = sh12 / gam;
				}
				else {
					*d1 = *d1 / (gam * gam);
					*x = *x * gam;
					sh11 = sh11 * gam;
					sh12 = sh12 * gam;
				}
			}
		}
		if (*d2 != 0) {
			while (std::abs(*d2) <= rgamsq || std::abs(*d2) >= gamsq) {
				if (sflag == 0) {
					sh11 = 1;
					sh22 = 1;
					sflag = -1;
				}
				else {
					sh21 = -1;
					sh12 = 1;
					sflag = -1;
				}
				if (std::abs(*d2) <= rgamsq) {
					*d2 = *d2 * (gam * gam);
					sh21 = sh21 / gam;
					sh22 = sh22 / gam;
				}
				else {
					*d2 = *d2 / (gam * gam);
					sh21 = sh21 * gam;
					sh22 = sh22 * gam;
				}
			}
		}
	}
	if (sflag < 0) {
		param[1] = sh11;
		param[2] = sh21;
		param[3] = sh12;
		param[4] = sh22;
	}
	else if (sflag == 0) {
		param[2] = sh21;
		param[3] = sh12;
	}
	else {
		param[1] = sh11;
		param[4] = sh22;
	}

	param[0] = sflag;
}

} // namespace
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_ROT_KERNELS_HPP

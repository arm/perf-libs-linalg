/*
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

// Tests for ?spffrt2 and ?spffrtx: comparison with a long double reference, L D L^T
// reconstruction, zero pivots, work arrays and quick returns.

#include "pl_linalg_lapack.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <vector>

using int_t = pl_linalg_int_t;

static void spffrt2(const int_t *n, const int_t *k, float *a, float *w, float *w2) { sspffrt2_(a, n, k, w, w2); }
static void spffrt2(const int_t *n, const int_t *k, double *a, double *w, double *w2) { dspffrt2_(a, n, k, w, w2); }
static void spffrt2(const int_t *n, const int_t *k, std::complex<float> *a, std::complex<float> *w, std::complex<float> *w2) { cspffrt2_(a, n, k, w, w2); }
static void spffrt2(const int_t *n, const int_t *k, std::complex<double> *a, std::complex<double> *w, std::complex<double> *w2) { zspffrt2_(a, n, k, w, w2); }
static void spffrtx(const int_t *n, const int_t *k, float *a, float *w, float *w2) { sspffrtx_(a, n, k, w, w2); }
static void spffrtx(const int_t *n, const int_t *k, double *a, double *w, double *w2) { dspffrtx_(a, n, k, w, w2); }
static void spffrtx(const int_t *n, const int_t *k, std::complex<float> *a, std::complex<float> *w, std::complex<float> *w2) { cspffrtx_(a, n, k, w, w2); }
static void spffrtx(const int_t *n, const int_t *k, std::complex<double> *a, std::complex<double> *w, std::complex<double> *w2) { zspffrtx_(a, n, k, w, w2); }

template<typename T> struct type_name;
template<> struct type_name<float> { static constexpr const char *value = "S"; };
template<> struct type_name<double> { static constexpr const char *value = "D"; };
template<> struct type_name<std::complex<float>> { static constexpr const char *value = "C"; };
template<> struct type_name<std::complex<double>> { static constexpr const char *value = "Z"; };

template<typename T> using real_t = decltype(std::abs(T{}));
// reference arithmetic: long double, complex<long double> for complex types
template<typename T> using wide_t = std::conditional_t<std::is_floating_point_v<T>, long double, std::complex<long double>>;

static int failures = 0;
static int checks = 0;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static unsigned long long rng_state = 88172645463325252ULL;
static double uniform() {
	rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
	return static_cast<double>(rng_state >> 11) / 9007199254740992.0 - 0.5;
}
template<typename T> static T random_value() {
	if constexpr (std::is_floating_point_v<T>) return static_cast<T>(uniform());
	else return T(static_cast<real_t<T>>(uniform()), static_cast<real_t<T>>(uniform()));
}

// packed offset of A(i,j), i >= j, lower triangle packed by columns
static std::size_t at(int_t i, int_t j, int_t n) {
	return static_cast<std::size_t>(j) * n - static_cast<std::size_t>(j) * (j - 1) / 2 + (i - j);
}

// symmetric (complex symmetric) diagonally dominant matrix, lower packed
template<typename T>
static std::vector<T> random_packed(int_t n) {
	std::vector<T> a(static_cast<std::size_t>(n) * (n + 1) / 2);
	for (int_t j = 0; j < n; j++) {
		for (int_t i = j; i < n; i++) a[at(i, j, n)] = random_value<T>();
		a[at(j, j, n)] += T(static_cast<real_t<T>>(n + 1));
	}
	return a;
}

// reference elimination in long double: inverse_diagonal selects the ?spffrt2 form
template<typename T>
static std::vector<wide_t<T>> reference(const std::vector<T>& a0, int_t n, int_t k, bool inverse_diagonal) {
	using W = wide_t<T>;
	std::vector<W> a(a0.begin(), a0.end());
	for (int_t c = 0; c < k; c++) {
		const W d    = a[at(c, c, n)];
		const W rcpd = (d == W(0)) ? W(0) : W(1) / d;   // a zero pivot updates nothing
		for (int_t j = c + 1; j < n; j++) {
			const W ljc = a[at(j, c, n)] * rcpd;
			for (int_t i = j; i < n; i++) a[at(i, j, n)] -= a[at(i, c, n)] * ljc;
		}
		if (inverse_diagonal) a[at(c, c, n)] = rcpd;
		else for (int_t i = c + 1; i < n; i++) a[at(i, c, n)] *= rcpd;
	}
	return a;
}

// max over packed columns j of max_i |x(i,j) - y(i,j)| / max_i |y(i,j)|, and whether every x is finite
template<typename T, typename U>
static double max_rel_diff(const std::vector<T>& x, const std::vector<U>& y, int_t n, bool *finite = nullptr) {
	double worst = 0; bool fin = true;
	for (int_t j = 0; j < n; j++) {
		long double d = 0, s = 0;
		for (int_t i = j; i < n; i++) {
			const std::size_t p = at(i, j, n);
			fin = fin && std::isfinite(static_cast<double>(std::abs(x[p])));
			d = std::max(d, static_cast<long double>(std::abs(wide_t<T>(x[p]) - wide_t<T>(y[p]))));
			s = std::max(s, static_cast<long double>(std::abs(wide_t<T>(y[p]))));
		}
		worst = std::max(worst, static_cast<double>(s > 0 ? d / s : d));
	}
	if (finite) *finite = fin;
	return worst;
}

template<typename T>
static void test_random() {
	const int_t cases[][2] = {
		{ 1, 1 }, { 2, 1 }, { 5, 3 }, { 17, 17 }, { 40, 0 }, { 40, 11 },
		{ 95, 40 }, { 96, 15 }, { 96, 16 }, { 130, 64 }, { 200, 65 }, { 257, 100 }, { 300, 150 }, { 300, 300 },
		{ 400, 128 }, { 400, 400 }
	};
	const double eps = std::numeric_limits<real_t<T>>::epsilon();
	for (const auto& c : cases) {
		const int_t n = c[0], k = c[1];
		const auto a0 = random_packed<T>(n);
		for (int form = 0; form < 2; form++) {
			auto a = a0;
			std::vector<T> w(n + 1, T(-777)), w2(n + 1, T(-777));
			if (form == 0) spffrt2(&n, &k, a.data(), w.data(), w2.data());
			else           spffrtx(&n, &k, a.data(), w.data(), w2.data());
			const auto ref = reference(a0, n, k, form == 0);
			const double e = max_rel_diff(a, ref, n);
			CHECK(e < 30 * eps * std::sqrt(static_cast<double>(n)), "%s%s n=%d ncolm=%d: max relative difference %.3e",
			      type_name<T>::value, form == 0 ? "SPFFRT2" : "SPFFRTX", (int)n, (int)k, e);
			CHECK(std::all_of(w.begin(), w.end(), [](T v) { return v == T(-777); }) &&
			      std::all_of(w2.begin(), w2.end(), [](T v) { return v == T(-777); }),
			      "%s%s n=%d: work arrays referenced", type_name<T>::value, form == 0 ? "SPFFRT2" : "SPFFRTX", (int)n);

			// complete factorization: rebuild A = L * D * L^T from the ?spffrtx output
			if (form == 1 && k == n) {
				long double emax = 0, amax = 0;
				for (int_t j = 0; j < n; j++) {
					for (int_t i = j; i < n; i++) {
						wide_t<T> s = 0;
						for (int_t p = 0; p <= j; p++) {
							const wide_t<T> lip = (i == p) ? wide_t<T>(1) : wide_t<T>(a[at(i, p, n)]);
							const wide_t<T> ljp = (j == p) ? wide_t<T>(1) : wide_t<T>(a[at(j, p, n)]);
							s += lip * wide_t<T>(a[at(p, p, n)]) * ljp;
						}
						emax = std::max(emax, static_cast<long double>(std::abs(s - wide_t<T>(a0[at(i, j, n)]))));
						amax = std::max(amax, static_cast<long double>(std::abs(wide_t<T>(a0[at(i, j, n)]))));
					}
				}
				const double r = static_cast<double>(emax / amax);
				CHECK(r < 30 * eps * std::sqrt(static_cast<double>(n)), "%sSPFFRTX n=%d: ||L D L^T - A|| / ||A|| = %.3e",
				      type_name<T>::value, (int)n, r);
			}
		}
	}
}

// zero pivots in random data on the blocked path
template<typename T>
static void test_zero_pivot_blocked() {
	const int_t cases[][3] = { { 255, 200, 0 }, { 255, 200, 15 }, { 255, 200, 16 }, { 255, 200, 63 }, { 255, 200, 64 },
	                           { 255, 200, 100 }, { 255, 200, 199 }, { 400, 128, 70 }, { 96, 16, 15 } };
	const double eps = std::numeric_limits<real_t<T>>::epsilon();
	for (const auto& c : cases) {
		const int_t n = c[0], k = c[1], z = c[2];
		auto a0 = random_packed<T>(n);
		for (int_t j = 0; j < z; j++) a0[at(z, j, n)] = T(0);
		a0[at(z, z, n)] = T(0);
		for (int form = 0; form < 2; form++) {
			auto a = a0;
			std::vector<T> w(n), w2(n);
			if (form == 0) spffrt2(&n, &k, a.data(), w.data(), w2.data());
			else           spffrtx(&n, &k, a.data(), w.data(), w2.data());
			bool finite = false;
			const double e = max_rel_diff(a, reference(a0, n, k, form == 0), n, &finite);
			CHECK(finite && e < 30 * eps * std::sqrt(static_cast<double>(n)), "%s%s n=%d ncolm=%d zero pivot %d: difference %.3e, finite %d",
			      type_name<T>::value, form == 0 ? "SPFFRT2" : "SPFFRTX", (int)n, (int)k, (int)z, e, (int)finite);
		}
	}
}

// exactly zero pivots: small cases with known results
template<typename T>
static void test_zero_pivot() {
	struct zcase { int_t n, k; std::vector<double> a, out2, outx; };
	const zcase cases[] = {
		{ 1, 1, { 0 },                  { 0 },                        { 0 } },
		{ 2, 2, { 0, 1, 2 },            { 0, 1, 0.5 },                { 0, 0, 2 } },
		{ 3, 3, { 1, 2, 3, 4, 6, 9 },   { 1, 2, 3, 0, 0, 0 },         { 1, 2, 3, 0, 0, 0 } },
		{ 3, 1, { 1, 2, 3, 4, 6, 9 },   { 1, 2, 3, 0, 0, 0 },         { 1, 2, 3, 0, 0, 0 } },
		{ 3, 3, { 4, 2, 2, 1, 1, 1 },   { 0.25, 2, 2, 0, 0, 0 },      { 4, 0.5, 0.5, 0, 0, 0 } },
		{ 3, 3, { 0, 1, 1, 2, 1, 3 },   { 0, 1, 1, 0.5, 1, 0.4 },     { 0, 0, 0, 2, 0.5, 2.5 } },
	};
	const double eps = std::numeric_limits<real_t<T>>::epsilon();
	for (const auto& c : cases) {
		for (int form = 0; form < 2; form++) {
			std::vector<T> a(c.a.begin(), c.a.end()), w(c.n), w2(c.n);
			if (form == 0) spffrt2(&c.n, &c.k, a.data(), w.data(), w2.data());
			else           spffrtx(&c.n, &c.k, a.data(), w.data(), w2.data());
			const auto& expect = form == 0 ? c.out2 : c.outx;
			bool ok = true;
			for (std::size_t i = 0; i < a.size(); i++) {
				ok = ok && std::isfinite(std::abs(a[i])) && std::abs(a[i] - T(static_cast<real_t<T>>(expect[i]))) <= 4 * eps * std::max(1.0, std::abs(expect[i]));
			}
			CHECK(ok, "%s%s zero pivot n=%d ncolm=%d: result differs from the expected values", type_name<T>::value, form == 0 ? "SPFFRT2" : "SPFFRTX", (int)c.n, (int)c.k);
		}
	}
}

template<typename T>
static void test_quick_return() {
	const auto a0 = random_packed<T>(7);
	const int_t zero = 0, seven = 7;
	std::vector<T> w(8), w2(8);
	auto a = a0;
	spffrt2(&zero, &seven, a.data(), w.data(), w2.data());
	spffrtx(&zero, &seven, a.data(), w.data(), w2.data());
	spffrt2(&seven, &zero, a.data(), w.data(), w2.data());
	spffrtx(&seven, &zero, a.data(), w.data(), w2.data());
	CHECK(a == a0, "%sSPFFRT2/X: n = 0 or ncolm = 0 changed A", type_name<T>::value);
}

template<typename T>
static void test_all() {
	test_random<T>();
	test_zero_pivot<T>();
	test_zero_pivot_blocked<T>();
	test_quick_return<T>();
}

int main() {
	test_all<float>();
	test_all<double>();
	test_all<std::complex<float>>();
	test_all<std::complex<double>>();
	std::printf("%d of %d checks failed\n", failures, checks);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

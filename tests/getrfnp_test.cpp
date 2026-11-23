/*
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

// Tests for ?getrfnp and ?getrfnpi: reconstruction, lda padding, zero pivots, quick returns
// and argument checks.

#include "pl_linalg_lapack.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using int_t = pl_linalg_int_t;

// Override the weak xerbla in the library to record the reported errors.
#if (defined(__clang__) || __GNUC__ > 7) && !defined(__APPLE__)
typedef size_t charlen_t;
#else
typedef int charlen_t;
#endif
static std::string xerbla_name;
static int_t xerbla_info = 0;
extern "C" void xerbla_(const char *srname, const int_t *info, charlen_t len) {
	xerbla_name.assign(srname, static_cast<size_t>(len));
	while (!xerbla_name.empty() && xerbla_name.back() == ' ') xerbla_name.pop_back();
	xerbla_info = *info;
}

static void getrfnp(const int_t *m, const int_t *n, float *a, const int_t *lda, int_t *info) { sgetrfnp_(m, n, a, lda, info); }
static void getrfnp(const int_t *m, const int_t *n, double *a, const int_t *lda, int_t *info) { dgetrfnp_(m, n, a, lda, info); }
static void getrfnp(const int_t *m, const int_t *n, std::complex<float> *a, const int_t *lda, int_t *info) { cgetrfnp_(m, n, a, lda, info); }
static void getrfnp(const int_t *m, const int_t *n, std::complex<double> *a, const int_t *lda, int_t *info) { zgetrfnp_(m, n, a, lda, info); }
static void getrfnpi(const int_t *m, const int_t *n, const int_t *k, float *a, const int_t *lda, int_t *info) { sgetrfnpi_(m, n, k, a, lda, info); }
static void getrfnpi(const int_t *m, const int_t *n, const int_t *k, double *a, const int_t *lda, int_t *info) { dgetrfnpi_(m, n, k, a, lda, info); }
static void getrfnpi(const int_t *m, const int_t *n, const int_t *k, std::complex<float> *a, const int_t *lda, int_t *info) { cgetrfnpi_(m, n, k, a, lda, info); }
static void getrfnpi(const int_t *m, const int_t *n, const int_t *k, std::complex<double> *a, const int_t *lda, int_t *info) { zgetrfnpi_(m, n, k, a, lda, info); }

template<typename T> struct type_name;
template<> struct type_name<float> { static constexpr const char *value = "S"; };
template<> struct type_name<double> { static constexpr const char *value = "D"; };
template<> struct type_name<std::complex<float>> { static constexpr const char *value = "C"; };
template<> struct type_name<std::complex<double>> { static constexpr const char *value = "Z"; };

template<typename T> using real_t = decltype(std::abs(T{}));

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

constexpr double pad_value = 777.0;

// column-major m x n matrix with leading dimension lda; rows m..lda-1 hold pad_value
template<typename T>
static std::vector<T> diagonally_dominant(int_t m, int_t n, int_t lda) {
	std::vector<T> a(static_cast<size_t>(lda) * std::max<int_t>(n, 1), T(pad_value));
	for (int_t j = 0; j < n; j++)
		for (int_t i = 0; i < m; i++)
			a[j * lda + i] = random_value<T>();
	for (int_t i = 0; i < std::min(m, n); i++)
		a[i * lda + i] += T(static_cast<real_t<T>>(std::max(m, n)));
	return a;
}

// max | [L 0; L2 I][U U2; 0 S] - A0 | / (max(m,n) * eps * max|A0|) with k factored columns
template<typename T>
static double reconstruction_error(int_t m, int_t n, int_t k, const std::vector<T>& f, const std::vector<T>& a0, int_t lda) {
	double amax = 0, emax = 0;
	for (int_t j = 0; j < n; j++) {
		for (int_t i = 0; i < m; i++) {
			T r = (i >= k && j >= k) ? f[j * lda + i] : T(0);
			for (int_t p = 0; p < k && p <= i && p <= j; p++) {
				T l = (i == p) ? T(1) : f[p * lda + i];
				r += l * f[j * lda + p];
			}
			emax = std::max(emax, static_cast<double>(std::abs(r - a0[j * lda + i])));
			amax = std::max(amax, static_cast<double>(std::abs(a0[j * lda + i])));
		}
	}
	const double eps = std::numeric_limits<real_t<T>>::epsilon();
	return amax == 0 ? emax : emax / (std::max<int_t>(std::max(m, n), 1) * eps * amax);
}

template<typename T>
static bool padding_untouched(int_t m, int_t n, int_t lda, const std::vector<T>& f) {
	for (int_t j = 0; j < n; j++)
		for (int_t i = m; i < lda; i++)
			if (f[j * lda + i] != T(pad_value)) return false;
	return true;
}

template<typename T>
static void test_random() {
	const int_t shapes[][2] = { {1, 1}, {2, 3}, {3, 2}, {5, 5}, {16, 16}, {17, 17}, {40, 25}, {25, 40},
	                            {64, 64}, {100, 100}, {257, 257}, {300, 120}, {120, 300} };
	const double threshold = 30.0;
	for (const auto& s : shapes) {
		const int_t m = s[0], n = s[1], k = std::min(m, n);
		for (int_t pad : { 0, 5 }) {
			const int_t lda = std::max<int_t>(m + pad, 1);
			const auto a0 = diagonally_dominant<T>(m, n, lda);

			auto f = a0; int_t info = -99;
			getrfnp(&m, &n, f.data(), &lda, &info);
			const double e = reconstruction_error(m, n, k, f, a0, lda);
			CHECK(info == 0, "%sgetrfnp m=%d n=%d lda=%d: info=%d", type_name<T>::value, (int)m, (int)n, (int)lda, (int)info);
			CHECK(e < threshold, "%sgetrfnp m=%d n=%d lda=%d: error %.2f", type_name<T>::value, (int)m, (int)n, (int)lda, e);
			CHECK(padding_untouched(m, n, lda, f), "%sgetrfnp m=%d n=%d lda=%d: padding changed", type_name<T>::value, (int)m, (int)n, (int)lda);

			for (int_t nfact : { int_t(0), int_t(1), k / 3, k / 2, k }) {
				if (nfact > k) continue;
				auto g = a0; info = -99;
				getrfnpi(&m, &n, &nfact, g.data(), &lda, &info);
				const double ei = reconstruction_error(m, n, nfact, g, a0, lda);
				CHECK(info == 0, "%sgetrfnpi m=%d n=%d nfact=%d lda=%d: info=%d", type_name<T>::value, (int)m, (int)n, (int)nfact, (int)lda, (int)info);
				CHECK(ei < threshold, "%sgetrfnpi m=%d n=%d nfact=%d lda=%d: error %.2f", type_name<T>::value, (int)m, (int)n, (int)nfact, (int)lda, ei);
				CHECK(padding_untouched(m, n, lda, g), "%sgetrfnpi m=%d n=%d nfact=%d lda=%d: padding changed", type_name<T>::value, (int)m, (int)n, (int)nfact, (int)lda);
			}
		}
	}
}

// A = L*U with small integers and U(z,z) = 0, so the factorization is exact and the zero pivot is exact.
template<typename T>
static std::vector<T> exact_zero_pivot_matrix(int_t n, int_t z) {
	std::vector<T> l(static_cast<size_t>(n) * n, T(0)), u(static_cast<size_t>(n) * n, T(0)), a(static_cast<size_t>(n) * n, T(0));
	unsigned s = 12345;
	for (int_t j = 0; j < n; j++) {
		for (int_t i = 0; i < n; i++) {
			s = s * 1103515245u + 12345u;
			const int r = static_cast<int>((s >> 16) % 3) - 1;
			if (i > j && i - j < 3) l[j * n + i] = T(r);
			if (i < j && j - i < 3) u[j * n + i] = T(r);
		}
		l[j * n + j] = T(1);
		u[j * n + j] = (j == z) ? T(0) : T(2);
	}
	for (int_t j = 0; j < n; j++)
		for (int_t i = 0; i < n; i++)
			for (int_t p = 0; p < n; p++)
				a[j * n + i] += l[p * n + i] * u[j * n + p];
	return a;
}

template<typename T>
static void test_zero_pivot() {
	const int_t cases[][2] = { {10, 4}, {16, 0}, {16, 15}, {200, 7}, {200, 50}, {200, 120} };
	for (const auto& c : cases) {
		const int_t n = c[0], z = c[1];
		const auto a0 = exact_zero_pivot_matrix<T>(n, z);

		auto f = a0; int_t info = -99;
		getrfnp(&n, &n, f.data(), &n, &info);
		CHECK(info == z + 1, "%sgetrfnp n=%d zero pivot %d: info=%d", type_name<T>::value, (int)n, (int)z + 1, (int)info);

		auto g = a0; int_t ginfo = -99, nfact = z;
		getrfnpi(&n, &n, &nfact, g.data(), &n, &ginfo);
		CHECK(ginfo == 0, "%sgetrfnpi n=%d nfact=%d: info=%d", type_name<T>::value, (int)n, (int)nfact, (int)ginfo);
		CHECK(f == g, "%sgetrfnp n=%d zero pivot %d: result differs from getrfnpi(nfact=%d)", type_name<T>::value, (int)n, (int)z + 1, (int)z);
		CHECK(reconstruction_error(n, n, z, f, a0, n) == 0.0, "%sgetrfnp n=%d zero pivot %d: not exact", type_name<T>::value, (int)n, (int)z + 1);

		// getrfnpi asked for more columns than it can factor stops at the same place
		auto h = a0; int_t hinfo = -99, nfull = n;
		getrfnpi(&n, &n, &nfull, h.data(), &n, &hinfo);
		CHECK(hinfo == z + 1 && h == f, "%sgetrfnpi n=%d nfact=%d zero pivot %d: info=%d", type_name<T>::value, (int)n, (int)n, (int)z + 1, (int)hinfo);
	}
}

template<typename T>
static void test_arguments() {
	std::vector<T> a(64, T(1));
	const std::string np = std::string(type_name<T>::value) + "GETRFNP";
	const std::string npi = np + "I";
	struct { int_t m, n, lda, expect; } np_cases[] = { {-1, 4, 4, -1}, {4, -1, 4, -2}, {4, 4, 3, -4} };
	for (const auto& c : np_cases) {
		int_t info = 0; xerbla_name.clear(); xerbla_info = 0;
		getrfnp(&c.m, &c.n, a.data(), &c.lda, &info);
		CHECK(info == c.expect && xerbla_name == np && xerbla_info == -c.expect,
		      "%s m=%d n=%d lda=%d: info=%d xerbla=%s/%d", np.c_str(), (int)c.m, (int)c.n, (int)c.lda, (int)info, xerbla_name.c_str(), (int)xerbla_info);
	}
	struct { int_t m, n, k, lda, expect; } npi_cases[] = { {-1, 4, 0, 4, -1}, {4, -1, 0, 4, -2}, {4, 4, -1, 4, -3}, {4, 3, 4, 4, -3}, {4, 4, 2, 3, -5} };
	for (const auto& c : npi_cases) {
		int_t info = 0; xerbla_name.clear(); xerbla_info = 0;
		getrfnpi(&c.m, &c.n, &c.k, a.data(), &c.lda, &info);
		CHECK(info == c.expect && xerbla_name == npi && xerbla_info == -c.expect,
		      "%s m=%d n=%d nfact=%d lda=%d: info=%d xerbla=%s/%d", npi.c_str(), (int)c.m, (int)c.n, (int)c.k, (int)c.lda, (int)info, xerbla_name.c_str(), (int)xerbla_info);
	}
	// quick returns
	const int_t zero = 0, four = 4;
	int_t info = -99; xerbla_name.clear();
	getrfnp(&zero, &four, a.data(), &four, &info);
	CHECK(info == 0 && xerbla_name.empty(), "%s m=0: info=%d", np.c_str(), (int)info);
	info = -99;
	getrfnpi(&four, &four, &zero, a.data(), &four, &info);
	CHECK(info == 0 && xerbla_name.empty() && a[0] == T(1), "%s nfact=0: info=%d", npi.c_str(), (int)info);
}

// random matrix with row z zero in columns 0..z, so U(z,z) is exactly zero
template<typename T>
static std::vector<T> zero_row_matrix(int_t m, int_t n, int_t lda, int_t z, T pivot) {
	auto a = diagonally_dominant<T>(m, n, lda);
	for (int_t j = 0; j < z; j++) a[j * lda + z] = T(0);
	a[z * lda + z] = pivot;
	return a;
}

// max over columns of max_i |x(i,j) - y(i,j)| / max_i |y(i,j)|
template<typename T>
static double column_difference(int_t m, int_t n, int_t lda, const std::vector<T>& x, const std::vector<T>& y) {
	double worst = 0;
	for (int_t j = 0; j < n; j++) {
		double d = 0, s = 0;
		for (int_t i = 0; i < m; i++) {
			d = std::max(d, static_cast<double>(std::abs(x[j * lda + i] - y[j * lda + i])));
			s = std::max(s, static_cast<double>(std::abs(y[j * lda + i])));
		}
		worst = std::max(worst, s > 0 ? d / s : d);
	}
	return worst;
}

// zero pivots in random data: info, and the result of getrfnpi(nfact = info-1)
template<typename T>
static void test_zero_pivot_random() {
	struct zcase { int_t m, n, nfact, lda, z; bool negative_zero; };
	const zcase cases[] = {
		{ 200, 200, 120, 200,  50, false }, { 300, 300, 200, 300, 137, false }, { 100, 100, 100, 100, 99, false },
		{ 120,  80,  80, 120,  40, false }, {  80, 120,  80,  80,  40, false }, {  40,  40,  40,  45, 13, false },
		{ 200, 200, 120, 200,  50, true  },
	};
	const double eps = std::numeric_limits<real_t<T>>::epsilon();
	for (const auto& c : cases) {
		const auto a0 = zero_row_matrix<T>(c.m, c.n, c.lda, c.z, c.negative_zero ? T(static_cast<real_t<T>>(-0.0)) : T(0));
		auto f = a0; int_t info = -99, nfact = c.nfact;
		if (c.nfact == std::min(c.m, c.n)) getrfnp(&c.m, &c.n, f.data(), &c.lda, &info);
		else                                getrfnpi(&c.m, &c.n, &nfact, f.data(), &c.lda, &info);
		CHECK(info == c.z + 1, "%sgetrfnp%s m=%d n=%d nfact=%d zero pivot %d: info=%d", type_name<T>::value,
		      c.nfact == std::min(c.m, c.n) ? "" : "i", (int)c.m, (int)c.n, (int)c.nfact, (int)c.z + 1, (int)info);

		auto g = a0; int_t ginfo = -99, gk = c.z;
		getrfnpi(&c.m, &c.n, &gk, g.data(), &c.lda, &ginfo);
		const double d = column_difference(c.m, c.n, c.lda, f, g);
		CHECK(d < 64 * eps * std::max(c.m, c.n), "%s m=%d n=%d nfact=%d zero pivot %d: differs from getrfnpi(nfact=%d) by %.2e",
		      type_name<T>::value, (int)c.m, (int)c.n, (int)c.nfact, (int)c.z + 1, (int)c.z, d);
		CHECK(reconstruction_error(c.m, c.n, c.z, f, a0, c.lda) < 30.0, "%s m=%d n=%d nfact=%d zero pivot %d: reconstruction",
		      type_name<T>::value, (int)c.m, (int)c.n, (int)c.nfact, (int)c.z + 1);
		CHECK(padding_untouched(c.m, c.n, c.lda, f), "%s m=%d n=%d lda=%d zero pivot: padding changed", type_name<T>::value, (int)c.m, (int)c.n, (int)c.lda);
	}
}

template<typename T>
static void test_all() {
	test_random<T>();
	test_zero_pivot<T>();
	test_zero_pivot_random<T>();
	test_arguments<T>();
}

int main() {
	test_all<float>();
	test_all<double>();
	test_all<std::complex<float>>();
	test_all<std::complex<double>>();
	std::printf("%d of %d checks failed\n", failures, checks);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

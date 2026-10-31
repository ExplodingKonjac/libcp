#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "def.hpp"

namespace acm
{

// Calculates a^b mod p
template <std::signed_integral T>
T qpow(T a, T b, T p) {
    static_assert(sizeof(T) <= 8, "qpow supports up to 64-bit integers");
    using W = i128;
    T r = 1;
    for (; b; b >>= 1, a = T(W(a) * a % p))
        if (b & 1) r = T(W(r) * a % p);
    return r;
}

// Returns gcd(a,b) and assigned x,y to one solution of ax+by=gcd(a,b)
template <std::signed_integral T>
T exgcd(T a, T b, T& x, T& y) {
    if (!b) return x = 1, y = 0, a;
    T d = exgcd(b, a % b, y, x);
    return y -= a / b * x, d;
}

// Solve ax+by=c. returns {x,y}, making x>=0 and minimizing x.
template <std::signed_integral T>
std::pair<T, T> biequation(T a, T b, T c) {
    T x, y, g = exgcd(a, b, x, y);
    if (c % g) return {-1, -1};
    T dx = b / g, dy = a / g;
    i128 xx = x, yy = y, k = 0;
    c /= g, xx *= c, yy *= c;
    if (xx < 0) k = (-xx + dx - 1) / dx;
    if (xx > 0) k = -xx / dx;
    return {T(xx + k * dx), T(yy - k * dy)};
}

// Checks if n passes test_num Miller-Rabin tests.
inline bool is_prime(i64 n) {
    if (n < 2) return false;
    for (i64 p: {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}) {
        if (n % p == 0) return n == p;
    }
    i64 d = n - 1, s = 0;
    while (!(d & 1)) d >>= 1, s++;
    for (i64 a: {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}) {
        if (a % n == 0) continue;
        i64 x = qpow(a % n, d, n);
        if (x == 1 || x == n - 1) continue;
        bool ok = false;
        for (i64 i = 1; i < s; i++) {
            x = i64(i128(x) * x % n);
            if (x == n - 1) {
                ok = true;
                break;
            }
        }
        if (!ok) return false;
    }
    return true;
}

// Returns one of the non-trivial factor of n, using Pollard-Rho algorithm.
inline i64 pollard_rho(i64 n) {
    if (n % 2 == 0) return 2;
    if (n % 3 == 0) return 3;
    static std::mt19937_64 rng(712367821);
    std::uniform_int_distribution<i64> dist(0, n - 1);
    while (true) {
        i64 c = dist(rng) % (n - 1) + 1;
        i64 x = dist(rng), y = x, d = 1;
        auto f = [&](i64 v) { return i64((i128(v) * v + c) % n); };
        while (d == 1) {
            x = f(x);
            y = f(f(y));
            d = std::gcd(std::llabs(x - y), n);
        }
        if (d != n) return d;
    }
}

// Factorize n and put the prime factors into res.
inline std::vector<i64> factorize(i64 n) {
    assert(n > 0);
    std::vector<i64> res, st{n};
    while (!st.empty()) {
        i64 x = st.back();
        st.pop_back();
        if (x == 1) continue;
        if (is_prime(x)) {
            res.push_back(x);
            continue;
        }
        i64 p = pollard_rho(x);
        st.push_back(p);
        st.push_back(x / p);
    }
    std::sort(res.begin(), res.end());
    return res;
}

// Returns the Legendre Symbol (a|p), assuming p is a odd prime number.
inline i64 legendre(i64 a, i64 p) {
    a %= p;
    if (a < 0) a += p;
    if (a == 0) return 0;
    if (p == 2) return 1;
    i64 r = qpow(a, (p - 1) / 2, p);
    return r == p - 1 ? -1 : 1;
}

// Returns x that x^2 mod p=n, or -1 if such x doesn't exists, assuming p is a
// odd prime number.
inline std::optional<i64> cipolla(i64 n, i64 p) {
    n %= p;
    if (n < 0) n += p;
    if (n == 0) return 0;
    if (p == 2) return n;
    if (legendre(n, p) != 1) return std::nullopt;

    static std::mt19937_64 rng(987654321);
    std::uniform_int_distribution<i64> dist(0, p - 1);
    i64 a = dist(rng);
    while (legendre(i64((i128(a) * a % p - n + p) % p), p) != -1) a = dist(rng);

    i64 i2 = i64((i128(a) * a % p - n + p) % p);
    using P = std::pair<i128, i128>;
    auto mul = [&](P x, P y) {
        return P{
            (x.first * y.first % p + i2 * x.second % p * y.second) % p,
            (x.first * y.second % p + x.second * y.first % p) % p,
        };
    };
    P base{a, 1}, res{1, 0};
    for (i64 k = (p + 1) / 2; k; k >>= 1, base = mul(base, base))
        if (k & 1) res = mul(res, base);
    return i64(res.first);
}

// Universal-Uclidean Algorithm. Computes the semigroup product of the segments:
// let f(x) = ⌊(p x + r) / q⌋, then
//   (0, f(0)) -> (1, f(0)) -> ... -> (1, f(1)) -> ... ->
//   (i, f(i)) -> (i + 1, f(i)) -> ... -> (i + 1, f(i + 1)) -> ... ->
//   (n - 1, f(n - 1)) -> (n, f(n - 1)) -> ... -> (n, f(n))
// where going up denotes a `U` and going right denotes an `R`.
template <typename D, std::signed_integral T, typename Op = std::multiplies<D>>
D uniclidean(T p, T q, T r, T n, D R, D U, D init = {}, Op mul = Op{}) {
    if (n == 0) return init;
    auto qpow = [&](D x, T y, D init) {
        for (; y; y >>= 1, x = mul(x, x))
            if (y & 1) init = mul(init, x);
        return init;
    };
    if (r >= q) return uniclidean(p, q, r % q, n, R, U, init, mul);
    if (p >= q)
        return uniclidean(p % q, q, r, n, qpow(U, p / q, R), U, init, mul);
    T m = (p * n + r) / q;
    if (m == 0) return qpow(R, n, init);
    init = qpow(R, (q - r - 1) / p + 1, init);
    init = uniclidean(q, p, (q - r - 1) % p, m - 1, U, R, init, mul);
    return qpow(R, n - (q * m - r - 1) / p - 1, mul(init, U));
}

}  // namespace acm

#pragma once

#include <concepts>
#include <functional>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "cp/def.hpp"

namespace cp
{

namespace detail
{

template <std::size_t SIZE>
using __nt_wide_int_t =
    std::conditional_t<(SIZE <= 4), i64, std::enable_if_t<(SIZE <= 8), i128>>;

template <std::size_t SIZE>
using __nt_wide_uint_t =
    std::conditional_t<(SIZE <= 4), u64, std::enable_if_t<(SIZE <= 8), u128>>;

template <std::integral IntT>
using nt_wide_t = std::conditional_t<
    std::is_signed<IntT>::value,
    __nt_wide_int_t<sizeof(IntT)>,
    __nt_wide_uint_t<sizeof(IntT)>
>;

};  // namespace detail

// Calculates a^b mod p
template <std::signed_integral T>
inline T qpow(T a, T b, T p) {
    using L = detail::nt_wide_t<T>;
    T res = 1;
    for (; b; a = (L)a * a % p, b >>= 1)
        if (b & 1) res = (L)res * a % p;
    return res;
}

// Returns gcd(a,b) and assigned x,y to one solution of ax+by=gcd(a,b)
template <std::signed_integral T>
inline T exgcd(T a, T b, T& x, T& y) {
    if (!b) return x = 1, y = 0, a;
    T res = exgcd(b, a % b, y, x);
    return y -= a / b * x, res;
}

// Solve ax+by=c. returns {x,y}, making x>=0 and minimizing x.
template <std::signed_integral T>
inline std::pair<T, T> biequation(T a, T b, T c) {
    using L = detail::nt_wide_t<T>;
    T x, y, g = exgcd(a, b, x, y);
    if (c % g) return {-1, -1};
    T dx = b / g, dy = a / g;
    L xx = x, yy = y, k = 0;
    c /= g, xx *= c, yy *= c;
    if (xx < 0) k = (-xx + dx - 1) / dx;
    if (xx > 0) k = -xx / dx;
    return {xx + k * dx, yy - k * dy};
}

// Checks if n passes test_num Miller-Rabin tests.
template <std::signed_integral T, typename Gen>
inline bool miller_rabin(T n, Gen&& gen, usize test_num = 8) {
    using L = detail::nt_wide_t<T>;
    for (auto& i: {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}) {
        if (n == i) return true;
        else if (n % i == 0) return false;
    }
    std::uniform_int_distribution<T> rng(2, n - 1);
    T u = n - 1, k = 0;
    while (!(u & 1)) u >>= 1, k++;
    for (; test_num; test_num--) {
        T a = rng(gen), v = qpow(a, u, n), j;
        if (v == 1) continue;
        for (j = 0; v != n - 1 && j < k; j++) v = (L)v * v % n;
        if (j == k) return false;
    }
    return true;
}

// Returns one of the non-trivial factor of n, using Pollard-Rho algorithm.
template <std::signed_integral T, typename Gen>
inline T pollard_rho(T n, Gen&& gen) {
    using L = detail::nt_wide_t<T>;
    std::uniform_int_distribution<T> rng(1, n - 1);
    if (n == 4) return 2;
    T s = 0, t = 0, val = 1, d, C = rng(gen);
    for (usize goal = 1;; goal <<= 1, s = t, val = 1) {
        for (usize step = 0; step < goal; step++) {
            t = ((L)t * t + C) % n;
            if (s == t) return pollard_rho(n, gen);
            val = (L)val * abs(t - s) % n;
            if (!val) return std::gcd(abs(t - s), n);
            if (step % 127 == 0) {
                d = std::gcd(val, n);
                if (d > 1) return d;
            }
        }
        d = std::gcd(val, n);
        if (d > 1) return d;
    }
}

// Factorize n and put the prime factors into res.
template <std::signed_integral T, typename Gen>
inline std::vector<T> factorize(T n, Gen&& gen) {
    std::vector<T> res;
    auto work = [&](auto&& self, T n) -> void {
        if (n == 1) return;
        if (miller_rabin(n, gen)) return res.push_back(n);
        T p = pollard_rho(n, gen);
        self(self, p);
        self(self, n / p);
    };
    work(work, n);
    return res;
}

// Returns the Legendre Symbol (a|p), assuming p is a odd prime number.
template <std::signed_integral T>
inline T legendre(T a, T p) {
    T res = qpow(a, (p - 1) / 2, p);
    if (res == p - 1) return -1;
    else return res;
}

// Returns x that x^2 mod p=n, or -1 if such x doesn't exists, assuming p is a
// odd prime number.
template <std::signed_integral T, typename Gen>
inline std::optional<T> cipolla(T n, T p, Gen&& gen) {
    using L = detail::nt_wide_t<T>;
    if (!(n %= p)) return 0;
    if (legendre(n, p) != 1) return std::nullopt;
    std::uniform_int_distribution<T> rng(0, p - 1);
    T a = rng(gen);
    while (legendre(T(((L)a * a % p + p - n) % p), p) != T(-1)) a = rng(gen);
    auto mul = [&, I2 = ((L)a * a % p + p - n) % p](auto& x, auto& y) {
        return std::pair<L, L>(
            (x.first * y.first % p + I2 * x.second % p * y.second) % p,
            (x.first * y.second % p + x.second * y.first % p) % p
        );
    };
    std::pair<L, L> base{a, 1}, res{1, 0};
    for (T k = (p + 1) / 2; k; base = mul(base, base), k >>= 1)
        if (k & 1) res = mul(res, base);
    return res.first;
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
    if (p >= q) {
        // each climbing up would have ⌊p / q⌋ `U`s, so add that after `R`.
        return uniclidean(p % q, q, r, n, qpow(U, p / q, R), U, init, mul);
    }
    T m = (p * n + r) / q;
    if (m == 0) return qpow(R, n, init);
    // when p < q, transpose
    init = qpow(R, (q - r - 1) / p + 1, init);
    init = uniclidean(q, p, (q - r - 1) % p, m - 1, U, R, init, mul);
    return qpow(R, n - (q * m - r - 1) / p - 1, mul(init, U));
}

}  // namespace cp

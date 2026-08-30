#pragma once

#include <cassert>
#include <concepts>
#include <optional>
#include <utility>

#include "def.hpp"

namespace acm
{

namespace detail
{

template <typename D>
class ModintBase {
    u32 v = 0;

    constexpr D& self() { return static_cast<D&>(*this); }

public:
    constexpr ModintBase() = default;
    template <std::integral T>
    constexpr ModintBase(T x) {
        i64 y = i64(x % i64(D::mod()));
        v = y < 0 ? y + D::mod() : y;
    }

    constexpr u32 val() const { return v; }
    constexpr explicit operator bool() const { return v; }
    friend constexpr bool operator==(D a, D b) { return a.v == b.v; }

    constexpr D& operator+=(D x) {
        return (v += x.v) >= D::mod() ? v -= D::mod() : 0, self();
    }
    constexpr D& operator-=(D x) {
        return v += (v < x.v ? D::mod() - x.v : -x.v), self();
    }
    constexpr D& operator*=(D x) { return v = u64(v) * x.v % D::mod(), self(); }
    constexpr D& operator/=(D x) { return *this *= x.inv(); }

    constexpr D inv() const {
        assert(v);
        i64 a = v, x = 1, y = 0, t;
        for (i64 b = D::mod(); b; std::swap(a, b), std::swap(x, y))
            t = a / b, a -= t * b, x -= t * y;
        return x < 0 ? x + D::mod() : x;
    }

    friend constexpr D operator+(D a, D b) { return a += b; }
    friend constexpr D operator-(D a, D b) { return a -= b; }
    friend constexpr D operator*(D a, D b) { return a *= b; }
    friend constexpr D operator/(D a, D b) { return a /= b; }
    friend constexpr D operator-(D a) { return a ? D(D::mod() - a.v) : D{}; }
};

}  // namespace detail

template <u32 MOD>
struct SModint: detail::ModintBase<SModint<MOD>> {
    using detail::ModintBase<SModint<MOD>>::ModintBase;
    static constexpr u32 mod() { return MOD; }
};

struct DModint: detail::ModintBase<DModint> {
    using detail::ModintBase<DModint>::ModintBase;
    inline static u32 MOD = 998244353;
    static void set_mod(u32 mod) { MOD = mod; }
    static u32 mod() { return MOD; }
};

template <typename M>
    requires std::derived_from<M, detail::ModintBase<M>>
constexpr M pow(M a, u64 n) {
    M r = 1;
    for (; n; n >>= 1, a *= a)
        if (n & 1) r *= a;
    return r;
}

template <typename M>
    requires std::derived_from<M, detail::ModintBase<M>>
constexpr int legendre(M x) {
    return x == M{} ? 0 : pow(x, (M::mod() - 1) / 2) == M{1} ? 1 : -1;
}

template <typename M>
    requires std::derived_from<M, detail::ModintBase<M>>
constexpr std::optional<M> sqrt(M a) {
    if (!a) return M{};
    u32 p = M::mod();
    if (legendre(a) < 0) return std::nullopt;
    if (p % 4 == 3) return pow(a, (p + 1) / 4);
    u32 q = p - 1, s = 0;
    while (!(q & 1)) q >>= 1, s++;
    M z = 2;
    while (legendre(z) != -1) z += 1;
    M c = pow(z, q), x = pow(a, (q + 1) / 2), t = pow(a, q);
    for (u32 m = s; t != M{1};) {
        u32 i = 1;
        M y = t * t;
        while (y != M{1}) y *= y, i++;
        M b = pow(c, u64{1} << (m - i - 1));
        x *= b, c = b * b, t *= c, m = i;
    }
    return x;
}

}  // namespace acm

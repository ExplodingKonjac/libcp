#pragma once

#include <cassert>
#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>

#include "def.hpp"

namespace acm
{

namespace detail
{

template <typename D>
class ModintBase {
protected:
    i32 v = 0;

    // v is always in [0, mod). qmo maps a value in (-mod, mod) back to [0, mod).
    constexpr static inline i32 qmo(i32 x) {
        return x + ((x >> 31) & D::mod());
    }
    // Let CRTP base member functions return D / D& without explicit casts.
    constexpr operator D() const { return static_cast<const D&>(*this); }
    constexpr operator D&() { return static_cast<D&>(*this); }

public:
    constexpr ModintBase() = default;

    template <std::integral T>
    constexpr ModintBase(T x): v(qmo(x % D::mod())) {}

    constexpr i32 operator()() const { return v; }
    constexpr explicit operator bool() const { return v != 0; }

    friend constexpr bool operator==(D a, D b) { return a.v == b.v; }
    friend constexpr bool operator!=(D a, D b) { return a.v != b.v; }

    constexpr D& operator+=(D x) { return v = qmo(v - D::mod() + x.v), *this; }
    constexpr D& operator-=(D x) { return v = qmo(v - x.v), *this; }
    constexpr D& operator*=(D x) { return v = i64(v) * x.v % D::mod(), *this; }
    constexpr D& operator/=(D x) { return *this *= x.inv(); }

    friend constexpr D operator+(D a, D b) { return a += b; }
    friend constexpr D operator-(D a, D b) { return a -= b; }
    friend constexpr D operator*(D a, D b) { return a *= b; }
    friend constexpr D operator/(D a, D b) { return a /= b; }

    constexpr D operator+() const { return *this; }
    constexpr D operator-() const { return D(0) - *this; }

    constexpr D& operator++() { return *this += D(1); }
    constexpr D& operator--() { return *this -= D(1); }
    constexpr D operator++(int) {
        D t = *this;
        return ++*this, t;
    }
    constexpr D operator--(int) {
        D t = *this;
        return --*this, t;
    }

    constexpr D inv() const {
        assert(v != 0);
        i64 a = v, b = D::mod(), x = 1, y = 0;
        while (b) {
            i64 q = a / b;
            std::swap(a -= q * b, b);
            std::swap(x -= q * y, y);
        }
        return D(x < 0 ? x + D::mod() : x);
    }
};

}  // namespace detail

template <i32 P>
struct SModint: detail::ModintBase<SModint<P>> {
    static_assert(P > 0 && u32(P) < (1u << 31), "P must be in (0, 2^31)");

    using detail::ModintBase<SModint<P>>::ModintBase;
    static constexpr i32 mod() { return P; }
};

struct DModint: detail::ModintBase<DModint> {
    inline static i32 M = 998244353;

    using detail::ModintBase<DModint>::ModintBase;
    static void set_mod(i32 p) { assert(p > 0 && u32(p) < (1u << 31)), M = p; }
    static i32 mod() { return M; }
};

template <typename M, std::integral T>
constexpr M pow(M a, T b) {
    if constexpr (std::is_signed_v<T>)
        if (b < 0) return pow(a.inv(), -b);
    M r = 1;
    for (; b; b >>= 1, a *= a)
        if (b & 1) r *= a;
    return r;
}

template <typename M>
inline int legendre(M x) {
    i32 p = M::mod();
    if (x() == 0) return 0;
    if (p == 2) return 1;
    M r = pow(x, (p - 1) / 2);
    return r() == p - 1 ? -1 : 1;
}

template <typename M>
inline std::optional<M> sqrt(M a) {
    i32 p = M::mod();
    if (a() == 0) return a;
    if (p == 2) return a;
    if (legendre(a) != 1) return std::nullopt;
    if (p % 4 == 3) return pow(a, (p + 1) / 4);

    i32 q = p - 1;
    u32 s = 0;
    while (!(q & 1)) q >>= 1, s++;
    M z = 2;
    while (legendre(z) != -1) z += 1;
    M c = pow(z, q), x = pow(a, (q + 1) / 2), t = pow(a, q);
    for (u32 m = s; t != M(1);) {
        u32 i = 1;
        M y = t * t;
        while (y != M(1)) y *= y, i++;
        M b = pow(c, u64(1) << (m - i - 1));
        x *= b;
        c = b * b;
        t *= c;
        m = i;
    }
    return x;
}

}  // namespace acm

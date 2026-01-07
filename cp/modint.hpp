#pragma once
#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <print>
#include <stdexcept>

namespace cp
{

using i32 = std::int32_t;
using u32 = std::uint32_t;
using i64 = std::int64_t;
using u64 = std::uint64_t;

namespace detail
{

struct MontInfo {
    u32 P;
    u32 P2 = P * 2;
    u32 P_INV = [](u32 P) {
        u32 x = P % 2;
        for (int i = 0; i < 5; i++) x *= (2u - P * x);
        return -x;
    }(P);
    u32 R = (1ull << 32) % P;
    u32 R2 = (u64)R * R % P;
    u32 R3 = (u64)R2 * R % P;

    constexpr u32 toMont(u32 x) const { return mul(x, R2); }
    constexpr u32 fromMont(u32 x) const { return mul(x, 1); }
    constexpr u32 add(u32 x, u32 y) const {
        return (x += y) >= P2 ? x - P2 : x;
    }
    constexpr u32 sub(u32 x, u32 y) const { return x < y ? x + P2 - y : x - y; }
    constexpr u32 mul(u32 x, u32 y) const {
        u64 t = (u64)x * y;
        u32 m = (u32)t * P_INV;
        return (t + (u64)m * P) >> 32;
    }
    template <std::integral T>
    constexpr u32 qpow(u32 x, T y) const {
        u32 res = toMont(1);
        for (; y; y >>= 1, x = mul(x, x))
            if (y & 1) res = mul(res, x);
        return res;
    }
    constexpr u32 inv(u32 a) const {
        i64 c = a, x = 1, y = 0, t;
        for (i64 b = P; b; std::swap(c, b), std::swap(x, y))
            t = c / b, c -= t * b, x -= t * y;
        return mul(x < 0 ? x + P : x, R3);
    }
};

template <typename D>
class ModintBase {
public:
    constexpr ModintBase(): _val{0} {}
    template <std::integral T>
    constexpr ModintBase(T x):
        _val{m.toMont((x %= (i32)m.P) < 0 ? x + m.P : x)} {}
    template <std::integral T>
    explicit constexpr operator T() const {
        return m.fromMont(_val);
    }
    constexpr operator D&() { return *static_cast<D*>(this); }

#define DEF_OP_ARI(op, expr)                                   \
    constexpr D operator op(D rhs) const { return raw(expr); } \
    constexpr D& operator op## = (D rhs) { return _val = (expr), *this; }
#define DEF_OP_INC(op, expr)                                    \
    constexpr D& operator op() { return _val = (expr), *this; } \
    constexpr D operator op(int) {                              \
        D res(*this);                                           \
        return op(*this), res;                                  \
    }
#define DEF_OP_UNARY(op, expr) \
    constexpr D operator op() const { return raw(expr); }

    DEF_OP_ARI(+, m.add(_val, rhs._val))
    DEF_OP_ARI(-, m.sub(_val, rhs._val))
    DEF_OP_ARI(*, m.mul(_val, rhs._val))
    DEF_OP_ARI(/, m.mul(_val, m.inv(rhs._val)))
    DEF_OP_INC(++, m.add(_val, m.R))
    DEF_OP_INC(--, m.sub(_val, m.R))
    DEF_OP_UNARY(+, _val)
    DEF_OP_UNARY(-, _val ? m.P - _val : 0)
    constexpr D inv() const { return D(m.inv(_val)); }

#undef DEF_OP_ARI
#undef DEF_OP_INC
#undef DEF_OP_UNARY

private:
    static constexpr const MontInfo& m = D::mont;
    static D raw(u32 x) { return reinterpret_cast<D&>(x); }

    u32 _val;
};

}  // namespace detail

template <u32 P>
class SModint: public detail::ModintBase<SModint<P>> {
    static_assert(P > 0 && P < (1 << 30), "P must be in [0, 2^{30})");

private:
    using Base = detail::ModintBase<SModint<P>>;
    friend Base;
    static constexpr detail::MontInfo mont{P};

public:
    using Base::Base;
};

class DModint: public detail::ModintBase<DModint> {
private:
    using Base = detail::ModintBase<DModint>;
    friend Base;
    inline static detail::MontInfo mont{998244353};

public:
    using Base::Base;
    static void setMod(u32 P) {
        if (P == 0 || P >= (1 << 30)) {
            throw std::out_of_range("P must be in [0, 2^{30})");
        }
        mont = detail::MontInfo{P};
    }
    static u32 getMod() { return mont.P; }
};

}  // namespace cp

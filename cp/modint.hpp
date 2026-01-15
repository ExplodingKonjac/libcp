#pragma once
#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <stdexcept>

namespace cp
{

using i32 = std::int32_t;
using u32 = std::uint32_t;
using i64 = std::int64_t;
using u64 = std::uint64_t;

namespace detail
{

#define C constexpr
struct MontInfo {
    u32 P, P2 = P * 2, P_INV = [](u32 P) {
        u32 x = P % 2;
        for (int i = 0; i < 5; i++) x *= (2u - P * x);
        return -x;
    }(P);
    u32 R = (1ull << 32) % P, R2 = (u64)R * R % P, R3 = (u64)R2 * R % P;

    C u32 toMont(u32 x) const { return mul(x, R2); }
    C u32 fromMont(u32 x) const { return x = redc(x), x < P ? x : x - P; }
    C u32 redc(u64 t) const { return (t + u64(u32(t) * P_INV) * P) >> 32; }
    C u32 add(u32 x, u32 y) const { return x += y, x >= P2 ? x - P2 : x; }
    C u32 sub(u32 x, u32 y) const { return x < y ? x + P2 - y : x - y; }
    C u32 mul(u32 x, u32 y) const { return redc(u64(x) * y); }
    C u32 inv(u32 a) const {
        i64 c = a, x = 1, y = 0, t;
        for (i64 b = P; b; std::swap(c, b), std::swap(x, y))
            t = c / b, c -= t * b, x -= t * y;
        return mul(x < 0 ? x + P : x, R3);
    }
};

template <typename D>
class ModintBase {
public:
    C ModintBase(): _val{0} {}
    C ModintBase(i32 x): _val{m.toMont(x + (1 << 31) / m.P * m.P)} {}
    C ModintBase(u32 x): _val{m.toMont(x)} {}
    C ModintBase(i64 x): _val{m.toMont(x % m.P + m.P)} {}
    C ModintBase(u64 x): _val{m.toMont(x % m.P)} {}
    C bool operator==(D other) const {
        u32 delta = _val >= other._val ? _val - other._val : other._val - _val;
        return delta == 0 || delta == m.P;
    }
    C bool operator!=(D other) const { return !(*this == other); }
    C u32 operator()() const { return m.fromMont(_val); }
    C D inv() const { return D(0, m.inv(_val)); }
    C D strict() const { return D(0, _val < m.P ? _val : _val - m.P); }
    C u32 raw() const { return _val; }

#define DEF_OP_ARI(op, expr)                                     \
    C D operator op(D rhs) const { return D(*this) op## = rhs; } \
    C D& operator op## = (D rhs) { return _val = (expr), *this; }
#define DEF_OP_INC(op, expr)                            \
    C D& operator op() { return _val = (expr), *this; } \
    C D operator op(int) {                              \
        D res(*this);                                   \
        return op(*this), res;                          \
    }
#define DEF_OP_UNARY(op, expr) \
    C D operator op() const { return D(0, (expr)); }

    DEF_OP_ARI(+, m.add(_val, rhs._val))
    DEF_OP_ARI(-, m.sub(_val, rhs._val))
    DEF_OP_ARI(*, m.mul(_val, rhs._val))
    DEF_OP_ARI(/, m.mul(_val, m.inv(rhs._val)))
    DEF_OP_INC(++, m.add(_val, m.R))
    DEF_OP_INC(--, m.sub(_val, m.R))
    DEF_OP_UNARY(+, _val)
    DEF_OP_UNARY(-, _val ? m.P2 - _val : 0)

#undef DEF_OP_ARI
#undef DEF_OP_INC
#undef DEF_OP_UNARY

private:
    static C const MontInfo& m = D::mont;

    C ModintBase(int, u32 x): _val{x} {}
    C operator D() const { return *static_cast<const D*>(this); }
    C operator D&() { return *static_cast<D*>(this); }

    u32 _val;
};

}  // namespace detail

template <u32 P>
struct SModint: detail::ModintBase<SModint<P>> {
    static_assert(P > 0 && P < (1 << 30), "P must be in [0, 2^{30})");

    static C detail::MontInfo mont{P};
    using detail::ModintBase<SModint<P>>::ModintBase;
};

struct DModint: public detail::ModintBase<DModint> {
    inline static detail::MontInfo mont{998244353};
    using detail::ModintBase<DModint>::ModintBase;

    static void setMod(u32 P) {
        if (P == 0 || P >= (1 << 30))
            throw std::out_of_range("P must be in [0, 2^{30})");
        mont = detail::MontInfo{P};
    }
    static u32 getMod() { return mont.P; }
};

template <typename T, std::integral U>
    requires std::derived_from<T, detail::ModintBase<T>>
C auto qpow(T x, U y) {
    T res{1};
    for (; y; y >>= 1, x = x * x)
        if (y & 1) res = res * x;
    return res;
}
#undef C

}  // namespace cp

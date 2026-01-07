#pragma once
#include <immintrin.h>

#include "modint.hpp"

#pragma GCC target("avx2", "fma")

namespace cp
{

using m256i = __m256i;
using v8si = i32 __attribute__((vector_size(32)));

namespace detail
{

constexpr m256i vset1(i32 x) { return (m256i)(v8si{x, x, x, x, x, x, x, x}); }

struct VMontInfo: MontInfo {
    m256i V_1 = vset1(1);
    m256i V_P = vset1(P);
    m256i V_P_INV = vset1(P_INV);
    m256i V_R = vset1(R);
    m256i V_R2 = vset1(R2);

    m256i toMont(m256i x) const { return mul(x, V_R2); }
    m256i fromMont(m256i x) const { return mul(x, V_1); }
    m256i add(m256i x, m256i y) const {
        x = _mm256_add_epi32(x, y);
        y = _mm256_sub_epi32(x, V_P);
        return _mm256_min_epu32(x, y);
    }
    m256i sub(m256i x, m256i y) const {
        x = _mm256_sub_epi32(x, y);
        y = _mm256_add_epi32(x, V_P);
        return _mm256_min_epu32(x, y);
    }
    m256i mul(m256i x, m256i y) const {
        // even
        m256i t0 = _mm256_mul_epu32(x, y);
        m256i m0 = _mm256_mul_epu32(t0, V_P_INV);
        m256i res0 = _mm256_add_epi64(t0, _mm256_mul_epu32(m0, V_P));
        // odd
        x = _mm256_shuffle_epi32(x, 0xf5);
        y = _mm256_shuffle_epi32(y, 0xf5);
        m256i t1 = _mm256_mul_epu32(x, y);
        m256i m1 = _mm256_mul_epu32(t1, V_P_INV);
        m256i res1 = _mm256_add_epi64(t1, _mm256_mul_epu32(m1, V_P));
        // blend
        m256i mask = _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0);
        res0 = _mm256_srli_epi64(res0, 32);
        res1 = _mm256_and_si256(res1, mask);
        return _mm256_or_si256(res0, res1);
    }
};

template <typename D, typename S>
class VModintBase {
public:
    VModintBase(): _val{} {}
    VModintBase(S v0, S v1, S v2, S v3, S v4, S v5, S v6, S v7) {
        _val = _mm256_set_epi32((int&)v7, (int&)v6, (int&)v5, (int&)v4,
                                (int&)v3, (int&)v2, (int&)v1, (int&)v0);
    }

    static D load(S* ptr) { return raw(_mm256_load_si256(ptr)); }
    static D loadu(S* ptr) { return raw(_mm256_loadu_si256(ptr)); }
    D store(S* ptr) const { _mm256_store_si256(ptr, _val); }
    D storeu(S* ptr) const { _mm256_storeu_si256(ptr, _val); }
    S& operator[](size_t i) { return reinterpret_cast<S*>(&_val)[i]; }
    S operator[](size_t i) const { return reinterpret_cast<S*>(&_val)[i]; }

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
    DEF_OP_INC(++, m.add(_val, m.V_R))
    DEF_OP_INC(--, m.sub(_val, m.V_R))
    DEF_OP_UNARY(+, _val)
    DEF_OP_UNARY(-, m.sub(vset1(0), _val))

#undef DEF_OP_ARI
#undef DEF_OP_INC
#undef DEF_OP_UNARY

private:
    static constexpr const VMontInfo& m = D::mont;
    static D raw(m256i _v) { return reinterpret_cast<D&>(_v); }

    m256i _val;
};

}  // namespace detail

template <u32 P>
class SVModint: public detail::VModintBase<SVModint<P>, SModint<P>> {
private:
    using Base = detail::VModintBase<SVModint<P>, SModint<P>>;
    friend Base;
    static constexpr detail::VMontInfo mont{detail::MontInfo{P}};

public:
    using Base::Base;
};

}  // namespace cp

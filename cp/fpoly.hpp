#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "modint.hpp"

#pragma GCC push_options
#pragma GCC target("avx2")
namespace cp
{

namespace detail
{

template <typename T, size_t A = alignof(T)>
class AlignedPool {
private:
    static inline struct _Vec: std::vector<T*> {
        ~_Vec() {
#ifdef _WIN32
            for (auto ptr: *this) _aligned_free(ptr);
#else
            for (auto ptr: *this) std::free(ptr);
#endif
        }
    } _pool[32];

public:
    class pointer_type {
    public:
        pointer_type() = default;
        pointer_type(T* p, size_t c): _p(p), _c(c) {}
        pointer_type(const pointer_type& other) = delete;
        pointer_type(pointer_type&& other): _p(other._p), _c(other._c) {
            other._p = nullptr;
            other._c = 0;
        }
        pointer_type& operator=(const pointer_type& other) = delete;
        pointer_type& operator=(pointer_type&& other) {
            std::swap(_p, other._p);
            std::swap(_c, other._c);
            return *this;
        }
        ~pointer_type() { AlignedPool::deallocate(*this); }

        operator auto*() { return _p; }
        operator auto*() const { return _p; }
        auto& operator*() { return *_p; }
        auto& operator*() const { return *_p; }
        auto operator->() { return _p; }
        auto operator->() const { return _p; }
        auto& operator[](size_t i) { return _p[i]; }
        auto& operator[](size_t i) const { return _p[i]; }
        auto capacity() const { return _c; }

    private:
        T* _p = nullptr;
        size_t _c = 0;
    };

    static pointer_type allocate(size_t n) {
        n = std::max(A / sizeof(T), std::bit_ceil(n));
        int k = std::countr_zero(n);
        if (!_pool[k].empty()) {
            T* p = _pool[k].back();
            _pool[k].pop_back();
            return {p, n};
        }
#ifdef _WIN32
        T* p = static_cast<T*>(_aligned_malloc(n * sizeof(T), A));
#else
        T* p = static_cast<T*>(std::aligned_alloc(A, n * sizeof(T)));
#endif
        return {p, n};
    }
    static void deallocate(pointer_type& p) {
        if (p) _pool[std::countr_zero(p.capacity())].push_back(p);
    }
};

template <u32 P, size_t _MAXN>
struct PolyUtils {
    using Mint = SModint<P>;
    using Pool = AlignedPool<Mint, 32>;
    using m256i = __m256i;

    static m256i vload(const void* p) {
        return _mm256_load_si256((const m256i*)p);
    }
    static m256i vloadu(const void* p) {
        return _mm256_loadu_si256((const m256i*)p);
    }
    static void vstore(void* p, m256i a) { _mm256_store_si256((m256i*)p, a); }
    static void vstoreu(void* p, m256i a) { _mm256_storeu_si256((m256i*)p, a); }
    static constexpr m256i vset1(u32 x) {
        return (m256i)__v8su{x, x, x, x, x, x, x, x};
    }
    static constexpr m256i vsetr(u32 v0, u32 v1, u32 v2, u32 v3, u32 v4, u32 v5,
                                 u32 v6, u32 v7) {
        return (m256i)__v8su{v0, v1, v2, v3, v4, v5, v6, v7};
    }
    static constexpr m256i vset1(Mint x) { return vset1(x.strict().raw()); }
    static constexpr m256i vsetr(Mint v0, Mint v1, Mint v2, Mint v3, Mint v4,
                                 Mint v5, Mint v6, Mint v7) {
        return vsetr(v0.strict().raw(), v1.strict().raw(), v2.strict().raw(),
                     v3.strict().raw(), v4.strict().raw(), v5.strict().raw(),
                     v6.strict().raw(), v7.strict().raw());
    }
    template <int imm>
    static m256i vshuffle(m256i a) {
        return _mm256_shuffle_epi32(a, imm);
    }
    template <int imm>
    static m256i vpermute(m256i a) {
        return _mm256_permute4x64_epi64(a, imm);
    }
    template <int control>
    static m256i vblend(m256i x, m256i y) {
        return _mm256_blend_epi32(x, y, control);
    }

    static constexpr bool is_prime = [] {
        for (u32 i = 2; (u64)i * i <= P; i++)
            if (P % i == 0) return false;
        return true;
    }();
    static constexpr size_t LG_MAXN = std::countr_zero(P - 1),
                            MAXN = std::min(_MAXN, size_t(1) << LG_MAXN);
    static constexpr const auto& M = Mint::mont;

    static_assert(is_prime, "P must be prime number");
    static_assert(MAXN >= 8, "MAXN must be at least 8");

    static constexpr Mint G = [] {
        u32 phi = P - 1, tmp = phi;
        std::vector<u32> divs;
        for (u32 i = 2; (u64)i * i <= tmp; i++) {
            if (tmp % i == 0) divs.push_back(i);
            while (tmp % i == 0) tmp /= i;
        }
        if (tmp > 1) divs.push_back(tmp);
        for (u32 g = 2; g < P; g++) {
            bool ok = true;
            for (auto i: divs) {
                if (qpow(Mint{g}, phi / i) == Mint{1}) {
                    ok = false;
                    break;
                }
            }
            if (ok) return Mint{g};
        }
        return Mint{};
    }();

    static constexpr struct _DFTInfo {
        Mint rt[LG_MAXN + 1], irt[LG_MAXN + 1];
        Mint w4d[LG_MAXN - 2], iw4d[LG_MAXN - 2];
        alignas(32) Mint w8d[LG_MAXN - 3][8], iw8d[LG_MAXN - 3][8];

        static constexpr void fillpow(Mint* a, Mint x, int k) {
            a[0] = 1;
            for (int i = 1; i < k; i++) a[i] = a[i - 1] * x;
        }
        constexpr _DFTInfo() {
            Mint prd = qpow(G, (P - 1) >> LG_MAXN), iprd = prd.inv();
            for (size_t i = LG_MAXN; ~i; i--) {
                rt[i] = prd, irt[i] = iprd;
                prd *= prd, iprd *= iprd;
            }
            prd = iprd = 1;
            for (size_t i = 0; i + 3 <= LG_MAXN; i++) {
                w4d[i] = rt[i + 3] * prd, prd *= irt[i + 3];
                iw4d[i] = irt[i + 3] * iprd, iprd *= rt[i + 3];
            }
            prd = iprd = 1;
            for (size_t i = 0; i + 4 <= LG_MAXN; i++) {
                fillpow(w8d[i], rt[i + 4] * prd, 8), prd *= irt[i + 4];
                fillpow(iw8d[i], irt[i + 4] * iprd, 8), iprd *= rt[i + 4];
            }
        }
    } dft_info{};

    static inline struct _InvInfo {
        alignas(32) Mint inv[MAXN];
        size_t inv_len = 2;

        _InvInfo() { inv[1] = 1; }
        void prepare(size_t len) {
            for (size_t i = inv_len; i <= len; i++)
                inv[i] = -Mint{P / i} * inv[P % i];
            inv_len = len;
        }
    } inv_info;

    static constexpr m256i V_P = vset1(P), V_P2 = vset1(2 * P),
                           V_R = vset1(M.R), V_R2 = vset1(M.R2),
                           V_P_INV = vset1(M.P_INV),
                           V_I = vset1(dft_info.rt[2]),
                           V_I_INV = vset1(dft_info.irt[2]);

    static m256i shrink(m256i x) {
        return _mm256_min_epu32(x, _mm256_sub_epi32(x, V_P2));
    }
    template <bool strict = true>
    static m256i add(m256i x, m256i y) {
        m256i t = _mm256_add_epi32(x, y);
        return strict ? shrink(t) : t;
    }
    template <bool strict = true>
    static m256i sub(m256i x, m256i y) {
        m256i s = _mm256_sub_epi32(x, y);
        m256i t = _mm256_add_epi32(s, V_P2);
        return strict ? _mm256_min_epu32(s, t) : t;
    }
    template <bool strict = true, int mask = 0xFF>
    static m256i neg(m256i x) {
        m256i y = _mm256_sub_epi32(V_P2, x);
        if constexpr (strict) {
            m256i eq = _mm256_cmpeq_epi32(x, _mm256_setzero_si256());
            y = _mm256_andnot_si256(eq, y);
        }
        return mask == 0xFF ? y : vblend<mask>(x, y);
    }
    static m256i redc(m256i t) {
        m256i m = _mm256_mul_epu32(t, V_P_INV);
        m256i v = _mm256_mul_epu32(m, V_P);
        return _mm256_add_epi64(t, v);
    }
    static m256i mul(m256i x, m256i y) {
        m256i x1 = vshuffle<0xF5>(x);
        m256i y1 = vshuffle<0xF5>(y);
        m256i res0 = redc(_mm256_mul_epu32(x, y));
        m256i res1 = redc(_mm256_mul_epu32(x1, y1));
        res0 = vshuffle<0xF5>(res0);
        return vblend<0xAA>(res0, res1);
    }

    static void clear(Mint* a, size_t len) {
        std::memset((void*)a, 0, len * sizeof(Mint));
    }
    static void copy(const Mint* a, size_t len, Mint* out, size_t pad = 0) {
        std::memcpy((void*)out, (const void*)a, len * sizeof(Mint));
        if (pad) clear(out + len, pad - len);
    }

    // out <- a + b
    static void add(const Mint* a, const Mint* b, size_t len, Mint* out) {
        size_t i = 0;
        for (; i + 7 < len; i += 8)
            vstore(out + i, add(vload(a + i), vload(b + i)));
        for (; i < len; i++) out[i] = a[i] + b[i];
    }
    // out <- a - b
    static void sub(const Mint* a, const Mint* b, size_t len, Mint* out) {
        size_t i = 0;
        for (; i + 7 < len; i += 8)
            vstore(out + i, sub(vload(a + i), vload(b + i)));
        for (; i < len; i++) out[i] = a[i] - b[i];
    }
    // out <- -a
    static void neg(const Mint* a, size_t len, Mint* out) {
        size_t i = 0;
        for (; i + 7 < len; i += 8) vstore(out + i, neg(vload(a + i)));
        for (; i < len; i++) out[i] = -a[i];
    }
    // out <- a \dot b
    static void dot(const Mint* a, const Mint* b, size_t len, Mint* out) {
        size_t i = 0;
        for (; i + 7 < len; i += 8)
            vstore(out + i, mul(vload(a + i), vload(b + i)));
        for (; i < len; i++) out[i] = a[i] * b[i];
    }
    // out <- k * a
    static void scale(const Mint* a, Mint k, size_t len, Mint* out) {
        size_t i = 0;
        const m256i vk = vset1(k);
        for (; i + 7 < len; i += 8) vstore(out + i, mul(vload(a + i), vk));
        for (; i < len; i++) out[i] = a[i] * k;
    }

    template <bool inv, bool strict = true>
    static m256i butterfly8(m256i v) {
        constexpr auto w8 = inv ? dft_info.irt[3] : dft_info.rt[3],
                       w4 = inv ? dft_info.irt[2] : dft_info.rt[2];
        constexpr auto W1 = vsetr(1, 1, 1, 1, 1, w8, w4, w8 * w4),
                       W2 = vsetr(1, 1, 1, w4, 1, 1, 1, w4);
        if constexpr (!inv) {
            v = mul(add<false>(vpermute<0x4E>(v), neg<false, 0xF0>(v)), W1);
            v = mul(add<false>(vshuffle<0x4E>(v), neg<false, 0xCC>(v)), W2);
            return add<strict>(vshuffle<0xB1>(v), neg<false, 0xAA>(v));
        } else {
            v = mul(add<false>(vshuffle<0xB1>(v), neg<false, 0xAA>(v)), W2);
            v = mul(add<false>(vshuffle<0x4E>(v), neg<false, 0xCC>(v)), W1);
            return add<strict>(vpermute<0x4E>(v), neg<false, 0xF0>(v));
        }
    }
    static void DIF(Mint* a, size_t len) {
        if (len == 1) return;
        if (len == 2) {
            Mint x = a[0], y = a[1];
            a[0] = x + y, a[1] = x - y;
        } else if (len == 4) {
            constexpr Mint I = dft_info.rt[2];
            Mint A = a[0], B = a[1], C = a[2], D = a[3];
            Mint t0 = A + C, t1 = A - C, t2 = B + D, t3 = I * (B - D);
            a[0] = t0 + t2, a[1] = t1 + t3;
            a[2] = t0 - t2, a[3] = t1 - t3;
        } else {
            size_t i = len;
            if ((std::countr_zero(len) & 1) == 0) {
                i >>= 1;
#pragma GCC unroll(8)
                for (size_t k = 0; k < i; k += 8) {
                    auto x = vload(a + k), y = vload(a + i + k);
                    vstore(a + k, add(x, y));
                    vstore(a + i + k, sub(x, y));
                }
            }
            for (size_t s = i >> 2; i >= 32; i >>= 2, s >>= 2) {
                Mint _w1{1}, _w2{1}, _w3{1};
                for (size_t j = 0, jc = 0; j < len; j += i, jc++) {
                    auto w1 = vset1(_w1), w2 = vset1(_w2), w3 = vset1(_w3);
                    auto pA = a + j, pB = pA + s, pC = pB + s, pD = pC + s;
#pragma GCC unroll(8)
                    for (size_t k = 0; k < s; k += 8) {
                        auto A = shrink(vload(pA + k));
                        auto C = mul(vload(pC + k), w2);
                        auto B = mul(vload(pB + k), w1);
                        auto D = mul(vload(pD + k), w3);
                        auto t0 = add(A, C), t1 = sub(A, C), t2 = add(B, D),
                             t3 = mul(V_I, sub<false>(B, D));
                        vstore(pA + k, add<false>(t0, t2));
                        vstore(pB + k, sub<false>(t0, t2));
                        vstore(pC + k, add<false>(t1, t3));
                        vstore(pD + k, sub<false>(t1, t3));
                    }
                    _w1 *= dft_info.w4d[std::countr_one(jc)];
                    _w2 = _w1 * _w1;
                    _w3 = _w2 * _w1;
                }
            }
            auto w = V_R;
#pragma GCC unroll(8)
            for (size_t j = 0; j < len; j += 8) {
                vstore(a + j, butterfly8<false>(mul(vload(a + j), w)));
                w = mul(w, vload(dft_info.w8d[std::countr_one(j >> 3)]));
            }
        }
    }
    static void DIT(Mint* a, size_t len) {
        if (len == 1) return;
        if (len == 2) {
            constexpr Mint i2 = Mint{2}.inv();
            Mint x = a[0], y = a[1];
            a[0] = (x + y) * i2, a[1] = (x - y) * i2;
        } else if (len == 4) {
            constexpr Mint i4 = Mint{4}.inv(), I = dft_info.irt[2];
            Mint A = a[0], B = a[1], C = a[2], D = a[3];
            Mint t0 = A + C, t1 = A - C, t2 = B + D, t3 = I * (B - D);
            a[0] = (t0 + t2) * i4, a[1] = (t1 + t3) * i4;
            a[2] = (t0 - t2) * i4, a[3] = (t1 - t3) * i4;
        } else {
            auto w = V_R;
#pragma GCC unroll(8)
            for (size_t j = 0; j < len; j += 8) {
                vstore(a + j, mul(butterfly8<true>(vload(a + j)), w));
                w = mul(w, vload(dft_info.iw8d[std::countr_one(j >> 3)]));
            }
            for (size_t i = 32, s = i >> 2; i <= len; i <<= 2, s <<= 2) {
                Mint _w1{1}, _w2{1}, _w3{1};
                for (size_t j = 0, jc = 0; j < len; j += i, jc++) {
                    auto w1 = vset1(_w1), w2 = vset1(_w2), w3 = vset1(_w3);
                    auto pA = a + j, pB = pA + s, pC = pB + s, pD = pC + s;
#pragma GCC unroll(8)
                    for (size_t k = 0; k < s; k += 8) {
                        auto A = vload(pA + k);
                        auto B = vload(pB + k);
                        auto C = vload(pC + k);
                        auto D = vload(pD + k);
                        auto t0 = add(A, B), t1 = sub(A, B), t2 = add(C, D),
                             t3 = mul(V_I_INV, sub<false>(C, D));
                        vstore(pA + k, add(t0, t2));
                        vstore(pB + k, mul(add<false>(t1, t3), w1));
                        vstore(pC + k, mul(sub<false>(t0, t2), w2));
                        vstore(pD + k, mul(sub<false>(t1, t3), w3));
                    }
                    _w1 *= dft_info.iw4d[std::countr_one(jc)];
                    _w2 = _w1 * _w1;
                    _w3 = _w2 * _w1;
                }
            }
            if ((std::countr_zero(len) & 1) == 0) {
                size_t s = len >> 1;
#pragma GCC unroll(8)
                for (size_t k = 0; k < s; k += 8) {
                    auto x = vload(a + k), y = vload(a + k + s);
                    vstore(a + k, add(x, y));
                    vstore(a + k + s, sub(x, y));
                }
            }
            scale(a, Mint{len}.inv(), len, a);
        }
    }

    // a <- b'
    static void polyder(const Mint* f, size_t len, Mint* out) {
        constexpr auto init = vsetr(Mint{1}, 2, 3, 4, 5, 6, 7, 8),
                       step = vset1(Mint{8});
        size_t i = 0;
        for (auto v = init; i + 8 < len; i += 8, v = add(v, step))
            vstore(out + i, mul(v, vloadu(f + i + 1)));
        for (; i + 1 < len; i++) out[i] = Mint{i + 1} * f[i + 1];
        out[len - 1] = 0;
    }
    // a <- \int b \dd x
    static void polyint(const Mint* f, size_t len, Mint* out, Mint C = 0) {
        size_t i = len - 1;
        inv_info.prepare(len);
        for (; i > 0 && (i & 7); i--) out[i] = f[i - 1] * inv_info.inv[i];
        for (; i > 0; i -= 8) {
            auto x = vload(f + i - 8), y = vloadu(inv_info.inv + i - 7);
            vstoreu(out + i - 7, mul(x, y));
        }
        out[0] = C;
    }
    // f <- f * g, assume f, g can both be modified
    static void polymul(Mint* f, Mint* g, size_t len) {
        DIF(f, len);
        if (f != g) DIF(g, len);
        dot(f, g, len, f);
        DIT(f, len);
    }
    // out <- f^{-1}
    static void polyinv(const Mint* f, size_t len_f, Mint* out) {
        if (!len_f || f[0] == 0) throw std::invalid_argument("[x^0] is 0");
        out[0] = f[0].inv();
        size_t len = std::bit_ceil(len_f);
        auto t1 = Pool::allocate(len), t2 = Pool::allocate(len);
        for (size_t k = 1, k2 = 2; k < len; k = k2, k2 <<= 1) {
            copy(f, std::min(k2, len_f), t1, k2);
            copy(out, k, t2, k2);
            DIF(t1, k2), DIF(t2, k2), dot(t1, t2, k2, t1), DIT(t1, k2);
            clear(t1, k), DIF(t1, k2), dot(t1, t2, k2, t1), DIT(t1, k2);
            neg(t1 + k, k, out + k);
        }
    }
    // out <- ln(f)
    static void polyln(const Mint* f, size_t len_f, Mint* out) {
        if (!len_f || f[0] != 1) throw std::invalid_argument("[x^0] is not 1");
        size_t len = std::bit_ceil(len_f);
        auto d = Pool::allocate(len), g = Pool::allocate(len),
             t1 = Pool::allocate(len), t2 = Pool::allocate(len),
             t3 = Pool::allocate(len);
        polyder(f, len_f, d), clear(d + len_f, len - len_f);
        out[0] = d[0], g[0] = 1;
        for (size_t k = 1, k2 = 2; k < len; k = k2, k2 <<= 1) {
            copy(g, k, t1, k2);
            copy(f, std::min(k2, len_f), t2, k2);
            DIF(t1, k2), DIF(t2, k2), dot(t1, t2, k2, t2);
            DIT(t2, k2), clear(t2, k), DIF(t2, k2);
            copy(g, k, t3, k2);
            DIF(t3, k2), dot(t2, t3, k2, t3), DIT(t3, k2);
            neg(t3 + k, k, g + k);
            copy(d, k2, t3), DIF(t3, k2), dot(t3, t1, k2, t1);
            copy(out, k, t3, k2), DIF(t3, k2), dot(t3, t2, k2, t2);
            sub(t1, t2, k2, t3), DIT(t3, k2);
            copy(t3 + k, k, out + k);
        }
        polyint(out, len, out);
    }
    // out <- exp(f)
    static void polyexp(const Mint* f, size_t len_f, Mint* out) {
        if (!len_f) return;
        if (f[0] != 0) throw std::invalid_argument("[x^0] is not 0");
        size_t len = std::bit_ceil(len_f);
        auto g = Pool::allocate(len), t1 = Pool::allocate(len),
             t2 = Pool::allocate(len), t3 = Pool::allocate(len),
             t4 = Pool::allocate(len);
        out[0] = g[0] = 1;
        for (size_t k = 1, k2 = 2; k < len; k = k2, k2 <<= 1) {
            copy(out, k, t1, k2), DIF(t1, k2);
            copy(g, k, t2, k2), DIF(t2, k2);
            for (size_t i = 0; i < k2; i += 8) {
                m256i x = vload(t1 + i), y = vload(t2 + i);
                vstore(t3 + i, mul(neg(x), mul(y, y)));
            }
            DIT(t3, k2), copy(g, k, t3), DIF(t3, k2);
            polyder(out, k, t4), clear(t4 + k, k);
            DIF(t4, k2), dot(t4, t3, k2, t4), DIT(t4, k2);
            polyint(t4, k2, t4);
            sub(t4 + k, f + k, std::min(len_f, k2) - k, t4 + k);
            clear(t4, k), DIF(t4, k2);
            for (size_t i = 0; i < k2; i += 8) {
                m256i d = vload(t4 + i);
                vstore(t1 + i, mul(vload(t1 + i), sub(V_R, d)));
                vstore(t2 + i, add(vload(t3 + i), mul(vload(t2 + i), d)));
            }
            DIT(t1, k2), copy(t1 + k, k, out + k);
            DIT(t2, k2), copy(t2 + k, k, g + k);
        }
    }
    // out <- sqrt(f)
    static void polysqrt(const Mint* f, size_t len_f, Mint* out) {
        if (!len_f || f[0]() == 0) throw std::invalid_argument("[x^0] is 0");
        auto out0 = sqrt(f[0]);
        if (!out0) throw std::invalid_argument("sqrt does not exist");
        size_t len = std::bit_ceil(len_f);
        auto h = Pool::allocate(len), t1 = Pool::allocate(len),
             t2 = Pool::allocate(len), t3 = Pool::allocate(len);
        out[0] = out0.transform([](auto x) { return std::min(x(), P - x()); })
                     .value();
        h[0] = out[0].inv();
        for (size_t k = 1, k2 = 2; k < len; k = k2, k2 <<= 1) {
            copy(f, std::min(k2, len_f), t1, k2), DIF(t1, k2);
            copy(out, k, t2, k2), DIF(t2, k2);
            copy(h, k, t3, k2), DIF(t3, k2);
            for (size_t i = 0; i < k2; i += 8) {
                constexpr auto C = vset1(-Mint{2}.inv());
                auto vf = vload(t1 + i), vg = vload(t2 + i), vh = vload(t3 + i);
                vstore(t1 + i, mul(sub(mul(vg, vg), vf), mul(vh, C)));
            }
            DIT(t1, k2), copy(out, k, t1), copy(t1 + k, k, out + k);
            DIF(t1, k2), dot(t1, t3, k2, t1), DIT(t1, k2);
            clear(t1, k), DIF(t1, k2), dot(t1, t3, k2, t1), DIT(t1, k2);
            neg(t1 + k, k, h + k);
        }
    }
};

}  // namespace detail

template <u32 P, size_t _MAXN = size_t(-1)>
class FPoly {
public:
    using U = detail::PolyUtils<P, _MAXN>;
    using Mint = U::Mint;
    using Pool = U::Pool;

private:
    size_t _len = 0;
    mutable Pool::pointer_type _data{};

    template <std::ranges::input_range R,
              typename T = std::ranges::range_value_t<R>>
    static constexpr bool can_fast_init =
        (std::is_same_v<T, u32> || std::is_same_v<T, i32>) &&
        std::ranges::contiguous_range<R>;

public:
    FPoly() = default;
    explicit FPoly(size_t n, bool no_init = false):
        _len{n}, _data{Pool::allocate(n)} {
        if (!no_init) U::clear(_data, n);
    }
    FPoly(const std::initializer_list<Mint>& init):
        _len{init.size()}, _data{Pool::allocate(_len)} {
        U::copy(init.begin(), init.size(), _data);
    }
    template <std::ranges::input_range R>
        requires can_fast_init<R>
    FPoly(R&& r): _len{std::ranges::size(r)}, _data{Pool::allocate(_len)} {
        size_t i = 0;
        for (; i + 7 < _len; i += 8) {
            auto v = U::vloadu(r.data() + i);
            if constexpr (std::is_signed_v<std::ranges::range_value_t<R>>)
                v = _mm256_add_epi32(v, U::vset1((1u << 31) / P * P));
            U::vstore(_data + i, U::mul(v, U::V_R2));
        }
        for (; i < _len; i++) _data[i] = Mint{r.data()[i]};
    }
    template <std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_value_t<R>, Mint> &&
                 (!can_fast_init<R> &&
                  !std::same_as<std::remove_cvref_t<R>, FPoly>)
    FPoly(R&& r) {
        std::vector<Mint> tmp{};
        for (auto&& x: r) tmp.emplace_back(x);
        _len = tmp.size();
        _data = Pool::allocate(_len);
        U::copy(tmp.data(), _len, _data);
    }
    template <std::input_iterator Iter, std::sentinel_for<Iter> Sent>
    FPoly(Iter begin, Sent end): FPoly(std::ranges::subrange(begin, end)) {}

    FPoly(FPoly&& other) = default;
    FPoly(const FPoly& other): _data() {
        _len = other._len;
        _data = Pool::allocate(_len);
        U::copy(other._data, _len, _data);
    }
    FPoly& operator=(FPoly&& other) = default;
    FPoly& operator=(const FPoly& other) { return *this = FPoly(other); }

    void resize(size_t sz) {
        reserve(sz);
        if (sz > _len) U::clear(_data + _len, sz - _len);
        _len = sz;
    }
    void reserve(size_t sz) const {
        if (sz > _data.capacity()) {
            auto new_data = Pool::allocate(sz);
            U::copy(_data, _len, new_data);
            _data = std::move(new_data);
        }
    }
    void clear() { _len = 0; }

    auto size() const { return _len; }
    auto data() { return (Mint*)_data; }
    auto data() const { return (const Mint*)_data; }
    auto begin() { return (Mint*)_data; }
    auto begin() const { return (const Mint*)_data; }
    auto end() { return begin() + _len; }
    auto end() const { return begin() + _len; }
    auto& operator[](size_t idx) { return _data[idx]; }
    auto operator[](size_t idx) const { return _data[idx]; }

    FPoly& operator+=(const FPoly& other) {
        if (other._len > _len) resize(other._len);
        U::add(_data, other._data, other._len, _data);
        return *this;
    }
    FPoly& operator-=(const FPoly& other) {
        if (other._len > _len) resize(other._len);
        U::sub(_data, other._data, other._len, _data);
        return *this;
    }
    FPoly& operator*=(FPoly other) {
        if (_len == 0 || other._len == 0) return clear(), *this;
        size_t n = _len + other._len - 1, nn = std::bit_ceil(n);
        resize(nn);
        other.resize(nn);
        U::polymul(_data, other._data, nn);
        return resize(n), *this;
    }
    FPoly& operator*=(Mint k) {
        U::scale(_data, k, _len, _data);
        return *this;
    }
    friend FPoly operator-(FPoly f) {
        return U::neg(f._data, f._len, f._data), f;
    }
    friend FPoly operator+(FPoly f, const FPoly& g) { return f += g, f; }
    friend FPoly operator-(FPoly f, const FPoly& g) { return f -= g, f; }
    friend FPoly operator*(Mint k, FPoly f) { return f *= k, f; }
    friend FPoly operator*(FPoly f, Mint k) { return f *= k, f; }
    friend FPoly operator*(FPoly f, FPoly g) { return f *= g, f; }

    template <u32 Q, size_t N>
    friend FPoly<Q, N> inv(const FPoly<Q, N>&);
    template <u32 Q, size_t N>
    friend FPoly<Q, N> ln(const FPoly<Q, N>&);
    template <u32 Q, size_t N>
    friend FPoly<Q, N> exp(const FPoly<Q, N>&);
    template <u32 Q, size_t N>
    friend FPoly<Q, N> sqrt(const FPoly<Q, N>&);
};

#define Poly FPoly<Q, N>
#define U Poly::U
template <u32 Q, size_t N>
Poly inv(const Poly& f) {
    Poly res(f._len, true);
    U::polyinv(f._data, f._len, res._data);
    return res;
}

template <u32 Q, size_t N>
Poly ln(const Poly& f) {
    Poly res(f._len, true);
    U::polyln(f._data, f._len, res._data);
    return res;
}

template <u32 Q, size_t N>
Poly exp(const Poly& f) {
    Poly res(f._len, true);
    U::polyexp(f._data, f._len, res._data);
    return res;
}

template <u32 Q, size_t N>
Poly sqrt(const Poly& f) {
    int k = std::ranges::find_if(f, [](auto x) { return x(); }) - f.begin();
    if (k % 2 != 0) throw std::invalid_argument("sqrt does not exist");
    Poly res(f._len, true);
    if (k == 0) U::polysqrt(f._data, f._len, res._data);
    else {
        auto tmp = U::Pool::allocate(f._len);
        U::copy(f._data + k, f._len - k, tmp, f._len);
        U::polysqrt(tmp, f._len, res._data);
        std::memmove(res._data + k / 2, res._data, f._len - k / 2);
        U::clear(res._data, k / 2);
    }
    return res;
}

template <u32 Q, size_t N>
std::pair<Poly, Poly> div(const Poly& f, const Poly& g) {
    size_t n = f.size(), m = g.size();
    if (m == 0) throw std::invalid_argument("divider is empty");
    if (n < m) return {{}, f};
    Poly h(n - m + 1, true), q(n - m + 1, true), r{};
    for (size_t i = 0; i < n - m + 1; i++) {
        q[i] = f[n - 1 - i];
        h[i] = i > m - 1 ? 0 : g[m - 1 - i];
    }
    q *= inv(h), q.resize(n - m + 1), std::ranges::reverse(q);
    r = f - q * g, r.resize(m - 1);
    return {std::move(q), std::move(r)};
}
#undef Poly
#undef U

}  // namespace cp
#pragma GCC pop_options

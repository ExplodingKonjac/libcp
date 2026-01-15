#include <emmintrin.h>
#include <immintrin.h>

#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <print>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "modint.hpp"

#pragma GCC target("avx2", "fma")

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

    static constexpr bool is_prime = [] {
        for (u32 i = 2; (u64)i * i <= P; i++)
            if (P % i == 0) return false;
        return true;
    }();
    static constexpr size_t MAXN =
        std::min(_MAXN, size_t(1) << std::countr_zero(P - 1));
    static constexpr size_t B = 8192;

    static_assert(is_prime, "P must be prime number");
    static_assert(MAXN >= 4, "MAXN must be at least 4");

    static constexpr const auto& M = Mint::mont;

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
    static constexpr Mint I = qpow(G, (P - 1) / 4).strict(),
                          I_INV = (-I).strict();
    static constexpr m256i V_P = vset1(P), V_P2 = vset1(2 * P),
                           V_P_INV = vset1(M.P_INV), V_R = vset1(M.R),
                           V_R2 = vset1(M.R2), V_I = vset1(I.raw());

    alignas(32) Mint g[MAXN * 2], inv[MAXN + 1];
    size_t g_len = 4, inv_len = 2;

    PolyUtils() { inv[1] = 1; }
    void prepareG(size_t len) {
        constexpr Mint one = Mint{1}.strict();
        for (size_t& i = g_len; i <= len; i <<= 1) {
            size_t s = i / 4;
            auto w0 = g + i, w1 = w0 + s, w2 = w1 + s, w3 = w2 + s;
            Mint cur = one, gn = qpow(G, (P - 1) / i);
            std::fill(w0, w0 + s, one);
            for (size_t k = 0; k < s; k++) w1[k] = cur, cur *= gn;
            dot(w1, w1, s, w2);
            dot(w1, w2, s, w3);
            for (size_t k = s; k < i; k++) w0[k] = w0[k].strict();
        }
    }
    void prepareInv(size_t len) {
        for (size_t& i = inv_len; i <= len; i++)
            inv[i] = -Mint{P / i} * inv[P % i];
    }

    template <bool lazy = false>
    static m256i add(m256i x, m256i y) {
        m256i t = _mm256_add_epi32(x, y);
        return lazy ? t : _mm256_min_epu32(t, _mm256_sub_epi32(t, V_P2));
    }
    template <bool lazy = false>
    static m256i sub(m256i x, m256i y) {
        m256i s = _mm256_sub_epi32(x, y);
        m256i t = _mm256_add_epi32(s, V_P2);
        return lazy ? t : _mm256_min_epu32(s, t);
    }
    static m256i neg(m256i x) {
        m256i mask = _mm256_cmpeq_epi32(x, _mm256_setzero_si256());
        m256i y = _mm256_sub_epi32(V_P2, x);
        return _mm256_andnot_si256(mask, y);
    }
    static m256i redc(m256i t) {
        m256i m = _mm256_mul_epu32(t, V_P_INV);
        m256i v = _mm256_mul_epu32(m, V_P);
        return _mm256_add_epi64(t, v);
    }
    static m256i mul(m256i x, m256i y) {
        m256i x1 = _mm256_shuffle_epi32(x, 0xF5);
        m256i y1 = _mm256_shuffle_epi32(y, 0xF5);
        m256i res0 = redc(_mm256_mul_epu32(x, y));
        m256i res1 = redc(_mm256_mul_epu32(x1, y1));
        res0 = _mm256_shuffle_epi32(res0, 0xF5);
        return _mm256_blend_epi32(res0, res1, 0xAA);
    }

    static void clear(Mint* a, size_t len) {
        std::memset((void*)a, 0, len * sizeof(Mint));
    }
    static void copy(const Mint* a, size_t len, Mint* out, size_t pad = 0) {
        std::memcpy((void*)out, (const void*)a, len * sizeof(Mint));
        if (pad) clear(out + len, pad - len);
    }
    static void rev(Mint* begin, Mint* end) {
        constexpr m256i rev_mask = (m256i)__v8si{7, 6, 5, 4, 3, 2, 1, 0};
        for (; end - begin >= 16; begin += 8, end -= 8) {
            m256i x = vloadu(begin), y = vloadu(end - 8);
            x = _mm256_permutevar8x32_epi32(x, rev_mask);
            y = _mm256_permutevar8x32_epi32(y, rev_mask);
            vstoreu(begin, y), vstoreu(end - 8, x);
        }
        std::reverse(begin, end);
    }

    // out <- a + b
    static void add(const Mint* a, const Mint* b, size_t len, Mint* out) {
        size_t i = 0;
        for (; i + 7 < len; i += 8)
            vstore(&out[i], add(vload(&a[i]), vload(&b[i])));
        for (; i < len; i++) out[i] = a[i] + b[i];
    }
    // out <- a - b
    static void sub(const Mint* a, const Mint* b, size_t len, Mint* out) {
        size_t i = 0;
        for (; i + 7 < len; i += 8)
            vstore(&out[i], sub(vload(&a[i]), vload(&b[i])));
        for (; i < len; i++) out[i] = a[i] - b[i];
    }
    // out <- -a
    static void neg(const Mint* a, size_t len, Mint* out) {
        size_t i = 0;
        for (; i + 7 < len; i += 8) vstore(&out[i], neg(vload(&a[i])));
        for (; i < len; i++) out[i] = -a[i];
    }
    // out <- a \dot b
    static void dot(const Mint* a, const Mint* b, size_t len, Mint* out) {
        size_t i = 0;
        for (; i + 7 < len; i += 8)
            vstore(&out[i], mul(vload(&a[i]), vload(&b[i])));
        for (; i < len; i++) out[i] = a[i] * b[i];
    }
    // out <- k * a
    static void scale(const Mint* a, Mint k, size_t len, Mint* out) {
        size_t i = 0;
        const m256i vk = vset1(reinterpret_cast<int&>(k));
        for (; i + 7 < len; i += 8) vstore(&out[i], mul(vload(&a[i]), vk));
        for (; i < len; i++) out[i] = a[i] * k;
    }

    template <bool inv = false>
    static void vbutterfly(m256i& A, m256i& B, m256i& C, m256i& D) {
        m256i t0 = add(A, C), t1 = sub(A, C), t2 = add(B, D),
              t3 = mul(sub<inv>(B, D), V_I);
        A = add(t0, t2), B = add<!inv>(t1, t3);
        C = sub<!inv>(t0, t2), D = sub<!inv>(t1, t3);
    }
    template <bool inv = false>
    static m256i vbutterfly2(m256i v) {
        m256i v_swp = _mm256_shuffle_epi32(v, 0xB1);
        m256i s = add(v_swp, v);
        m256i t = sub(v_swp, v);
        return _mm256_blend_epi32(s, t, 0xAA);
    }
    template <bool inv = false>
    static m256i vbutterfly4(m256i v) {
        constexpr m256i c = (m256i)__v8su{M.R, I.raw(), P - M.R, I_INV.raw(),
                                          M.R, I.raw(), P - M.R, I_INV.raw()};
        m256i v_swp = _mm256_shuffle_epi32(v, 0x4E);
        m256i s = add(v, v_swp);
        m256i t = sub(v, v_swp);
        m256i u = _mm256_unpacklo_epi64(s, t);
        m256i L = _mm256_shuffle_epi32(u, 0x88);
        m256i R = mul(_mm256_shuffle_epi32(u, 0xDD), c);
        return add(L, R);
    }
    template <bool inv = false>
    static m256i vbutterfly8(m256i v) {
        constexpr m256i c = (m256i)__v8su{M.R,     M.R,     I.raw(), I.raw(),
                                          P - M.R, P - M.R, I.raw(), I.raw()};
        m256i v_swp = _mm256_permute4x64_epi64(v, 0x4E);
        m256i s = add(v, v_swp), t = sub(v, v_swp);
        m256i L = _mm256_unpacklo_epi64(s, t);
        m256i R = _mm256_unpackhi_epi64(s, t);
        m256i L_fixed = _mm256_permute4x64_epi64(L, 0x44);
        m256i R_fixed = mul(R, c);
        return add<!inv>(L_fixed, R_fixed);
    }
    template <bool inv = false>
    static void vbutterfly16(m256i& v0, m256i& v1) {
        constexpr m256i c = (m256i)__v8su{M.R,     M.R,     M.R,     M.R,
                                          I.raw(), I.raw(), I.raw(), I.raw()};
        m256i s = add(v0, v1);
        m256i t = mul(sub<true>(v0, v1), c);
        m256i L = _mm256_permute2f128_si256(s, t, 0x20);
        m256i R = _mm256_permute2f128_si256(s, t, 0x31);
        v0 = add<!inv>(L, R), v1 = sub<!inv>(L, R);
    }
    void DIFLayer(Mint* a, size_t len, size_t& i) {
        if (i == 2) {
#pragma GCC unroll 8
            for (size_t j = 0; j < len; j += 8)
                vstore(a + j, vbutterfly2(vload(a + j)));
            i = 0;
        } else if (i == 4) {
#pragma GCC unroll 8
            for (size_t j = 0; j < len; j += 8)
                vstore(a + j, vbutterfly4(vload(a + j)));
            i = 0;
        } else if (i == 8) {
            m256i rot = vload(g + 8);
#pragma GCC unroll 8
            for (size_t j = 0; j < len; j += 8) {
                m256i v = vload(a + j);
                v = mul(vbutterfly8(v), rot);
                v = vbutterfly2(v);
                vstore(a + j, v);
            }
            i = 0;
        } else if (i == 16) {
            m256i rot0 = vload(g + 16), rot1 = vload(g + 24);
#pragma GCC unroll 8
            for (size_t j = 0; j < len; j += 16) {
                m256i v0 = vload(a + j), v1 = vload(a + j + 8);
                vbutterfly16(v0, v1);
                v0 = vbutterfly4(mul(v0, rot0));
                v1 = vbutterfly4(mul(v1, rot1));
                vstore(a + j, v0), vstore(a + j + 8, v1);
            }
            i = 0;
        } else {
            size_t s = i >> 2;
            auto w1 = g + i + s, w2 = w1 + s, w3 = w2 + s;
            for (size_t j = 0; j < len; j += i) {
                auto pA = a + j, pB = pA + s, pC = pB + s, pD = pC + s;
#pragma GCC unroll 8
                for (size_t k = 0; k < s; k += 8) {
                    auto A = vload(pA + k), B = vload(pB + k),
                         C = vload(pC + k), D = vload(pD + k);
                    vbutterfly(A, B, C, D);
                    vstore(pA + k, A);
                    vstore(pB + k, mul(B, vload(w1 + k)));
                    vstore(pC + k, mul(C, vload(w2 + k)));
                    vstore(pD + k, mul(D, vload(w3 + k)));
                }
            }
            i >>= 2;
        }
    }
    void DITLayer(Mint* a, size_t len, size_t& i) {
        if (i == 2 && len >= 8) {
            m256i rot = vload(g + 8);
#pragma GCC unroll 8
            for (size_t j = 0; j < len; j += 8) {
                m256i v = vload(a + j);
                v = vbutterfly2<true>(v);
                v = vbutterfly8<true>(mul(v, rot));
                vstore(a + j, v);
            }
            i = 32;
        } else if (i == 4 && len >= 16) {
            m256i rot0 = vload(g + 16), rot1 = vload(g + 24);
#pragma GCC unroll 8
            for (size_t j = 0; j < len; j += 16) {
                m256i v0 = vload(a + j), v1 = vload(a + j + 8);
                v0 = mul(vbutterfly4<true>(v0), rot0);
                v1 = mul(vbutterfly4<true>(v1), rot1);
                vbutterfly16<true>(v0, v1);
                vstore(a + j, v0), vstore(a + j + 8, v1);
            }
            i = 64;
        } else if (i == 2) {
#pragma GCC unroll 8
            for (size_t j = 0; j < len; j += 8)
                vstore(a + j, vbutterfly2<true>(vload(a + j)));
            i = 8;
        } else if (i == 4) {
#pragma GCC unroll 8
            for (size_t j = 0; j < len; j += 8)
                vstore(a + j, vbutterfly4<true>(vload(a + j)));
            i = 16;
        } else {
            size_t s = i >> 2;
            auto w1 = g + i + s, w2 = w1 + s, w3 = w2 + s;
            for (size_t j = 0; j < len; j += i) {
                auto pA = a + j, pB = pA + s, pC = pB + s, pD = pC + s;
#pragma GCC unroll 8
                for (size_t k = 0; k < s; k += 8) {
                    auto A = vload(pA + k);
                    auto B = mul(vload(pB + k), vload(w1 + k));
                    auto C = mul(vload(pC + k), vload(w2 + k));
                    auto D = mul(vload(pD + k), vload(w3 + k));
                    vbutterfly<true>(A, B, C, D);
                    vstore(pA + k, A), vstore(pB + k, B);
                    vstore(pC + k, C), vstore(pD + k, D);
                }
            }
            i <<= 2;
        }
    }
    void DIF(Mint* a, size_t len) {
        size_t i = len;
        for (; i > B; DIFLayer(a, len, i));
        for (size_t j = 0; j < len; j += i)
            for (size_t k = i; k > 0; DIFLayer(a + j, i, k));
    }
    void DIT(Mint* a, size_t len) {
        size_t st = (std::countr_zero(len) & 1) ? 2 : 4,
               i = std::min(len, B << (std::countr_zero(B / st) & 1));
        for (size_t j = 0; j < len; j += i)
            for (size_t k = st; k <= i; DITLayer(a + j, i, k));
        for (i <<= 2; i <= len; DITLayer(a, len, i));
        scale(a, Mint{len}.inv(), len, a);
        rev(a + 1, a + len);
    }

    // a <- b'
    void polyder(const Mint* f, size_t len, Mint* out) {
        constexpr m256i init =
            (m256i)__v8su{M.toMont(1), M.toMont(2), M.toMont(3), M.toMont(4),
                          M.toMont(5), M.toMont(6), M.toMont(7), M.toMont(8)};
        constexpr m256i step = vset1(M.toMont(8));
        size_t i = 0;
        for (m256i v = init; i + 8 < len; i += 8, v = add(v, step))
            vstore(out + i, mul(v, vloadu(f + i + 1)));
        for (; i + 1 < len; i++) out[i] = Mint{i + 1} * f[i + 1];
        out[len - 1] = 0;
    }
    // a <- \int b \dd x
    void polyint(const Mint* f, size_t len, Mint* out, Mint C = 0) {
        size_t i = len - 1;
        prepareInv(len);
        for (; i > 0 && (i & 7); i--) out[i] = f[i - 1] * inv[i];
        for (; i > 0; i -= 8)
            vstoreu(out + i - 7, mul(vload(f + i - 8), vloadu(inv + i - 7)));
        out[0] = C;
    }
    // f <- f * g, assume f, g can both be modified
    void polymul(Mint* f, Mint* g, size_t len) {
        DIF(f, len);
        if (f != g) DIF(g, len);
        dot(f, g, len, f);
        DIT(f, len);
    }
    // out <- f^{-1}
    void polyinv(const Mint* f, size_t len_f, Mint* out) {
        if (!len_f || f[0] == 0) throw std::runtime_error("inv does not exist");
        out[0] = f[0].inv();
        size_t len = std::bit_ceil(len_f);
        auto t1 = Pool::allocate(len), t2 = Pool::allocate(len);
        prepareG(len);
        for (size_t k = 1, k2 = 2; k < len; k = k2, k2 <<= 1) {
            copy(f, std::min(k2, len_f), t1, k2);
            copy(out, k, t2, k2);
            DIF(t1, k2), DIF(t2, k2), dot(t1, t2, k2, t1), DIT(t1, k2);
            clear(t1, k), DIF(t1, k2), dot(t1, t2, k2, t1), DIT(t1, k2);
            neg(t1 + k, k, out + k);
        }
    }
    // out <- ln(f)
    void polyln(const Mint* f, size_t len_f, Mint* out) {
        if (!len_f || f[0] != 1) throw std::runtime_error("ln does not exist");
        size_t len = std::bit_ceil(len_f);
        auto d = Pool::allocate(len), g = Pool::allocate(len),
             t1 = Pool::allocate(len), t2 = Pool::allocate(len),
             t3 = Pool::allocate(len);
        prepareG(len);
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
    void polyexp(const Mint* f, size_t len_f, Mint* out) {
        if (!len_f) return;
        if (f[0] != 0) throw std::runtime_error("exp does not exist");
        size_t len = std::bit_ceil(len_f);
        auto g = Pool::allocate(len), t1 = Pool::allocate(len),
             t2 = Pool::allocate(len), t3 = Pool::allocate(len),
             t4 = Pool::allocate(len);
        prepareG(len);
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
};

}  // namespace detail

template <u32 P, size_t _MAXN = -1>
class FPoly {
private:
    using Mint = SModint<P>;
    using Pool = detail::AlignedPool<Mint, 32>;

    static inline detail::PolyUtils<P, _MAXN> u{};

    size_t _len = 0;
    Pool::pointer_type _data{};

public:
    FPoly() = default;
    explicit FPoly(size_t n, bool no_init = false):
        _len{n}, _data{Pool::allocate(n)} {
        if (!no_init) u.clear(_data, n);
    }
    FPoly(const std::initializer_list<Mint>& init):
        _len{init.size()}, _data{Pool::allocate(_len)} {
        u.copy(init.begin(), init.size(), _data);
    }
    template <std::ranges::contiguous_range R,
              typename T = std::ranges::range_value_t<R>>
        requires std::same_as<T, u32> || std::same_as<T, i32>
    FPoly(R&& r): _len{std::ranges::size(r)}, _data{Pool::allocate(_len)} {
        size_t i = 0;
        for (; i + 7 < _len; i += 8) {
            auto v = u.vloadu(r.data() + i);
            if constexpr (std::is_signed_v<T>)
                v = u.add(v, u.vset1((1 << 31) / P * P));
            u.vstore(_data + i, u.mul(u.vloadu(r.data() + i), u.V_R2));
        }
        for (; i < _len; i++) _data[i] = Mint{r.data()[i]};
    }
    template <std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_value_t<R>, Mint> &&
                 (!std::ranges::contiguous_range<R>) &&
                 (!std::same_as<std::remove_cvref_t<R>, FPoly>)
    FPoly(R&& r) {
        std::vector<Mint> tmp{};
        for (auto&& x: r) tmp.emplace_back(x);
        _len = tmp.size();
        _data = Pool::allocate(_len);
        u.copy(tmp.data(), _len, _data);
    }
    template <std::input_iterator Iter, std::sentinel_for<Iter> Sent>
    FPoly(Iter begin, Sent end): FPoly(std::ranges::subrange(begin, end)) {}

    FPoly(FPoly&& other) = default;
    FPoly(const FPoly& other): _data() {
        _len = other._len;
        _data = Pool::allocate(_len);
        u.copy(other._data, _len, _data);
    }
    FPoly& operator=(FPoly&& other) = default;
    FPoly& operator=(const FPoly& other) { return *this = FPoly(other); }

    void resize(size_t sz) {
        reserve(sz);
        if (sz > _len) u.clear(_data + _len, sz - _len);
        _len = sz;
    }
    void reserve(size_t sz) const {
        if (sz > _data.capacity()) {
            auto new_data = Pool::allocate(sz);
            u.copy(_data, _len, new_data);
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
        u.add(_data, other._data, other._len, _data);
        return *this;
    }
    FPoly& operator-=(const FPoly& other) {
        if (other._len > _len) resize(other._len);
        u.sub(_data, other._data, other._len, _data);
        return *this;
    }
    FPoly& operator*=(FPoly other) {
        if (_len == 0 || other._len == 0) return clear(), *this;
        size_t n = _len + other._len - 1, nn = std::bit_ceil(n);
        resize(nn);
        other.resize(nn);
        u.polymul(_data, other._data, nn);
        return resize(n), *this;
    }
    FPoly& operator*=(Mint k) {
        u.scale(_data, k, _len, _data);
        return *this;
    }
    friend FPoly operator-(FPoly f) {
        return u.neg(f._data, f._len, f._data), f;
    }
    friend FPoly operator+(FPoly f, const FPoly& g) { return f += g, f; }
    friend FPoly operator-(FPoly f, const FPoly& g) { return f -= g, f; }
    friend FPoly operator*(Mint k, FPoly f) { return f *= k, f; }
    friend FPoly operator*(FPoly f, Mint k) { return f *= k, f; }
    friend FPoly operator*(FPoly f, FPoly g) { return f *= g, f; }

    FPoly inv() const {
        FPoly res(_len, true);
        u.polyinv(_data, _len, res._data);
        return res;
    }

    template <u32 Q, size_t N>
    friend FPoly<Q, N> ln(const FPoly<Q, N>&);
    template <u32 Q, size_t N>
    friend FPoly<Q, N> exp(const FPoly<Q, N>&);
};

template <u32 Q, size_t N>
FPoly<Q, N> ln(const FPoly<Q, N>& f) {
    FPoly<Q, N> res(f._len, true);
    f.u.polyln(f._data, f._len, res._data);
    return res;
}
template <u32 Q, size_t N>
FPoly<Q, N> exp(const FPoly<Q, N>& f) {
    FPoly<Q, N> res(f._len, true);
    f.u.polyexp(f._data, f._len, res._data);
    return res;
}

}  // namespace cp

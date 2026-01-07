#pragma once
#include <immintrin.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#pragma GCC target("avx2", "fma")

namespace cp
{

using i32 = std::int32_t;
using u32 = std::uint32_t;
using i64 = std::int64_t;
using u64 = std::uint64_t;
using m256i = __m256i;
using v8si = i32 __attribute__((vector_size(32)));

namespace detail
{

template <typename T>
struct AlignedAllocator {
    static constexpr size_t A = alignof(T) < 32 ? 32 : alignof(T);
    using value_type = T;

    T* allocate(size_t n) {
        size_t size = (n * sizeof(T) + A - 1) & ~(A - 1);
#ifdef _WIN32
        return (T*)_aligned_malloc(size, A);
    }
    void deallocate(T* p, size_t n) { _aligned_free(p); }
#else
        return (T*)std::aligned_alloc(A, size);
    }
    void deallocate(T* p, size_t n) { std::free(p); }
#endif
};

constexpr m256i vload(const void* p) { return _mm256_load_si256((m256i*)p); }
constexpr void vstore(void* p, m256i a) { _mm256_store_si256((m256i*)p, a); }
constexpr m256i vset1(i32 x) { return (m256i)(v8si{x, x, x, x, x, x, x, x}); }

template <i32 P>
struct Montgomery {
    static constexpr i32 P_INV = [] {
        u32 x = P % 2;
        for (int i = 0; i < 5; i++) x *= (2u - P * x);
        return -x;
    }();
    static constexpr i32 R = (1ll << 32) % P;
    static constexpr i32 R2 = (long long)R * R % P;
    static constexpr m256i V_1 = vset1(1);
    static constexpr m256i V_P = vset1(P);
    static constexpr m256i V_P_INV = vset1(P_INV);
    static constexpr m256i V_R = vset1(R);
    static constexpr m256i V_R2 = vset1(R2);

    static constexpr i32 add(i32 x, i32 y) { return (x += y) >= P ? x - P : x; }
    static constexpr i32 sub(i32 x, i32 y) { return (x -= y) < 0 ? x + P : x; }
    static constexpr i32 mul(i32 x, i32 y) {
        i64 t = (i64)x * y;
        u32 m = (u32)t * P_INV;
        u32 res = (t + (u64)m * P) >> 32;
        return res >= P ? res - P : res;
    }
    template <typename T>
    static constexpr i32 qpow(i32 x, T y) {
        i32 res = toMont(1);
        for (; y; y >>= 1, x = mul(x, x))
            if (y & 1) res = mul(res, x);
        return res;
    }

    static m256i add(m256i x, m256i y) {
        x = _mm256_add_epi32(x, y);
        y = _mm256_sub_epi32(x, V_P);
        return _mm256_min_epu32(x, y);
    }
    static m256i sub(m256i x, m256i y) {
        x = _mm256_sub_epi32(x, y);
        y = _mm256_add_epi32(x, V_P);
        return _mm256_min_epu32(x, y);
    }
    static m256i mul(m256i x, m256i y) {
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
        m256i res = _mm256_or_si256(res0, res1);
        // strict reduction
        return _mm256_min_epu32(res, _mm256_sub_epi32(res, V_P));
    }

    static constexpr i32 toMont(i32 x) { return mul(x, R2); }
    static constexpr i32 fromMont(i32 x) { return mul(x, 1); }
    static m256i toMont(m256i x) { return mul(x, V_R2); }
    static m256i fromMont(m256i x) { return mul(x, V_1); }

    static void toMont(i32* a, size_t len) {
        size_t i = 0;
        for (; i + 7 < len; i += 8) vstore(a + i, toMont(vload(a + i)));
        for (; i < len; i++) a[i] = toMont(a[i]);
    }
    static void fromMont(i32* a, size_t len) {
        size_t i = 0;
        for (; i + 7 < len; i += 8) vstore(a + i, fromMont(vload(a + i)));
        for (; i < len; i++) a[i] = fromMont(a[i]);
    }
};

class Buffer {
public:
    Buffer(size_t size) {
        _p = reinterpret_cast<i32*>(_pool_p);
        _pool_p += (size * sizeof(i32) + 31) & ~31;
    }
    Buffer(const Buffer& buf) = delete;
    Buffer(Buffer&& buf) = delete;
    Buffer& operator=(const Buffer& other) = delete;
    Buffer& operator=(Buffer&& other) = delete;
    ~Buffer() { _pool_p = (_p ? reinterpret_cast<char*>(_p) : _pool_p); }
    auto get() { return _p; }
    auto get() const { return static_cast<const i32*>(_p); }
    i32& operator[](size_t i) { return _p[i]; }
    i32 operator[](size_t i) const { return _p[i]; }

private:
    alignas(32) static inline char _pool[(1 << 23) * sizeof(i32) * 2] = {};
    static inline char* _pool_p = _pool;

    Buffer() = default;
    i32* _p = nullptr;
};

template <i32 P, size_t _MAXN>
struct PolyUtils {
    using M = Montgomery<P>;

    static constexpr bool is_prime = [] {
        for (i32 i = 2; (i64)i * i <= P; i++)
            if (P % i == 0) return false;
        return true;
    }();
    static constexpr i32 G = [] {
        i32 phi = P - 1, tmp = phi, cnt = 0, divs[32] = {};
        for (i32 i = 2; (i64)i * i <= tmp; i++) {
            if (tmp % i == 0) divs[cnt++] = i;
            while (tmp % i == 0) tmp /= i;
        }
        if (tmp > 1) divs[cnt++] = tmp;
        for (i32 g = 2; g < P; g++) {
            i32 gm = M::toMont(g);
            bool ok = true;
            for (i32 i = 0; i < cnt; i++) {
                if (M::qpow(gm, phi / divs[i]) == M::R) {
                    ok = false;
                    break;
                }
            }
            if (ok) return g;
        }
        return 0;
    }();
    static constexpr size_t MAXN = [] {
        size_t n = 1;
        for (i32 i = P - 1; i % 2 == 0; i >>= 1) n <<= 1;
        return _MAXN < n ? _MAXN : n;
    }();
    static inline const struct _Info: std::array<i32, MAXN> {
        _Info() {
            auto g = this->data();
            for (size_t i = 2; i <= MAXN; i <<= 1) {
                g[i >> 1] = M::toMont(1);
                i32 gn = M::qpow(M::toMont(G), (P - 1) / i);
                for (size_t j = (i >> 1) + 1; j < i; j++) {
                    g[j] = M::mul(g[j - 1], gn);
                }
            }
        }
    } g;

    static size_t extend(size_t n) {
        return n < 3 ? n : (2 << std::__lg(n - 1));
    }
    static void copy(i32* a, const i32* b, size_t len, size_t pad_len = 0) {
        std::memcpy(a, b, len * sizeof(i32));
        if (pad_len && len < pad_len) std::fill(a + len, a + pad_len, 0);
    }
    static void clear(i32* a, size_t len) { std::fill_n(a, len, 0); }

    // a <- a + b
    static void add(i32* a, const i32* b, size_t len) {
        size_t i = 0;
        for (; i + 7 < len; i += 8)
            vstore(&a[i], M::add(vload(&a[i]), vload(&b[i])));
#pragma GCC unroll(8)
        for (; i < len; i++) a[i] = M::add(a[i], b[i]);
    }

    // a <- a - b
    static void sub(i32* a, const i32* b, size_t len) {
        size_t i = 0;
        for (; i + 7 < len; i += 8)
            vstore(&a[i], M::sub(vload(&a[i]), vload(&b[i])));
#pragma GCC unroll(8)
        for (; i < len; i++) a[i] = M::sub(a[i], b[i]);
    }

    // a <- a \dot b
    static void dot(i32* a, const i32* b, size_t len) {
        size_t i = 0;
        for (; i + 7 < len; i += 8)
            vstore(&a[i], M::mul(vload(&a[i]), vload(&b[i])));
#pragma GCC unroll(8)
        for (; i < len; i++) a[i] = M::mul(a[i], b[i]);
    }

    // a <- k * a
    static void scale(i32* a, i32 k, size_t len) {
        m256i vk = vset1(k);
        size_t i = 0;
        for (; i + 7 < len; i += 8) vstore(&a[i], M::mul(vload(&a[i]), vk));
#pragma GCC unroll(8)
        for (; i < len; i++) a[i] = M::mul(a[i], k);
    }

    static void DIF(i32* a, size_t len) {
        size_t i = len >> 1;
        for (; i >= 8; i >>= 1) {
            for (size_t j = 0; j < len; j += i << 1) {
                for (size_t k = 0; k < i; k += 8) {
                    m256i u = vload(&a[j + k]);
                    m256i v = vload(&a[j + k + i]);
                    m256i w = vload(&g[i + k]);
                    vstore(&a[j + k], M::add(u, v));
                    vstore(&a[j + k + i], M::mul(w, M::sub(u, v)));
                }
            }
        }
        for (; i > 0; i >>= 1) {
            for (size_t j = 0; j < len; j += i << 1) {
                for (size_t k = 0; k < i; k++) {
                    i32 u = a[j + k], v = a[j + k + i];
                    a[j + k] = M::add(u, v);
                    a[j + k + i] = M::mul(g[i + k], M::sub(u, v));
                }
            }
        }
    }

    static void DIT(i32* a, size_t len) {
        size_t i = 1;
        for (; i < len && i < 8; i <<= 1) {
            for (size_t j = 0; j < len; j += i << 1) {
                for (size_t k = 0; k < i; k++) {
                    i32 u = a[j + k], v = M::mul(g[i + k], a[j + k + i]);
                    a[j + k] = M::add(u, v);
                    a[j + k + i] = M::sub(u, v);
                }
            }
        }
        for (i = 8; i < len; i <<= 1) {
            for (size_t j = 0; j < len; j += i << 1) {
                for (size_t k = 0; k < i; k += 8) {
                    m256i u = vload(&a[j + k]);
                    m256i w = vload(&g[i + k]);
                    m256i v = M::mul(vload(&a[j + k + i]), w);
                    vstore(&a[j + k], M::add(u, v));
                    vstore(&a[j + k + i], M::sub(u, v));
                }
            }
        }
        scale(a, M::qpow(M::toMont(len), P - 2), len);
        std::reverse(a + 1, a + len);
    }

    // a <- a * b, assume b can be modified
    static void polymul(i32* a, i32* b, size_t len) {
        DIF(a, len);
        if (a != b) DIF(b, len);
        dot(a, b, len);
        DIT(a, len);
    }

    // a <- b^{-1}
    static void inv(i32* a, const i32* b, size_t len, size_t len_b) {
        if (b[0] == 0) throw std::runtime_error("inverse does not exist");
        a[0] = M::qpow(b[0], P - 2);
        Buffer f(len), g(len);
        clear(f.get(), len);
        clear(g.get(), len);
        for (size_t k = 1, k2 = 2; k < len; k = k2, k2 <<= 1) {
            copy(f.get(), b, k2 == len ? len_b : k2, k2);
            copy(g.get(), a, k);
            DIF(f.get(), k2);
            DIF(g.get(), k2);
            dot(f.get(), g.get(), k2);
            DIT(f.get(), k2);
            clear(f.get(), k);
            DIF(f.get(), k2);
            dot(f.get(), g.get(), k2);
            DIT(f.get(), k2);
            if (k >= 8) {
                constexpr m256i v0 = vset1(0);
                for (size_t i = k; i < k2; i += 8)
                    vstore(&a[i], M::sub(v0, vload(&f[i])));
            } else {
#pragma GCC unroll(8)
                for (size_t i = k; i < k2; i++) a[i] = M::sub(0, f[i]);
            }
        }
    }
};

};  // namespace detail

template <i32 P, size_t MAXN = 1 << 23>
class FPoly {
private:
    using AlignedVector = std::vector<i32, detail::AlignedAllocator<i32>>;
    using M = detail::Montgomery<P>;
    using U = detail::PolyUtils<P, MAXN>;
    using Buffer = detail::Buffer;

    static_assert(U::is_prime, "P is not a prime");

    class Proxy {
    public:
        Proxy& operator=(int val) { return _val = M::toMont(val), *this; }
        operator int() const { return M::fromMont(_val); }

    private:
        int _val;
    };

    AlignedVector _data;

public:
    FPoly() = default;
    explicit FPoly(size_t n): _data(n) {}
    FPoly(const std::initializer_list<i32>& init): _data(init) {
        M::toMont(_data.data(), init.size());
    }
    template <std::ranges::input_range R>
        requires std::is_same_v<std::ranges::range_value_t<R>, i32>
    FPoly(R&& range) {
        for (auto x: range) _data.push_back(x);
        M::toMont(_data.data(), size());
    }

    auto size() const { return _data.size(); }
    auto resize(size_t sz) { _data.resize(sz); }
    auto data() { return _data.data(); }
    auto data() const { return _data.data(); }
    auto begin() { return (Proxy*)_data.data(); }
    auto end() { return begin() + _data.size(); }
    auto begin() const { return (const Proxy*)_data.data(); }
    auto end() const { return begin() + _data.size(); }
    auto& operator[](size_t idx) { return (Proxy&)_data[idx]; }
    auto operator[](size_t idx) const { return M::fromMont(_data[idx]); }

    template <std::common_reference_with<FPoly> T>
    FPoly& operator*=(T&& other) {
        if (_data.empty() || other._data.empty()) {
            _data.clear();
            return *this;
        }
        size_t n = size() + other.size() - 1, nn = U::extend(n);
        resize(nn);
        if (&other == this) {
            U::polymul(data(), data(), nn);
        } else if constexpr (std::is_rvalue_reference_v<T&&>) {
            other.resize(nn);
            U::polymul(data(), other.data(), nn);
        } else {
            Buffer buf(nn);
            U::copy(buf.get(), other.data(), other.size(), nn);
            U::polymul(data(), buf.get(), nn);
        }
        resize(n);
        return *this;
    }

    template <std::common_reference_with<FPoly> T>
    FPoly& operator+=(T&& other) {
        if (other.size() > size()) resize(other.size());
        U::add(data(), other.data(), other.size());
        return *this;
    }

    template <std::common_reference_with<FPoly> T>
    FPoly& operator-=(T&& other) {
        if (other.size() > size()) resize(other.size());
        U::sub(data(), other.data(), other.size());
        return *this;
    }

    FPoly inv() const {
        size_t n = size(), nn = U::extend(n);
        FPoly res(nn);
        U::inv(res.data(), data(), nn, n);
        res.resize(n);
        return res;
    }

#define DEF_OP(op)                                                      \
    template <std::common_reference_with<FPoly> T,                      \
              std::common_reference_with<FPoly> U>                      \
    friend FPoly operator op(T&& lhs, U&& rhs) {                        \
        return FPoly(std::forward<T>(lhs)) op## = std::forward<U>(rhs); \
    }
    DEF_OP(*) DEF_OP(+) DEF_OP(-)
#undef DEF_OP
};

}  // namespace cp

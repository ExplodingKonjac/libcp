#include <immintrin.h>

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "modint.hpp"

#pragma GCC target("avx2", "fma")

namespace cp
{

namespace detail
{

template <typename T, size_t A>
class AlignedPool {
private:
    static inline std::vector<T*> _pool[32];

public:
    class pointer_type {
    public:
        pointer_type() = default;
        pointer_type(T* p, size_t len): _p(p), _len(len) {}
        pointer_type(const pointer_type& other) = delete;
        pointer_type(pointer_type&& other): _p(other._p), _len(other._len) {
            other._p = nullptr;
            other._len = 0;
        }
        pointer_type& operator=(const pointer_type& other) = delete;
        pointer_type& operator=(pointer_type&& other) {
            std::swap(_p, other._p);
            std::swap(_len, other._len);
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
        auto capacity() const { return _len; }

    private:
        T* _p = nullptr;
        size_t _len = 0;
    };

    static pointer_type allocate(size_t len) {
        len = std::max(A / sizeof(T), std::bit_ceil(len));
        int k = std::bit_width(len);
        if (!_pool[k].empty()) {
            T* p = _pool[k].back();
            _pool[k].pop_back();
            return {p, len};
        }
#ifdef _WIN32
        T* p = (T*)_aligned_malloc(len * sizeof(T), A);
#else
        T* p = (T*)std::aligned_alloc(A, len * sizeof(T));
#endif
        return {p, len};
    }
    static void deallocate(pointer_type& p) {
        if (!p) return;
        _pool[std::bit_width(p.capacity())].push_back(p);
    }
};

using m256i = __m256i;
using v8si = int __attribute__((vector_size(32)));

constexpr m256i vload(const void* p) { return _mm256_load_si256((m256i*)p); }
constexpr void vstore(void* p, m256i a) { _mm256_store_si256((m256i*)p, a); }
constexpr m256i vset1(i32 x) { return (m256i)v8si{x, x, x, x, x, x, x, x}; }

template <u32 P, size_t _MAXN>
struct PolyUtils {
    using Mint = SModint<P>;
    using Pool = AlignedPool<Mint, 32>;

    static constexpr m256i V_1 = vset1(1);
    static constexpr m256i V_P = vset1(P);
    static constexpr m256i V_P2 = vset1(Mint::mont.P2);
    static constexpr m256i V_P_INV = vset1(Mint::mont.P_INV);
    static constexpr m256i V_R = vset1(Mint::mont.R);
    static constexpr m256i V_R2 = vset1(Mint::mont.R2);

    static constexpr bool is_prime = [] {
        for (u32 i = 2; (u64)i * i <= P; i++)
            if (P % i == 0) return false;
        return true;
    }();
    static constexpr u32 G = [] {
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
                if (qpow(Mint(g), phi / i) == Mint{1}) {
                    ok = false;
                    break;
                }
            }
            if (ok) return g;
        }
        return 0u;
    }();
    static constexpr size_t MAXN = (1 << std::countr_zero(P - 1));

    static inline const struct _Info: std::array<Mint, MAXN> {
        _Info() {
            auto g = this->data();
            for (size_t i = 2; i <= MAXN; i <<= 1) {
                g[i >> 1] = Mint{1};
                Mint gn = qpow(Mint(G), (P - 1) / i);
                for (size_t j = (i >> 1) + 1; j < i; j++) {
                    g[j] = g[j - 1] * gn;
                }
            }
        }
    } g;

    static m256i add(m256i x, m256i y) {
        x = _mm256_add_epi32(x, y);
        y = _mm256_sub_epi32(x, V_P2);
        return _mm256_min_epu32(x, y);
    }
    static m256i sub(m256i x, m256i y) {
        x = _mm256_sub_epi32(x, y);
        y = _mm256_add_epi32(x, V_P2);
        return _mm256_min_epu32(x, y);
    }
    static m256i neg(m256i x) {
        m256i mask = _mm256_cmpeq_epi32(x, _mm256_setzero_si256());
        x = _mm256_sub_epi32(V_P2, x);
        return _mm256_andnot_si256(mask, x);
    }
    static m256i mul(m256i x, m256i y) {
        // even
        m256i t0 = _mm256_mul_epu32(x, y);
        m256i m0 = _mm256_mul_epu32(t0, V_P_INV);
        m256i res0 = _mm256_add_epi64(t0, _mm256_mul_epu32(m0, V_P));
        // odd
        x = _mm256_srli_epi64(x, 32);
        y = _mm256_srli_epi64(y, 32);
        m256i t1 = _mm256_mul_epu32(x, y);
        m256i m1 = _mm256_mul_epu32(t1, V_P_INV);
        m256i res1 = _mm256_add_epi64(t1, _mm256_mul_epu32(m1, V_P));
        // blend
        m256i mask = _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0);
        res0 = _mm256_srli_epi64(res0, 32);
        res1 = _mm256_and_si256(res1, mask);
        return _mm256_or_si256(res0, res1);
    }

    static void clear(Mint* a, size_t len) {
        std::memset((void*)a, 0, len * sizeof(Mint));
    }
    static void copy(Mint* a, const Mint* b, size_t len, size_t pad_len = -1) {
        std::memcpy((void*)a, (const void*)b, len * sizeof(Mint));
        if (pad_len && ~pad_len) clear(a + len, pad_len - len);
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

    static void DIF(Mint* a, size_t len) {
        size_t i = len >> 1;
        for (; i >= 8; i >>= 1) {
            for (size_t j = 0; j < len; j += i << 1) {
                for (size_t k = 0; k < i; k += 8) {
                    auto u = vload(&a[j + k]);
                    auto v = vload(&a[j + k + i]);
                    vstore(&a[j + k], add(u, v));
                    vstore(&a[j + k + i], mul(vload(&g[i + k]), sub(u, v)));
                }
            }
        }
        for (; i > 0; i >>= 1) {
            for (size_t j = 0; j < len; j += i << 1) {
                for (size_t k = 0; k < i; k++) {
                    Mint u = a[j + k], v = a[j + k + i];
                    a[j + k] = u + v;
                    a[j + k + i] = g[i + k] * (u - v);
                }
            }
        }
    }

    static void DIT(Mint* a, size_t len) {
        size_t i = 1;
        for (; i < 8 && i < len; i <<= 1) {
            for (size_t j = 0; j < len; j += i << 1) {
                for (size_t k = 0; k < i; k++) {
                    Mint u = a[j + k], v = g[i + k] * a[j + k + i];
                    a[j + k] = u + v;
                    a[j + k + i] = u - v;
                }
            }
        }
        for (i = 8; i < len; i <<= 1) {
            for (size_t j = 0; j < len; j += i << 1) {
                for (size_t k = 0; k < i; k += 8) {
                    auto u = vload(&a[j + k]);
                    auto v = mul(vload(&g[i + k]), vload(&a[j + k + i]));
                    vstore(&a[j + k], add(u, v));
                    vstore(&a[j + k + i], sub(u, v));
                }
            }
        }
        scale(a, Mint(len).inv(), len, a);
        std::reverse(a + 1, a + len);
    }

    // a <- a * b, assume b can be modified
    static void polymul(Mint* a, Mint* b, size_t len) {
        DIF(a, len);
        // DIT(a, len);
        if (a != b) DIF(b, len);
        dot(a, b, len, a);
        DIT(a, len);
    }

    // a <- b^{-1}
    static void inv(Mint* a, const Mint* b, size_t len_b) {
        if (!len_b || b[0] == 0)
            throw std::runtime_error("inverse does not exist");

        size_t len = std::bit_ceil(len_b);
        auto f = Pool::allocate(len), g = Pool::allocate(len);
        clear(f, len);
        clear(g, len);
        a[0] = b[0].inv();
        for (size_t k = 1, k2 = 2; k < len; k = k2, k2 <<= 1) {
            copy(f, b, k2 == len ? len_b : k2, k2);
            copy(g, a, k);
            DIF(f, k2), DIF(g, k2), dot(f, g, k2, f), DIT(f, k2);
            clear(f, k), DIF(f, k2), dot(f, g, k2, f), DIT(f, k2);
            neg(f + k, k, a + k);
        }
    }
};

}  // namespace detail

template <u32 P, size_t _MAXN>
class FPoly {
private:
    using U = detail::PolyUtils<P, _MAXN>;
    using Mint = typename U::Mint;
    using Pool = typename U::Pool;

    static_assert(U::is_prime, "P must be a prime number");
    static_assert(U::MAXN >= 2, "MAXN must be at least 2");

    Pool::pointer_type _data{};
    size_t _len = 0;

public:
    FPoly() = default;
    explicit FPoly(size_t n): _data{Pool::allocate(n)}, _len(n) {
        U::clear(_data, n);
    }
    FPoly(const std::initializer_list<Mint>& init):
        _data{Pool::allocate(init.size())}, _len{init.size()} {
        U::copy(_data, init.begin(), init.size());
    }
    template <std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_value_t<R>, Mint>
    FPoly(R&& range) {
        std::vector<Mint> tmp{};
        for (auto&& x: range) tmp.push_back(x);
        _len = tmp.size();
        _data = Pool::allocate(_len);
        U::copy(_data, tmp.data(), _len);
    }

    void resize(size_t sz) {
        if (sz > _data.capacity()) {
            auto new_data = Pool::allocate(sz);
            U::copy(new_data, _data, _len);
            _data = std::move(new_data);
        }
        if (sz > _len) U::clear(_data + _len, sz - _len);
        _len = sz;
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

    template <std::common_reference_with<FPoly> T>
    FPoly& operator*=(T&& other) {
        if (_len == 0 || other._len == 0) return clear(), *this;
        size_t n = _len + other._len - 1, nn = std::bit_ceil(n);
        resize(nn);
        if (&other == this) {
            U::polymul(_data, _data, nn);
        } else if constexpr (std::is_rvalue_reference_v<T&&>) {
            other.resize(nn);
            U::polymul(_data, other._data, nn);
        } else {
            auto tmp = Pool::allocate(nn);
            U::copy(tmp, other._data, other._len, nn);
            U::polymul(_data, tmp, nn);
        }
        return resize(n), *this;
    }

    FPoly& operator*=(Mint k) {
        U::scale(_data, k, _len, _data);
        return *this;
    }
    template <std::common_reference_with<FPoly> T>
    friend FPoly operator*(Mint k, T&& f) {
        return FPoly(std::forward<T>(f)) *= k;
    }
    template <std::common_reference_with<FPoly> T>
    friend FPoly operator*(T&& f, Mint k) {
        return FPoly(std::forward<T>(f)) *= k;
    }

    template <std::common_reference_with<FPoly> T>
    FPoly& operator+=(T&& other) {
        if (other._len > _len) resize(other._len);
        U::add(_data, other._data, other._len, _data);
        return *this;
    }

    template <std::common_reference_with<FPoly> T>
    FPoly& operator-=(T&& other) {
        if (other._len > _len) resize(other._len);
        U::sub(_data, other._data, other._len, _data);
        return *this;
    }

    FPoly operator-() const {
        FPoly res(_len);
        U::neg(_data, _len, res._data);
        return res;
    }

    FPoly inv() const {
        FPoly res(_len);
        U::inv(res._data, _data, _len);
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

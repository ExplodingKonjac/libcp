#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <charconv>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <format>
#include <initializer_list>
#include <limits>
#include <optional>
#include <random>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

#define CP_FASTIO_USE_BUF
// #include "cp/fast_io.hpp"
// #include "cp/fpoly.hpp"
// #include "cp/modint.hpp"

namespace cp
{

constexpr size_t IN_BUF_SIZE = 1 << 21;
constexpr size_t OUT_BUF_SIZE = 1 << 21;

#ifndef CP_FORMAT_STRING
#define CP_FORMAT_STRING
template <size_t N>
struct FixedString: std::array<char, N> {
    consteval FixedString(const char (&str)[N]) {
        std::ranges::copy(str, this->data());
    }
    constexpr auto view() const {
        return std::string_view{this->data(), N - 1};
    }
};

template <FixedString S>
struct FormatString {
    constexpr operator auto() const { return S.view(); }
};

namespace literals
{

template <FixedString S>
consteval auto operator""_fmt() {
    return FormatString<S>{};
}

}  // namespace literals
#endif

class FastIOBase {
protected:
    FILE* _target;
#ifdef CP_FASTIO_USE_BUF
    char* _buf = nullptr;
    size_t _size = 0, _pos = -1;
#endif

public:
    FastIOBase(FILE* target): _target(target) {}
};

class FastInput final: FastIOBase {
private:
#if defined(CP_FASTIO_USE_BUF) && defined(__linux__)
    bool _is_mmap = false;
#endif

    struct BufIterator {
        using value_type = char;
        using difference_type = ptrdiff_t;

        [[gnu::always_inline]] BufIterator& operator++() {
#ifdef CP_FASTIO_USE_BUF
            c = (++t->_pos == t->_size ? t->sync() : t->_buf[t->_pos]);
#else
            c = fgetc(t->_target);
#endif
            return *this;
        }
        char operator*() { return c; }
        void operator++(int) { ++(*this); }
        void skipws() {
            while (c <= 32) ++(*this);
        }
        bool eof() { return c == EOF; }

        FastInput* t = nullptr;
#ifdef CP_FASTIO_USE_BUF
        char c = t ? (~t->_pos ? t->_buf[t->_pos] : t->sync()) : EOF;
#else
        char c = t ? std::fgetc(t->_target) : EOF;
        ~BufIterator() { std::ungetc(c, t->_target); }
#endif
    };

public:
    FastInput(FILE* target): FastIOBase(target) {
#ifdef CP_FASTIO_USE_BUF
#ifdef __linux__
        int fd = fileno(target);
        struct stat sb;
        if (fstat(fd, &sb) == 0 && S_ISREG(sb.st_mode) && sb.st_size > 0) {
            _buf =
                (char*)mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (_buf != MAP_FAILED) {
                _size = sb.st_size;
                _pos = 0;
                _is_mmap = true;
                return;
            }
        }
#endif
        _buf = new char[IN_BUF_SIZE];
        _size = IN_BUF_SIZE;
#endif
    }
#ifdef CP_FASTIO_USE_BUF
    char sync() {
        if (_is_mmap) return EOF;
        size_t read_sz = std::fread(_buf, 1, IN_BUF_SIZE, _target);
        _pos = 0;
        if (read_sz < IN_BUF_SIZE) _buf[read_sz] = EOF;
        return _buf[0];
    }
#endif

    operator bool() { return !eof(); }
    bool eof() { return BufIterator{this}.eof(); }

    template <std::integral T>
    std::optional<T> scan() {
        BufIterator it{this};
        bool neg = false;
        it.skipws();
        if constexpr (std::is_signed_v<T>) {
            if (*it == '-') neg = true, it++;
            else if (*it == '+') it++;
        }
        if (!std::isdigit(*it)) return std::nullopt;
        T res = 0;
        do res = res * 10 + (*it ^ 48), it++;
        while (std::isdigit(*it));
        return neg ? -res : res;
    }

    template <std::same_as<std::string> T>
    std::optional<T> scan() {
        BufIterator it{this};
        it.skipws();
        if (it.eof()) return std::nullopt;
        T res{};
        do res.push_back(*it), it++;
        while (std::isgraph(*it));
        return res;
    }

    template <std::floating_point T>
    std::optional<T> scan() {
        BufIterator it{this};
        T res = 0;
        bool neg = false, ok = false;
        it.skipws();
        if (*it == '-') neg = true, it++;
        else if (*it == '+') it++;
        if (std::isdigit(*it)) {
            ok = true;
            do res = res * 10 + (*it ^ 48), it++;
            while (std::isdigit(*it));
        }
        if (*it == '.' && std::isdigit(*(++it))) {
            ok = true;
            T mul = 0.1;
            do res += mul * (*it ^ 48), mul *= 0.1, it++;
            while (std::isdigit(*it));
        }
        return ok ? std::optional(neg ? -res : res) : std::nullopt;
    }

    template <typename... Args>
        requires(sizeof...(Args) > 1)
    std::optional<std::tuple<Args...>> scan() {
        std::tuple<std::optional<Args>...> tmp;
        auto try_scan = [this](auto&... r) {
            return ((r = scan<std::remove_cvref_t<decltype(r.value())>>()) &&
                    ...);
        };
        auto unwrap = [](auto&&... items) {
            return std::tuple(std::move(items).value()...);
        };
        if (std::apply(try_scan, tmp)) return std::apply(unwrap, tmp);
        return std::nullopt;
    }
};

class FastOutput final: FastIOBase {
private:
    static constexpr std::array<char, 200> lut = [] {
        std::array<char, 200> res;
        for (size_t i = 0; i < 100; i++) {
            res[i * 2] = i / 10 + '0';
            res[i * 2 + 1] = i % 10 + '0';
        }
        return res;
    }();

    struct BufIterator {
        using difference_type = ptrdiff_t;

        [[gnu::always_inline]] BufIterator& operator=(char c) {
#ifdef CP_FASTIO_USE_BUF
            t->_buf[t->_pos++] = c;
            if (t->_pos == t->_size) [[unlikely]] {
                t->flush();
            }
#else
            fputc(c, t->_target);
#endif
            return *this;
        }
        BufIterator& operator*() { return *this; }
        BufIterator& operator++() { return *this; }
        BufIterator& operator++(int) { return ++*this; }

        FastOutput* t;
    };

public:
    FastOutput(FILE* target): FastIOBase{target} {
#ifdef CP_FASTIO_USE_BUF
        _buf = new char[OUT_BUF_SIZE];
        _size = OUT_BUF_SIZE;
        _pos = 0;
    }
    ~FastOutput() { flush(); }

    void flush() {
        fwrite(_buf, 1, _pos, _target);
        _pos = 0;
#endif
    }

    template <FixedString S, typename... Args>
    void print(FormatString<S> fmt, Args&&... args) {
        using std::format_to;
        format_to(BufIterator{this}, fmt, args...);
    }

    void print(char c) { BufIterator{this} = c; }

    template <std::integral T>
    void print(T x) {
        using U = std::make_unsigned_t<T>;
        char stk[std::numeric_limits<T>::digits10], *top = stk;
        BufIterator it{this};
        U u = (x < 0 ? print('-'), -x : x);
        while (u >= 100) {
            int pos = u % 100 * 2;
            *(top++) = lut[pos + 1];
            *(top++) = lut[pos];
            u /= 100;
        }
        if (u >= 10) {
            *(top++) = lut[u * 2 + 1];
            *(top++) = lut[u * 2];
        } else {
            *(top++) = u | 48;
        }
        while (top > stk) it = *(--top);
    }

    template <std::floating_point T>
    void print(T x) {
        constexpr size_t S = sizeof(T) > 8 ? 64 : 32;
        char buf[S];
        auto [p, e] =
            std::to_chars(buf, buf + S, x, std::chars_format::general);
        if (e == std::errc{}) {
            BufIterator it{this};
            for (auto i = buf; i != p; i++) it = *i;
        }
    }

    template <std::convertible_to<std::string_view> T>
    void print(T&& x) {
        std::ranges::copy(std::string_view(x), BufIterator{this});
    }

    template <typename First, typename... Args>
        requires(sizeof...(Args) > 0)
    void print(First&& first, Args&&... args) {
        print(first), print(' '), print(args...);
    }

    template <typename... Args>
    void println(Args&&... args) {
        print(args...), print('\n');
    }
};

inline cp::FastInput qin(stdin);
inline cp::FastOutput qout(stdout);

}  // namespace cp

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

#define DEF_OP_ARI(op, expr)                                        \
    friend C D operator op(D lhs, D rhs) { return lhs op## = rhs; } \
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

template <typename T>
concept modint = std::derived_from<T, detail::ModintBase<T>>;

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

template <modint T, std::integral U>
C T qpow(T x, U y) {
    if (y < 0) return qpow(x.inv(), std::make_signed_t<U>(-y));
    T res{1};
    for (; y; y >>= 1, x = x * x)
        if (y & 1) res = res * x;
    return res;
}

template <modint T>
C int legendre(T x) {
    auto r = qpow(x, (x.mont.P - 1) / 2)();
    return r == x.mont.P - 1 ? -1 : r;
}

template <modint T>
C std::optional<T> sqrt(T x) {
    static std::default_random_engine rng(std::random_device{}());
    if (x == T{0}) return x;
    if (legendre(x) != 1) return std::nullopt;
    T r{}, g{};
    do r = rng(), g = r * r - x;
    while (legendre(g) != -1);
    auto mul = [&](auto& x, auto& y) {
        return std::pair{x.second * y.second * g + x.first * y.first,
                         x.first * y.second + y.first * x.second};
    };
    std::pair<T, T> res{1, 0}, base{r, 1};
    for (u32 t = (x.mont.P + 1) / 2; t; t >>= 1) {
        if (t & 1) res = mul(res, base);
        base = mul(base, base);
    }
    return res.first;
}
#undef C

}  // namespace cp

#pragma GCC push_options
#pragma GCC target("avx2")
namespace cp
{

namespace detail
{

template <typename T, size_t A = alignof(T)>
class AlignedPool {
private:
    static inline bool _cleaned = false;
    static inline struct _PoolObj: std::array<std::vector<T*>, 32> {
        ~_PoolObj() {
            _cleaned = true;
            for (auto& vec: *this) std::ranges::for_each(vec, free);
        }
    } _pool;

    static void free(T* p) {
#ifdef _WIN32
        _aligned_free(p);
#else
        std::free(p);
#endif
    }
    static T* alloc(size_t n) {
#ifdef _WIN32
        return static_cast<T*>(_aligned_malloc(n * sizeof(T), A));
#else
        return static_cast<T*>(std::aligned_alloc(A, n * sizeof(T)));
#endif
    }

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
        return {alloc(n), n};
    }
    static void deallocate(pointer_type& p) {
        if (!p) return;
        if (_cleaned) return free(p);
        _pool[std::countr_zero(p.capacity())].push_back(p);
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

    template <bool strict = false>
    static m256i shrink(m256i x) {
        return _mm256_min_epu32(x, _mm256_sub_epi32(x, strict ? V_P : V_P2));
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
                w = shrink<true>(w);
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

template <typename T, typename Mint>
concept init_friendly_type =
    std::same_as<T, u32> || std::same_as<T, i32> || std::same_as<T, Mint>;

template <typename R, typename Mint>
concept can_fast_init = std::ranges::contiguous_range<R> &&
                        init_friendly_type<std::ranges::range_value_t<R>, Mint>;

}  // namespace detail

template <u32 P, size_t _MAXN = size_t(-1)>
class FPoly {
public:
    using U = detail::PolyUtils<P, _MAXN>;
    using Mint = U::Mint;
    using Pool = U::Pool;

private:
    size_t _len = 0;
    Pool::pointer_type _data{};

public:
    FPoly() = default;
    FPoly(const std::initializer_list<Mint>& init):
        FPoly(std::views::all(init)) {}
    explicit FPoly(size_t n, bool no_init = false):
        _len{n}, _data{Pool::allocate(n)} {
        if (!no_init) U::clear(_data, n);
    }
    template <detail::can_fast_init<Mint> R>
        requires(!std::same_as<std::remove_cvref_t<R>, FPoly>)
    FPoly(R&& r): FPoly(std::ranges::size(r), true) {
        using T = std::ranges::range_value_t<R>;
        auto data = std::ranges::data(r);
        if constexpr (std::same_as<T, Mint>) {
            U::copy(data, _len, _data);
        } else {
            size_t i = 0;
            for (; i + 7 < _len; i += 8) {
                auto v = U::vloadu(data + i);
                if constexpr (std::is_signed_v<T>) {
                    v = _mm256_add_epi32(v, U::vset1((1u << 31) / P * P));
                }
                U::vstore(_data + i, U::mul(v, U::V_R2));
            }
            for (; i < _len; i++) _data[i] = Mint{data[i]};
        }
    }
    template <std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_value_t<R>, Mint> &&
                 (!detail::can_fast_init<R, Mint>) &&
                 (!std::same_as<std::remove_cvref_t<R>, FPoly>)
    FPoly(R&& r) {
        if constexpr (std::ranges::sized_range<R>) {
            reserve(std::ranges::size(r));
        }
        for (auto&& x: r) push_back(std::forward<decltype(x)>(x));
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
    void reserve(size_t sz) {
        if (sz > _data.capacity()) {
            auto new_data = Pool::allocate(sz);
            U::copy(_data, _len, new_data);
            _data = std::move(new_data);
        }
    }
    void push_back(Mint x) { resize(_len + 1), _data[_len - 1] = x; }
    void pop_back() { _len--; }
    void clear() { _len = 0; }
    size_t size() const { return _len; }

    auto data() { return (Mint*)_data; }
    auto data() const { return (const Mint*)_data; }
    auto begin() { return (Mint*)_data; }
    auto begin() const { return (const Mint*)_data; }
    auto end() { return begin() + _len; }
    auto end() const { return begin() + _len; }
    Mint& operator[](size_t idx) { return _data[idx]; }
    Mint operator[](size_t idx) const { return _data[idx]; }

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
    FPoly& operator/=(const FPoly& other) {
        size_t sz = std::max(_len, other._len);
        (*this) *= inv(other);
        return resize(sz), *this;
    }
    friend FPoly operator-(FPoly f) {
        return U::neg(f._data, f._len, f._data), std::move(f);
    }
    friend FPoly operator+(FPoly f, const FPoly& g) {
        return std::move(f += g);
    }
    friend FPoly operator-(FPoly f, const FPoly& g) {
        return std::move(f -= g);
    }
    friend FPoly operator*(FPoly f, FPoly g) {
        return std::move(f *= std::move(g));
    }
    friend FPoly operator/(FPoly f, const FPoly& g) {
        return std::move(f /= g);
    }
    friend FPoly operator*(Mint k, FPoly f) { return std::move(f *= k); }
    friend FPoly operator*(FPoly f, Mint k) { return std::move(f *= k); }
};

#define Poly FPoly<Q, N>
#define U Poly::U
template <u32 Q, size_t N>
Poly integrate(Poly f) {
    f.resize(f.size() + 1);
    U::polyint(f.data(), f.size(), f.data());
    return f;
}

template <u32 Q, size_t N>
Poly derivative(Poly f) {
    if (f.size() == 0) return f;
    U::polyder(f.data(), f.size(), f.data());
    f.resize(f.size() - 1);
    return f;
}

template <u32 Q, size_t N>
Poly inv(const Poly& f) {
    Poly res(f.size(), true);
    U::polyinv(f.data(), f.size(), res.data());
    return res;
}

template <u32 Q, size_t N>
Poly ln(const Poly& f) {
    Poly res(f.size(), true);
    U::polyln(f.data(), f.size(), res.data());
    return res;
}

template <u32 Q, size_t N>
Poly exp(const Poly& f) {
    Poly res(f.size(), true);
    U::polyexp(f.data(), f.size(), res.data());
    return res;
}

template <u32 Q, size_t N>
Poly sqrt(const Poly& f) {
    auto k = std::ranges::find_if(f, [](auto x) { return x(); }) - f.begin();
    if (k == f.size()) return Poly(f.size(), true);
    if (k % 2 != 0) throw std::invalid_argument("sqrt does not exist");
    Poly res(f.size(), true);
    if (k == 0) U::polysqrt(f.data(), f.size(), res.data());
    else {
        auto tmp = U::Pool::allocate(f.size());
        U::copy(f.data() + k, f.size() - k, tmp, f.size());
        U::polysqrt(tmp, f.size(), res.data());
        std::memmove(res.data() + k / 2, res.data(),
                     (res.size() - k / 2) * sizeof(typename Poly::Mint));
        U::clear(res.data(), k / 2);
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

using cp::qin, cp::qout;
using namespace cp::literals;

constexpr int MOD = 998244353, MAXN = 500000;
using Mint = cp::SModint<MOD>;
using Poly = cp::FPoly<MOD, (1 << 19)>;

unsigned a[MAXN + 5];

int main() {
    int n = qin.scan<int>().value();
    for (int i = 0; i < n; i++) a[i] = qin.scan<unsigned>().value();
    try {
        Poly ans = sqrt(Poly(a, a + n));
        for (int i = 0; i < n; i++) qout.print(ans[i](), "");
        qout.print('\n');
    } catch (...) {
        qout.println("-1");
    }
    return 0;
}
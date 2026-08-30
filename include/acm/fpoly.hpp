#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

#include "modint.hpp"

namespace acm
{

template <u32 P>
class FPoly {
public:
    using Mint = SModint<P>;

private:
    std::vector<Mint> a;

    static constexpr bool is_prime = [] {
        for (u32 i = 2; u64(i) * i <= P; i++)
            if (P % i == 0) return false;
        return P >= 2;
    }();
    static constexpr usize LG_MAXN = std::countr_zero(P - 1);
    static_assert(is_prime, "P must be prime");
    static_assert(LG_MAXN > 0, "P - 1 must be even");

    static constexpr Mint primitive_root = [] {
        u32 phi = P - 1, n = phi;
        std::vector<u32> factors;
        for (u32 i = 2; u64(i) * i <= n; i++) {
            if (n % i == 0) factors.push_back(i);
            while (n % i == 0) n /= i;
        }
        if (n > 1) factors.push_back(n);
        for (u32 g = 2; g < P; g++) {
            bool ok = true;
            for (u32 factor: factors)
                if (acm::pow(Mint(g), phi / factor) == Mint(1)) {
                    ok = false;
                    break;
                }
            if (ok) return Mint(g);
        }
        return Mint{};
    }();

    struct NttInfo {
        std::array<Mint, LG_MAXN + 1> root{}, iroot{};
        std::array<Mint, LG_MAXN> rate{}, irate{};

        constexpr NttInfo() {
            Mint r = acm::pow(primitive_root, (P - 1) >> LG_MAXN);
            Mint ir = r.inv();
            for (usize i = LG_MAXN; ~i; i--) {
                root[i] = r;
                iroot[i] = ir;
                r *= r;
                ir *= ir;
            }
            Mint prd = 1, inv = 1;
            for (usize i = 0; i + 2 <= LG_MAXN; i++) {
                rate[i] = root[i + 2] * prd;
                irate[i] = iroot[i + 2] * inv;
                prd *= iroot[i + 2];
                inv *= root[i + 2];
            }
        }
    };
    static constexpr NttInfo ntt_info{};

    static void dft(Mint* a, usize n) {
        for (usize i = n; i > 1; i >>= 1) {
            usize s = i >> 1;
            Mint r = 1;
            for (usize j = 0, jc = 0; j < n; j += i, jc++) {
                for (usize k = 0; k < s; k++) {
                    Mint x = a[j + k], y = a[j + s + k] * r;
                    a[j + k] = x + y;
                    a[j + s + k] = x - y;
                }
                r *= ntt_info.rate[std::countr_one(jc)];
            }
        }
    }

    static void idft(Mint* a, usize n) {
        for (usize i = 2; i <= n; i <<= 1) {
            usize s = i >> 1;
            Mint r = 1;
            for (usize j = 0, jc = 0; j < n; j += i, jc++) {
                for (usize k = 0; k < s; k++) {
                    Mint x = a[j + k], y = a[j + s + k];
                    a[j + k] = x + y;
                    a[j + s + k] = (x - y) * r;
                }
                r *= ntt_info.irate[std::countr_one(jc)];
            }
        }
        Mint z = Mint(n).inv();
        for (usize i = 0; i < n; i++) a[i] *= z;
    }

    static std::vector<Mint> convolution(
        std::vector<Mint> x, std::vector<Mint> y
    ) {
        if (x.empty() || y.empty()) return {};
        usize need = x.size() + y.size() - 1;
        if (std::min(x.size(), y.size()) < 32) {
            std::vector<Mint> z(need);
            for (usize i = 0; i < x.size(); i++)
                for (usize j = 0; j < y.size(); j++) z[i + j] += x[i] * y[j];
            return z;
        }
        usize n = 1;
        while (n < need) n <<= 1;
        if ((P - 1) % n) {
            std::vector<Mint> z(need);
            for (usize i = 0; i < x.size(); i++)
                for (usize j = 0; j < y.size(); j++) z[i + j] += x[i] * y[j];
            return z;
        }
        x.resize(n);
        y.resize(n);
        dft(x.data(), n);
        dft(y.data(), n);
        for (usize i = 0; i < n; i++) x[i] *= y[i];
        idft(x.data(), n);
        x.resize(need);
        return x;
    }

    static void clear(Mint* x, usize n) { std::fill_n(x, n, Mint{}); }
    static void copy(const Mint* x, usize n, Mint* y, usize pad = 0) {
        std::copy_n(x, n, y);
        if (pad) clear(y + n, pad - n);
    }
    static void dot(const Mint* x, const Mint* y, usize n, Mint* z) {
        for (usize i = 0; i < n; i++) z[i] = x[i] * y[i];
    }
    static void polyder(const Mint* f, usize n, Mint* g) {
        for (usize i = 1; i < n; i++) g[i - 1] = Mint(i) * f[i];
        if (n) g[n - 1] = 0;
    }
    static void polyint(Mint* f, usize n) {
        static std::vector<Mint> inverses{0, 1};
        while (inverses.size() <= n) {
            usize i = inverses.size();
            inverses.push_back(-Mint(P / i) * inverses[P % i]);
        }
        for (usize i = n - 1; i; i--) f[i] = f[i - 1] * inverses[i];
        f[0] = 0;
    }
    static std::vector<Mint> polyinv(const FPoly& f) {
        if (f.empty() || !f[0])
            throw std::invalid_argument("constant term is zero");
        usize n = f.size(), len = std::bit_ceil(n);
        std::vector<Mint> out(len), t1(len), t2(len);
        out[0] = f[0].inv();
        for (usize k = 1, k2 = 2; k < len; k = k2, k2 <<= 1) {
            copy(f.data(), std::min(k2, n), t1.data(), k2);
            copy(out.data(), k, t2.data(), k2);
            dft(t1.data(), k2);
            dft(t2.data(), k2);
            dot(t1.data(), t2.data(), k2, t1.data());
            idft(t1.data(), k2);
            clear(t1.data(), k);
            dft(t1.data(), k2);
            dot(t1.data(), t2.data(), k2, t1.data());
            idft(t1.data(), k2);
            for (usize i = 0; i < k; i++) out[k + i] = -t1[k + i];
        }
        out.resize(n);
        return out;
    }
    static std::vector<Mint> polyln(const FPoly& f) {
        if (f.empty() || f[0] != Mint(1))
            throw std::invalid_argument("constant term is not one");
        usize n = f.size(), len = std::bit_ceil(n);
        std::vector<Mint> out(len), d(len), g(len), t1(len), t2(len), t3(len);
        polyder(f.data(), n, d.data());
        out[0] = d[0];
        g[0] = 1;
        for (usize k = 1, k2 = 2; k < len; k = k2, k2 <<= 1) {
            copy(g.data(), k, t1.data(), k2);
            copy(f.data(), std::min(k2, n), t2.data(), k2);
            dft(t1.data(), k2);
            dft(t2.data(), k2);
            dot(t1.data(), t2.data(), k2, t2.data());
            idft(t2.data(), k2);
            clear(t2.data(), k);
            dft(t2.data(), k2);
            copy(g.data(), k, t3.data(), k2);
            dft(t3.data(), k2);
            dot(t2.data(), t3.data(), k2, t3.data());
            idft(t3.data(), k2);
            for (usize i = 0; i < k; i++) g[k + i] = -t3[k + i];
            copy(d.data(), k2, t3.data());
            dft(t3.data(), k2);
            dot(t3.data(), t1.data(), k2, t1.data());
            copy(out.data(), k, t3.data(), k2);
            dft(t3.data(), k2);
            dot(t3.data(), t2.data(), k2, t2.data());
            for (usize i = 0; i < k2; i++) t3[i] = t1[i] - t2[i];
            idft(t3.data(), k2);
            copy(t3.data() + k, k, out.data() + k);
        }
        polyint(out.data(), len);
        out.resize(n);
        return out;
    }
    static std::vector<Mint> polyexp(const FPoly& f) {
        usize n = f.size();
        if (!n) return {};
        if (!f.empty() && f[0])
            throw std::invalid_argument("constant term is not zero");
        usize len = std::bit_ceil(n);
        std::vector<Mint> out(len), g(len), t1(len), t2(len), t3(len), t4(len);
        out[0] = g[0] = 1;
        for (usize k = 1, k2 = 2; k < len; k = k2, k2 <<= 1) {
            copy(out.data(), k, t1.data(), k2);
            dft(t1.data(), k2);
            copy(g.data(), k, t2.data(), k2);
            dft(t2.data(), k2);
            for (usize i = 0; i < k2; i++) t3[i] = -t1[i] * t2[i] * t2[i];
            idft(t3.data(), k2);
            copy(g.data(), k, t3.data());
            dft(t3.data(), k2);
            polyder(out.data(), k, t4.data());
            clear(t4.data() + k, k);
            dft(t4.data(), k2);
            dot(t4.data(), t3.data(), k2, t4.data());
            idft(t4.data(), k2);
            polyint(t4.data(), k2);
            for (usize i = k; i < k2; i++) t4[i] -= i < n ? f[i] : Mint{};
            clear(t4.data(), k);
            dft(t4.data(), k2);
            for (usize i = 0; i < k2; i++) {
                Mint d = t4[i];
                t1[i] *= Mint(1) - d;
                t2[i] = t3[i] + t2[i] * d;
            }
            idft(t1.data(), k2);
            copy(t1.data() + k, k, out.data() + k);
            idft(t2.data(), k2);
            copy(t2.data() + k, k, g.data() + k);
        }
        out.resize(n);
        return out;
    }
    static std::vector<Mint> polysqrt(const FPoly& f) {
        if (f.empty() || !f[0])
            throw std::invalid_argument("constant term is zero");
        auto root = acm::sqrt(f[0]);
        if (!root) throw std::invalid_argument("square root does not exist");
        usize n = f.size(), len = std::bit_ceil(n);
        std::vector<Mint> out(len), h(len), t1(len), t2(len), t3(len);
        out[0] = root->val() <= P - root->val() ? *root : -*root;
        h[0] = out[0].inv();
        Mint c = -Mint(2).inv();
        for (usize k = 1, k2 = 2; k < len; k = k2, k2 <<= 1) {
            copy(f.data(), std::min(k2, n), t1.data(), k2);
            dft(t1.data(), k2);
            copy(out.data(), k, t2.data(), k2);
            dft(t2.data(), k2);
            copy(h.data(), k, t3.data(), k2);
            dft(t3.data(), k2);
            for (usize i = 0; i < k2; i++)
                t1[i] = (t2[i] * t2[i] - t1[i]) * t3[i] * c;
            idft(t1.data(), k2);
            copy(out.data(), k, t1.data());
            copy(t1.data() + k, k, out.data() + k);
            dft(t1.data(), k2);
            dot(t1.data(), t3.data(), k2, t1.data());
            idft(t1.data(), k2);
            clear(t1.data(), k);
            dft(t1.data(), k2);
            dot(t1.data(), t3.data(), k2, t1.data());
            idft(t1.data(), k2);
            for (usize i = 0; i < k; i++) h[k + i] = -t1[k + i];
        }
        out.resize(n);
        return out;
    }

    template <u32 Q>
    friend FPoly<Q> derivative(FPoly<Q>);
    template <u32 Q>
    friend FPoly<Q> integrate(FPoly<Q>);
    template <u32 Q>
    friend FPoly<Q> inv(const FPoly<Q>&);
    template <u32 Q>
    friend FPoly<Q> ln(const FPoly<Q>&);
    template <u32 Q>
    friend FPoly<Q> exp(const FPoly<Q>&);
    template <u32 Q>
    friend FPoly<Q> sqrt(const FPoly<Q>&);

public:
    FPoly() = default;
    FPoly(std::initializer_list<Mint> x): a(x) {}
    explicit FPoly(usize n): a(n) {}
    explicit FPoly(std::vector<Mint> x): a(std::move(x)) {}
    template <typename It>
    FPoly(It first, It last): a(first, last) {}
    usize size() const { return a.size(); }
    bool empty() const { return a.empty(); }
    void resize(usize n) { a.resize(n); }
    void push_back(Mint x) { a.push_back(x); }
    void pop_back() { a.pop_back(); }
    void clear() { a.clear(); }
    Mint* data() { return a.data(); }
    const Mint* data() const { return a.data(); }
    auto begin() { return a.begin(); }
    auto end() { return a.end(); }
    auto begin() const { return a.begin(); }
    auto end() const { return a.end(); }
    Mint& operator[](usize i) { return a[i]; }
    Mint operator[](usize i) const { return a[i]; }
    void trim() {
        while (!a.empty() && !a.back()) a.pop_back();
    }
    FPoly& operator+=(const FPoly& b) {
        a.resize(std::max(size(), b.size()));
        for (usize i = 0; i < b.size(); i++) a[i] += b[i];
        return *this;
    }
    FPoly& operator-=(const FPoly& b) {
        a.resize(std::max(size(), b.size()));
        for (usize i = 0; i < b.size(); i++) a[i] -= b[i];
        return *this;
    }
    FPoly& operator*=(const FPoly& b) {
        a = convolution(std::move(a), b.a);
        return *this;
    }
    FPoly& operator*=(Mint k) {
        for (auto& x: a) x *= k;
        return *this;
    }
    FPoly& operator/=(const FPoly& b) { return *this = div(*this, b).first; }
    friend FPoly operator+(FPoly x, const FPoly& y) { return x += y; }
    friend FPoly operator-(FPoly x, const FPoly& y) { return x -= y; }
    friend FPoly operator*(FPoly x, const FPoly& y) { return x *= y; }
    friend FPoly operator*(FPoly x, Mint y) { return x *= y; }
    friend FPoly operator*(Mint y, FPoly x) { return x *= y; }
    friend FPoly operator/(FPoly x, const FPoly& y) { return x /= y; }
};

template <u32 P>
FPoly<P> derivative(FPoly<P> f) {
    if (f.empty()) return f;
    FPoly<P>::polyder(f.data(), f.size(), f.data());
    f.pop_back();
    return f;
}

template <u32 P>
FPoly<P> integrate(FPoly<P> f) {
    f.resize(f.size() + 1);
    FPoly<P>::polyint(f.data(), f.size());
    return f;
}

template <u32 P>
FPoly<P> inv(const FPoly<P>& f) {
    return FPoly<P>(FPoly<P>::polyinv(f));
}

template <u32 P>
FPoly<P> ln(const FPoly<P>& f) {
    return FPoly<P>(FPoly<P>::polyln(f));
}

template <u32 P>
FPoly<P> exp(const FPoly<P>& f) {
    return FPoly<P>(FPoly<P>::polyexp(f));
}

template <u32 P>
FPoly<P> sqrt(const FPoly<P>& f) {
    usize n = f.size();
    usize k = 0;
    while (k < f.size() && !f[k]) k++;
    if (k == f.size()) return FPoly<P>(n);
    if (k & 1) throw std::invalid_argument("square root does not exist");
    usize off = k / 2;
    if (off >= n) return FPoly<P>(n);
    FPoly<P> h(f.begin() + k, f.end());
    h.resize(n);
    FPoly<P> q(FPoly<P>::polysqrt(h)), g(n);
    for (usize i = 0; i + off < n; i++) g[off + i] = q[i];
    return g;
}

template <u32 P>
std::pair<FPoly<P>, FPoly<P>> div(const FPoly<P>& f, const FPoly<P>& g) {
    if (g.empty()) throw std::invalid_argument("divider is empty");
    if (f.size() < g.size()) return {{}, f};
    usize n = f.size() - g.size() + 1;
    FPoly<P> rf(f), rg(g);
    std::reverse(rf.begin(), rf.end());
    std::reverse(rg.begin(), rg.end());
    rf.resize(n);
    rg.resize(n);
    FPoly<P> q = rf * inv(rg);
    q.resize(n);
    std::reverse(q.begin(), q.end());
    FPoly<P> r = f - q * g;
    r.resize(g.size() - 1);
    r.trim();
    return {q, r};
}

}  // namespace acm

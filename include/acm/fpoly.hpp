#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <initializer_list>
#include <utility>
#include <vector>

#include "modint.hpp"

namespace acm
{

template <i32 P>
struct FPoly;

template <i32 P>
std::pair<FPoly<P>, FPoly<P>> div(const FPoly<P>&, const FPoly<P>&);

template <i32 P>
struct FPoly: std::vector<SModint<P>> {
    using Mint = SModint<P>;
    using Base = std::vector<Mint>;
    using Base::Base;

    static constexpr bool is_prime_const(i32 x) {
        if (x < 2) return false;
        for (i32 i = 2; i64(i) * i <= x; i++)
            if (x % i == 0) return false;
        return true;
    }

    static_assert(P >= 3 && is_prime_const(P), "P must be an odd prime");
    static constexpr int LG = __builtin_ctz(P - 1);

    static constexpr Mint primitive_root_const() {
        i32 phi = P - 1, x = phi;
        std::array<i32, 32> fac{};
        int cnt = 0;
        for (i32 i = 2; i64(i) * i <= x; i++) {
            if (x % i) continue;
            fac[cnt++] = i;
            while (x % i == 0) x /= i;
        }
        if (x > 1) fac[cnt++] = x;
        for (i32 g = 2;; g++) {
            bool ok = true;
            for (int i = 0; i < cnt; i++)
                if (pow(Mint(g), phi / fac[i]) == Mint(1)) {
                    ok = false;
                    break;
                }
            if (ok) return Mint(g);
        }
        return Mint(0);
    }

    // Primitive root of P, computed at compile time.
    static constexpr Mint G = primitive_root_const();

    // w[k] is a primitive 2^k-th root, iw[k] its inverse.
    static constexpr std::array<Mint, 32> make_w() {
        std::array<Mint, 32> a{};
        a[LG] = pow(G, (P - 1) >> LG);
        for (int i = LG - 1; i >= 0; i--) a[i] = a[i + 1] * a[i + 1];
        return a;
    }

    static constexpr std::array<Mint, 32> make_iw() {
        std::array<Mint, 32> a{};
        a[LG] = w[LG].inv();
        for (int i = LG - 1; i >= 0; i--) a[i] = a[i + 1] * a[i + 1];
        return a;
    }

    inline static constexpr std::array<Mint, 32> w = make_w();
    inline static constexpr std::array<Mint, 32> iw = make_iw();

    // Transposed NTT: forward DIF (bit-reversed output), inverse DIT
    // (bit-reversed input). No explicit bit-reversal permutation is needed.
    static void dft(Mint* a, usize n) {
        for (usize len = n; len > 1; len >>= 1) {
            usize h = len >> 1;
            Mint wl = w[std::countr_zero(len)];
            for (usize i = 0; i < n; i += len) {
                Mint x = 1;
                for (usize j = 0; j < h; j++) {
                    Mint u = a[i + j], v = a[i + j + h];
                    a[i + j] = u + v;
                    a[i + j + h] = (u - v) * x;
                    x *= wl;
                }
            }
        }
    }

    static void idft(Mint* a, usize n) {
        for (usize len = 2; len <= n; len <<= 1) {
            usize h = len >> 1;
            Mint wl = iw[std::countr_zero(len)];
            for (usize i = 0; i < n; i += len) {
                Mint x = 1;
                for (usize j = 0; j < h; j++) {
                    Mint u = a[i + j], v = a[i + j + h] * x;
                    a[i + j] = u + v;
                    a[i + j + h] = u - v;
                    x *= wl;
                }
            }
        }
        Mint z = Mint(n).inv();
        for (usize i = 0; i < n; i++) a[i] *= z;
    }

    static std::vector<Mint> convolution(
        const std::vector<Mint>& a, const std::vector<Mint>& b
    ) {
        usize n = a.size(), m = b.size();
        if (!n || !m) return {};
        usize need = n + m - 1;
        if (std::min(n, m) <= 32) {
            std::vector<Mint> c(need);
            for (usize i = 0; i < n; i++)
                for (usize j = 0; j < m; j++) c[i + j] += a[i] * b[j];
            return c;
        }
        usize len = std::bit_ceil(need);
        assert(len <= (usize(1) << LG));
        std::vector<Mint> x(a), y(b);
        x.resize(len);
        y.resize(len);
        dft(x.data(), len);
        dft(y.data(), len);
        for (usize i = 0; i < len; i++) x[i] *= y[i];
        idft(x.data(), len);
        x.resize(need);
        return x;
    }

    static std::vector<Mint> polyinv(const std::vector<Mint>& f) {
        usize n = f.size();
        assert(n && f[0] != Mint(0));
        usize len = std::bit_ceil(n);
        std::vector<Mint> out(len), t1(len), t2(len);
        out[0] = f[0].inv();
        for (usize k = 1, k2 = 2; k < len; k = k2, k2 <<= 1) {
            usize c = std::min(k2, n);
            std::copy(f.begin(), f.begin() + c, t1.begin());
            std::fill(t1.begin() + c, t1.begin() + k2, Mint(0));
            std::copy(out.begin(), out.begin() + k, t2.begin());
            std::fill(t2.begin() + k, t2.begin() + k2, Mint(0));
            dft(t1.data(), k2);
            dft(t2.data(), k2);
            for (usize i = 0; i < k2; i++) t1[i] *= t2[i];
            idft(t1.data(), k2);
            std::fill(t1.begin(), t1.begin() + k, Mint(0));
            dft(t1.data(), k2);
            for (usize i = 0; i < k2; i++) t1[i] *= t2[i];
            idft(t1.data(), k2);
            for (usize i = 0; i < k; i++) out[k + i] = -t1[k + i];
        }
        out.resize(n);
        return out;
    }

    static std::vector<Mint> polyder(const std::vector<Mint>& f) {
        usize n = f.size();
        if (!n) return {};
        std::vector<Mint> g(n - 1);
        for (usize i = 1; i < n; i++) g[i - 1] = f[i] * Mint(i);
        return g;
    }

    static std::vector<Mint> polyint(const std::vector<Mint>& f) {
        usize n = f.size();
        assert(n < usize(P));
        std::vector<Mint> g(n + 1);
        static std::vector<Mint> inv{0, 1};
        while (inv.size() <= n) {
            usize i = inv.size();
            inv.push_back(-Mint(P / i) * inv[P % i]);
        }
        for (usize i = 1; i <= n; i++) g[i] = f[i - 1] * inv[i];
        return g;
    }

    static std::vector<Mint> polyln(const std::vector<Mint>& f) {
        usize n = f.size();
        if (!n) return {};
        assert(f[0] == Mint(1));
        auto d = polyder(f);
        auto g = polyinv(f);
        auto q = convolution(d, g);
        q.resize(n);
        auto out = polyint(q);
        out.resize(n);
        return out;
    }

    static std::vector<Mint> polyexp(const std::vector<Mint>& f) {
        usize n = f.size();
        if (!n) return {};
        assert(f[0] == Mint(0));
        std::vector<Mint> g{1};
        while (g.size() < n) {
            usize k = std::min<usize>(g.size() * 2, n);
            std::vector<Mint> gk(g);
            gk.resize(k);
            auto lg = polyln(gk);
            for (usize i = 0; i < k; i++)
                lg[i] = (i < n ? f[i] : Mint(0)) - lg[i];
            lg[0] += Mint(1);
            g = convolution(g, lg);
            g.resize(k);
        }
        g.resize(n);
        return g;
    }

    static std::vector<Mint> polysqrt(const std::vector<Mint>& f) {
        usize n = f.size();
        if (!n) return {};
        assert(f[0] != Mint(0));
        auto r = acm::sqrt(f[0]);
        assert(r);
        i32 rv = (*r)();
        Mint root = rv <= P - rv ? *r : -*r;
        Mint inv2 = Mint(2).inv();
        std::vector<Mint> g{root};
        while (g.size() < n) {
            usize k = std::min<usize>(g.size() * 2, n);
            std::vector<Mint> gk(g);
            gk.resize(k);
            auto gi = polyinv(gk);
            std::vector<Mint> fk(k);
            for (usize i = 0; i < k && i < n; i++) fk[i] = f[i];
            auto q = convolution(fk, gi);
            q.resize(k);
            g.resize(k);
            for (usize i = 0; i < k; i++) g[i] = (g[i] + q[i]) * inv2;
        }
        g.resize(n);
        return g;
    }

    FPoly() = default;
    FPoly(std::initializer_list<Mint> l): Base(l) {}
    FPoly(const Base& b): Base(b) {}
    FPoly(Base&& b): Base(std::move(b)) {}

    void trim() {
        while (!this->empty() && this->back() == Mint(0)) this->pop_back();
    }

    FPoly& operator+=(const FPoly& b) {
        if (this->size() < b.size()) this->resize(b.size());
        for (usize i = 0; i < b.size(); i++) (*this)[i] += b[i];
        return *this;
    }
    FPoly& operator-=(const FPoly& b) {
        if (this->size() < b.size()) this->resize(b.size());
        for (usize i = 0; i < b.size(); i++) (*this)[i] -= b[i];
        return *this;
    }
    FPoly& operator*=(const FPoly& b) {
        *this = FPoly(convolution(*this, b));
        return *this;
    }
    FPoly& operator*=(Mint k) {
        for (auto& x: *this) x *= k;
        return *this;
    }
    FPoly& operator/=(const FPoly& b) {
        *this = div(*this, b).first;
        return *this;
    }

    friend FPoly operator-(FPoly f) {
        for (auto& x: f) x = -x;
        return f;
    }
    friend FPoly operator+(FPoly a, const FPoly& b) { return a += b; }
    friend FPoly operator-(FPoly a, const FPoly& b) { return a -= b; }
    friend FPoly operator*(FPoly a, const FPoly& b) { return a *= b; }
    friend FPoly operator*(FPoly a, Mint k) { return a *= k; }
    friend FPoly operator*(Mint k, FPoly a) { return a *= k; }
    friend FPoly operator/(FPoly a, const FPoly& b) { return a /= b; }

    friend bool operator==(const FPoly& a, const FPoly& b) {
        return static_cast<const Base&>(a) == static_cast<const Base&>(b);
    }
    friend bool operator!=(const FPoly& a, const FPoly& b) {
        return !(a == b);
    }
};

template <i32 P>
FPoly<P> inv(const FPoly<P>& f) {
    return FPoly<P>(FPoly<P>::polyinv(f));
}

template <i32 P>
FPoly<P> ln(const FPoly<P>& f) {
    if (f.empty()) return {};
    return FPoly<P>(FPoly<P>::polyln(f));
}

template <i32 P>
FPoly<P> exp(const FPoly<P>& f) {
    if (f.empty()) return {};
    return FPoly<P>(FPoly<P>::polyexp(f));
}

template <i32 P>
FPoly<P> sqrt(const FPoly<P>& f) {
    using Mint = SModint<P>;
    usize n = f.size();
    if (!n) return {};
    usize k = 0;
    while (k < n && f[k] == Mint(0)) k++;
    if (k == n) return FPoly<P>(n);
    assert(k % 2 == 0);
    FPoly<P> h(f.begin() + k, f.end());
    h.resize(n);
    auto s = FPoly<P>::polysqrt(h);
    s.resize(n);
    FPoly<P> res(n);
    for (usize i = 0; i + k / 2 < n; i++) res[i + k / 2] = s[i];
    return res;
}

template <i32 P>
FPoly<P> derivative(FPoly<P> f) {
    if (f.empty()) return f;
    return FPoly<P>(FPoly<P>::polyder(f));
}

template <i32 P>
FPoly<P> integrate(FPoly<P> f) {
    return FPoly<P>(FPoly<P>::polyint(f));
}

template <i32 P>
std::pair<FPoly<P>, FPoly<P>> div(const FPoly<P>& f, const FPoly<P>& g) {
    assert(!g.empty());
    if (f.size() < g.size()) return {FPoly<P>{}, f};
    usize n = f.size() - g.size() + 1;
    FPoly<P> rf(f.rbegin(), f.rend());
    FPoly<P> rg(g.rbegin(), g.rend());
    rf.resize(n);
    rg.resize(n);
    FPoly<P> q = rf * inv(rg);
    q.resize(n);
    std::reverse(q.begin(), q.end());
    FPoly<P> r = f - q * g;
    r.resize(g.size() - 1);
    r.trim();
    return {std::move(q), std::move(r)};
}

}  // namespace acm

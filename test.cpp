#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <queue>
#include <vector>

const int N = 400001;
const int p = 998244353;
typedef std::vector<int> Poly;

char buf[1 << 25], *p1 = buf, *p2 = buf;
#define getchar()                                                            \
    (p1 == p2 && (p2 = (p1 = buf) + fread(buf, 1, 1 << 21, stdin), p1 == p2) \
         ? EOF                                                               \
         : *p1++)
inline int read() {
    int x = 0, f = 1;
    char ch = getchar();
    while (ch > '9' || ch < '0') {
        if (ch == '-') f = -1;
        ch = getchar();
    }
    while (ch >= '0' && ch <= '9') x = x * 10 + ch - 48, ch = getchar();
    return x * f;
}

inline int pls(int a, int b) { return a + b >= p ? a + b - p : a + b; }
inline int mus(int a, int b) { return a - b < 0 ? a - b + p : a - b; }
inline int prd(int a, int b) { return 1ll * a * b % p; }
inline int fastpow(int a, int b) {
    int r = 1;
    while (b) {
        if (b & 1) r = 1ll * r * a % p;
        a = 1ll * a * a % p;
        b >>= 1;
    }
    return r;
}

int rev[N];
void NTT(Poly& a) {
    int N = a.size();
    for (int i = 0; i < N; i++) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) * N / 2);
    for (int i = 0; i < N; i++)
        if (i > rev[i]) std::swap(a[i], a[rev[i]]);
    for (int n = 2, m = 1; n <= N; m = n, n <<= 1) {
        int g1 = fastpow(3, (p - 1) / n), t1, t2;
        for (int l = 0; l < N; l += n) {
            int g = 1;
            for (int i = l; i < l + m; i++) {
                t1 = a[i], t2 = prd(a[i + m], g);
                a[i] = pls(t1, t2), a[i + m] = mus(t1, t2);
                g = prd(g, g1);
            }
        }
    }
    return;
}

void INTT(Poly& a) {
    NTT(a), std::reverse(a.begin() + 1, a.end());
    int invN = fastpow(a.size(), p - 2);
    for (int i = 0; i < (int)a.size(); i++) a[i] = prd(a[i], invN);
}

Poly Mul(Poly a, Poly b) {
    int n = a.size() + b.size() - 1, N = 1;
    while (N < (int)(a.size() + b.size())) N <<= 1;
    a.resize(N), b.resize(N), NTT(a), NTT(b);
    for (int i = 0; i < N; i++) a[i] = prd(a[i], b[i]);
    INTT(a), a.resize(n);
    return a;
}

Poly MulT(Poly a, Poly b) {
    int n = a.size(), m = b.size();
    std::reverse(b.begin(), b.end()), b = Mul(a, b);
    for (int i = 0; i < n; i++) a[i] = b[i + m - 1];
    return a;
}

Poly tmp;
Poly Inv(Poly a, int n) {
    if (n == 1) return Poly(1, fastpow(a[0], p - 2));
    Poly b = Inv(a, (n + 1) / 2);
    int N = 1;
    while (N <= 2 * n) N <<= 1;
    tmp.resize(N), b.resize(N);
    for (int i = 0; i < n; i++) tmp[i] = a[i];
    for (int i = n; i < N; i++) tmp[i] = 0;
    NTT(tmp), NTT(b);
    for (int i = 0; i < N; i++)
        b[i] = mus(prd(2, b[i]), prd(prd(b[i], b[i]), tmp[i]));
    INTT(b), b.resize(n);
    return b;
}

Poly Dervt(Poly a) {
    for (int i = 0; i < (int)a.size() - 1; i++) a[i] = prd(i + 1, a[i + 1]);
    a.pop_back();
    return a;
}

#define lc(k) k << 1
#define rc(k) k << 1 | 1
Poly Q[N];

void MultiInit(Poly& a, int k, int l, int r) {
    if (l == r) {
        Q[k].resize(2);
        Q[k][0] = 1, Q[k][1] = mus(0, a[l]);
        return;
    }
    int m = (l + r) / 2;
    MultiInit(a, lc(k), l, m), MultiInit(a, rc(k), m + 1, r);
    Q[k] = Mul(Q[lc(k)], Q[rc(k)]);
    return;
}

void Multipoints(int k, int l, int r, Poly F, Poly& g) {
    F.resize(r - l + 1);
    if (l == r) return void(g[l] = F[0]);
    int m = (l + r) / 2;
    Multipoints(lc(k), l, m, MulT(F, Q[rc(k)]), g);
    Multipoints(rc(k), m + 1, r, MulT(F, Q[lc(k)]), g);
    return;
}

void Multipoint(Poly f, Poly a, Poly& v, int n) {
    f.resize(n + 1), a.resize(n);
    MultiInit(a, 1, 0, n - 1), v.resize(n);
    Multipoints(1, 0, n - 1, MulT(f, Inv(Q[1], n + 1)), v);
    return;
}

int n;
Poly x, y, t, f;

Poly F[N];
void InterInit(int k, int l, int r) {
    if (l == r) {
        F[k].resize(2);
        F[k][0] = mus(0, x[l]), F[k][1] = 1;
        return;
    }
    int m = (l + r) / 2;
    InterInit(lc(k), l, m);
    InterInit(rc(k), m + 1, r);
    F[k] = Mul(F[lc(k)], F[rc(k)]);
    return;
}

Poly InterSolve(int k, int l, int r, Poly& t) {
    if (l == r) return Poly(1, t[l]);
    int m = (l + r) / 2;
    Poly L(InterSolve(lc(k), l, m, t));
    Poly R(InterSolve(rc(k), m + 1, r, t));
    R = Mul(R, F[lc(k)]);
    L = Mul(L, F[rc(k)]);
    for (int i = 0; i < (int)R.size(); i++) L[i] = pls(L[i], R[i]);
    return L;
}

void Interpolate(Poly x, Poly y, Poly& f, int n) {
    InterInit(1, 0, n - 1);
    F[1] = Dervt(F[1]), Multipoint(F[1], x, t, n);
    for (int i = 0; i < n; i++) t[i] = prd(y[i], fastpow(t[i], p - 2));
    f = InterSolve(1, 0, n - 1, t);
    return;
}

int main() {
    n = read();
    for (int i = 0; i < n; i++) x.push_back(read()), y.push_back(read());
    Interpolate(x, y, f, n);
    for (int i = 0; i < n; i++) std::printf("%d ", f[i]);
    return 0;
}

#include <cassert>
#include <random>
#include <vector>

#include "acm/fpoly.hpp"

int main() {
    using P = acm::FPoly<998244353>;
    using M = P::Mint;
    std::mt19937 rng(3);
    for (int tc = 0; tc < 50; tc++) {
        std::vector<M> x(40), y(35), want(74);
        for (auto& v: x) v = rng() % 1000;
        for (auto& v: y) v = rng() % 1000;
        for (int i = 0; i < 40; i++)
            for (int j = 0; j < 35; j++) want[i + j] += x[i] * y[j];
        P z = P(x.begin(), x.end()) * P(y.begin(), y.end());
        for (int i = 0; i < 74; i++) assert(z[i] == want[i]);
    }
    P f{1, 2, 3, 4};
    auto one = f * acm::inv(f);
    one.resize(4);
    assert(one[0] == M(1) && !one[1] && !one[2] && !one[3]);
    P h{0, 2, 3, 4};
    assert(acm::exp(acm::ln(f))[0] == M(1));
    auto [q, r] = (f * h).size() ? acm::div(f * h, f) : std::pair<P, P>{};
    assert(r.empty() && q.size() == h.size());

    constexpr int n = 257;
    P u(n), v(n);
    u[0] = 1;
    for (int i = 1; i < n; i++) u[i] = rng() % 1000;
    auto iu = acm::inv(u);
    auto id = u * iu;
    id.resize(n);
    assert(id[0] == M(1));
    for (int i = 1; i < n; i++) assert(!id[i]);

    v[0] = 0;
    for (int i = 1; i < n; i++) v[i] = rng() % 1000;
    auto ev = acm::exp(v);
    auto lv = acm::ln(ev);
    for (int i = 0; i < n; i++) assert(lv[i] == v[i]);

    for (auto& x: u) x = rng() % 1000;
    u[0] = 1;
    auto square = u * u;
    square.resize(n);
    auto root = acm::sqrt(square);
    auto check = root * root;
    check.resize(n);
    for (int i = 0; i < n; i++) assert(check[i] == square[i]);

    P shifted_root(n);
    shifted_root[3] = 1;
    for (int i = 4; i < n; i++) shifted_root[i] = rng() % 1000;
    auto shifted_square = shifted_root * shifted_root;
    shifted_square.resize(n);
    auto shifted = acm::sqrt(shifted_square);
    check = shifted * shifted;
    check.resize(n);
    for (int i = 0; i < n; i++) assert(check[i] == shifted_square[i]);
}

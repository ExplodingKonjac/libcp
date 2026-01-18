#include <bit>
#include <vector>

#include "cp/fast_io.hpp"
#include "cp/fpoly.hpp"

using cp::qin, cp::qout;

constexpr int MOD = 998244353;
using Poly = cp::FPoly<MOD, (1 << 17)>;

int main() {
    auto [n, m] = qin.scan<int, int>().value();
    Poly f(n, true), g(n, true);
    for (int i = 0; i < n; i++) f[i] = qin.scan<int>().value();
    for (int i = 0; i < m; i++) g[i] = qin.scan<int>().value();
    f *= g;
    for (auto& i: f) qout.print(i()), qout.print(' ');
    qout.print('\n');
    return 0;
}
/*
16 16
1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0

16 16
0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0

*/
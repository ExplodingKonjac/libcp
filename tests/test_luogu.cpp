#define CP_FASTIO_USE_BUF

#include "cp/fast_io.hpp"
#include "cp/fpoly.hpp"
#include "cp/modint.hpp"

using cp::qin, cp::qout;
using namespace cp::literals;

constexpr int MOD = 998244353, MAXN = 500000;
using Mint = cp::SModint<MOD>;
using Poly = cp::FPoly<MOD>;

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
#include <algorithm>
#include <format>
#include <functional>
#include <random>
#include <ranges>

#include "cp/fast_io.hpp"
#include "cp/fpoly.hpp"
#include "cp/modint.hpp"

using namespace cp::literals;
using cp::qin, cp::qout;

constexpr int MOD = 998244353;
using Mint = cp::SModint<MOD>;
using Poly = cp::FPoly<MOD, (1 << 20)>;

template <std::uint32_t P, typename CharT>
struct std::formatter<cp::SModint<P>, CharT>: std::formatter<int, CharT> {
    constexpr auto format(auto v, auto& ctx) const {
        return std::formatter<int>::format(int(v), ctx);
    }
};

template <std::uint32_t P, size_t MAXN, typename CharT>
struct std::formatter<cp::FPoly<P, MAXN>, CharT>
    : std::range_formatter<cp::SModint<P>, CharT> {};

unsigned a[1000005];

unsigned next() {
    static unsigned seed = 114514;
    seed ^= seed << 5;
    seed ^= seed >> 13;
    seed ^= seed << 17;
    return seed;
}

int main() {
    // freopen("P4238_1.in", "r", stdin);
    // freopen("output.out", "w", stdout);

    int T = 100;
    while (T--) {
        int n = 1000000;

        for (int i = 0; i < n; i++) a[i] = next();
        Poly ans = Poly(a, a + n).inv();

        int checksum = 0;
        for (auto& i: ans) checksum ^= i();
        qout.println(checksum);
    }

    return 0;
}

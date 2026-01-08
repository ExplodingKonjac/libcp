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

std::mt19937 mt_rnd(19260817);

template <std::uint32_t P, typename CharT>
struct std::formatter<cp::SModint<P>, CharT>: std::formatter<int, CharT> {
    constexpr auto format(auto v, auto& ctx) const {
        return std::formatter<int>::format(int(v), ctx);
    }
};

template <std::uint32_t P, size_t MAXN, typename CharT>
struct std::formatter<cp::FPoly<P, MAXN>, CharT>
    : std::range_formatter<cp::SModint<P>, CharT> {};

int main() {
    // freopen("P4238_1.in", "r", stdin);
    // freopen("output.out", "w", stdout);

    // Poly f{1, 1, 4, 5, 1}, g{1, 9, 1, 9, 8};
    // f *= g;
    // qout.println("{}"_fmt, f);

    int T = 10;
    while (T--) {
        int n = 1000000;

        std::uniform_int_distribution dist(1, MOD - 1);
        Poly f(std::views::iota(0, n) |
               std::views::transform([&](auto...) { return dist(mt_rnd); }));

        f = f.inv();
        qout.println(std::ranges::fold_left(
            f | std::views::transform([](auto& x) { return int(x); }), 0,
            std::bit_xor<>{}));
    }
    return 0;
}

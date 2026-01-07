#include "cp/fast_io.hpp"
#include "cp/modint.hpp"
#include "cp/vmodint.hpp"

using namespace cp::literals;
using cp::qin, cp::qout;

constexpr int MOD = 998244353;
using Mint = cp::SModint<MOD>;
using VMint = cp::SVModint<MOD>;

int main() {
    VMint a{-1, 114, 998244352, 0, 0, 0, 0, 0};
    VMint b{-2, 514, 2, 0, 0, 0, 0, 123};
    VMint c = a * b;

    qout.println("[{}, {}, {}, {}, {}, {}, {}, {}]"_fmt, int(c[0]), int(c[1]),
                 int(c[2]), int(c[3]), int(c[4]), int(c[5]), int(c[6]),
                 int(c[7]));
    return 0;
}
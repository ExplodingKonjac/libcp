#include <cassert>

#include "cp/modint.hpp"

using namespace cp;

int main() {
    using Mint = SModint<7>;

    assert(legendre(Mint{0}) == 0);
    assert(legendre(Mint{2}) == 1);
    assert(legendre(Mint{3}) == -1);

    assert(sqrt(Mint{0}) == Mint{0});
    auto root = sqrt(Mint{2});
    assert(root && *root * *root == Mint{2});
    assert(!sqrt(Mint{3}));
}

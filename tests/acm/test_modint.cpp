#include <cassert>
#include "acm/modint.hpp"

int main() {
    using M = acm::SModint<998244353>;
    assert((M(2) / M(3) * M(3)).val() == 2);
    assert(acm::pow(M(2), 10).val() == 1024);
    auto r = acm::sqrt(M(4));
    assert(r && *r * *r == M(4));
    acm::DModint::set_mod(17);
    assert((acm::DModint(5) * acm::DModint(7)).val() == 1);
}

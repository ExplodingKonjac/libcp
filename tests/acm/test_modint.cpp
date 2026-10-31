#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

#include "acm/modint.hpp"

using acm::i128;
using acm::i32;
using acm::i64;
using acm::u32;
using acm::u64;

int main() {
    using M = acm::SModint<998244353>;
    constexpr u32 MOD = M::mod();

    static_assert(std::is_same_v<decltype(M(0)()), i32>);
    static_assert(std::is_same_v<decltype(M::mod()), i32>);
    static_assert(M(2) + M(3) == M(5));
    static_assert(M(2) * M(3) == M(6));
    static_assert(M(0) - M(1) == M(MOD - 1));
    static_assert(M(MOD - 1) + M(1) == M(0));
    static_assert(M(MOD - 1) + M(MOD - 1) == M(MOD - 2));
    static_assert(M(1) - M(MOD - 1) == M(2));
    static_assert(M(-1) == M(MOD - 1));
    static_assert(M(u64(1) << 40) == M((u64(1) << 40) % MOD));
    static_assert(M(123).operator()() == 123);
    static_assert(acm::pow(M(2), 10) == M(1024));
    static_assert(M(2).inv() * M(2) == M(1));

    constexpr i64 I64MIN = std::numeric_limits<i64>::min();
    constexpr u64 U64MAX = std::numeric_limits<u64>::max();
    static_assert(M(I64MIN) == M(i32((i128(I64MIN) % MOD + MOD) % MOD)));
    static_assert(M(U64MAX) == M(i32(U64MAX % MOD)));

    assert((M(2) / M(3) * M(3))() == 2);
    assert(acm::pow(M(2), 10)() == 1024);
    assert(acm::pow(M(2), u64(40))() == (1099511627776ULL % MOD));
    assert((acm::pow(M(2), -3) * M(8))() == 1);

    assert(M(I64MIN)() == i32((i128(I64MIN) % MOD + MOD) % MOD));
    assert(M(U64MAX)() == i32(U64MAX % MOD));
    assert((M(MOD - 1) * M(MOD - 1))() == 1);
    assert((M(3).inv() * M(3))() == 1);
    assert(M(1).inv()() == 1);
    assert(!M(0) && M(1));
    assert(M(1) != M(2));

    M x = MOD - 1;
    ++x;
    assert(x() == 0);
    x--;
    assert(x() == MOD - 1);
    assert((-M(0))() == 0);
    assert((-M(1))() == MOD - 1);
    assert((+M(7))() == 7);

    // legendre / sqrt on a small odd prime.
    using S = acm::SModint<7>;
    assert(acm::legendre(S(0)) == 0);
    assert(acm::legendre(S(2)) == 1);
    assert(acm::legendre(S(3)) == -1);
    auto r2 = acm::sqrt(S(2));
    assert(r2 && *r2 * *r2 == S(2));
    assert(!acm::sqrt(S(3)));
    assert(acm::sqrt(S(0)) == S(0));

    // p = 2 is a special case.
    using B = acm::SModint<2>;
    assert(acm::sqrt(B(0)) == B(0));
    assert(acm::sqrt(B(1)) == B(1));

    // A prime just below 2^31 stresses the i32 qmo path.
    using L = acm::SModint<2147483647>;
    static_assert(L(2147483646) + L(1) == L(0));
    static_assert(L(0) - L(1) == L(2147483646));
    static_assert(L(2147483646) + L(2147483646) == L(2147483645));
    static_assert(L(2147483646) * L(2147483646) == L(1));
    assert(L(2147483646).inv() == L(2147483646));
    assert((L(2).inv() * L(2))() == 1);

    // Dynamic modulus.
    acm::DModint::set_mod(2147483647);
    assert((acm::DModint(2147483646) + acm::DModint(1))() == 0);
    assert((acm::DModint(0) - acm::DModint(1))() == 2147483646);
    assert((acm::DModint(2147483646) * acm::DModint(2147483646))() == 1);
    acm::DModint::set_mod(17);
    assert(acm::DModint::mod() == 17);
    assert((acm::DModint(5) * acm::DModint(7))() == 1);
    assert((acm::DModint(16) + acm::DModint(2))() == 1);
    assert((acm::DModint(0) - acm::DModint(1))() == 16);
    assert(acm::pow(acm::DModint(3), 16)() == 1);
    acm::DModint::set_mod(998244353);
}

#include <cassert>
#include <iostream>
#include <tuple>

#include "cp/utils/hash_value.hpp"

using namespace cp;

void test_default_construct() {
    HashValue<1000000007> a;
    HashValue<1000000007> b(0);
    // default leaves val_ indeterminate, but += should not crash
    a += b;
    // no assertion — just verify it doesn't blow up
}

void test_integral_construct() {
    HashValue<1000000007> a(42);
    auto [v] = a.tuple();
    assert(v == 42);

    // value equal to MOD → wraps to 0
    HashValue<1000000007> b(1000000007);
    auto [v2] = b.tuple();
    assert(v2 == 0);

    // value larger than MOD
    HashValue<1000000007> c(1000000009);
    auto [v3] = c.tuple();
    assert(v3 == 2);

    // negative value
    HashValue<1000000007> d(-1);
    auto [v4] = d.tuple();
    assert(v4 == 1000000006);
}

void test_addition() {
    HashValue<1000000007> a(5);
    HashValue<1000000007> b(3);
    auto c = a + b;
    assert((c.tuple() == std::tuple(8)));

    // wrap-around
    HashValue<1000000007> x(1000000006);
    HashValue<1000000007> y(5);
    auto z = x + y;
    assert((z.tuple() == std::tuple(4)));
}

void test_subtraction() {
    HashValue<1000000007> a(10);
    HashValue<1000000007> b(3);
    auto c = a - b;
    assert((c.tuple() == std::tuple(7)));

    // underflow
    HashValue<1000000007> x(0);
    HashValue<1000000007> y(1);
    auto z = x - y;
    assert((z.tuple() == std::tuple(1000000006)));
}

void test_multiplication() {
    HashValue<1000000007> a(1000000);
    HashValue<1000000007> b(2000000);
    auto c = a * b;
    assert((c.tuple() == std::tuple((i64(1000000) * 2000000) % 1000000007)));

    // large multiplication near MOD
    HashValue<1000000007> x(1000000005);
    HashValue<1000000007> y(1000000005);
    auto z = x * y;
    assert(
        (z.tuple() == std::tuple((i64(1000000005) * 1000000005) % 1000000007))
    );
}

void test_compound_assign() {
    HashValue<1000000007> a(5);
    HashValue<1000000007> b(3);

    a += b;
    assert((a.tuple() == std::tuple(8)));

    a -= b;
    assert((a.tuple() == std::tuple(5)));

    a *= b;
    assert((a.tuple() == std::tuple(15)));
}

void test_comparison() {
    HashValue<1000000007> a(42);
    HashValue<1000000007> b(42);
    HashValue<1000000007> c(43);

    assert(a == b);
    assert(a != c);
    assert(a < c);
    assert(c > a);
    assert(a <= b);
    assert(a >= b);
}

void test_multiple_mods() {
    constexpr i32 M1 = 1000000007;
    constexpr i32 M2 = 1000000009;

    HashValue<M1, M2> a(42);
    auto [v1, v2] = a.tuple();
    assert(v1 == 42);
    assert(v2 == 42);

    // value larger than M1 but smaller than M2
    HashValue<M1, M2> b(1000000008);
    auto [v3, v4] = b.tuple();
    assert(v3 == 1);           // 1000000008 % M1
    assert(v4 == 1000000008);  // 1000000008 % M2 = 1000000008 (< M2)

    // addition with wrap on both mods
    HashValue<M1, M2> x(1000000006);
    HashValue<M1, M2> y(5);
    auto z = x + y;
    auto [v5, v6] = z.tuple();
    assert(v5 == 4);  // (1000000006 + 5) % M1
    assert(v6 == 2);  // (1000000006 + 5) % M2

    // subtraction with underflow on both mods
    HashValue<M1, M2> p(0);
    HashValue<M1, M2> q(1);
    auto r = p - q;
    auto [v7, v8] = r.tuple();
    assert(v7 == M1 - 1);
    assert(v8 == M2 - 1);

    // multiplication with modulo
    HashValue<M1, M2> u(M1 - 1);
    auto w = u * u;
    auto [v9, v10] = w.tuple();
    assert(v9 == (i64(M1 - 1) * (M1 - 1)) % M1);
    assert(v10 == (i64(M1 - 1) * (M1 - 1)) % M2);
}

void test_componentwise_construct() {
    constexpr i32 M1 = 998244353;
    constexpr i32 M2 = 1000000007;
    constexpr i32 M3 = 1000000009;

    HashValue<M1, M2, M3> a(1, -2, 1000000011LL);
    auto [v1, v2, v3] = a.tuple();
    assert(v1 == 1);
    assert(v2 == M2 - 2);
    assert(v3 == 2);

    HashValue<M1, M2> b(M1 + 5LL, M2 + 7LL);
    auto [v4, v5] = b.tuple();
    assert(v4 == 5);
    assert(v5 == 7);
}

void test_three_mods() {
    HashValue<998244353, 1000000007, 1000000009> a(123456789);
    auto [v1, v2, v3] = a.tuple();
    assert(v1 == 123456789 % 998244353);
    assert(v2 == 123456789 % 1000000007);
    assert(v3 == 123456789 % 1000000009);
}

void test_chained_operations() {
    HashValue<1000000007> a(10);
    HashValue<1000000007> b(3);
    HashValue<1000000007> c(2);

    auto r = (a + b) * c;
    assert((r.tuple() == std::tuple(26)));

    auto s = a - b + c;
    assert((s.tuple() == std::tuple(9)));
}

void test_commutativity() {
    HashValue<1000000007> a(1000000005);
    HashValue<1000000007> b(1000000003);

    assert(a + b == b + a);
    assert(a * b == b * a);
    // subtraction is NOT commutative
    assert(a - b != b - a);
}

void test_associativity() {
    HashValue<1000000007> a(1000000001);
    HashValue<1000000007> b(1000000002);
    HashValue<1000000007> c(1000000003);

    assert((a + b) + c == a + (b + c));
    assert((a * b) * c == a * (b * c));
}

void test_identity() {
    HashValue<1000000007> a(42);
    HashValue<1000000007> zero(0);
    HashValue<1000000007> one(1);

    assert(a + zero == a);
    assert(a - zero == a);
    assert(a * one == a);
}

void test_negative_integral() {
    HashValue<998244353> a(-100);
    auto [v] = a.tuple();
    assert(v == 998244353 - 100);

    HashValue<998244353> b(-998244353);
    auto [v2] = b.tuple();
    assert(v2 == 0);
}

void test_inverse() {
    HashValue<1000000007> a(3);
    auto ai = a.inv();
    assert(a * ai == HashValue<1000000007>(1));

    constexpr i32 M1 = 998244353;
    constexpr i32 M2 = 1000000007;
    HashValue<M1, M2> b(2, 3);
    auto bi = b.inv();
    assert((b * bi == HashValue<M1, M2>(1)));
    assert((bi.tuple() == std::tuple((M1 + 1) / 2, 333333336)));
}

int main() {
    test_default_construct();
    test_integral_construct();
    test_addition();
    test_subtraction();
    test_multiplication();
    test_compound_assign();
    test_comparison();
    test_multiple_mods();
    test_componentwise_construct();
    test_three_mods();
    test_chained_operations();
    test_commutativity();
    test_associativity();
    test_identity();
    test_negative_integral();
    test_inverse();

    std::cout << "All tests passed.\n";
    return 0;
}

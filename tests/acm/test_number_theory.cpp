#include <algorithm>
#include <cassert>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "acm/number_theory.hpp"

using acm::i64;

void test_qpow() {
    assert(acm::qpow<i64>(2, 10, 1000) == 24);
    assert(acm::qpow<i64>(123, 0, 17) == 1);
    constexpr i64 mod = 4'000'000'000'000'000'003;
    assert(acm::qpow(mod - 1, i64{2}, mod) == 1);
}

void test_gcd_and_equation() {
    i64 x, y;
    assert(acm::exgcd<i64>(240, 46, x, y) == 2);
    assert(240 * x + 46 * y == 2);

    assert((acm::biequation<i64>(6, 3, 3) == std::pair<i64, i64>(0, 1)));
    assert((acm::biequation<i64>(9, 6, 3) == std::pair<i64, i64>(1, -1)));
    assert((acm::biequation<i64>(6, 9, 4) == std::pair<i64, i64>(-1, -1)));
}

void test_primes_and_factors() {
    std::mt19937_64 gen(123456789);
    assert(acm::miller_rabin<i64>(1'000'000'007, gen));
    assert(!acm::miller_rabin<i64>(1'000'000'016'000'000'063, gen));

    assert(acm::pollard_rho<i64>(4, gen) == 2);
    i64 factor = acm::pollard_rho<i64>(8051, gen);
    assert(factor != 1 && factor != 8051 && 8051 % factor == 0);

    assert(acm::factorize<i64>(1, gen).empty());
    auto factors = acm::factorize<i64>(2 * 2 * 3 * 83 * 97, gen);
    std::ranges::sort(factors);
    assert(factors == std::vector<i64>({2, 2, 3, 83, 97}));
}

void test_quadratic_residues() {
    assert(acm::legendre<i64>(0, 7) == 0);
    assert(acm::legendre<i64>(2, 7) == 1);
    assert(acm::legendre<i64>(3, 7) == -1);

    std::mt19937_64 gen(987654321);
    assert(acm::cipolla<i64>(0, 13, gen) == 0);
    auto root = acm::cipolla<i64>(10, 13, gen);
    assert(root && *root * *root % 13 == 10);
    assert(!acm::cipolla<i64>(2, 13, gen));
}

std::string direct_path(i64 p, i64 q, i64 r, i64 n) {
    std::string result = "I";
    i64 last = r / q;
    for (i64 i = 1; i <= n; i++) {
        i64 next = (p * i + r) / q;
        result += 'R';
        result.append(next - last, 'U');
        last = next;
    }
    return result;
}

void test_uniclidean() {
    using namespace std::string_literals;
    assert(
        acm::uniclidean(4, 7, 1, 10, "R"s, "U"s, ""s, std::plus{}) ==
        "RRURRURURRURRUR"
    );

    for (i64 p = 0; p <= 12; p++)
        for (i64 q = 1; q <= 10; q++)
            for (i64 r = 0; r <= 20; r++)
                for (i64 n = 0; n <= 12; n++)
                    assert(
                        acm::uniclidean(
                            p, q, r, n, "R"s, "U"s, "I"s, std::plus{}
                        ) == direct_path(p, q, r, n)
                    );
}

int main() {
    test_qpow();
    test_gcd_and_equation();
    test_primes_and_factors();
    test_quadratic_residues();
    test_uniclidean();
}

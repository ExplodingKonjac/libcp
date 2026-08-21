#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include "cp/interpolation.hpp"
#include "cp/modint.hpp"

using namespace cp;

namespace
{

template <typename T>
bool near(T lhs, T rhs, T eps = static_cast<T>(1e-9)) {
    return std::abs(lhs - rhs) <=
        eps * std::max<T>({1, std::abs(lhs), std::abs(rhs)});
}

void test_empty_and_singleton() {
    const std::vector<std::pair<int, int>> empty;
    assert(polynomial_interpolation(empty, 42) == 0);
    assert(
        (polynomial_coefficients<int>(empty, std::equal_to<>{}) ==
         std::vector<int>{0})
    );

    const std::vector<std::pair<double, double>> point{{3, 10}};
    assert(polynomial_interpolation(point, -100.0) == 10.0);
    assert(
        (polynomial_coefficients<double>(point, std::equal_to<>{}) ==
         std::vector<double>{10.0})
    );
}

void test_interpolation() {
    const std::vector<std::pair<int, int>> points{{-1, 10}, {0, 5}, {2, 7}};
    for (const auto& [x, y]: points)
        assert(polynomial_interpolation(points, x) == y);
    assert(polynomial_interpolation(points, 1) == 4);
    assert(
        polynomial_interpolation(std::vector<std::pair<int, int>>(points), 3) ==
        14
    );
}

void test_coefficients() {
    // f(x) = 2x^2 - 3x + 5, represented in ascending order.
    const std::vector<std::pair<int, double>> floating_points{
        {-1, 10.0}, {0, 5.0}, {2, 7.0}};
    const auto coefficients =
        polynomial_coefficients<double>(floating_points, std::equal_to<>{});
    assert(coefficients.size() == 3);
    assert(near(coefficients[0], 5.0));
    assert(near(coefficients[1], -3.0));
    assert(near(coefficients[2], 2.0));
}

void test_zero_x_and_custom_equality() {
    const std::vector<std::pair<double, double>> points_with_small_zero{
        {1e-12, 5.0}, {1.0, 4.0}, {2.0, 7.0}};
    const auto coefficients = polynomial_coefficients<double>(
        points_with_small_zero,
        [](double x, double) { return std::abs(x) < 1e-9; }
    );
    assert(coefficients.size() == 3);
    assert(near(coefficients[0], 5.0));
    assert(near(coefficients[1], -3.0));
    assert(near(coefficients[2], 2.0));
}

void test_smodint() {
    using Mint = SModint<998244353>;
    const std::vector<std::pair<Mint, Mint>> points{
        {Mint{0}, Mint{5}}, {Mint{1}, Mint{4}}, {Mint{2}, Mint{7}}};

    assert(polynomial_interpolation(points, Mint{3}) == Mint{14});

    const auto coefficients =
        polynomial_coefficients<Mint>(points, std::equal_to<>{});
    assert((coefficients == std::vector<Mint>{Mint{5}, Mint{-3}, Mint{2}}));
}

}  // namespace

int main() {
    test_empty_and_singleton();
    test_interpolation();
    test_coefficients();
    test_zero_x_and_custom_equality();
    test_smodint();
    std::cerr << "All tests passed!";
}

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "cp/utils/lambda.hpp"

using namespace cp;

void test_placeholder_basic() {
    auto f = _1;
    assert(f(42) == 42);
    assert(f(3.14) == 3.14);

    auto g = _2;
    assert(g(10, 20) == 20);
    assert(g(1, 2, 3) == 2);

    auto h = _3;
    assert(h(1, 2, 3) == 3);
    assert(h(1, 2, 3, 4, 5) == 3);
}

void test_arithmetic() {
    assert((_1 + _2)(3, 5) == 8);
    assert((_1 - _2)(10, 3) == 7);
    assert((_1 * _2)(4, 5) == 20);
    assert((_1 / _2)(15, 4) == 3);
    assert((_1 % _2)(15, 4) == 3);
}

void test_comparison() {
    assert((_1 == _2)(3, 3) == true);
    assert((_1 == _2)(3, 4) == false);
    assert((_1 != _2)(3, 4) == true);
    assert((_1 < _2)(3, 5) == true);
    assert((_1 < _2)(5, 3) == false);
    assert((_1 > _2)(5, 3) == true);
    assert((_1 <= _2)(3, 3) == true);
    assert((_1 <= _2)(3, 4) == true);
    assert((_1 >= _2)(4, 3) == true);
    assert((_1 >= _2)(3, 4) == false);
}

void test_logical() {
    assert((_1 && _2)(true, true) == true);
    assert((_1 && _2)(true, false) == false);
    assert((_1 || _2)(false, true) == true);
    assert((_1 || _2)(false, false) == false);
}

void test_bitwise() {
    assert((_1 & _2)(6, 3) == 2);  // 110 & 011 = 010
    assert((_1 | _2)(6, 3) == 7);  // 110 | 011 = 111
    assert((_1 ^ _2)(6, 3) == 5);  // 110 ^ 011 = 101
    assert((_1 << _2)(1, 3) == 8);
    assert((_1 >> _2)(16, 2) == 4);
}

void test_unary() {
    assert((!_1)(false) == true);
    assert((!_1)(true) == false);
    assert((-_1)(5) == -5);
    assert((+_1)(-3) == -3);
    assert((~_1)(0) == -1);
    assert((~_1)(1) == -2);
}

void test_chained() {
    auto expr = (_1 + _2) * _3;
    assert(expr(2, 3, 4) == 20);  // (2 + 3) * 4

    auto expr2 = _1 + _2 * _3;
    assert(expr2(2, 3, 4) == 14);  // 2 + (3 * 4)
}

void test_mixed_placeholders() {
    auto square = _1 * _1;
    assert(square(5) == 25);
    assert(square(10) == 100);

    auto sum_sq = _1 * _1 + _2 * _2;
    assert(sum_sq(3, 4) == 25);
}

void test_sort_with_lambda() {
    std::vector<int> v{3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    std::sort(v.begin(), v.end(), _1 > _2);
    assert(v[0] == 9);
    for (usize i = 1; i < v.size(); ++i) assert(v[i - 1] >= v[i]);

    std::sort(v.begin(), v.end(), _1 < _2);
    for (usize i = 1; i < v.size(); ++i) assert(v[i - 1] <= v[i]);
}

void test_transform() {
    std::vector<int> v{1, 2, 3, 4, 5};
    std::vector<int> out(v.size());
    std::transform(v.begin(), v.end(), out.begin(), _1 * 2);
    assert(out[0] == 2 && out[1] == 4 && out[2] == 6);
    assert(out[3] == 8 && out[4] == 10);
}

void test_predicate_composition() {
    auto is_even = _1 % 2 == 0;
    assert(is_even(4) == true);
    assert(is_even(5) == false);

    auto in_range = _1 >= 10 && _1 <= 20;
    assert(in_range(5) == false);
    assert(in_range(10) == true);
    assert(in_range(15) == true);
    assert(in_range(20) == true);
    assert(in_range(25) == false);
}

void test_spaceship() {
    auto cmp = _1 <=> _2;
    auto r = cmp(5, 3);
    assert(r > 0);
    r = cmp(3, 3);
    assert(r == 0);
    r = cmp(1, 4);
    assert(r < 0);
}

void test_multiple_args_with_unused() {
    auto expr = _2 + _5;
    assert(expr(0, 10, 0, 0, 20) == 30);
}

void test_nested_expression_types() {
    auto expr = ((_1 + _2) * (_3 - _1)) / _2;
    assert(expr(1, 2, 3) == 3);
}

void test_negative_numbers() {
    assert((_1 + _2)(-3, 5) == 2);
    assert((_1 - _2)(-3, -5) == 2);
    assert((_1 * _2)(-2, 3) == -6);
}

void test_double_values() {
    assert((_1 + _2)(2.5, 3.5) == 6.0);
    assert((_1 * _2)(2.0, 3.0) == 6.0);
    assert((_1 / _2)(7.0, 2.0) == 3.5);
    assert((_1 == _2)(3.5, 3.5) == true);
}

int main() {
    test_placeholder_basic();
    test_arithmetic();
    test_comparison();
    test_logical();
    test_bitwise();
    test_unary();
    test_chained();
    test_mixed_placeholders();
    test_sort_with_lambda();
    test_transform();
    test_predicate_composition();
    test_spaceship();
    test_multiple_args_with_unused();
    test_nested_expression_types();
    test_negative_numbers();
    test_double_values();

    std::cout << "All tests passed.\n";
    return 0;
}

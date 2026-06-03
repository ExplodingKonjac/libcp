#include <array>
#include <cassert>
#include <compare>
#include <concepts>
#include <iostream>
#include <utility>

#include "cp/bitset.hpp"

using cp::Bitset;
using cp::usize;

template <typename X, typename Y>
concept HasBitsetComparisons = requires(X&& x, Y&& y) {
    { std::forward<X>(x) == std::forward<Y>(y) } -> std::same_as<bool>;
    { std::forward<X>(x) != std::forward<Y>(y) } -> std::same_as<bool>;
    {
        std::forward<X>(x) <=> std::forward<Y>(y)
    } -> std::same_as<std::partial_ordering>;
};

using BitsetExpr80 = decltype(~std::declval<Bitset<80>&>());
using BinaryBitsetExpr80 =
    decltype(std::declval<Bitset<80>&>() | std::declval<Bitset<80>&>());

static_assert(cp::BitsetExpr<Bitset<80>&>);
static_assert(cp::BitsetExpr<BitsetExpr80>);
static_assert(cp::BitsetExpr<BinaryBitsetExpr80>);
static_assert(HasBitsetComparisons<Bitset<80>&, Bitset<80>&>);
static_assert(HasBitsetComparisons<Bitset<80>&, BitsetExpr80>);
static_assert(HasBitsetComparisons<BinaryBitsetExpr80, BinaryBitsetExpr80>);
static_assert(!HasBitsetComparisons<Bitset<80>&, Bitset<81>&>);

template <usize N>
using Bits = std::array<bool, N>;

template <usize N>
usize count_bits(const Bits<N>& bits) {
    usize result = 0;
    for (bool bit: bits)
        if (bit) ++result;
    return result;
}

template <usize N>
void assert_matches(const Bitset<N>& bitset, const Bits<N>& expected) {
    for (usize i = 0; i != N; ++i) {
        assert(bitset[i] == expected[i]);
        assert(bitset(i) == expected[i]);
    }
    assert(bitset.count() == count_bits(expected));
    assert(bitset.popcnt() == count_bits(expected));
    assert(bitset.any() == (count_bits(expected) != 0));
    assert(bitset.none() == (count_bits(expected) == 0));
    assert(bitset.all() == (count_bits(expected) == N));
    assert(!bitset[N]);
    assert(!bitset(N));
}

template <usize N>
Bits<N> shifted_left(const Bits<N>& bits, usize step) {
    Bits<N> result{};
    if (step >= N) return result;
    for (usize i = 0; i + step < N; ++i) result[i + step] = bits[i];
    return result;
}

template <usize N>
Bits<N> shifted_right(const Bits<N>& bits, usize step) {
    Bits<N> result{};
    if (step >= N) return result;
    for (usize i = step; i < N; ++i) result[i - step] = bits[i];
    return result;
}

template <usize N>
void set_range(Bits<N>& bits, usize position, usize length) {
    if (position + length > N || length == 0) return;
    for (usize i = position; i != position + length; ++i) bits[i] = true;
}

template <usize N>
void unset_range(Bits<N>& bits, usize position, usize length) {
    if (position + length > N || length == 0) return;
    for (usize i = position; i != position + length; ++i) bits[i] = false;
}

template <usize N>
void flip_range(Bits<N>& bits, usize position, usize length) {
    if (position + length > N || length == 0) return;
    for (usize i = position; i != position + length; ++i) bits[i] = !bits[i];
}

void test_default_and_single_bit_operations() {
    std::cout << "test_default_and_single_bit_operations... ";

    Bitset<130> bits;
    Bits<130> expected{};
    assert(bits.size() == 130);
    assert(bits.length() == 130);
    assert_matches(bits, expected);

    for (usize position: std::array<usize, 6>{0, 1, 63, 64, 65, 129}) {
        bits.set_bit(position);
        expected[position] = true;
    }
    bits.set_bit(130);
    assert_matches(bits, expected);

    bits.unset_bit(64);
    expected[64] = false;
    bits.unset_bit(130);
    assert_matches(bits, expected);

    bits.flip_bit(1);
    expected[1] = false;
    bits.flip_bit(128);
    expected[128] = true;
    bits.flip_bit(130);
    assert_matches(bits, expected);

    std::cout << "OK\n";
}

void test_all_operations_and_trimmed_tail() {
    std::cout << "test_all_operations_and_trimmed_tail... ";

    Bitset<130> bits;
    Bits<130> expected{};

    bits.set_all();
    expected.fill(true);
    assert_matches(bits, expected);
    assert(bits.find_first_unset(0) == usize(-1));

    bits.flip_all();
    expected.fill(false);
    assert_matches(bits, expected);

    bits.flip_all();
    expected.fill(true);
    assert_matches(bits, expected);

    bits.unset_all();
    expected.fill(false);
    assert_matches(bits, expected);
    assert(bits.find_first_set(0) == usize(-1));

    std::cout << "OK\n";
}

void test_range_operations() {
    std::cout << "test_range_operations... ";

    Bitset<320> bits;
    Bits<320> expected{};

    bits.set_range(3, 70);
    set_range(expected, 3, 70);
    assert_matches(bits, expected);

    bits.set_range(128, 160);
    set_range(expected, 128, 160);
    assert_matches(bits, expected);

    bits.unset_range(60, 90);
    unset_range(expected, 60, 90);
    assert_matches(bits, expected);

    bits.flip_range(124, 132);
    flip_range(expected, 124, 132);
    assert_matches(bits, expected);

    bits.set_range(256, 64);
    set_range(expected, 256, 64);
    assert_matches(bits, expected);

    bits.unset_range(0, 0);
    bits.set_range(319, 2);
    bits.flip_range(319, 2);
    assert_matches(bits, expected);

    std::cout << "OK\n";
}

void test_bitwise_expressions_and_assignments() {
    std::cout << "test_bitwise_expressions_and_assignments... ";

    Bitset<260> a;
    Bitset<260> b;
    Bits<260> ae{};
    Bits<260> be{};

    for (usize i = 0; i < 260; i += 3) {
        a.set_bit(i);
        ae[i] = true;
    }
    for (usize i = 1; i < 260; i += 5) {
        b.set_bit(i);
        be[i] = true;
    }
    b.set_bit(259);
    be[259] = true;

    Bitset<260> c = a & b;
    Bits<260> ce{};
    for (usize i = 0; i < 260; ++i) ce[i] = ae[i] && be[i];
    assert_matches(c, ce);

    c = a | b;
    for (usize i = 0; i < 260; ++i) ce[i] = ae[i] || be[i];
    assert_matches(c, ce);

    c = a ^ b;
    for (usize i = 0; i < 260; ++i) ce[i] = ae[i] != be[i];
    assert_matches(c, ce);

    c = a - b;
    for (usize i = 0; i < 260; ++i) ce[i] = ae[i] && !be[i];
    assert_matches(c, ce);

    c = ~a;
    for (usize i = 0; i < 260; ++i) ce[i] = !ae[i];
    assert_matches(c, ce);

    c = (a | b) - (a & b);
    for (usize i = 0; i < 260; ++i) ce[i] = ae[i] != be[i];
    assert_matches(c, ce);

    Bitset<260> d = a;
    Bits<260> de = ae;
    d &= b;
    for (usize i = 0; i < 260; ++i) de[i] = de[i] && be[i];
    assert_matches(d, de);

    d = a;
    de = ae;
    d |= b;
    for (usize i = 0; i < 260; ++i) de[i] = de[i] || be[i];
    assert_matches(d, de);

    d = a;
    de = ae;
    d ^= b;
    for (usize i = 0; i < 260; ++i) de[i] = de[i] != be[i];
    assert_matches(d, de);

    d = a;
    de = ae;
    d -= b;
    for (usize i = 0; i < 260; ++i) de[i] = de[i] && !be[i];
    assert_matches(d, de);

    std::cout << "OK\n";
}

void test_find_first_operations() {
    std::cout << "test_find_first_operations... ";

    Bitset<260> bits;
    assert(bits.find_first_set(0) == usize(-1));
    assert(bits.find_first_unset(0) == 0);

    bits.set_bit(0);
    bits.set_bit(64);
    bits.set_bit(129);
    bits.set_bit(255);
    bits.set_bit(259);

    assert(bits.find_first_set(0) == 0);
    assert(bits.find_first_set(1) == 64);
    assert(bits.find_first_set(65) == 129);
    assert(bits.find_first_set(130) == 255);
    assert(bits.find_first_set(256) == 259);
    assert(bits.find_first_set(260) == usize(-1));

    bits.set_all();
    bits.unset_bit(0);
    bits.unset_bit(64);
    bits.unset_bit(129);
    bits.unset_bit(255);
    bits.unset_bit(259);

    assert(bits.find_first_unset(0) == 0);
    assert(bits.find_first_unset(1) == 64);
    assert(bits.find_first_unset(65) == 129);
    assert(bits.find_first_unset(130) == 255);
    assert(bits.find_first_unset(256) == 259);
    assert(bits.find_first_unset(260) == usize(-1));

    std::cout << "OK\n";
}

void test_shifts() {
    std::cout << "test_shifts... ";

    Bitset<260> bits;
    Bits<260> expected{};
    for (
        usize position:
        std::array<usize, 9>{0, 1, 63, 64, 127, 128, 191, 255, 259}
    ) {
        bits.set_bit(position);
        expected[position] = true;
    }

    for (
        usize step:
        std::array<usize, 10>{0, 1, 7, 63, 64, 65, 128, 255, 260, 400}
    ) {
        Bitset<260> left = bits << step;
        assert_matches(left, shifted_left(expected, step));

        Bitset<260> right = bits >> step;
        assert_matches(right, shifted_right(expected, step));

        Bitset<260> inplace_left = bits;
        inplace_left <<= step;
        assert_matches(inplace_left, shifted_left(expected, step));

        Bitset<260> inplace_right = bits;
        inplace_right >>= step;
        assert_matches(inplace_right, shifted_right(expected, step));
    }

    std::cout << "OK\n";
}

void test_comparisons() {
    std::cout << "test_comparisons... ";

    Bitset<80> empty;
    Bitset<80> subset;
    Bitset<80> superset;
    Bitset<80> overlap;

    subset.set_bit(3);
    subset.set_bit(64);
    superset = subset;
    superset.set_bit(79);
    overlap.set_bit(3);
    overlap.set_bit(10);

    assert(empty == Bitset<80>{});
    assert(subset != superset);
    assert(empty < subset);
    assert(subset <= superset);
    assert(superset > subset);
    assert(superset >= subset);
    assert((subset <=> superset) == std::partial_ordering::less);
    assert((superset <=> subset) == std::partial_ordering::greater);
    assert((subset <=> subset) == std::partial_ordering::equivalent);
    assert((superset <=> overlap) == std::partial_ordering::unordered);

    Bitset<80> intersection = subset & overlap;
    Bitset<80> union_bits = subset | overlap;
    Bitset<80> left_only = subset - overlap;
    Bitset<80> right_only = overlap - subset;

    assert(union_bits == (subset | overlap));
    assert((subset | overlap) == union_bits);
    assert((subset | overlap) == (overlap | subset));
    assert((subset & overlap) == intersection);
    assert((subset & overlap) != (subset ^ overlap));
    assert(union_bits != (subset & overlap));

    assert((subset & overlap) < subset);
    assert(((subset & overlap) <=> subset) == std::partial_ordering::less);
    assert(((subset | overlap) <=> subset) == std::partial_ordering::greater);
    assert(
        ((subset | overlap) <=> (overlap | subset))
        == std::partial_ordering::equivalent
    );
    assert(
        ((subset - overlap) <=> (overlap - subset))
        == std::partial_ordering::unordered
    );
    assert(left_only != right_only);

    std::cout << "OK\n";
}

int main() {
    test_default_and_single_bit_operations();
    test_all_operations_and_trimmed_tail();
    test_range_operations();
    test_bitwise_expressions_and_assignments();
    test_find_first_operations();
    test_shifts();
    test_comparisons();
    std::cout << "All bitset tests passed!\n";
    return 0;
}

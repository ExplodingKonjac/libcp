#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "cp/suffix_array.hpp"

using namespace cp;

template <typename T>
std::vector<usize> expected_sa(const std::vector<T>& text) {
    std::vector<usize> sa(text.size());
    std::iota(sa.begin(), sa.end(), 0);
    std::ranges::sort(sa, [&](usize x, usize y) {
        return std::ranges::lexicographical_compare(
            text.begin() + x, text.end(), text.begin() + y, text.end()
        );
    });
    return sa;
}

template <typename T>
usize lcp(const std::vector<T>& text, usize x, usize y) {
    usize result = 0;
    while (
        x + result < text.size() &&
        y + result < text.size() &&
        text[x + result] == text[y + result]
    ) {
        ++result;
    }
    return result;
}

template <typename R>
void assert_matches_oracle(const R& range) {
    std::vector<std::ranges::range_value_t<R>> text(range.begin(), range.end());
    SuffixArray suffix_array(range);
    auto sa = expected_sa(text);

    for (usize rank = 0; rank < text.size(); ++rank) {
        assert(suffix_array.sa(rank) == sa[rank]);
        assert(suffix_array.rk(sa[rank]) == rank);
        assert(
            suffix_array.height(rank) ==
            (rank == 0 ? 0 : lcp(text, sa[rank - 1], sa[rank]))
        );
    }
}

void test_empty() {
    std::cout << "test_empty... ";
    std::vector<int> text;
    assert_matches_oracle(text);
    std::cout << "OK\n";
}

void test_known_string() {
    std::cout << "test_known_string... ";
    std::string text = "banana";
    SuffixArray suffix_array(text);

    assert(suffix_array.sa(0) == 5);
    assert(suffix_array.sa(1) == 3);
    assert(suffix_array.sa(2) == 1);
    assert(suffix_array.sa(3) == 0);
    assert(suffix_array.sa(4) == 4);
    assert(suffix_array.sa(5) == 2);

    assert(suffix_array.rk(0) == 3);
    assert(suffix_array.rk(1) == 2);
    assert(suffix_array.rk(2) == 5);
    assert(suffix_array.rk(3) == 1);
    assert(suffix_array.rk(4) == 4);
    assert(suffix_array.rk(5) == 0);

    assert(suffix_array.height(0) == 0);
    assert(suffix_array.height(1) == 1);
    assert(suffix_array.height(2) == 3);
    assert(suffix_array.height(3) == 0);
    assert(suffix_array.height(4) == 0);
    assert(suffix_array.height(5) == 2);
    std::cout << "OK\n";
}

void test_small_inputs() {
    std::cout << "test_small_inputs... ";
    assert_matches_oracle(std::vector<int>{42});
    assert_matches_oracle(std::vector<int>{7, 7, 7, 7});
    assert_matches_oracle(std::vector<int>{3, 1, 3, 1, 3});
    assert_matches_oracle(std::vector<int>{4, 2, 1, 3, 0});
    assert_matches_oracle(std::string{"abacaba"});
    std::cout << "OK\n";
}

void test_exhaustive_binary_strings() {
    std::cout << "test_exhaustive_binary_strings... ";
    for (usize length = 1; length <= 7; ++length) {
        for (usize mask = 0; mask < (usize{1} << length); ++mask) {
            std::string text(length, 'a');
            for (usize i = 0; i < length; ++i) {
                if (mask & (usize{1} << i)) text[i] = 'b';
            }
            assert_matches_oracle(text);
        }
    }
    std::cout << "OK\n";
}

int main() {
    test_empty();
    test_known_string();
    test_small_inputs();
    test_exhaustive_binary_strings();
    std::cout << "All tests passed!\n";
}

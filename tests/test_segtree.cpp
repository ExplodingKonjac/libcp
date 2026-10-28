#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "cp/segtree.hpp"

using namespace cp;

void test_basic_sum() {
    usize n = 10;
    SegTree seg(n, 0, std::plus<int>{});
    for (int i = 0; i < static_cast<int>(n); ++i) seg.modify(i, i + 1);

    for (int l = 0; l < static_cast<int>(n); ++l) {
        for (int r = l + 1; r <= static_cast<int>(n); ++r) {
            int expected = 0;
            for (int k = l; k < r; ++k) expected += (k + 1);
            auto actual = seg.query(l, r);
            if (!actual || *actual != expected) {
                std::cerr
                    << "test_basic_sum failed at query("
                    << l
                    << ", "
                    << r
                    << "): expected "
                    << expected
                    << ", got "
                    << (actual ? std::to_string(*actual) : "nullopt")
                    << "\n";
                exit(1);
            }
        }
    }
}

void test_modify_updates() {
    SegTree seg(5, 0, std::plus<int>{});
    for (int i = 0; i < 5; ++i) seg.modify(i, 1);
    if (seg.query(0, 5) != 5) {
        std::cerr << "test_modify_updates failed: sum should be 5\n";
        exit(1);
    }
    seg.modify(2, 10);
    if (seg.query(0, 5) != 14) {
        std::cerr
            << "test_modify_updates failed after modify(2, 10): expected 14, "
               "got "
            << *seg.query(0, 5)
            << "\n";
        exit(1);
    }
    seg.modify(0, 100);
    if (seg.query(0, 5) != 113) {
        std::cerr
            << "test_modify_updates failed after modify(0, 100): expected 113, "
               "got "
            << *seg.query(0, 5)
            << "\n";
        exit(1);
    }
}

void test_single_element() {
    SegTree seg(1, 0, std::plus<int>{});
    seg.modify(0, 42);
    if (seg.query(0, 1) != 42) {
        std::cerr << "test_single_element failed\n";
        exit(1);
    }
}

void test_mul_semigroup() {
    SegTree seg(5, 1, std::multiplies<int>{});
    for (int i = 0; i < 5; ++i) seg.modify(i, i + 1);
    if (seg.query(0, 5) != 120) {
        std::cerr
            << "test_mul_semigroup failed: expected 120, got "
            << *seg.query(0, 5)
            << "\n";
        exit(1);
    }
    if (seg.query(1, 4) != 24) {
        std::cerr
            << "test_mul_semigroup failed at query(1,4): expected 24, got "
            << *seg.query(1, 4)
            << "\n";
        exit(1);
    }
}

void test_min_semigroup() {
    SegTree seg(5, 100, [](int a, int b) { return std::min(a, b); });
    seg.modify(0, 10);
    seg.modify(1, 3);
    seg.modify(2, 7);
    seg.modify(3, 1);
    seg.modify(4, 9);

    if (seg.query(0, 5) != 1) {
        std::cerr
            << "test_min_semigroup failed: expected 1, got "
            << *seg.query(0, 5)
            << "\n";
        exit(1);
    }
    if (seg.query(0, 2) != 3) {
        std::cerr
            << "test_min_semigroup failed at query(0,2): expected 3, got "
            << *seg.query(0, 2)
            << "\n";
        exit(1);
    }
    if (seg.query(3, 5) != 1) {
        std::cerr
            << "test_min_semigroup failed at query(3,5): expected 1, got "
            << *seg.query(3, 5)
            << "\n";
        exit(1);
    }
}

void test_max_semigroup() {
    SegTree seg(5, -1, [](int a, int b) { return std::max(a, b); });
    seg.modify(0, 10);
    seg.modify(1, 3);
    seg.modify(2, 7);
    seg.modify(3, 1);
    seg.modify(4, 9);

    if (seg.query(0, 5) != 10) {
        std::cerr
            << "test_max_semigroup failed: expected 10, got "
            << *seg.query(0, 5)
            << "\n";
        exit(1);
    }
    if (seg.query(1, 4) != 7) {
        std::cerr
            << "test_max_semigroup failed at query(1,4): expected 7, got "
            << *seg.query(1, 4)
            << "\n";
        exit(1);
    }
    if (seg.query(4, 5) != 9) {
        std::cerr
            << "test_max_semigroup failed at query(4,5): expected 9, got "
            << *seg.query(4, 5)
            << "\n";
        exit(1);
    }
}

void test_large_n() {
    usize n = 100000;
    SegTree seg(n, 0, std::plus<int>{});
    for (usize i = 0; i < n; ++i) seg.modify(i, static_cast<int>(i % 100));

    int expected = 0;
    for (usize i = 0; i < n; ++i) expected += static_cast<int>(i % 100);
    auto actual = seg.query(0, n);
    if (!actual || *actual != expected) {
        std::cerr
            << "test_large_n failed: expected "
            << expected
            << ", got "
            << (actual ? std::to_string(*actual) : "nullopt")
            << "\n";
        exit(1);
    }
}

void test_power_of_two_boundary() {
    for (usize n: {usize(8), usize(9)}) {
        SegTree seg(n, 0, std::plus<int>{});
        for (usize i = 0; i < n; ++i) seg.modify(i, 1);
        if (seg.query(0, n) != static_cast<int>(n)) {
            std::cerr
                << "test_power_of_two_boundary failed for n="
                << n
                << "\n";
            exit(1);
        }
    }
}

void test_all() {
    SegTree seg(5, 0, std::plus<int>{});
    for (int i = 0; i < 5; ++i) seg.modify(i, i + 1);
    if (seg.all() != 15) {
        std::cerr << "test_all failed: expected 15, got " << seg.all() << "\n";
        exit(1);
    }
}

void test_query_returns_nullopt_on_invalid_range() {
    SegTree seg(5, 0, std::plus<int>{});
    if (seg.query(3, 3).has_value()) {
        std::cerr << "test_query_returns_nullopt_on_empty_interval: expected "
                     "nullopt\n";
        exit(1);
    }
    if (seg.query(0, 10).has_value()) {
        std::cerr
            << "test_query_returns_nullopt_on_out_of_range: expected nullopt\n";
        exit(1);
    }
}

void test_modify_returns_false_on_out_of_range() {
    SegTree seg(5, 0, std::plus<int>{});
    if (seg.modify(10, 1)) {
        std::cerr
            << "test_modify_returns_false_on_out_of_range: expected false\n";
        exit(1);
    }
}

void test_randomized_large_sum_queries() {
    constexpr usize n = 4096;
    constexpr int operations = 20000;
    std::mt19937 rng(20260527);
    std::uniform_int_distribution<int> value_dist(-1000, 1000);
    std::uniform_int_distribution<usize> index_dist(0, n - 1);
    std::bernoulli_distribution update_dist(0.45);

    std::vector<int> values(n);
    for (auto& value: values) value = value_dist(rng);

    SegTree seg(n, [&](usize i) { return values[i]; }, std::plus<int>{});

    for (int step = 0; step < operations; ++step) {
        if (update_dist(rng)) {
            usize index = index_dist(rng);
            int value = value_dist(rng);
            values[index] = value;
            if (!seg.modify(index, value)) {
                std::cerr
                    << "test_randomized_large_sum_queries failed: modify("
                    << index
                    << ", "
                    << value
                    << ") returned false at step "
                    << step
                    << "\n";
                exit(1);
            }
            continue;
        }

        usize l = index_dist(rng);
        usize r = index_dist(rng);
        if (l > r) std::swap(l, r);
        ++r;

        int expected = std::accumulate(
            values.begin() + static_cast<std::ptrdiff_t>(l),
            values.begin() + static_cast<std::ptrdiff_t>(r), 0
        );
        auto actual = seg.query(l, r);
        if (!actual || *actual != expected) {
            std::cerr
                << "test_randomized_large_sum_queries failed at step "
                << step
                << " for query("
                << l
                << ", "
                << r
                << "): expected "
                << expected
                << ", got "
                << (actual ? std::to_string(*actual) : "nullopt")
                << "\n";
            exit(1);
        }
    }

    int total_expected = std::accumulate(values.begin(), values.end(), 0);
    auto total_actual = seg.query(0, n);
    if (!total_actual
        || *total_actual
        != total_expected
        || seg.all()
        != total_expected) {
        std::cerr
            << "test_randomized_large_sum_queries final check failed: expected "
            << total_expected
            << ", got query="
            << (total_actual ? std::to_string(*total_actual) : "nullopt")
            << ", all="
            << seg.all()
            << "\n";
        exit(1);
    }
}

void test_randomized_large_min_queries() {
    constexpr usize n = 2048;
    constexpr int operations = 12000;
    std::mt19937 rng(31415926);
    std::uniform_int_distribution<int> value_dist(-5000, 5000);
    std::uniform_int_distribution<usize> index_dist(0, n - 1);
    std::bernoulli_distribution update_dist(0.4);

    std::vector<int> values(n);
    for (auto& value: values) value = value_dist(rng);

    auto min_op = [](int a, int b) { return std::min(a, b); };
    SegTree seg(n, [&](usize i) { return values[i]; }, min_op);

    for (int step = 0; step < operations; ++step) {
        if (update_dist(rng)) {
            usize index = index_dist(rng);
            int value = value_dist(rng);
            values[index] = value;
            if (!seg.modify(index, value)) {
                std::cerr
                    << "test_randomized_large_min_queries failed: modify("
                    << index
                    << ", "
                    << value
                    << ") returned false at step "
                    << step
                    << "\n";
                exit(1);
            }
            continue;
        }

        usize l = index_dist(rng);
        usize r = index_dist(rng);
        if (l > r) std::swap(l, r);
        ++r;

        int expected = *std::min_element(
            values.begin() + static_cast<std::ptrdiff_t>(l),
            values.begin() + static_cast<std::ptrdiff_t>(r)
        );
        auto actual = seg.query(l, r);
        if (!actual || *actual != expected) {
            std::cerr
                << "test_randomized_large_min_queries failed at step "
                << step
                << " for query("
                << l
                << ", "
                << r
                << "): expected "
                << expected
                << ", got "
                << (actual ? std::to_string(*actual) : "nullopt")
                << "\n";
            exit(1);
        }
    }
}

int main() {
    test_basic_sum();
    test_modify_updates();
    test_single_element();
    test_mul_semigroup();
    test_min_semigroup();
    test_max_semigroup();
    test_large_n();
    test_power_of_two_boundary();
    test_all();
    test_query_returns_nullopt_on_invalid_range();
    test_modify_returns_false_on_out_of_range();
    test_randomized_large_sum_queries();
    test_randomized_large_min_queries();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
